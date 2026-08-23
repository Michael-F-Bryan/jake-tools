"""Write this run's products into the tier-owned sections of an Obsidian note.

This is the only stage that writes the note, and the note is co-owned by a
human. The note has three ownership tiers (decisions [E15], final):

- **pipeline** (``## Chapters``, ``## Transcript``): replaced wholesale
  every run.
- **pipeline-seeded, human-editable** (the ``> [!summary]`` callout,
  ``## Discussion Notes``): a three-way merge against a cached baseline -
  never delete or rewrite content a human may have edited; when in doubt,
  add alongside rather than modify.
- **human** (frontmatter, ``## Meeting Prep``): untouched here.

The merge invariant for the middle tier is absolute, and its enforcement
mechanism is the point of this module, not an implementation detail:

1. Every write is a **three-way textual comparison** (current section
   content vs. the baseline the pipeline last wrote vs. this run's fresh
   content) - never an LLM. A unit whose current text still matches its
   baseline is "human untouched" and may be replaced; a unit that no
   longer matches (edited or newly added by a human) is preserved
   verbatim, and genuinely new pipeline content is appended after it,
   never interleaved into it. With no baseline at all (an evicted cache,
   or a different machine) every existing unit is presumed human-owned:
   append-only, nothing modified.
2. **Frontmatter is preserved as raw text, never re-serialised.** Round-
   tripping it through ``yaml.dump`` risks reordering keys or reformatting
   values a human wrote by hand - :func:`~.note.split_raw_frontmatter`
   keeps the original bytes and this module never touches them.
3. **A pre-write reconstruction assertion.** The section model's
   byte-for-byte reconstruction property (``note.py``) is what makes a
   surgical write to one section safe for every other section - this
   module asserts that property holds for the file being written *before*
   doing any surgery, and again that the untouched ``## Meeting Prep``
   section is byte-identical *after* surgery, before the file is ever
   opened for writing. Either check failing raises without writing a
   single byte: a clobbered human edit cannot be re-run, so this stage
   would rather fail loudly than guess.

Everything else in this pipeline can be re-run if it goes wrong. This
cannot, which is why the mechanism above is built exactly as specified
rather than approximated.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .cache import RunCache
from .minutes import MINUTES_CACHE_NAME, MinutesResult
from .models import PolishedChapter, TranscriptProducts
from .note import (
    MEETING_PREP_HEADING,
    NoteSection,
    parse_note,
    render_body,
    split_raw_frontmatter,
)
from .polish import POLISHED_CACHE_NAME, PolishedChapterList

# Baseline names under `RunCache.store_text`/`load_text` - these land on
# disk as `<name>.txt` (`RunCache.store_text` appends `.txt` itself), never
# `.md` despite the plan's cosmetic phrasing; the binding thing is the
# three-way mechanism, not the extension.
BASELINE_SUMMARY_NAME = "baseline_summary"
BASELINE_NOTES_NAME = "baseline_notes"

_DISCUSSION_NOTES_HEADING = "discussion notes"
_CHAPTERS_HEADING = "chapters"
_TRANSCRIPT_HEADING = "transcript"
_SUMMARY_LABEL = "summary callout"  # report label; not a real heading

_TOP_LEVEL_BULLET_RE = re.compile(r"^[-*]\s+")

# Matches this module's own `### <ts> — <title>` chapter headings
# (`format_timestamp` output, an em-dash, then the title) - unambiguous
# pipeline output wherever it appears, per the legacy-shape sweep ruling.
_CHAPTER_HEADING_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})? — ")


class IntegrateError(RuntimeError):
    """Base for integrate domain errors."""


class MissingPolishedChaptersError(IntegrateError):
    """Raised when no `polished.json` is cached for a run id."""

    def __init__(self, run_id: str) -> None:
        super().__init__(
            f"no cached polished chapters for run {run_id!r}; run "
            f"`transcript polish --run-id {run_id} ...` first."
        )
        self.run_id = run_id


class MissingMinutesError(IntegrateError):
    """Raised when no `minutes.json` is cached for a run id."""

    def __init__(self, run_id: str) -> None:
        super().__init__(
            f"no cached minutes for run {run_id!r}; run "
            f"`transcript minutes --run-id {run_id} ...` first."
        )
        self.run_id = run_id


class ReconstructionError(IntegrateError):
    """Raised when the section model can't reconstruct the note byte-for-byte.

    This is the abort-without-writing guard: every surgical write in this
    module depends on `note.py`'s reconstruction property (untouched
    sections round-trip through `render_body` unchanged) actually holding
    for the file at hand. If it doesn't - a corrupted section boundary, or
    any other break in that invariant - `integrate` raises this *before*
    touching the file on disk, rather than risk writing something that
    silently mangles a human-owned region.
    """


# --- Step 1: rendering (pure, no I/O) -----------------------------------------


def format_timestamp(seconds: float) -> str:
    """Render `seconds` as `MM:SS` (zero-padded) below one hour, `HH:MM:SS` at/above.

    Exemplars: `3` -> `"00:03"`, `415` -> `"06:55"`, `3762` -> `"01:02:42"`.
    """
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _unique_labels(chapters: Sequence[PolishedChapter]) -> list[str]:
    """`"<ts> — <title>"` per chapter, with a `" (N)"` suffix on repeats.

    Obsidian heading links (`[[#label]]`) need uniqueness within a note;
    two chapters can only collide on this exact label if they share both
    the same timestamp and the same title, but the pipeline never assumes
    that can't happen.
    """
    seen: dict[str, int] = {}
    labels: list[str] = []
    for chapter in chapters:
        base = f"{format_timestamp(chapter.start_seconds)} — {chapter.title}"
        seen[base] = seen.get(base, 0) + 1
        labels.append(base if seen[base] == 1 else f"{base} ({seen[base]})")
    return labels


def render_summary_callout(text: str) -> str:
    """`> [!summary]` followed by `> `-prefixed lines of `text`.

    A blank line inside `text` still gets a bare `>` (no trailing space) so
    it stays part of the callout - a genuinely blank line (no `>` at all)
    would end an Obsidian blockquote early.
    """
    lines = text.splitlines() if text else []
    rendered = ["> [!summary]"]
    rendered.extend(f"> {line}" if line else ">" for line in lines)
    return "\n".join(rendered) + "\n"


def render_chapter_index(chapters: Sequence[PolishedChapter]) -> str:
    """One `- [[#<ts> — <title>]]` bullet per chapter, in order."""
    labels = _unique_labels(chapters)
    bullets = "".join(f"- [[#{label}]]\n" for label in labels)
    return f"\n{bullets}\n"


def render_transcript(chapters: Sequence[PolishedChapter]) -> str:
    """The full `## Transcript` body: per chapter, a heading, summary callout, and turns.

    Each chapter renders as `### <ts> — <title>`, then its `> [!summary]`
    callout, then `**Name:** text` turns separated by blank lines
    (`**Unknown:**` for an unresolved speaker - just another `turn.speaker`
    value, nothing special-cased here).
    """
    labels = _unique_labels(chapters)
    blocks: list[str] = []
    for label, chapter in zip(labels, chapters, strict=True):
        turns = "\n\n".join(
            f"**{turn.speaker}:** {turn.text}" for turn in chapter.turns
        )
        callout = render_summary_callout(chapter.summary)
        blocks.append(f"### {label}\n\n{callout}\n{turns}\n")
    return "\n" + "\n".join(blocks)


# --- Step 2: the three-way merge for tier-b sections ---------------------------


def _normalize(unit: str) -> str:
    """Collapse all whitespace to single spaces for a textual, not semantic, compare."""
    return " ".join(unit.split())


def _split_summary_unit(text: str) -> list[str]:
    stripped = text.strip()
    return [stripped] if stripped else []


def _join_summary_units(units: Sequence[str]) -> str:
    return "\n\n".join(units)


def _split_discussion_units(text: str) -> list[str]:
    """One unit per top-level bullet, each carrying its indented sub-bullets."""
    stripped = text.strip("\n")
    if not stripped.strip():
        return []
    units: list[str] = []
    for line in stripped.splitlines():
        if _TOP_LEVEL_BULLET_RE.match(line) or not units:
            units.append(line + "\n")
        else:
            units[-1] += line + "\n"
    return units


def _join_discussion_units(units: Sequence[str]) -> str:
    return "".join(units)


Mode = Literal["created", "replaced", "first_run", "merged", "degraded_append_only"]


class SectionOutcome(BaseModel):
    """What happened to one tier-owned section/region during one `integrate` call."""

    heading: str
    mode: Mode
    preserved_units: int = 0
    appended_units: int = 0
    deleted_units: int = 0


class IntegrationReport(BaseModel):
    """What `integrate` did, for the run report and the CLI's JSON output."""

    run_id: str
    note_path: str
    sections: list[SectionOutcome]


def _merge_tier_b(
    label: str,
    *,
    current_text: str,
    baseline_text: str | None,
    new_text: str,
    split: Callable[[str], list[str]],
    join: Callable[[Sequence[str]], str],
) -> tuple[str, str, SectionOutcome]:
    """The three-way merge, generic over how a section splits into logical units.

    Shared by the summary callout (one unit: the whole text) and Discussion
    Notes (one unit per top-level bullet, sub-bullets included) - the
    granularity lives entirely in `split`/`join`, the merge logic itself is
    identical for both per the plan.

    **Deletions are a human edit too.** A unit present in the baseline but
    absent (by normalised comparison) from the current section is treated
    as human-deleted, not as "the pipeline just hasn't written it back
    yet": a fresh pipeline unit whose normalised text matches a deleted
    baseline unit is never appended, so the pipeline can't silently
    resurrect something a human removed. This is "preserve on doubt"
    applied to absence as well as presence - the accepted cost is that an
    accidentally-deleted bullet stays gone until a human manually retypes
    it; the pipeline will never offer it back on its own. The deleted
    unit's normalised fingerprint is carried forward into the new baseline
    (alongside genuinely new appended content) so this suppression
    persists across future runs too, not just the one where the deletion
    was first noticed.

    Returns `(merged_body, new_baseline, outcome)`. `new_baseline` is the
    pipeline-owned part of the merge (newly appended units) plus any
    baseline units still being suppressed as deleted - never preserved
    human content, which would make this module mistake human edits for
    its own on a later run.
    """
    current_units = split(current_text)
    new_units = split(new_text)

    if baseline_text is None:
        if not current_units:
            merged = join(new_units)
            return (
                merged,
                merged,
                SectionOutcome(heading=label, mode="first_run"),
            )
        # Degraded append-only mode: no baseline (evicted cache, or a run
        # on a different machine) but the section already has content -
        # presume every existing unit is human-owned. Add only units that
        # are clearly new; never modify what's there. There is no baseline
        # to detect a deletion against, so deletion-suppression doesn't
        # apply here - there's nothing to remember was ever deleted.
        preserved = current_units
        preserved_norms = {_normalize(unit) for unit in preserved}
        appended = [
            unit for unit in new_units if _normalize(unit) not in preserved_norms
        ]
        return (
            join(preserved + appended),
            join(appended),
            SectionOutcome(
                heading=label,
                mode="degraded_append_only",
                preserved_units=len(preserved),
                appended_units=len(appended),
            ),
        )

    baseline_units = split(baseline_text)
    baseline_norms = {_normalize(unit) for unit in baseline_units}
    current_norms = {_normalize(unit) for unit in current_units}

    preserved = [
        unit for unit in current_units if _normalize(unit) not in baseline_norms
    ]
    preserved_norms = {_normalize(unit) for unit in preserved}

    # A baseline unit no longer present (by normalised text) among the
    # current units was removed by a human - deduped by normalised text so
    # a baseline with an accidental literal duplicate doesn't double-count.
    deleted_norms = baseline_norms - current_norms
    deleted_units: list[str] = []
    seen_deleted_norms: set[str] = set()
    for unit in baseline_units:
        norm = _normalize(unit)
        if norm in deleted_norms and norm not in seen_deleted_norms:
            seen_deleted_norms.add(norm)
            deleted_units.append(unit)

    excluded_norms = preserved_norms | deleted_norms
    appended = [unit for unit in new_units if _normalize(unit) not in excluded_norms]

    mode: Mode = "merged" if preserved else "replaced"
    return (
        join(preserved + appended),
        join(deleted_units + appended),
        SectionOutcome(
            heading=label,
            mode=mode,
            preserved_units=len(preserved),
            appended_units=len(appended),
            deleted_units=len(deleted_units),
        ),
    )


# --- Step 2 (continued): locating sections in the model -------------------------


def _find_heading(sections: Sequence[NoteSection], name: str) -> int | None:
    return next(
        (
            index
            for index, section in enumerate(sections)
            if section.heading is not None and section.heading.strip().lower() == name
        ),
        None,
    )


def _is_chapter_heading(heading: str | None) -> bool:
    """Whether `heading` is unambiguously one of this module's own chapter headings.

    `render_transcript` only ever produces `### <ts> — <title>` headings
    (`format_timestamp` output, an em-dash, the title) - no human-authored
    heading looks like this by convention, so a match is treated as
    pipeline output wherever it appears, per the ruling on legacy shapes.
    """
    return heading is not None and _CHAPTER_HEADING_RE.match(heading) is not None


def _find_transcript_sweep_indices(sections: Sequence[NoteSection]) -> list[int]:
    """Every existing section the tier-a Transcript replacement must remove.

    Two shapes occur in practice, and both must be fully swept, not just
    the first contiguous run:

    - This module's own prior output: a `## Transcript` heading with its
      `### <ts> — <title>` chapter sections immediately following it (on a
      re-run, `## Transcript`'s own body is empty - `parse_note` splits
      each chapter heading it contains into its own section).
    - The **legacy** shape: `### <ts> — <title>` chapter sections sitting
      directly after `## Chapters` with **no** `## Transcript` heading at
      all - the real shape of existing vault notes that predate this
      stage. Re-running on one of these must not orphan the old chapters
      or duplicate their content.

    A `### <ts> — <title>` heading is unambiguous pipeline output wherever
    it occurs in the document - contiguous with `## Transcript`/`##
    Chapters` or not (an unrelated section can interrupt the run without
    stopping the sweep) - so every matching section is collected,
    regardless of position. A human section's heading never matches this
    shape (or `## Transcript` itself) and is never included here, so it's
    never touched by the replacement that follows.

    Returns the matching indices in ascending document order (possibly
    empty, if this is a first run with neither shape present).
    """
    return [
        index
        for index, section in enumerate(sections)
        if (
            section.heading is not None
            and section.heading.strip().lower() == _TRANSCRIPT_HEADING
        )
        or _is_chapter_heading(section.heading)
    ]


def _split_summary_callout(body: str) -> tuple[str, str, str]:
    """Split a preamble body into `(before, callout_block, after)`.

    `callout_block` is the `> [!summary]` line plus every immediately
    following line that still starts with `>` (an Obsidian blockquote ends
    at the first non-`>` line, blank or not). `before`/`after` are
    everything else, byte-exact, so reattaching them around a new callout
    never disturbs other preamble content (an ops-log note's opening
    bullet and audio embeds, for instance).
    """
    lines = body.splitlines(keepends=True)
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.lstrip().startswith(">") and "[!summary]" in line.lower()
        ),
        None,
    )
    if start is None:
        return body, "", ""
    end = start + 1
    while end < len(lines) and lines[end].lstrip().startswith(">"):
        end += 1
    return "".join(lines[:start]), "".join(lines[start:end]), "".join(lines[end:])


def _callout_content(callout_block: str) -> str:
    """The plain text inside a `> [!summary]` callout block (no `>` prefixes)."""
    if not callout_block.strip():
        return ""
    lines = callout_block.splitlines()[1:]  # drop "> [!summary]" itself
    body_lines: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("> "):
            body_lines.append(stripped[2:])
        elif stripped.startswith(">"):
            body_lines.append(stripped[1:])
        else:
            body_lines.append(stripped)
    return "\n".join(body_lines).strip("\n")


# --- Step 3: run orchestration --------------------------------------------------


def load_products(note_path: Path, run_id: str, cache: RunCache) -> TranscriptProducts:
    """Assemble `TranscriptProducts` from this run's cached polished chapters and minutes.

    Errors name the exact prerequisite command to run, matching every
    other stage's `Missing*Error` convention.
    """
    polished = cache.load(run_id, POLISHED_CACHE_NAME, PolishedChapterList)
    if polished is None:
        raise MissingPolishedChaptersError(run_id)

    minutes_result = cache.load(run_id, MINUTES_CACHE_NAME, MinutesResult)
    if minutes_result is None:
        raise MissingMinutesError(run_id)

    note = parse_note(note_path)
    return TranscriptProducts(
        context=note.context,
        meeting_summary=minutes_result.meeting_summary,
        discussion_notes=minutes_result.discussion_notes,
        chapters=polished.chapters,
    )


def run_integrate(
    note_path: Path,
    products: TranscriptProducts,
    *,
    cache: RunCache,
    run_id: str,
) -> IntegrationReport:
    """Write `products` into `note_path`'s tier-owned sections and return a report.

    1. Parses the note, and independently re-derives its body from the raw
       file text via `split_raw_frontmatter` - if these two disagree
       (`render_body(sections) != raw_body`), the section model can't be
       trusted for a surgical write and this raises `ReconstructionError`
       *before* anything else happens: no baseline load, no write.
    2. Tier b (the summary callout, `## Discussion Notes`): three-way
       merge against this run id's cached baseline, per `_merge_tier_b`.
       The new baseline *contents* are computed here but not yet
       persisted - see step 5.
    3. Tier a (`## Chapters`, `## Transcript`): replaced wholesale,
       created in the exemplar order (right after `## Meeting Prep` when
       present, otherwise right after the preamble) when absent. The
       `## Transcript` replacement sweeps every section that looks like
       this module's own chapter output, wherever it sits in the
       document - see `_find_transcript_sweep_indices`.
    4. Before writing, re-asserts that `## Meeting Prep` (if present) is
       still byte-identical to what was parsed in step 1 - a second,
       independent guard on the same invariant, this time checking that
       *this run's own surgery* didn't touch it.
    5. Only once every guard has passed does this write anything: the note
       file first, then the two tier-b baselines. Either guard in step 1
       or step 4 failing raises before *any* write - note or baseline -
       so an aborted run never leaves a baseline pointing at content that
       was never actually written to the file (which would make an
       untouched note look human-edited on the next run).
    """
    text = note_path.read_text()
    frontmatter_prefix, raw_body = split_raw_frontmatter(text)
    note = parse_note(note_path)

    if render_body(note.sections) != raw_body:
        raise ReconstructionError(
            f"{note_path}: the section model failed to reconstruct the note "
            "body byte-for-byte; aborting without writing rather than risk "
            "a surgical write on an untrustworthy section boundary."
        )

    sections = list(note.sections)
    outcomes: list[SectionOutcome] = []

    meeting_prep_index = _find_heading(sections, MEETING_PREP_HEADING)
    original_meeting_prep = (
        sections[meeting_prep_index] if meeting_prep_index is not None else None
    )

    # --- Tier b: the summary callout, in the preamble ---
    preamble = sections[0]
    before, callout_block, after = _split_summary_callout(preamble.body)
    current_summary = _callout_content(callout_block)
    baseline_summary = cache.load_text(run_id, BASELINE_SUMMARY_NAME)

    merged_summary, new_baseline_summary, summary_outcome = _merge_tier_b(
        _SUMMARY_LABEL,
        current_text=current_summary,
        baseline_text=baseline_summary,
        new_text=products.meeting_summary,
        split=_split_summary_unit,
        join=_join_summary_units,
    )
    # Not persisted yet - see the docstring's step 5. `new_baseline_summary`
    # is stored only after the note write below has actually committed.
    # `after` is only empty when the callout sat at the very end of the
    # preamble (or is being created fresh there) - fall back to a blank
    # line so a heading that immediately follows isn't jammed against it.
    # A callout with real trailing preamble content is left exactly as-is.
    new_preamble_body = (
        before + render_summary_callout(merged_summary) + (after or "\n")
    )
    sections[0] = preamble.model_copy(update={"body": new_preamble_body})
    outcomes.append(summary_outcome)

    # --- Tier b: Discussion Notes ---
    anchor = (meeting_prep_index + 1) if meeting_prep_index is not None else 1
    dn_index = _find_heading(sections, _DISCUSSION_NOTES_HEADING)
    current_notes = sections[dn_index].body if dn_index is not None else ""
    baseline_notes = cache.load_text(run_id, BASELINE_NOTES_NAME)

    merged_notes, new_baseline_notes, notes_outcome = _merge_tier_b(
        "Discussion Notes",
        current_text=current_notes,
        baseline_text=baseline_notes,
        new_text=products.discussion_notes,
        split=_split_discussion_units,
        join=_join_discussion_units,
    )
    # Not persisted yet - see the docstring's step 5.
    new_dn_body = f"\n{merged_notes}\n" if merged_notes else "\n\n"

    if dn_index is not None:
        sections[dn_index] = sections[dn_index].model_copy(update={"body": new_dn_body})
        dn_final_index = dn_index
    else:
        dn_final_index = anchor
        sections.insert(
            dn_final_index,
            NoteSection(heading="Discussion Notes", level=2, body=new_dn_body),
        )
    outcomes.append(notes_outcome)
    cursor = dn_final_index + 1

    # --- Tier a: Chapters (the index) ---
    ch_index = _find_heading(sections, _CHAPTERS_HEADING)
    chapters_body = render_chapter_index(products.chapters)
    if ch_index is not None:
        sections[ch_index] = sections[ch_index].model_copy(
            update={"body": chapters_body}
        )
        ch_final_index = ch_index
        ch_mode: Mode = "replaced"
    else:
        ch_final_index = cursor
        sections.insert(
            ch_final_index, NoteSection(heading="Chapters", level=2, body=chapters_body)
        )
        ch_mode = "created"
    outcomes.append(SectionOutcome(heading="Chapters", mode=ch_mode))
    cursor = ch_final_index + 1

    # --- Tier a: Transcript (the full per-chapter content) ---
    sweep_indices = _find_transcript_sweep_indices(sections)
    transcript_body = render_transcript(products.chapters)
    new_transcript_section = NoteSection(
        heading="Transcript", level=2, body=transcript_body
    )
    if sweep_indices:
        sweep_set = set(sweep_indices)
        sections = [
            section for index, section in enumerate(sections) if index not in sweep_set
        ]
        # `sweep_indices[0]` is still correct in the filtered list: nothing
        # smaller than it was removed (it's the smallest swept index), so
        # everything before it kept its original position.
        sections.insert(sweep_indices[0], new_transcript_section)
        tr_mode: Mode = "replaced"
    else:
        sections.insert(cursor, new_transcript_section)
        tr_mode = "created"
    outcomes.append(SectionOutcome(heading="Transcript", mode=tr_mode))

    # --- Pre-write guard: Meeting Prep must be byte-identical ---
    # `meeting_prep_index` is only safe to reuse here because every
    # insertion/removal above happens at or after it (the exemplar order
    # places Meeting Prep before every tier-owned section this stage
    # touches) - it never shifts what sits at that index.
    if (
        meeting_prep_index is not None
        and sections[meeting_prep_index] != original_meeting_prep
    ):
        raise ReconstructionError(
            f"{note_path}: `## Meeting Prep` would change; aborting without "
            "writing - this section is human-owned and this stage never "
            "touches it."
        )

    # Only now, with every guard passed, is anything actually written:
    # the note file first, then the two tier-b baselines - never before,
    # and never if a guard raised above.
    new_text = frontmatter_prefix + render_body(sections)
    note_path.write_text(new_text)
    cache.store_text(run_id, BASELINE_SUMMARY_NAME, new_baseline_summary)
    cache.store_text(run_id, BASELINE_NOTES_NAME, new_baseline_notes)

    return IntegrationReport(run_id=run_id, note_path=str(note_path), sections=outcomes)
