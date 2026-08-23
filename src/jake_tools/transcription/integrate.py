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

    Returns `(merged_body, new_baseline, outcome)`. `new_baseline` is only
    ever the pipeline-owned part of the merge (newly appended units) - a
    baseline that included preserved human content would make this
    module mistake human edits for its own on a later run.
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
        # are clearly new; never modify what's there.
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
    preserved = [
        unit for unit in current_units if _normalize(unit) not in baseline_norms
    ]
    preserved_norms = {_normalize(unit) for unit in preserved}
    appended = [unit for unit in new_units if _normalize(unit) not in preserved_norms]

    mode: Mode = "merged" if preserved else "replaced"
    return (
        join(preserved + appended),
        join(appended),
        SectionOutcome(
            heading=label,
            mode=mode,
            preserved_units=len(preserved),
            appended_units=len(appended),
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


def _find_transcript_span(sections: Sequence[NoteSection]) -> tuple[int | None, int]:
    """The `## Transcript` section's index and the end of its trailing chapters.

    On a re-run, `## Transcript`'s body from the *previous* write is empty
    - the `### <ts> — <title>` chapter headings it contains were split into
    their own (deeper-level) sections by `parse_note`, exactly like
    `## Chapters`' index never nests anything. This finds the whole
    span - the `## Transcript` heading plus every section immediately
    after it at a deeper level - so it can be replaced as one wholesale
    unit, per the tier-a contract.
    """
    index = _find_heading(sections, _TRANSCRIPT_HEADING)
    if index is None:
        return None, 0
    level = sections[index].level
    end = index + 1
    while (
        end < len(sections)
        and sections[end].heading is not None
        and sections[end].level > level
    ):
        end += 1
    return index, end


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
       New baselines are stored immediately after each merge.
    3. Tier a (`## Chapters`, `## Transcript`): replaced wholesale,
       created in the exemplar order (right after `## Meeting Prep` when
       present, otherwise right after the preamble) when absent.
    4. Before writing, re-asserts that `## Meeting Prep` (if present) is
       still byte-identical to what was parsed in step 1 - a second,
       independent guard on the same invariant, this time checking that
       *this run's own surgery* didn't touch it. Either guard failing
       aborts without writing.
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
    cache.store_text(run_id, BASELINE_SUMMARY_NAME, new_baseline_summary)
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
    cache.store_text(run_id, BASELINE_NOTES_NAME, new_baseline_notes)
    new_dn_body = f"\n{merged_notes}\n" if merged_notes else "\n"

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
    tr_index, tr_end = _find_transcript_span(sections)
    transcript_body = render_transcript(products.chapters)
    if tr_index is not None:
        replacement = NoteSection(
            heading=sections[tr_index].heading,
            level=sections[tr_index].level,
            body=transcript_body,
        )
        sections[tr_index:tr_end] = [replacement]
        tr_mode: Mode = "replaced"
    else:
        sections.insert(
            cursor, NoteSection(heading="Transcript", level=2, body=transcript_body)
        )
        tr_mode = "created"
    outcomes.append(SectionOutcome(heading="Transcript", mode=tr_mode))

    # --- Pre-write guard: Meeting Prep must be byte-identical ---
    if (
        meeting_prep_index is not None
        and sections[meeting_prep_index] != original_meeting_prep
    ):
        raise ReconstructionError(
            f"{note_path}: `## Meeting Prep` would change; aborting without "
            "writing - this section is human-owned and this stage never "
            "touches it."
        )

    new_text = frontmatter_prefix + render_body(sections)
    note_path.write_text(new_text)

    return IntegrationReport(run_id=run_id, note_path=str(note_path), sections=outcomes)
