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

from pydantic import BaseModel, Field

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
# three-way mechanism, not the extension. A baseline holds *exactly* the
# pipeline-owned content as it stands in the file after the run that wrote
# it - never a unit absent from the file (see `DeletedFingerprints` below
# for how a deletion is remembered instead).
BASELINE_SUMMARY_NAME = "baseline_summary"
BASELINE_NOTES_NAME = "baseline_notes"

# Deleted-fingerprint names under `RunCache.store`/`load` - these land on
# disk as `<name>.json` (typed `DeletedFingerprints`, not raw text).
DELETED_SUMMARY_NAME = "deleted_summary"
DELETED_NOTES_NAME = "deleted_notes"

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


def _split_summary_units(text: str) -> list[str]:
    """One unit per top-level line (a bare lead sentence, or a bullet - each
    carrying its own indented continuation lines), blank lines dropped.

    `minutes.py`'s prompt always emits the summary callout in this shape: a
    plain-prose lead sentence, then zero or more top-level bullets. Splitting
    it the same way `_split_discussion_units` splits Discussion Notes (one
    merge unit per bullet, sub-bullets carried with their parent) means a
    human edit to *one* callout bullet only preserves that bullet - it no
    longer forces the whole callout to be treated as one giant edited unit,
    which previously made every re-run append a second, third, ... full
    fresh summary underneath the human-edited one once any edit had been
    made (and registered the entire original callout as one deleted
    fingerprint in the process).

    Any top-level (column-0) line starts a fresh unit, whether or not it's a
    bullet - not just the first line of the text. That matters once a merge
    has actually run: preserved (human-edited) units are re-assembled ahead
    of fresh appended ones (see `_merge_tier_b`), so a bare lead sentence can
    end up *after* a bullet in the merged text on a later run, not only
    before it. A split that only recognised a lead at the very start of the
    text would then glue that sentence onto the preceding bullet as a
    "continuation" on the next parse - silently merging two units into one,
    which both breaks preservation and manufactures a spurious "deletion"
    fingerprint for content nobody touched. Blank lines are structural only
    (dropped here, reinserted by `_join_summary_units` at unit-type
    boundaries) so they can never be misread as part of a unit's own text.
    """
    stripped = text.strip("\n")
    if not stripped.strip():
        return []
    units: list[str] = []
    for line in stripped.splitlines():
        if not line.strip():
            continue  # a blank line is a separator, never unit content
        if not units or not line[:1].isspace():
            units.append(line + "\n")  # a fresh top-level line: new unit
        else:
            units[-1] += line + "\n"  # an indented continuation
    return units


def _join_summary_units(units: Sequence[str]) -> str:
    """Reassemble summary units: a blank line between a lead paragraph and
    the bullets (or between two paragraph units), no blank line between
    consecutive bullets - the shape `_split_summary_units` expects back.
    No trailing newline, matching `render_summary_callout`'s expectation
    (and the pre-F2 join's own convention) of a bare block of text."""
    parts: list[str] = []
    for index, unit in enumerate(units):
        if index > 0:
            prev_is_bullet = _TOP_LEVEL_BULLET_RE.match(units[index - 1]) is not None
            this_is_bullet = _TOP_LEVEL_BULLET_RE.match(unit) is not None
            if not (prev_is_bullet and this_is_bullet):
                parts.append("\n")
        parts.append(unit)
    return "".join(parts).rstrip("\n")


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
    """What happened to one tier-owned section/region during one `integrate` call.

    `deleted_units` counts fingerprints newly detected as deleted *this*
    run (a baseline unit that just disappeared from the section).
    `suppressed_units` counts fresh pipeline units withheld this run
    because they matched *any* currently-active deleted fingerprint (new
    or carried over from an earlier run) - the visible signal that the
    pipeline wanted to write something back that a human had removed.
    """

    heading: str
    mode: Mode
    preserved_units: int = 0
    appended_units: int = 0
    deleted_units: int = 0
    suppressed_units: int = 0


class IntegrationReport(BaseModel):
    """What `integrate` did, for the run report and the CLI's JSON output."""

    run_id: str
    note_path: str
    sections: list[SectionOutcome]


class DeletedFingerprints(BaseModel):
    """Normalised fingerprints of tier-b units a human has deleted from one section.

    Loaded/stored as JSON via `RunCache.load`/`store` (`RunCache` needs a
    `BaseModel`) - deliberately a *separate* cache artefact from the
    baseline, not folded into it: a baseline that carried deleted
    fingerprints ("ghosts") made a human's later retyped content look like
    untouched pipeline content and get silently overwritten again - see
    `_merge_tier_b`'s docstring. A fingerprint here suppresses a matching
    fresh pipeline unit from being appended; it's removed the moment a
    human retypes matching text back into the section (restoration).
    """

    fingerprints: list[str] = Field(default_factory=list)


def _merge_tier_b(
    label: str,
    *,
    current_text: str,
    baseline_text: str | None,
    new_text: str,
    deleted_fingerprints: frozenset[str],
    split: Callable[[str], list[str]],
    join: Callable[[Sequence[str]], str],
) -> tuple[str, str, set[str], SectionOutcome]:
    """The three-way merge, generic over how a section splits into logical units.

    Shared by the summary callout (one unit: the whole text) and Discussion
    Notes (one unit per top-level bullet, sub-bullets included) - the
    granularity lives entirely in `split`/`join`, the merge logic itself is
    identical for both per the plan.

    **Deletions are a human edit too, tracked separately from the
    baseline.** A unit present in the baseline but absent (by normalised
    comparison) from the current section is treated as human-deleted: its
    normalised fingerprint is added to `deleted_fingerprints` (persisted
    by the caller as a `DeletedFingerprints` artefact, *not* folded into
    the baseline), and a fresh pipeline unit whose normalised text matches
    an active fingerprint is withheld from `appended` rather than silently
    resurrecting what a human removed.

    An earlier version of this mechanism carried deleted fingerprints
    *inside* the baseline text itself (`new_baseline = deleted + appended`).
    That was a bug, not a simplification: the baseline is also what decides
    whether a *current* unit counts as untouched pipeline content
    (`unit not in baseline_norms` -> preserved). A baseline holding a
    "ghost" fingerprint for content that isn't actually in the file made a
    human's later retyped bullet match the ghost, get classified as
    untouched pipeline content, and get silently dropped again the next
    time the pipeline regenerated without it - the escape hatch the
    original docstring promised ("stays gone until manually retyped")
    didn't actually work. The fix: a baseline holds *only* what's actually
    in the file after this run (`new_baseline` below is exactly
    `join(appended)` - never a ghost), and deletion-suppression is a
    wholly separate, explicit set threaded through this function instead.

    **Restoration**: if a fingerprint in `deleted_fingerprints` is present
    among the *current* units this run, a human has retyped it back - it's
    removed from the fingerprint set returned here (so future runs stop
    suppressing it), and the retyped unit is preserved verbatim like any
    other content that doesn't match the (ghost-free) baseline. The
    accepted cost, honestly documented: an accidentally-deleted unit stays
    suppressed only until a human manually retypes matching text - the
    pipeline itself will never offer it back on its own.

    Returns `(merged_body, new_baseline, new_deleted_fingerprints,
    outcome)`. `new_baseline` is exactly the pipeline-owned content
    actually landing in the file this run (`appended`) - never preserved
    human content and never a fingerprint for something absent from the
    file. `new_deleted_fingerprints` is the active suppression set to
    persist for next run (restorations removed, new deletions added).
    """
    current_units = split(current_text)
    new_units = split(new_text)
    current_norms = {_normalize(unit) for unit in current_units}

    # Restoration: a fingerprint the human has retyped back into the
    # section is no longer suppressed - garbage-collected out of the set
    # this function returns, regardless of which branch runs below.
    active_deleted = {
        norm for norm in deleted_fingerprints if norm not in current_norms
    }

    if baseline_text is None:
        if not current_units:
            merged = join(new_units)
            return (
                merged,
                merged,
                active_deleted,
                SectionOutcome(heading=label, mode="first_run"),
            )
        # Degraded append-only mode: no baseline (evicted cache, or a run
        # on a different machine) but the section already has content -
        # presume every existing unit is human-owned. Add only units that
        # are clearly new (and not actively suppressed as deleted); never
        # modify what's there. There is no baseline to detect a *new*
        # deletion against here - only suppression of what was already
        # known deleted, and restoration, still apply.
        preserved = current_units
        preserved_norms = {_normalize(unit) for unit in preserved}
        appended = [
            unit
            for unit in new_units
            if _normalize(unit) not in preserved_norms
            and _normalize(unit) not in active_deleted
        ]
        suppressed = [unit for unit in new_units if _normalize(unit) in active_deleted]
        return (
            join(preserved + appended),
            join(appended),
            active_deleted,
            SectionOutcome(
                heading=label,
                mode="degraded_append_only",
                preserved_units=len(preserved),
                appended_units=len(appended),
                suppressed_units=len(suppressed),
            ),
        )

    baseline_units = split(baseline_text)
    baseline_norms = {_normalize(unit) for unit in baseline_units}

    preserved = [
        unit for unit in current_units if _normalize(unit) not in baseline_norms
    ]
    preserved_norms = {_normalize(unit) for unit in preserved}

    # New deletions this run: a baseline unit no longer present (by
    # normalised text) among the current units. Baselines never carry a
    # fingerprint for content that isn't in the file (see above), so this
    # can never "re-detect" something already tracked in `active_deleted`.
    newly_deleted_norms = baseline_norms - current_norms
    active_deleted = active_deleted | newly_deleted_norms

    excluded_norms = preserved_norms | active_deleted
    appended = [unit for unit in new_units if _normalize(unit) not in excluded_norms]
    suppressed = [unit for unit in new_units if _normalize(unit) in active_deleted]

    mode: Mode = "merged" if preserved else "replaced"
    return (
        join(preserved + appended),
        join(appended),
        active_deleted,
        SectionOutcome(
            heading=label,
            mode=mode,
            preserved_units=len(preserved),
            appended_units=len(appended),
            deleted_units=len(newly_deleted_norms),
            suppressed_units=len(suppressed),
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
    """Whether `heading` has this module's own chapter-heading *shape*.

    `render_transcript` only ever produces `### <ts> — <title>` headings
    (`format_timestamp` output, an em-dash, the title) - but a human can
    coincidentally write a heading with the same shape (e.g. `### 10:30 —
    Standup round-robin` as a Meeting Prep sub-heading), so this shape
    match alone is *not* sufficient to treat a heading as pipeline output -
    see `_find_transcript_sweep_sections`, which additionally requires the
    heading's governing section to be `## Transcript`/`## Chapters` before
    sweeping it.
    """
    return heading is not None and _CHAPTER_HEADING_RE.match(heading) is not None


# The headings this module treats as structurally significant when scoping
# the tier-a sweep - see `_governing_heading`.
_GOVERNING_HEADINGS = frozenset(
    {
        MEETING_PREP_HEADING,
        _DISCUSSION_NOTES_HEADING,
        _CHAPTERS_HEADING,
        _TRANSCRIPT_HEADING,
    }
)


def _governing_heading(sections: Sequence[NoteSection], index: int) -> str | None:
    """The nearest preceding heading at `index` that this module recognises structurally.

    Walks backward from `index`, skipping any heading this module doesn't
    otherwise track (an arbitrary human aside, or another chapter-shaped
    `### <ts> — <title>` heading already swept elsewhere) - only
    `_GOVERNING_HEADINGS` count as establishing or ending a region for the
    purpose of `_find_transcript_sweep_sections`. An unrelated human section
    interrupting a run of swept chapters (see the interrupted-sweep test)
    is therefore transparent here: it doesn't reset which region a later
    chapter heading belongs to. Returns `None`, lower-cased/stripped
    otherwise, if no governing heading precedes `index` at all.
    """
    for probe in range(index - 1, -1, -1):
        heading = sections[probe].heading
        if heading is not None and heading.strip().lower() in _GOVERNING_HEADINGS:
            return heading.strip().lower()
    return None


def _find_transcript_sweep_sections(
    original_sections: Sequence[NoteSection],
) -> list[NoteSection]:
    """Every existing section (by identity) the tier-a Transcript replacement must remove.

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

    A literal `## Transcript` heading is always swept - unambiguous
    pipeline output wherever it occurs. A chapter-*shaped* `### <ts> —
    <title>` heading is swept only when its nearest governing heading
    (`_governing_heading`) is `## Transcript` or `## Chapters` - an
    unrelated human section interrupting a run of chapters doesn't stop
    the sweep (it isn't a governing heading, so it's skipped over), but a
    human heading that merely *looks* timestamp-shaped while actually
    nested under `## Meeting Prep` (or any other non-tier-a section) is
    never swept, because `## Meeting Prep` *is* a governing heading and it
    isn't `## Transcript`/`## Chapters`. This is what stops the sweep from
    silently deleting a human `### 10:30 — Standup round-robin` heading
    inside Meeting Prep - the Meeting Prep byte-identity guard alone can't
    catch that, since `parse_note` splits such a heading into its own
    section, outside Meeting Prep's own body.

    Takes and must be called against `original_sections` - the section
    list exactly as parsed, *before* this run's own Discussion Notes/
    Chapters insertions happen - and returns the matching section
    *objects*, not indices. `run_integrate` inserts fresh, empty ``##
    Discussion Notes``/``## Chapters`` sections earlier in the same call
    when they're absent, always right after ``## Meeting Prep``; if a
    chapter-shaped heading happened to sit immediately after ``##
    Meeting Prep`` in the original document (exactly the human-heading-in-
    Meeting-Prep case this function exists to protect), those insertions
    would land *between* Meeting Prep and it, making a freshly-created
    ``## Chapters`` look like its nearest governing heading if this were
    recomputed against the already-mutated list. Judging every section
    once, against the pristine original structure, and having the caller
    match the results back into the (by-then mutated) working list by
    object identity avoids that - the insertions can only ever change
    *positions*, never which original section object a heading actually
    was.

    Returns the matching section objects in ascending document order
    (possibly empty, if this is a first run with neither shape present).
    """
    result: list[NoteSection] = []
    for index, section in enumerate(original_sections):
        heading = section.heading
        if (
            heading is not None
            and heading.strip().lower() == _TRANSCRIPT_HEADING
            or _is_chapter_heading(heading)
            and _governing_heading(original_sections, index)
            in (_CHAPTERS_HEADING, _TRANSCRIPT_HEADING)
        ):
            result.append(section)
    return result


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
       merge against this run id's cached baseline and deleted-fingerprint
       set, per `_merge_tier_b`. The new baseline/fingerprint *contents*
       are computed here but not yet persisted - see step 5.
    3. Tier a (`## Chapters`, `## Transcript`): replaced wholesale,
       created in the exemplar order (right after `## Meeting Prep` when
       present, otherwise right after the preamble) when absent. The
       `## Transcript` replacement sweeps every section that looks like
       this module's own chapter output and is actually governed by `##
       Transcript`/`## Chapters` - see `_find_transcript_sweep_sections`,
       which is deliberately evaluated against the document exactly as
       parsed, before this step's own Discussion Notes/Chapters insertions
       above can shift what looks like a heading's nearest governing
       section.
    4. Before writing, re-asserts that `## Meeting Prep` (if present) is
       still byte-identical to what was parsed in step 1 - a second,
       independent guard on the same invariant, this time checking that
       *this run's own surgery* didn't touch it.
    5. Only once every guard has passed does this write anything: the note
       file first, then the tier-b baselines and deleted-fingerprint sets.
       Either guard in step 1 or step 4 failing raises before *any*
       write - note, baseline, or fingerprints - so an aborted run never
       leaves state pointing at content that was never actually written to
       the file (which would make an untouched note look human-edited, or
       a deletion look active, on the next run).
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
    # Judged once, against the document exactly as parsed - before this
    # function's own Discussion Notes/Chapters insertions below can land
    # between `## Meeting Prep` and a heading that was never actually
    # governed by `## Transcript`/`## Chapters`, which would otherwise make
    # a freshly-inserted section look like that heading's nearest governing
    # heading. See `_find_transcript_sweep_sections`'s docstring.
    sweep_targets = _find_transcript_sweep_sections(sections)
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
    deleted_summary_model = cache.load(
        run_id, DELETED_SUMMARY_NAME, DeletedFingerprints
    )
    deleted_summary_fingerprints = frozenset(
        deleted_summary_model.fingerprints if deleted_summary_model is not None else ()
    )

    merged_summary, new_baseline_summary, new_deleted_summary, summary_outcome = (
        _merge_tier_b(
            _SUMMARY_LABEL,
            current_text=current_summary,
            baseline_text=baseline_summary,
            new_text=products.meeting_summary,
            deleted_fingerprints=deleted_summary_fingerprints,
            split=_split_summary_units,
            join=_join_summary_units,
        )
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
    deleted_notes_model = cache.load(run_id, DELETED_NOTES_NAME, DeletedFingerprints)
    deleted_notes_fingerprints = frozenset(
        deleted_notes_model.fingerprints if deleted_notes_model is not None else ()
    )

    merged_notes, new_baseline_notes, new_deleted_notes, notes_outcome = _merge_tier_b(
        "Discussion Notes",
        current_text=current_notes,
        baseline_text=baseline_notes,
        new_text=products.discussion_notes,
        deleted_fingerprints=deleted_notes_fingerprints,
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
    # `sweep_targets` was judged against the original, pre-surgery document
    # (see its capture above); translate those section objects into their
    # (possibly shifted, by the insertions above) indices in the current
    # list by identity - never by re-judging position/governing-heading
    # against the now-mutated `sections`.
    sweep_target_ids = {id(section) for section in sweep_targets}
    sweep_indices = [
        index
        for index, section in enumerate(sections)
        if id(section) in sweep_target_ids
    ]
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
    # `meeting_prep_index` was captured once, before any surgery, and is
    # reused here rather than re-found. That's only a no-op check when the
    # document already follows the exemplar order (Meeting Prep before
    # every tier-owned section this stage touches): every insertion then
    # lands at or after `meeting_prep_index + 1`, so the index never
    # shifts and this comparison is trivially true. If the document
    # doesn't follow that order - e.g. `## Discussion Notes` already sits
    # before `## Meeting Prep` - an insertion can land exactly AT
    # `meeting_prep_index` and shift it, so `sections[meeting_prep_index]`
    # ends up pointing at whatever got inserted there instead, which
    # (almost certainly) fails this comparison. That is exactly why this
    # guard is load-bearing rather than decorative: on such a document it
    # correctly refuses to write instead of silently comparing the wrong
    # section and reporting success.
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
    # the note file first, then the tier-b baselines and deleted-
    # fingerprint sets - never before, and never if a guard raised above.
    new_text = frontmatter_prefix + render_body(sections)
    note_path.write_text(new_text)
    cache.store_text(run_id, BASELINE_SUMMARY_NAME, new_baseline_summary)
    cache.store_text(run_id, BASELINE_NOTES_NAME, new_baseline_notes)
    cache.store(
        run_id,
        DELETED_SUMMARY_NAME,
        DeletedFingerprints(fingerprints=sorted(new_deleted_summary)),
    )
    cache.store(
        run_id,
        DELETED_NOTES_NAME,
        DeletedFingerprints(fingerprints=sorted(new_deleted_notes)),
    )

    return IntegrationReport(run_id=run_id, note_path=str(note_path), sections=outcomes)
