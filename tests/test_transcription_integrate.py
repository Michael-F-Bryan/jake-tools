"""Behaviour of note integration (`transcription/integrate.py`) and the
`jake-tools transcript integrate` CLI command.

This is the effort's trust boundary: the note is co-owned by a human, and a
destroyed human edit is the project's defined trust-loss failure mode. Every
test below either proves a rendering convention exactly, or proves the
three-way merge / reconstruction-assertion mechanism that makes the
ownership tiers real rather than aspirational - the human-edit-preservation
cases are the point of the plan and are treated as the spec, not incidental
coverage. All tests run on `tmp_path` copies; there is no live-vault test
here (E22: this stage makes no LLM calls, so no `--slow` test is needed
either).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from jake_tools.cli import main
from jake_tools.transcription.cache import RunCache, StageManifest, stable_hash
from jake_tools.transcription.chapters import ChapterList
from jake_tools.transcription.integrate import (
    BASELINE_NOTES_NAME,
    BASELINE_SUMMARY_NAME,
    DELETED_NOTES_NAME,
    DELETED_SUMMARY_NAME,
    DeletedFingerprints,
    IntegrationReport,
    MissingMinutesError,
    MissingPolishedChaptersError,
    ReconstructionError,
    apply_integration_plan,
    format_timestamp,
    load_products,
    plan_integrate,
    render_chapter_index,
    render_summary_callout,
    render_transcript,
)
from jake_tools.transcription.minutes import MINUTES_CACHE_NAME, MinutesResult
from jake_tools.transcription.models import (
    ChapterSpan,
    PolishedChapter,
    PolishedTurn,
    RawTranscript,
    TranscriptProducts,
)
from jake_tools.transcription.note import (
    human_owned_note_context,
    parse_note,
    split_raw_frontmatter,
)
from jake_tools.transcription.polish import POLISHED_CACHE_NAME, PolishedChapterList

# `jake_tools.cli`'s __init__ rebinds the name `transcript` to the Click
# group, shadowing the submodule (see `test_transcription_minutes.py`) -
# fetch the actual module via importlib to monkeypatch its bindings.
transcript_cli = importlib.import_module("jake_tools.cli.transcript")
integrate_module = importlib.import_module("jake_tools.transcription.integrate")


def run_integrate(
    note_path: Path,
    products: TranscriptProducts,
    *,
    cache: RunCache,
    run_id: str,
) -> IntegrationReport:
    """Exercise renderer/merge tests without bypassing public apply in production."""
    return apply_integration_plan(
        note_path,
        plan_integrate(note_path, products, cache=cache, run_id=run_id),
        cache=cache,
        run_id=run_id,
    )


# --- fixtures/helpers ----------------------------------------------------------

MEETING_PREP_ONLY = """---
Date: "[[August 3, 2026]]"
Attendees:
  - "[[Ada Lovelace]]"
tags:
  - note/meeting
---

## Meeting Prep

- Agenda link: <https://example.test/agenda>
"""

OPS_LOG_PREAMBLE_ONLY = """---
tags:
  - ops-log
---

- Chat with [[Bob Example]] about scheduling

![[Recording 20260805183000.m4a]]
"""


def _write_note(tmp_path: Path, body: str, *, name: str = "note.md") -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


def _chapter(
    title: str = "Opening",
    start: float = 3.0,
    summary: str = "Kickoff.",
    turns: tuple[tuple[str, str], ...] = (("Ada Lovelace", "Let's begin."),),
) -> PolishedChapter:
    return PolishedChapter(
        title=title,
        start_seconds=start,
        summary=summary,
        turns=[PolishedTurn(speaker=s, text=t) for s, t in turns],
    )


def _products(
    *,
    context: str = "meeting",
    summary: str = "One sentence summary.",
    notes: str = "- First point\n  - detail\n- Second point\n",
    chapters: list[PolishedChapter] | None = None,
) -> TranscriptProducts:
    return TranscriptProducts(
        context=context,  # type: ignore[arg-type]
        meeting_summary=summary,
        discussion_notes=notes,
        chapters=chapters if chapters is not None else [_chapter()],
    )


def _store_current_product_provenance(
    note_path: Path,
    cache: RunCache,
    run_id: str,
    chapters: list[PolishedChapter],
    summary: str = "s",
    notes: str = "- b",
) -> None:
    note = parse_note(note_path)
    raw = RawTranscript(clips=[], utterances=[])
    chapter_list = ChapterList(
        chapters=[
            ChapterSpan(
                title=chapter.title,
                start_utterance=0,
                end_utterance=0,
                start_seconds=chapter.start_seconds,
            )
            for chapter in chapters
        ]
    )
    polished = PolishedChapterList(chapters=chapters)
    minutes = MinutesResult(
        run_id=run_id, meeting_summary=summary, discussion_notes=notes
    )
    human_hash = stable_hash(human_owned_note_context(note))
    lexicon_hash = "fixture-lexicon"
    cache.store(run_id, "resolved_transcript", raw)
    cache.store(run_id, "chapters", chapter_list)
    cache.store(run_id, POLISHED_CACHE_NAME, polished)
    cache.store(run_id, MINUTES_CACHE_NAME, minutes)
    cache.store_manifest(
        run_id,
        StageManifest(
            stage="polish",
            input_hash="polish-input",
            config_hash="polish-config",
            input_hashes={
                "resolved_transcript": stable_hash(raw.model_dump(mode="json")),
                "chapters": stable_hash(chapter_list.model_dump(mode="json")),
                "human_context": human_hash,
                "lexicon": lexicon_hash,
            },
            output_hash=stable_hash(polished.model_dump(mode="json")),
        ),
    )
    cache.store_manifest(
        run_id,
        StageManifest(
            stage="minutes",
            input_hash="minutes-input",
            config_hash="minutes-config",
            input_hashes={
                "polished": stable_hash(polished.model_dump(mode="json")),
                "human_context": human_hash,
                "lexicon": lexicon_hash,
            },
            output_hash=stable_hash(minutes.model_dump(mode="json")),
        ),
    )


def _meeting_prep_body(note_path: Path) -> str:
    note = parse_note(note_path)
    section = next(
        s
        for s in note.sections
        if s.heading and s.heading.strip().lower() == "meeting prep"
    )
    return section.body


def _outcome(report: IntegrationReport, heading: str):
    return next(s for s in report.sections if s.heading == heading)


# --- Step 1: rendering -----------------------------------------------------------


def test_format_timestamp_exemplars() -> None:
    assert format_timestamp(3) == "00:03"
    assert format_timestamp(415) == "06:55"
    assert format_timestamp(3762) == "01:02:42"


def test_render_summary_callout_prefixes_every_line() -> None:
    assert (
        render_summary_callout("A short summary.")
        == "> [!summary]\n> A short summary.\n"
    )
    # A blank line inside the text stays part of the callout (bare `>`).
    assert render_summary_callout("Line one.\n\nLine two.") == (
        "> [!summary]\n> Line one.\n>\n> Line two.\n"
    )


def test_render_chapter_index_produces_heading_link_bullets() -> None:
    chapters = [
        _chapter(title="Opening", start=3.0),
        _chapter(title="Wrap-up", start=415.0),
    ]

    index = render_chapter_index(chapters)

    assert "- [[#00:03 — Opening]]\n" in index
    assert "- [[#06:55 — Wrap-up]]\n" in index


def test_render_transcript_uses_bold_colon_turns_and_per_chapter_callout() -> None:
    chapter = _chapter(
        title="Opening",
        start=3.0,
        summary="Kickoff.",
        turns=(("Ada Lovelace", "Let's begin."), ("Unknown", "Who is this?")),
    )

    text = render_transcript([chapter])

    assert "### 00:03 — Opening" in text
    assert "> [!summary]\n> Kickoff.\n" in text
    assert "**Ada Lovelace:** Let's begin." in text
    assert "**Unknown:** Who is this?" in text


def test_render_transcript_disambiguates_duplicate_labels_in_heading() -> None:
    chapters = [
        _chapter(title="Standup", start=3.0),
        _chapter(title="Standup", start=3.0),  # identical timestamp + title
    ]

    text = render_transcript(chapters)
    index = render_chapter_index(chapters)

    assert "### 00:03 — Standup\n" in text
    assert "### 00:03 — Standup (2)\n" in text
    assert "- [[#00:03 — Standup]]\n" in index
    assert "- [[#00:03 — Standup (2)]]\n" in index


# --- first run: sections created in exemplar order, frontmatter/Meeting Prep untouched --


def test_first_run_into_prep_only_meeting_note_creates_sections_in_exemplar_order(
    tmp_path: Path,
) -> None:
    note_path = _write_note(tmp_path, MEETING_PREP_ONLY)
    original_prefix, _ = split_raw_frontmatter(note_path.read_text())
    original_meeting_prep = _meeting_prep_body(note_path)
    cache = RunCache(tmp_path / "cache")

    report = run_integrate(note_path, _products(), cache=cache, run_id="run-1")

    new_text = note_path.read_text()
    new_prefix, _ = split_raw_frontmatter(new_text)
    assert new_prefix == original_prefix
    assert _meeting_prep_body(note_path) == original_meeting_prep

    order = [
        new_text.index(marker)
        for marker in (
            "> [!summary]",
            "## Meeting Prep",
            "## Discussion Notes",
            "## Chapters",
            "## Transcript",
        )
    ]
    assert order == sorted(order)

    modes = {s.heading: s.mode for s in report.sections}
    assert modes["summary callout"] == "first_run"
    assert modes["Discussion Notes"] == "first_run"
    assert modes["Chapters"] == "created"
    assert modes["Transcript"] == "created"


def test_first_run_into_ops_log_note_creates_sections_after_preamble_with_no_meeting_prep(
    tmp_path: Path,
) -> None:
    note_path = _write_note(tmp_path, OPS_LOG_PREAMBLE_ONLY)
    cache = RunCache(tmp_path / "cache")

    report = run_integrate(
        note_path, _products(context="ops-log"), cache=cache, run_id="run-1"
    )

    text = note_path.read_text()
    # No Meeting Prep in an ops-log note: original preamble content (the
    # bullet and the audio embed) survives, and the new sections land
    # after it.
    assert "Chat with [[Bob Example]] about scheduling" in text
    assert "![[Recording 20260805183000.m4a]]" in text
    order = [
        text.index(marker)
        for marker in (
            "Bob Example",
            "> [!summary]",
            "## Discussion Notes",
            "## Chapters",
        )
    ]
    assert order == sorted(order)
    modes = {s.heading: s.mode for s in report.sections}
    assert modes["Chapters"] == "created"
    assert modes["Transcript"] == "created"


# --- re-run, human untouched: tier-b replaced; baselines updated -----------------


def test_rerun_with_human_untouched_content_is_replaced_and_baselines_updated_on_disk(
    tmp_path: Path,
) -> None:
    note_path = _write_note(tmp_path, MEETING_PREP_ONLY)
    cache = RunCache(tmp_path / "cache")
    run_integrate(
        note_path,
        _products(summary="First summary.", notes="- Point A\n- Point B\n"),
        cache=cache,
        run_id="run-1",
    )

    run_dir = cache.run_dir("run-1")
    # The literal on-disk filenames `store_text` produces (plan 007's
    # lesson: assert the real extension, not the plan's cosmetic `.md`
    # phrasing).
    assert (run_dir / "baseline_summary.txt").exists()
    assert (run_dir / "baseline_notes.txt").exists()
    assert not (run_dir / "baseline_summary.md").exists()
    assert not (run_dir / "baseline_notes.md").exists()
    # The deleted-fingerprint artefacts are typed JSON via `RunCache.store`
    # (a different mechanism from the text baselines), landing as `.json`.
    assert (run_dir / f"{DELETED_SUMMARY_NAME}.json").exists()
    assert (run_dir / f"{DELETED_NOTES_NAME}.json").exists()

    report = run_integrate(
        note_path,
        _products(summary="Second summary.", notes="- Point C\n- Point D\n"),
        cache=cache,
        run_id="run-1",
    )

    text = note_path.read_text()
    assert "Second summary." in text
    assert "First summary." not in text
    assert "Point C" in text
    assert "Point D" in text
    assert "Point A" not in text
    assert "Point B" not in text

    assert _outcome(report, "summary callout").mode == "replaced"
    assert _outcome(report, "Discussion Notes").mode == "replaced"

    assert cache.load_text("run-1", BASELINE_SUMMARY_NAME) == "Second summary."
    stored_notes = cache.load_text("run-1", BASELINE_NOTES_NAME)
    assert stored_notes is not None
    assert "Point C" in stored_notes
    assert "Point D" in stored_notes


# --- summary callout: bullet-granularity merge (F2) --------------------------


def test_summary_callout_human_edit_to_one_bullet_preserves_it_without_duplicating_across_reruns(
    tmp_path: Path,
) -> None:
    """Regression for the reviewer's exact repro: treating the whole callout
    as one merge unit meant a human edit to even one bullet made every
    subsequent run append a second, full fresh summary underneath it
    (v1..v4 pile-up), and registered the entire original callout as one
    deleted fingerprint. Splitting the callout the same way Discussion
    Notes are (lead sentence, then one unit per bullet) fixes both: only
    the edited bullet is preserved, the untouched lead and other bullets
    still update from fresh pipeline output, and two further re-runs with
    unchanged fresh content neither duplicate anything nor grow the
    fingerprint set."""
    note_path = _write_note(tmp_path, MEETING_PREP_ONLY)
    cache = RunCache(tmp_path / "cache")
    summary_v1 = "Kickoff meeting.\n\n- Budget approved\n- Timeline slipped\n"
    run_integrate(note_path, _products(summary=summary_v1), cache=cache, run_id="run-1")

    text = note_path.read_text()
    edited = text.replace("- Timeline slipped", "- Timeline slipped, EDITED BY A HUMAN")
    assert edited != text
    note_path.write_text(edited)

    summary_v2 = (
        "Kickoff meeting.\n\n- Budget approved\n- Timeline slipped\n- New risk raised\n"
    )
    report_2 = run_integrate(
        note_path, _products(summary=summary_v2), cache=cache, run_id="run-1"
    )

    text_after_2 = note_path.read_text()
    # Only one note-level summary callout (a per-chapter callout also
    # renders as "> [!summary]" further down in `## Transcript` - not the
    # thing under test here, so scope the count to the preamble).
    preamble_after_2 = text_after_2.split("## Meeting Prep")[0]
    assert preamble_after_2.count("> [!summary]") == 1  # no duplicate summary block
    assert text_after_2.count("Kickoff meeting.") == 1  # not piled up either
    assert text_after_2.count("Budget approved") == 1
    assert "Timeline slipped, EDITED BY A HUMAN" in text_after_2
    # The un-edited original bullet wasn't resurrected alongside the edit.
    assert "- Timeline slipped\n" not in text_after_2
    assert "New risk raised" in text_after_2

    summary_outcome = _outcome(report_2, "summary callout")
    assert summary_outcome.mode == "merged"
    assert summary_outcome.preserved_units == 1  # the edited bullet, alone

    fingerprints_after_2 = cache.load(
        "run-1", DELETED_SUMMARY_NAME, DeletedFingerprints
    )
    assert fingerprints_after_2 is not None

    # A further re-run with the exact same fresh content: byte-stable, and
    # the fingerprint set must not grow across runs with no new edits.
    run_integrate(note_path, _products(summary=summary_v2), cache=cache, run_id="run-1")
    text_after_3 = note_path.read_text()
    assert text_after_3 == text_after_2
    assert text_after_3.split("## Meeting Prep")[0].count("> [!summary]") == 1

    fingerprints_after_3 = cache.load(
        "run-1", DELETED_SUMMARY_NAME, DeletedFingerprints
    )
    assert fingerprints_after_3 is not None
    assert fingerprints_after_3.fingerprints == fingerprints_after_2.fingerprints


# --- re-run, human edited a bullet: preserved verbatim; new points appended -------


def test_rerun_with_human_edited_bullet_preserves_it_verbatim_and_appends_new_points(
    tmp_path: Path,
) -> None:
    note_path = _write_note(tmp_path, MEETING_PREP_ONLY)
    cache = RunCache(tmp_path / "cache")
    run_integrate(
        note_path,
        _products(notes="- First point\n  - detail\n- Second point\n"),
        cache=cache,
        run_id="run-1",
    )

    text = note_path.read_text()
    edited = text.replace(
        "- First point\n  - detail\n", "- First point, EDITED BY A HUMAN\n  - detail\n"
    )
    assert edited != text
    note_path.write_text(edited)

    report = run_integrate(
        note_path,
        _products(
            notes="- First point\n  - detail\n- Second point\n- Third point (new)\n"
        ),
        cache=cache,
        run_id="run-1",
    )

    new_text = note_path.read_text()
    assert "- First point, EDITED BY A HUMAN\n  - detail\n" in new_text
    assert "Third point (new)" in new_text

    outcome = _outcome(report, "Discussion Notes")
    assert outcome.mode == "merged"
    assert outcome.preserved_units == 1  # the edited bullet, with its sub-bullet
    assert outcome.appended_units >= 1  # the pipeline's fresh regeneration


# --- re-run, human added a bullet: preserved --------------------------------------


def test_rerun_with_human_added_bullet_preserves_it(tmp_path: Path) -> None:
    note_path = _write_note(tmp_path, MEETING_PREP_ONLY)
    cache = RunCache(tmp_path / "cache")
    run_integrate(
        note_path,
        _products(notes="- First point\n- Second point\n"),
        cache=cache,
        run_id="run-1",
    )

    text = note_path.read_text()
    edited = text.replace(
        "- Second point\n", "- Second point\n- A bullet the human added\n"
    )
    note_path.write_text(edited)

    report = run_integrate(
        note_path,
        _products(notes="- First point\n- Second point\n"),
        cache=cache,
        run_id="run-1",
    )

    new_text = note_path.read_text()
    assert "- A bullet the human added\n" in new_text

    outcome = _outcome(report, "Discussion Notes")
    assert outcome.preserved_units == 1


# --- re-run, human deleted a bullet: not resurrected; genuinely new still appends --


def test_rerun_with_human_deleted_bullet_does_not_resurrect_it_but_still_appends_new(
    tmp_path: Path,
) -> None:
    note_path = _write_note(tmp_path, MEETING_PREP_ONLY)
    cache = RunCache(tmp_path / "cache")
    run_integrate(
        note_path,
        _products(notes="- Point A\n- Point B\n"),
        cache=cache,
        run_id="run-1",
    )

    text = note_path.read_text()
    without_b = text.replace("- Point B\n", "")
    assert without_b != text
    note_path.write_text(without_b)

    # The pipeline regenerates BOTH points again (it has no memory of the
    # deletion), plus one genuinely new point.
    report = run_integrate(
        note_path,
        _products(notes="- Point A\n- Point B\n- Point C (genuinely new)\n"),
        cache=cache,
        run_id="run-1",
    )

    new_text = note_path.read_text()
    assert "Point A" in new_text
    assert "Point B" not in new_text  # deleted by a human - not resurrected
    assert "Point C (genuinely new)" in new_text  # genuinely new still appends

    outcome = _outcome(report, "Discussion Notes")
    assert outcome.deleted_units == 1
    assert outcome.suppressed_units == 1  # "Point B" was withheld, visibly

    # The baseline itself must never contain a unit absent from the note -
    # deletion-suppression lives in the separate deleted-fingerprint
    # artefact, not "carried" inside the baseline as a ghost entry.
    stored_baseline = cache.load_text("run-1", BASELINE_NOTES_NAME)
    assert stored_baseline is not None
    assert "Point B" not in stored_baseline


def test_deleted_bullet_suppression_persists_across_a_further_rerun(
    tmp_path: Path,
) -> None:
    """The deleted unit's normalised fingerprint is carried in the
    separate `DeletedFingerprints` artefact (never inside the baseline
    itself), so a *second* re-run - not just the one that first noticed
    the deletion - still doesn't resurrect it, matching the documented
    "stays gone until manually retyped" semantics."""
    note_path = _write_note(tmp_path, MEETING_PREP_ONLY)
    cache = RunCache(tmp_path / "cache")
    run_integrate(
        note_path,
        _products(notes="- Point A\n- Point B\n"),
        cache=cache,
        run_id="run-1",
    )
    note_path.write_text(note_path.read_text().replace("- Point B\n", ""))

    run_integrate(
        note_path,
        _products(notes="- Point A\n- Point B\n"),
        cache=cache,
        run_id="run-1",
    )
    assert "Point B" not in note_path.read_text()
    deleted = cache.load("run-1", DELETED_NOTES_NAME, DeletedFingerprints)
    assert deleted is not None
    assert deleted.fingerprints == ["- Point B"]

    # A further re-run, still regenerating "Point B" every time.
    run_integrate(
        note_path,
        _products(notes="- Point A\n- Point B\n"),
        cache=cache,
        run_id="run-1",
    )

    assert "Point B" not in note_path.read_text()


def test_human_retyping_a_deleted_bullet_restores_it_instead_of_vanishing(
    tmp_path: Path,
) -> None:
    """Re-reviewer's exact regression scenario: an earlier "carry the
    deleted fingerprint inside the baseline" design made a human's
    manually-retyped bullet silently vanish again, because the ghost
    baseline entry made the retyped text look like untouched pipeline
    content due for replacement. The fix keeps deletion tracking in a
    separate, restorable set: retyping a deleted bullet must make it
    survive a run whose fresh pipeline output doesn't even propose it."""
    note_path = _write_note(tmp_path, MEETING_PREP_ONLY)
    cache = RunCache(tmp_path / "cache")

    # 1. Write A + B.
    run_integrate(
        note_path,
        _products(notes="- Point A\n- Point B\n"),
        cache=cache,
        run_id="run-1",
    )

    # 2. Human deletes B.
    note_path.write_text(note_path.read_text().replace("- Point B\n", ""))

    # 3. Re-run regenerating both: B is suppressed.
    report_2 = run_integrate(
        note_path,
        _products(notes="- Point A\n- Point B\n"),
        cache=cache,
        run_id="run-1",
    )
    assert "Point B" not in note_path.read_text()
    assert _outcome(report_2, "Discussion Notes").suppressed_units == 1
    deleted_after_suppression = cache.load(
        "run-1", DELETED_NOTES_NAME, DeletedFingerprints
    )
    assert deleted_after_suppression is not None
    assert deleted_after_suppression.fingerprints == ["- Point B"]

    # 4. Human retypes B.
    note_path.write_text(
        note_path.read_text().replace("- Point A\n", "- Point A\n- Point B\n")
    )
    assert "- Point B" in note_path.read_text()

    # 5. Re-run with pipeline output that doesn't even propose B this
    # time: B must be PRESERVED in the file (not vanish), the deleted set
    # must no longer contain it, and the report must show it preserved.
    report_3 = run_integrate(
        note_path,
        _products(notes="- Point A\n"),
        cache=cache,
        run_id="run-1",
    )

    final_text = note_path.read_text()
    assert "- Point B" in final_text  # retyped content survives
    assert "- Point A" in final_text

    notes_outcome = _outcome(report_3, "Discussion Notes")
    assert notes_outcome.preserved_units == 1  # the retyped "Point B"
    assert notes_outcome.suppressed_units == 0  # nothing suppressed this run

    deleted_after_restoration = cache.load(
        "run-1", DELETED_NOTES_NAME, DeletedFingerprints
    )
    assert deleted_after_restoration is not None
    assert deleted_after_restoration.fingerprints == []

    # And the baseline never held a ghost for "Point B" at any point.
    stored_baseline = cache.load_text("run-1", BASELINE_NOTES_NAME)
    assert stored_baseline is not None
    assert "Point B" not in stored_baseline


# --- no baseline but section has content: append-only degraded mode ---------------


def _note_with_established_content(tmp_path: Path, name: str) -> Path:
    """A note that already has `## Discussion Notes` content from a prior
    run, but under a cache that doesn't know about it - simulating an
    evicted baseline or a run on a different machine."""
    note_path = _write_note(tmp_path, MEETING_PREP_ONLY, name=name)
    seeding_cache = RunCache(tmp_path / f"{name}-seed-cache")
    run_integrate(
        note_path,
        _products(summary="Original summary.", notes="- Point A\n- Point B\n"),
        cache=seeding_cache,
        run_id="run-1",
    )
    return note_path


def test_no_baseline_with_existing_content_and_nothing_clearly_new_modifies_nothing(
    tmp_path: Path,
) -> None:
    note_path = _note_with_established_content(tmp_path, "note-a.md")
    text_before = note_path.read_text()
    cache = RunCache(tmp_path / "fresh-cache-a")
    assert cache.load_text("run-1", BASELINE_NOTES_NAME) is None

    report = run_integrate(
        note_path,
        _products(summary="Original summary.", notes="- Point A\n- Point B\n"),
        cache=cache,
        run_id="run-1",
    )

    # Nothing "clearly new" this run (identical content): the note is
    # untouched byte-for-byte.
    assert note_path.read_text() == text_before
    notes_outcome = _outcome(report, "Discussion Notes")
    assert notes_outcome.mode == "degraded_append_only"
    assert notes_outcome.preserved_units == 2
    assert notes_outcome.appended_units == 0


def test_no_baseline_with_existing_content_appends_clearly_new_points_only(
    tmp_path: Path,
) -> None:
    note_path = _note_with_established_content(tmp_path, "note-b.md")
    cache = RunCache(tmp_path / "fresh-cache-b")
    assert cache.load_text("run-1", BASELINE_NOTES_NAME) is None

    report = run_integrate(
        note_path,
        _products(
            summary="Original summary.",
            notes="- Point A\n- Point B\n- Point C (clearly new)\n",
        ),
        cache=cache,
        run_id="run-1",
    )

    new_text = note_path.read_text()
    assert "Point A" in new_text
    assert "Point B" in new_text
    assert "Point C (clearly new)" in new_text
    outcome = _outcome(report, "Discussion Notes")
    assert outcome.mode == "degraded_append_only"
    assert outcome.preserved_units == 2
    assert outcome.appended_units == 1


# --- tier-a always replaced wholesale, even hand-edited ---------------------------


def test_tier_a_sections_are_always_replaced_wholesale_even_when_hand_edited(
    tmp_path: Path,
) -> None:
    note_path = _write_note(tmp_path, MEETING_PREP_ONLY)
    cache = RunCache(tmp_path / "cache")
    run_integrate(note_path, _products(), cache=cache, run_id="run-1")

    text = note_path.read_text()
    edited = text.replace("Let's begin.", "SOMEONE HAND-EDITED THIS TRANSCRIPT TURN")
    edited = edited.replace("00:03 — Opening", "00:03 — A Human Renamed This Chapter")
    note_path.write_text(edited)

    new_chapter = _chapter(title="Opening", start=3.0, summary="Kickoff.")
    report = run_integrate(
        note_path, _products(chapters=[new_chapter]), cache=cache, run_id="run-1"
    )

    final_text = note_path.read_text()
    assert "SOMEONE HAND-EDITED THIS TRANSCRIPT TURN" not in final_text
    assert "A Human Renamed This Chapter" not in final_text
    assert "**Ada Lovelace:** Let's begin." in final_text
    assert "### 00:03 — Opening" in final_text

    assert _outcome(report, "Chapters").mode == "replaced"
    assert _outcome(report, "Transcript").mode == "replaced"


# --- tier-a Transcript sweep: legacy shapes and non-contiguous chapters -----------

LEGACY_TRANSCRIPT_NOTE = """---
tags:
  - note/meeting
---

## Meeting Prep

- Agenda link: <https://example.test/agenda>

## Discussion Notes

- Old discussion point

## Chapters

- 00:00 — Opening
- 05:00 — Wrap-up

### 00:00 — Opening

**Ada Lovelace:** Let's get started.

### 05:00 — Wrap-up

**Ada Lovelace:** That's everything for today.
"""


def test_rerun_on_legacy_shape_with_no_transcript_heading_sweeps_old_chapters(
    tmp_path: Path,
) -> None:
    """The real shape of existing vault notes predating this stage:
    `### <ts> — <title>` chapter sections sit directly after `## Chapters`,
    with no `## Transcript` heading at all. Re-running must not orphan the
    old chapters or duplicate their content."""
    note_path = _write_note(tmp_path, LEGACY_TRANSCRIPT_NOTE)
    cache = RunCache(tmp_path / "cache")
    new_chapter = _chapter(
        title="Kickoff",
        start=3.0,
        summary="Kickoff.",
        turns=(("Ada Lovelace", "New content."),),
    )

    report = run_integrate(
        note_path, _products(chapters=[new_chapter]), cache=cache, run_id="run-1"
    )

    text = note_path.read_text()
    assert "Let's get started." not in text
    assert "That's everything for today." not in text
    assert text.count("### ") == 1  # exactly one fresh chapter heading, no orphans
    assert "### 00:03 — Kickoff" in text
    assert "New content." in text
    assert text.count("## Transcript") == 1

    assert _outcome(report, "Transcript").mode == "replaced"


def test_rerun_sweeps_transcript_chapters_interrupted_by_a_human_section(
    tmp_path: Path,
) -> None:
    """Chapter sections that aren't contiguous with `## Transcript` (an
    unrelated human section splits them) must still all be swept - and the
    human section, whose heading never matches the chapter shape, must
    survive untouched."""
    note_path = _write_note(
        tmp_path,
        """---
tags:
  - note/meeting
---

## Meeting Prep

- Agenda link: <https://example.test/agenda>

## Discussion Notes

- Old discussion point

## Chapters

- 00:00 — Opening
- 05:00 — Wrap-up

## Transcript

### 00:00 — Opening

**Ada Lovelace:** Let's get started.

## A Human's Own Aside

Michael's own note, unrelated to the transcript.

### 05:00 — Wrap-up

**Ada Lovelace:** That's everything for today.
""",
    )
    cache = RunCache(tmp_path / "cache")

    report = run_integrate(note_path, _products(), cache=cache, run_id="run-1")

    text = note_path.read_text()
    assert "Let's get started." not in text
    assert "That's everything for today." not in text
    assert "Michael's own note, unrelated to the transcript." in text
    assert "## A Human's Own Aside" in text
    assert text.count("## Transcript") == 1
    # The consolidated Transcript section lands where the swept content
    # started; the surviving human section ends up after it (there is no
    # way to keep it "in the middle" once the chapters it split are
    # collapsed into one fresh block).
    assert text.index("## Transcript") < text.index("## A Human's Own Aside")

    assert _outcome(report, "Transcript").mode == "replaced"


def test_time_like_human_heading_inside_meeting_prep_survives_byte_for_byte(
    tmp_path: Path,
) -> None:
    """F1 repro: a human heading that happens to share this module's own
    chapter shape (`### <ts> — <title>`) but sits inside `## Meeting Prep` -
    with no `## Transcript`/`## Chapters` heading anywhere else in the note
    yet - must never be swept. The `## Meeting Prep` byte-identity guard
    alone can't catch this: `parse_note` splits the nested heading into its
    own section, outside Meeting Prep's own body, so the guard never even
    looks at it."""
    note_body = (
        "---\n"
        "tags:\n"
        "  - note/meeting\n"
        "---\n"
        "\n"
        "## Meeting Prep\n"
        "\n"
        "- Agenda link: <https://example.test/agenda>\n"
        "\n"
        "### 10:30 — Standup round-robin\n"
        "\n"
        "- Ada: shipped the parser\n"
        "- Bob: blocked on review\n"
    )
    note_path = _write_note(tmp_path, note_body)
    standup_block = (
        "### 10:30 — Standup round-robin\n"
        "\n"
        "- Ada: shipped the parser\n"
        "- Bob: blocked on review\n"
    )
    assert standup_block in note_path.read_text()  # sanity: the fixture is as written
    cache = RunCache(tmp_path / "cache")

    run_integrate(note_path, _products(), cache=cache, run_id="run-1")

    assert standup_block in note_path.read_text()


# --- abort path: a broken reconstruction invariant aborts without writing --------


def test_broken_reconstruction_invariant_raises_and_leaves_the_file_unmodified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    note_path = _write_note(tmp_path, MEETING_PREP_ONLY)
    original_bytes = note_path.read_bytes()
    cache = RunCache(tmp_path / "cache")

    # Simulate a corrupted section boundary: `render_body` no longer
    # reconstructs the note's body byte-for-byte, which is the exact
    # invariant every surgical write in this module depends on.
    monkeypatch.setattr(
        integrate_module,
        "render_body",
        lambda sections: "this does not match the real body",
    )

    with pytest.raises(ReconstructionError):
        run_integrate(note_path, _products(), cache=cache, run_id="run-1")

    assert note_path.read_bytes() == original_bytes
    # No baseline was stored either - the abort happens before any I/O.
    assert cache.load_text("run-1", BASELINE_SUMMARY_NAME) is None
    assert cache.load_text("run-1", BASELINE_NOTES_NAME) is None


def test_baseline_writes_wait_until_every_guard_has_passed(tmp_path: Path) -> None:
    """Reviewer-constructed scenario: `## Discussion Notes` sits *before*
    `## Meeting Prep` in the note, and `## Chapters` is absent. Inserting a
    fresh `## Chapters` (and then `## Transcript`) section right after
    Discussion Notes lands exactly at `## Meeting Prep`'s captured index,
    shifting it - so the post-surgery "Meeting Prep unchanged" guard trips
    on an independently-reachable path, not the pre-surgery one. Both
    tier-b merges still ran and produced fresh baseline content before
    that guard fires: this proves neither baseline is persisted anyway -
    the note write and the guards that gate it come first, no exceptions.
    """
    note_path = _write_note(
        tmp_path,
        """---
tags:
  - note/meeting
---

## Discussion Notes

- Some notes

## Meeting Prep

- Agenda link: <https://example.test/agenda>
""",
    )
    original_bytes = note_path.read_bytes()
    cache = RunCache(tmp_path / "cache")

    with pytest.raises(ReconstructionError):
        run_integrate(note_path, _products(), cache=cache, run_id="run-1")

    assert note_path.read_bytes() == original_bytes
    assert cache.load_text("run-1", BASELINE_SUMMARY_NAME) is None
    assert cache.load_text("run-1", BASELINE_NOTES_NAME) is None


# --- load_products: prerequisite errors -------------------------------------------


def test_load_products_raises_missing_polished_chapters_error(tmp_path: Path) -> None:
    note_path = _write_note(tmp_path, MEETING_PREP_ONLY)
    cache = RunCache(tmp_path / "cache")

    with pytest.raises(MissingPolishedChaptersError) as excinfo:
        load_products(note_path, "run-1", cache)

    assert "run-1" in str(excinfo.value)
    assert "transcript polish" in str(excinfo.value)


def test_load_products_raises_missing_minutes_error(tmp_path: Path) -> None:
    note_path = _write_note(tmp_path, MEETING_PREP_ONLY)
    cache = RunCache(tmp_path / "cache")
    chapters = [_chapter()]
    _store_current_product_provenance(note_path, cache, "run-1", chapters)
    cache.invalidate_artefacts("run-1", {"minutes", "minutes.manifest"})

    with pytest.raises(MissingMinutesError) as excinfo:
        load_products(note_path, "run-1", cache)

    assert "run-1" in str(excinfo.value)
    assert "transcript minutes" in str(excinfo.value)


def test_load_products_refuses_products_without_current_manifests(
    tmp_path: Path,
) -> None:
    note_path = _write_note(tmp_path, MEETING_PREP_ONLY)
    cache = RunCache(tmp_path / "cache")
    cache.store(
        "run-1", POLISHED_CACHE_NAME, PolishedChapterList(chapters=[_chapter()])
    )
    cache.store(
        "run-1",
        MINUTES_CACHE_NAME,
        MinutesResult(run_id="run-1", meeting_summary="s", discussion_notes="- b"),
    )

    with pytest.raises(MissingPolishedChaptersError):
        load_products(note_path, "run-1", cache)


def test_load_products_assembles_context_and_products_from_the_cache(
    tmp_path: Path,
) -> None:
    note_path = _write_note(tmp_path, MEETING_PREP_ONLY)
    cache = RunCache(tmp_path / "cache")
    chapters = [_chapter()]
    _store_current_product_provenance(note_path, cache, "run-1", chapters)

    products = load_products(note_path, "run-1", cache)

    assert products.context == "meeting"
    assert products.meeting_summary == "s"
    assert products.discussion_notes == "- b"
    assert products.chapters == chapters


# --- CLI: delegation, error mapping, help, one full happy path ---------------------


def test_integrate_cli_help_exits_zero() -> None:
    result = CliRunner().invoke(main, ["transcript", "integrate", "--help"])

    assert result.exit_code == 0
    assert "integrate" in result.output.lower()


def test_integrate_cli_delegates_and_prints_the_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_report = IntegrationReport(run_id="run-1", note_path="note.md", sections=[])
    fake_products = _products()
    captured: dict[str, object] = {}

    def fake_load_products(
        note_path: Path, run_id: str, cache: object
    ) -> TranscriptProducts:
        captured["load_note_path"] = note_path
        captured["load_run_id"] = run_id
        return fake_products

    def fake_run_integrate(
        note_path: Path, products: TranscriptProducts, *, cache: object, run_id: str
    ) -> IntegrationReport:
        captured["note_path"] = note_path
        captured["products"] = products
        captured["run_id"] = run_id
        return fake_report

    monkeypatch.setattr(transcript_cli, "load_products", fake_load_products)
    monkeypatch.setattr(transcript_cli, "run_integrate", fake_run_integrate)
    note_path = _write_note(tmp_path, MEETING_PREP_ONLY)

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "integrate",
            str(note_path),
            "--run-id",
            "run-1",
            "--cache-root",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == json.loads(fake_report.model_dump_json())
    assert captured["note_path"] == note_path
    assert captured["run_id"] == "run-1"
    assert captured["products"] is fake_products


def test_integrate_cli_reports_a_missing_prerequisite_as_a_clean_click_exception(
    tmp_path: Path,
) -> None:
    note_path = _write_note(tmp_path, MEETING_PREP_ONLY)

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "integrate",
            str(note_path),
            "--run-id",
            "run-1",
            "--cache-root",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code != 0
    assert "run-1" in result.output
    assert "transcript polish" in result.output


def test_integrate_cli_refuses_without_an_accepted_product_review(
    tmp_path: Path,
) -> None:
    note_path = _write_note(tmp_path, MEETING_PREP_ONLY)
    cache = RunCache(tmp_path / "cache")
    chapters = [_chapter()]
    _store_current_product_provenance(
        note_path,
        cache,
        "run-1",
        chapters,
        summary="A real summary.",
        notes="- A real point\n",
    )

    before = note_path.read_bytes()
    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "integrate",
            str(note_path),
            "--run-id",
            "run-1",
            "--cache-root",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code != 0, result.output
    assert "accepted product review" in result.output
    assert note_path.read_bytes() == before
