from pathlib import Path

import pytest

from jake_tools.transcripts.errors import TranscriptError
from jake_tools.transcripts.verify import VerifyPrimitiveError, verify_note

DEFAULT_NOTE = (
    "# Meeting\n\n"
    "## Meeting Notes\n\n- Summary\n\n"
    "## Chapters\n\n- 00:00 — Kickoff\n\n"
    "## Transcript\n\n### 00:00 — Kickoff\n\nBody\n"
)


def test_verify_note_default_profile_passes_for_well_formed_note() -> None:
    report = verify_note(
        DEFAULT_NOTE,
        expected_chapter_count=1,
        affected_path=Path("note.md"),
    )

    assert report.status == "pass"
    assert report.failed_gate_ids == []


def test_verify_note_default_profile_rejects_missing_chapters_section() -> None:
    note = DEFAULT_NOTE.replace("## Chapters\n\n- 00:00 — Kickoff\n\n", "")

    report = verify_note(
        note,
        expected_chapter_count=1,
        affected_path=Path("note.md"),
    )

    assert "note.has-chapters" in report.failed_gate_ids


def test_verify_note_original_body_check_requires_single_transcript_heading() -> None:
    original = "# Meeting\n\n![[meeting.m4a]]\n"
    duplicated = original + "\n## Transcript\n\nA\n\n## Transcript\n\nB\n"

    report = verify_note(
        duplicated,
        expected_chapter_count=0,
        affected_path=Path("note.md"),
        original_body=original,
    )

    assert "note.single-transcript-heading" in report.failed_gate_ids


def test_verify_note_original_body_check_requires_recording_embed_preserved() -> None:
    original = "# Meeting\n\n![[meeting.m4a]]\n"
    merged_without_embed = (
        "# Meeting\n\n"
        "## Meeting Notes\n\n- Summary\n\n"
        "## Chapters\n\n- 00:00 — Kickoff\n\n"
        "## Transcript\n\n### 00:00 — Kickoff\n\nBody\n"
    )

    report = verify_note(
        merged_without_embed,
        expected_chapter_count=1,
        affected_path=Path("note.md"),
        original_body=original,
    )

    assert "note.recording-embeds-preserved" in report.failed_gate_ids


def test_verify_note_original_body_check_passes_when_embed_is_preserved() -> None:
    original = "# Meeting\n\n![[meeting.m4a]]\n"
    merged = original + "\n" + DEFAULT_NOTE

    report = verify_note(
        merged,
        expected_chapter_count=1,
        affected_path=Path("note.md"),
        original_body=original,
    )

    assert "note.single-transcript-heading" not in report.failed_gate_ids
    assert "note.recording-embeds-preserved" not in report.failed_gate_ids


def test_verify_note_rejects_unknown_profile_with_a_typed_error() -> None:
    with pytest.raises(VerifyPrimitiveError, match="unknown note verification profile"):
        verify_note(
            DEFAULT_NOTE,
            expected_chapter_count=1,
            affected_path=Path("note.md"),
            profile="bogus",  # type: ignore[arg-type]
        )


def test_verify_primitive_error_is_a_transcript_error() -> None:
    assert issubclass(VerifyPrimitiveError, TranscriptError)
