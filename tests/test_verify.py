from jake_tools.transcripts.models import Chapter
from jake_tools.transcripts.verify import VerificationError, verify_note

ORIGINAL = "# Meeting\n\n![[meeting.m4a]]\n"


def test_verify_note_passes_for_valid_minutes_note() -> None:
    merged = "# Meeting\n\n![[meeting.m4a]]\n\n## Meeting Notes\n\n- Summary\n\n## Chapters\n\n- 00:00 — Kickoff\n\n## Transcript\n\n### 00:00 — Kickoff\n\nBody\n"
    report = verify_note(
        merged,
        original_body=ORIGINAL,
        chapters=[Chapter(title="Kickoff", start=0, end=30, summary="Start")],
    )

    assert report.status == "pass"


def test_verify_note_requires_single_transcript_heading() -> None:
    try:
        verify_note(
            "## Transcript\n\nA\n\n## Transcript\n\nB\n",
            original_body=ORIGINAL,
            chapters=[],
        )
    except VerificationError as exc:
        assert str(exc) == "expected exactly one ## Transcript heading"
    else:
        raise AssertionError("expected VerificationError")
