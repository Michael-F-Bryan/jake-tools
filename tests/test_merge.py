from jake_tools.transcripts.merge import (
    merge_note,
    render_chaptered_transcript,
    render_transcript,
)
from jake_tools.transcripts.models import Chapter, MeetingMinutes, TranscriptTurn

ORIGINAL = "# Meeting\n\nAgenda line\n\n![[meeting.m4a]]\n"


def test_merge_note_preserves_original_content_and_embed() -> None:
    merged = merge_note(
        ORIGINAL,
        transcript_body="**Speaker** Hello",
        chapters=[Chapter(title="Kickoff", start=0, end=30, summary="Start")],
        minutes=MeetingMinutes(summary="Summary", key_points=["High-level note"]),
    )

    assert "Agenda line" in merged
    assert "![[meeting.m4a]]" in merged
    assert "## Meeting Notes" in merged
    assert "## Chapters" in merged
    assert "## Transcript" in merged


def test_merge_note_replaces_existing_generated_sections() -> None:
    original = ORIGINAL + "\n## Transcript\n\nold stuff\n"
    merged = merge_note(
        original,
        transcript_body="**Speaker** New text",
        chapters=[Chapter(title="Kickoff", start=0, end=30, summary="Start")],
        minutes=MeetingMinutes(summary="Summary", key_points=["High-level note"]),
    )

    assert "old stuff" not in merged
    assert merged.count("## Transcript") == 1


def test_render_minutes_uses_dot_points() -> None:
    merged = merge_note(
        ORIGINAL,
        transcript_body="**Speaker** Hello",
        chapters=[Chapter(title="Kickoff", start=0, end=30, summary="Start")],
        minutes=MeetingMinutes(
            summary="Summary", key_points=["First note", "Second note"]
        ),
    )

    assert "## Meeting Notes\n\n- First note\n- Second note" in merged


def test_render_transcript_omits_per_turn_timestamps() -> None:
    rendered = render_transcript(
        [TranscriptTurn(start=12, end=18, speaker="Speaker 1", text="Hello there")]
    )

    assert rendered == "**Speaker 1** Hello there"


def test_render_chaptered_transcript_keeps_chapter_timestamps_only() -> None:
    rendered = render_chaptered_transcript(
        [TranscriptTurn(start=12, end=18, speaker="Speaker 1", text="Hello there")],
        [Chapter(title="Kickoff", start=0, end=30, summary="Start")],
    )

    assert "### 00:00 — Kickoff" in rendered
    assert "**Speaker 1** Hello there" in rendered
    assert "[00:12]" not in rendered


def test_render_chaptered_transcript_assigns_boundary_turn_to_one_chapter() -> None:
    rendered = render_chaptered_transcript(
        [
            TranscriptTurn(start=1, end=2, speaker="Speaker 1", text="Intro"),
            TranscriptTurn(start=5, end=6, speaker="Speaker 1", text="Boundary turn"),
            TranscriptTurn(start=8, end=9, speaker="Speaker 2", text="Wrap up"),
        ],
        [
            Chapter(title="First", start=0, end=5, summary="Start"),
            Chapter(title="Second", start=5, end=10, summary="End"),
        ],
    )

    assert rendered.count("Boundary turn") == 1
