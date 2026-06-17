from jake_tools.transcripts.merge import merge_note
from jake_tools.transcripts.models import Chapter, MeetingMinutes


ORIGINAL = "# Meeting\n\nAgenda line\n\n![[meeting.m4a]]\n"


def test_merge_note_preserves_original_content_and_embed() -> None:
    merged = merge_note(
        ORIGINAL,
        mode="minutes",
        transcript_body="**Speaker** [00:00] Hello",
        chapters=[Chapter(title="Kickoff", start=0, end=30, summary="Start")],
        minutes=MeetingMinutes(summary="Summary"),
    )

    assert "Agenda line" in merged
    assert "![[meeting.m4a]]" in merged
    assert "## Meeting Minutes" in merged
    assert "## Chapters" in merged
    assert "## Transcript" in merged


def test_merge_note_replaces_existing_generated_sections() -> None:
    original = ORIGINAL + "\n## Transcript\n\nold stuff\n"
    merged = merge_note(
        original,
        mode="transcript",
        transcript_body="**Speaker** [00:00] New text",
    )

    assert "old stuff" not in merged
    assert merged.count("## Transcript") == 1
