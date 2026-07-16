from jake_tools.transcripts.merge import render_transcript
from jake_tools.transcripts.models import (
    SpeakerIdentity,
    SpeakerMapping,
    TranscriptArtifact,
    TranscriptTurn,
)
from jake_tools.transcripts.transform_primitives import draft_chapter_boundaries
from jake_tools.transcripts.transforms import merge_consecutive_turns, normalise_turns


def test_normalise_turns_collapses_duplicate_words_and_whitespace() -> None:
    turns = normalise_turns(
        [
            TranscriptTurn(
                start=0, end=1, speaker="SPEAKER_01", text="  eyes  looks looks   red  "
            ),
        ]
    )

    assert turns == [
        TranscriptTurn(start=0, end=1, speaker="SPEAKER_01", text="eyes looks red")
    ]


def test_normalise_turns_dedupes_overlapping_identical_turns() -> None:
    turns = normalise_turns(
        [
            TranscriptTurn(start=0, end=1, speaker="SPEAKER_01", text="Hello there"),
            TranscriptTurn(start=0.5, end=2, speaker="SPEAKER_01", text="Hello there"),
        ]
    )

    assert turns == [
        TranscriptTurn(start=0, end=2, speaker="SPEAKER_01", text="Hello there")
    ]


def test_normalise_turns_preserves_repetition_after_a_pause() -> None:
    repeated = TranscriptTurn(start=2, end=3, speaker="SPEAKER_01", text="Hello there")

    turns = normalise_turns(
        [
            TranscriptTurn(start=0, end=1, speaker="SPEAKER_01", text="Hello there"),
            repeated,
        ]
    )

    assert turns[-1] == repeated


def test_merge_consecutive_turns_combines_short_same_speaker_segments() -> None:
    turns = merge_consecutive_turns(
        [
            TranscriptTurn(start=0, end=1, speaker="SPEAKER_01", text="Hello there"),
            TranscriptTurn(start=1.5, end=2, speaker="SPEAKER_01", text="How are you?"),
            TranscriptTurn(start=5, end=6, speaker="SPEAKER_02", text="Fine"),
        ]
    )

    assert turns == [
        TranscriptTurn(
            start=0, end=2, speaker="SPEAKER_01", text="Hello there. How are you?"
        ),
        TranscriptTurn(start=5, end=6, speaker="SPEAKER_02", text="Fine"),
    ]


def test_render_transcript_uses_speaker_mapping_names() -> None:
    rendered = render_transcript(
        [TranscriptTurn(start=0, end=1, speaker="SPEAKER_01", text="Hello there")],
        SpeakerMapping(
            mapping={
                "SPEAKER_01": SpeakerIdentity(
                    name="Michael Bryan", confidence=0.9, reason="test"
                )
            }
        ),
    )

    assert rendered == "**Michael Bryan** Hello there"


def test_chapter_boundaries_close_at_next_turn_start() -> None:
    transcript = TranscriptArtifact(
        turns=[
            TranscriptTurn(
                start=0.0, end=10.0, speaker="Speaker", text="First window."
            ),
            TranscriptTurn(
                start=6.0, end=12.0, speaker="Speaker", text="Second window."
            ),
        ]
    )

    plan = draft_chapter_boundaries(transcript, window_minutes=0.1)

    assert plan.chapters[0].end == plan.chapters[1].start == 6.0
