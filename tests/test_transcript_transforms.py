from jake_tools.transcripts.models import SpeakerIdentity, SpeakerMapping, TranscriptTurn
from jake_tools.transcripts.transforms import merge_consecutive_turns, normalise_turns
from jake_tools.transcripts.merge import render_transcript


def test_normalise_turns_collapses_duplicate_words_and_whitespace() -> None:
    turns = normalise_turns(
        [
            TranscriptTurn(start=0, end=1, speaker="SPEAKER_01", text="  eyes  looks looks   red  "),
        ]
    )

    assert turns == [TranscriptTurn(start=0, end=1, speaker="SPEAKER_01", text="eyes looks red")]


def test_normalise_turns_dedupes_identical_adjacent_turns() -> None:
    turns = normalise_turns(
        [
            TranscriptTurn(start=0, end=1, speaker="SPEAKER_01", text="Hello there"),
            TranscriptTurn(start=1, end=2, speaker="SPEAKER_01", text="Hello there"),
        ]
    )

    assert turns == [TranscriptTurn(start=0, end=2, speaker="SPEAKER_01", text="Hello there")]


def test_merge_consecutive_turns_combines_short_same_speaker_segments() -> None:
    turns = merge_consecutive_turns(
        [
            TranscriptTurn(start=0, end=1, speaker="SPEAKER_01", text="Hello there"),
            TranscriptTurn(start=1.5, end=2, speaker="SPEAKER_01", text="How are you?"),
            TranscriptTurn(start=5, end=6, speaker="SPEAKER_02", text="Fine"),
        ]
    )

    assert turns == [
        TranscriptTurn(start=0, end=2, speaker="SPEAKER_01", text="Hello there. How are you?"),
        TranscriptTurn(start=5, end=6, speaker="SPEAKER_02", text="Fine"),
    ]


def test_render_transcript_uses_speaker_mapping_names() -> None:
    rendered = render_transcript(
        [TranscriptTurn(start=0, end=1, speaker="SPEAKER_01", text="Hello there")],
        SpeakerMapping(
            mapping={
                "SPEAKER_01": SpeakerIdentity(name="Michael Bryan", confidence=0.9, reason="test")
            }
        ),
    )

    assert rendered == "**Michael Bryan** Hello there"
