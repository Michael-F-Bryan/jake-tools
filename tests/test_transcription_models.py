from __future__ import annotations

from jake_tools.transcription.models import (
    ChapterSpan,
    PolishedChapter,
    PolishedTurn,
    RawTranscript,
    SnippetRequest,
    SourceClip,
    SpeakerAssignment,
    TranscriptProducts,
    Utterance,
)


def test_source_clip_round_trips() -> None:
    clip = SourceClip(path="clips/a.wav", offset_seconds=0.0, duration_seconds=12.5)
    assert SourceClip.model_validate_json(clip.model_dump_json()) == clip


def test_utterance_round_trips() -> None:
    utterance = Utterance(start=1.0, end=4.5, speaker="SPEAKER_00", text="hello there")
    assert Utterance.model_validate_json(utterance.model_dump_json()) == utterance


def test_raw_transcript_round_trips() -> None:
    transcript = RawTranscript(
        clips=[SourceClip(path="a.wav", offset_seconds=0.0, duration_seconds=10.0)],
        utterances=[Utterance(start=0.0, end=2.0, speaker="SPEAKER_00", text="hi")],
        audio_sha256="deadbeef",
    )
    assert RawTranscript.model_validate_json(transcript.model_dump_json()) == transcript


def test_raw_transcript_audio_sha256_defaults_to_none() -> None:
    transcript = RawTranscript(clips=[], utterances=[])
    assert transcript.audio_sha256 is None
    assert RawTranscript.model_validate_json(transcript.model_dump_json()) == transcript


def test_raw_transcript_preserves_clip_order_and_offsets() -> None:
    clips = [
        SourceClip(path="a.wav", offset_seconds=0.0, duration_seconds=10.0),
        SourceClip(path="b.wav", offset_seconds=10.0, duration_seconds=5.0),
        SourceClip(path="c.wav", offset_seconds=15.0, duration_seconds=7.5),
    ]
    transcript = RawTranscript(clips=clips, utterances=[])

    restored = RawTranscript.model_validate_json(transcript.model_dump_json())

    assert [clip.path for clip in restored.clips] == ["a.wav", "b.wav", "c.wav"]
    assert [clip.offset_seconds for clip in restored.clips] == [0.0, 10.0, 15.0]


def test_speaker_assignment_round_trips() -> None:
    assignment = SpeakerAssignment(cluster="SPEAKER_03", name="Nikki Staltari")
    assert (
        SpeakerAssignment.model_validate_json(assignment.model_dump_json())
        == assignment
    )


def test_snippet_request_round_trips() -> None:
    request = SnippetRequest(
        cluster="SPEAKER_01",
        clip_paths=["snippets/one.wav", "snippets/two.wav"],
        context="A: hi B\nB: hi A",
    )
    assert SnippetRequest.model_validate_json(request.model_dump_json()) == request


def test_chapter_span_round_trips() -> None:
    span = ChapterSpan(
        title="Intro", start_utterance=0, end_utterance=5, start_seconds=0.0
    )
    assert ChapterSpan.model_validate_json(span.model_dump_json()) == span


def test_polished_turn_round_trips() -> None:
    turn = PolishedTurn(speaker="Nikki Staltari", text="Welcome everyone.")
    assert PolishedTurn.model_validate_json(turn.model_dump_json()) == turn


def test_polished_chapter_round_trips() -> None:
    chapter = PolishedChapter(
        title="Intro",
        start_seconds=0.0,
        summary="Opening remarks.",
        turns=[PolishedTurn(speaker="Nikki Staltari", text="Welcome everyone.")],
    )
    assert PolishedChapter.model_validate_json(chapter.model_dump_json()) == chapter


def test_transcript_products_round_trips() -> None:
    products = TranscriptProducts(
        context="meeting",
        meeting_summary="Discussed the roadmap.",
        discussion_notes="- talked about X\n- decided Y",
        chapters=[
            PolishedChapter(
                title="Intro",
                start_seconds=0.0,
                summary="Opening remarks.",
                turns=[
                    PolishedTurn(speaker="Nikki Staltari", text="Welcome everyone.")
                ],
            )
        ],
    )
    assert (
        TranscriptProducts.model_validate_json(products.model_dump_json()) == products
    )
