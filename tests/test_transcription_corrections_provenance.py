from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from jake_tools.cli import main
from jake_tools.transcription.cache import RunCache
from jake_tools.transcription.chapters import ChapterList
from jake_tools.transcription.models import (
    PolishedChapter,
    PolishedTurn,
    RawTranscript,
    SnippetRequest,
    SpeakerCorrection,
    TimestampedSegment,
    Utterance,
)
from jake_tools.transcription.note import NoteSection, ParsedNote
from jake_tools.transcription.polish import (
    PolishedChapterList,
    validate_polished_chapter,
)
from jake_tools.transcription.speakers import (
    AssignmentSet,
    CorrectionSet,
    InvalidCorrectionError,
    SpeakerCorrectionConflictError,
    _merge_corrections,
    parse_correct,
    resolve,
)


class _Agent:
    def __init__(self, proposals: list[dict[str, object]]) -> None:
        self.proposals = proposals
        self.calls: list[object] = []

    async def run_structured(
        self, prompt: object, *args: object
    ) -> tuple[object, object]:
        self.calls.append(prompt)
        from jake_tools.transcription.speakers import SpeakerProposals

        return SpeakerProposals.model_validate({"proposals": self.proposals}), None


class _Audio:
    def cut(self, source: Path, start: float, end: float, out: Path) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"clip")
        return out


def _transcript() -> RawTranscript:
    return RawTranscript(
        clips=[],
        audio_sha256="audio-hash",
        utterances=[
            Utterance(start=0, end=2, speaker="SPEAKER_00", text="first exact quote"),
            Utterance(start=2, end=4, speaker="SPEAKER_00", text="second exact quote"),
            Utterance(start=4, end=6, speaker="SPEAKER_00", text="third exact quote"),
        ],
    )


def test_parse_correct_accepts_half_open_range() -> None:
    assert parse_correct("SPEAKER_00=1.5-4.0=Ada Lovelace") == SpeakerCorrection(
        cluster="SPEAKER_00", start_seconds=1.5, end_seconds=4.0, name="Ada Lovelace"
    )


def test_parse_correct_rejects_non_finite_or_empty_ranges() -> None:
    with pytest.raises(InvalidCorrectionError):
        parse_correct("SPEAKER_00=nan-4=Ada")
    with pytest.raises(InvalidCorrectionError):
        parse_correct("SPEAKER_00=4-4=Ada")


@pytest.mark.asyncio
async def test_range_only_relabels_fully_contained_utterance_and_preserves_uncovered_input(
    tmp_path: Path,
) -> None:
    transcript = _transcript()
    agent = _Agent(
        [
            {
                "cluster": "SPEAKER_00",
                "name": "Grace Hopper",
                "confidence": "high",
                "reasoning": "must be ignored for mixed cluster",
            }
        ]
    )
    resolved, requests = await resolve(
        _note(),
        transcript,
        agent=agent,  # type: ignore[arg-type]
        assignments=[],
        corrections=[
            SpeakerCorrection(
                cluster="SPEAKER_00",
                start_seconds=2.0,
                end_seconds=6.0,
                name="Ada Lovelace",
            )
        ],
        audio_tool=_Audio(),  # type: ignore[arg-type]
        merged_audio=tmp_path / "missing.m4a",
        snippet_dir=tmp_path / "snippets",
    )
    assert [u.speaker for u in resolved.utterances] == [
        "SPEAKER_00",
        "Ada Lovelace",
        "Ada Lovelace",
    ]
    assert [segment.start_seconds for segment in requests[0].segments] == [0]
    assert requests[0].segments[0].text == "first exact quote"
    assert agent.calls == []


@pytest.mark.asyncio
async def test_partial_overlap_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(InvalidCorrectionError, match="partial"):
        await resolve(
            _note(),
            _transcript(),
            agent=_Agent([]),  # type: ignore[arg-type]
            assignments=[],
            corrections=[
                SpeakerCorrection(
                    cluster="SPEAKER_00",
                    start_seconds=1,
                    end_seconds=3,
                    name="Ada Lovelace",
                )
            ],
            audio_tool=_Audio(),  # type: ignore[arg-type]
            merged_audio=tmp_path / "missing.m4a",
            snippet_dir=tmp_path / "snippets",
        )


def test_conflicting_corrections_fail_but_identical_duplicates_are_idempotent() -> None:
    correction = parse_correct("SPEAKER_00=0-2=Ada Lovelace")
    assert _merge_corrections([correction], [correction]) == [correction]
    with pytest.raises(SpeakerCorrectionConflictError):
        _merge_corrections(
            [correction],
            [parse_correct("SPEAKER_00=1-3=Grace Hopper")],
        )


def test_snippet_request_serialises_timestamped_evidence_additively() -> None:
    request = SnippetRequest(
        cluster="SPEAKER_00",
        clip_paths=[],
        context="quote",
        segments=[
            TimestampedSegment(
                start_seconds=1.0,
                end_seconds=2.0,
                speaker="SPEAKER_00",
                text="quote",
            )
        ],
    )
    payload = request.model_dump()
    assert payload["clip_paths"] == []
    assert payload["context"] == "quote"
    assert payload["segments"][0]["start_seconds"] == 1.0


def test_range_hint_contains_exact_quote_range_audio_identity_and_no_cluster_id() -> (
    None
):
    from jake_tools.transcription.speakers import _hint_lines_for_corrections

    lines = _hint_lines_for_corrections(
        [
            SpeakerCorrection(
                cluster="SPEAKER_00",
                start_seconds=0,
                end_seconds=4,
                name="Ada Lovelace",
            ),
            SpeakerCorrection(
                cluster="SPEAKER_00",
                start_seconds=4,
                end_seconds=6,
                name="Unknown",
            ),
        ],
        _transcript().utterances,
        audio_sha256="audio-hash",
        recording_identity="Recording 20260803",
    )
    assert lines == [
        'Ada Lovelace said "first exact quote second exact quote" from 0.0-4.0 seconds in Recording 20260803 (audio audio-hash)'
    ]
    assert "SPEAKER_00" not in lines[0]


def test_cache_invalidation_removes_downstream_but_preserves_evidence_and_baselines(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path)
    cache.store("run", "raw_transcript", _transcript())
    cache.store("run", "assignments", AssignmentSet(assignments=[]))
    cache.store("run", "corrections", CorrectionSet(corrections=[]))
    cache.store("run", "chapters", _chapters())
    cache.store("run", "polished", PolishedChapterList(chapters=[]))
    cache.store_text("run", "baseline_notes", "human baseline")
    cache.invalidate_downstream("run", reason="speaker correction")
    assert cache.load("run", "raw_transcript", RawTranscript) is not None
    assert cache.load("run", "assignments", AssignmentSet) is not None
    assert cache.load("run", "corrections", CorrectionSet) is not None
    assert cache.load("run", "chapters", type(_chapters())) is None
    assert cache.load_text("run", "baseline_notes") == "human baseline"
    assert (cache.run_dir("run") / "stale.json").exists()


def test_unknown_cannot_change_speaker_and_partition_is_exact() -> None:
    source = [
        _utterance(0, "Unknown", "unknown words"),
        _utterance(1, "Ada", "named words"),
    ]
    chapter = PolishedChapter(
        title="x",
        start_seconds=0,
        summary="s",
        turns=[
            PolishedTurn(
                speaker="Unknown", text="unknown words", source_turn_indices=[0]
            ),
            PolishedTurn(speaker="Ada", text="named words", source_turn_indices=[1]),
        ],
    )
    assert validate_polished_chapter(chapter, source) == chapter
    bad = chapter.model_copy(
        update={
            "turns": [
                chapter.turns[0].model_copy(update={"speaker": "Ada"}),
                chapter.turns[1],
            ]
        }
    )
    with pytest.raises(ValueError, match="speaker"):
        validate_polished_chapter(bad, source)


def test_legacy_polish_without_provenance_is_readable_but_not_verified() -> None:
    chapter = PolishedChapter(
        title="legacy",
        start_seconds=0,
        summary="s",
        turns=[PolishedTurn(speaker="Ada", text="x")],
    )
    assert chapter.turns[0].source_turn_indices == []
    with pytest.raises(ValueError, match="provenance"):
        validate_polished_chapter(chapter, [_utterance(0, "Ada", "x")])


def test_cli_keeps_assign_and_adds_repeatable_correct_option() -> None:
    result = CliRunner().invoke(main, ["transcript", "speakers", "--help"])
    assert result.exit_code == 0
    assert "--assign" in result.output
    assert "--correct" in result.output


def _utterance(index: int, speaker: str, text: str) -> Utterance:
    return Utterance(
        start=float(index), end=float(index) + 0.5, speaker=speaker, text=text
    )


def _chapters() -> ChapterList:
    return ChapterList(chapters=[])


def _note() -> ParsedNote:
    return ParsedNote(
        path="note.md",
        frontmatter={"Date": "[[Recording 20260803]]"},
        context="meeting",
        attendees=["Ada Lovelace", "Grace Hopper"],
        diarisation_hints=[],
        embeds=[],
        sections=[NoteSection(heading="Meeting Prep", level=2, body="")],
    )
