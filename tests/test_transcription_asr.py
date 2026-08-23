"""Behaviour of local ASR + diarisation and its `align()` pure function.

Per the plan, `align()` carries the real test weight: it's a pure function
(hand-built segment/turn lists in, `Utterance` list out) exercised without
any model. `transcribe_merged_audio` orchestration is tested next with an
injected fake `Transcriber` (per `memo-cli-options.md` rule 4 — behaviour
tests live at the library seam). The `jake-tools transcript asr` CLI test
stays thin: flag parsing -> options object -> delegation, monkeypatching the
library function, mirroring `test_transcript_cli.py`'s `merge-audio` tests.
The `@pytest.mark.live` test is the only one that touches real models.
"""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from jake_tools.cli import main
from jake_tools.cli.transcript_options import TranscriberOptions, transcriber_options
from jake_tools.transcription.asr import (
    UNKNOWN_SPEAKER,
    AsrSegment,
    DiarisedTurn,
    LocalTranscriber,
    align,
    transcribe_merged_audio,
)
from jake_tools.transcription.cache import RunCache
from jake_tools.transcription.models import RawTranscript, SourceClip, Utterance

# `jake_tools.cli`'s __init__ rebinds the name `transcript` to the Click
# group, shadowing the submodule (see `test_transcript_cli.py`) — fetch the
# actual module via importlib to monkeypatch its `transcribe_merged_audio`
# binding.
transcript_cli = importlib.import_module("jake_tools.cli.transcript")


# --- align(): clean alternation --------------------------------------------


def test_align_assigns_each_segment_to_its_single_overlapping_speaker() -> None:
    segments = [
        AsrSegment(start=0.0, end=2.0, text="hello there"),
        AsrSegment(start=2.0, end=4.0, text="how are you"),
    ]
    turns = [
        DiarisedTurn(start=0.0, end=2.0, speaker="SPEAKER_00"),
        DiarisedTurn(start=2.0, end=4.0, speaker="SPEAKER_01"),
    ]

    result = align(segments, turns)

    assert result == [
        Utterance(start=0.0, end=2.0, speaker="SPEAKER_00", text="hello there"),
        Utterance(start=2.0, end=4.0, speaker="SPEAKER_01", text="how are you"),
    ]


def test_align_output_is_monotonic_for_monotonic_input() -> None:
    segments = [
        AsrSegment(start=0.0, end=2.0, text="one"),
        AsrSegment(start=2.0, end=5.0, text="two"),
        AsrSegment(start=5.0, end=9.0, text="three"),
    ]
    turns = [
        DiarisedTurn(start=0.0, end=5.0, speaker="SPEAKER_00"),
        DiarisedTurn(start=5.0, end=9.0, speaker="SPEAKER_01"),
    ]

    result = align(segments, turns)

    starts = [utterance.start for utterance in result]
    ends = [utterance.end for utterance in result]
    assert starts == sorted(starts)
    assert ends == sorted(ends)
    assert all(u.start <= u.end for u in result)


# --- align(): overlap ties ---------------------------------------------


def test_align_splits_deterministically_on_an_exact_overlap_tie() -> None:
    """Two diarised turns overlap one segment by exactly the same duration.

    There's no single "maximal" overlap to pick, so this is treated as a
    genuine mid-segment switch: the segment splits at the turn boundary,
    with words divided in proportion to each side's (equal) share.
    """
    segments = [AsrSegment(start=0.0, end=4.0, text="one two three four")]
    turns = [
        DiarisedTurn(start=0.0, end=2.0, speaker="SPEAKER_00"),
        DiarisedTurn(start=2.0, end=4.0, speaker="SPEAKER_01"),
    ]

    result = align(segments, turns)

    assert result == [
        Utterance(start=0.0, end=2.0, speaker="SPEAKER_00", text="one two"),
        Utterance(start=2.0, end=4.0, speaker="SPEAKER_01", text="three four"),
    ]


def test_align_tie_split_is_order_independent_in_the_turns_list() -> None:
    """The tiebreak comes from each turn's own `start`, not list order."""
    segments = [AsrSegment(start=0.0, end=4.0, text="one two three four")]
    turns_reversed = [
        DiarisedTurn(start=2.0, end=4.0, speaker="SPEAKER_01"),
        DiarisedTurn(start=0.0, end=2.0, speaker="SPEAKER_00"),
    ]

    result = align(segments, turns_reversed)

    assert [u.speaker for u in result] == ["SPEAKER_00", "SPEAKER_01"]


# --- align(): mid-segment speaker switch --------------------------------


def test_align_splits_unevenly_in_proportion_to_overlap_duration() -> None:
    segments = [AsrSegment(start=0.0, end=10.0, text="a b c d e f g h i j")]
    turns = [
        DiarisedTurn(start=0.0, end=7.0, speaker="SPEAKER_00"),
        DiarisedTurn(start=7.0, end=10.0, speaker="SPEAKER_01"),
    ]

    result = align(segments, turns)

    assert result == [
        Utterance(start=0.0, end=7.0, speaker="SPEAKER_00", text="a b c d e f g"),
        Utterance(start=7.0, end=10.0, speaker="SPEAKER_01", text="h i j"),
    ]


def test_align_merges_consecutive_same_speaker_turns_before_splitting() -> None:
    """Two turns for the same speaker either side of a middle turn for a
    different speaker should not fragment the same-speaker text further
    than the genuine switches require."""
    segments = [AsrSegment(start=0.0, end=9.0, text="a b c d e f g h i")]
    turns = [
        DiarisedTurn(start=0.0, end=3.0, speaker="SPEAKER_00"),
        DiarisedTurn(start=3.0, end=6.0, speaker="SPEAKER_01"),
        DiarisedTurn(start=6.0, end=9.0, speaker="SPEAKER_00"),
    ]

    result = align(segments, turns)

    assert [u.speaker for u in result] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert [(u.start, u.end) for u in result] == [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0)]


# --- align(): diarised speech with no ASR text --------------------------


def test_align_drops_diarised_turns_with_no_overlapping_asr_text() -> None:
    """A diarised turn that no ASR segment ever touches produces no
    utterance. `align()` walks the ASR segments (the carriers of text), so
    diarisation-only spans are silently dropped rather than emitted as
    empty-text utterances — see the module docstring for why."""
    segments = [AsrSegment(start=0.0, end=2.0, text="hello")]
    turns = [
        DiarisedTurn(start=0.0, end=2.0, speaker="SPEAKER_00"),
        DiarisedTurn(start=5.0, end=7.0, speaker="SPEAKER_01"),  # never overlaps
    ]

    result = align(segments, turns)

    assert result == [Utterance(start=0.0, end=2.0, speaker="SPEAKER_00", text="hello")]


def test_align_assigns_unknown_speaker_when_no_turn_overlaps_at_all() -> None:
    segments = [AsrSegment(start=0.0, end=2.0, text="hello")]
    turns: list[DiarisedTurn] = []

    result = align(segments, turns)

    assert result == [
        Utterance(start=0.0, end=2.0, speaker=UNKNOWN_SPEAKER, text="hello")
    ]


# --- transcriber_options: flag parsing -> options object -------------------


@click.command()
@transcriber_options
def _transcriber_probe(transcriber_options: TranscriberOptions) -> None:
    click.echo(
        f"hf_token={transcriber_options.hf_token} "
        f"asr_model={transcriber_options.asr_model} "
        f"diarisation_model={transcriber_options.diarisation_model}"
    )


def test_transcriber_options_decorator_builds_options_from_flags() -> None:
    result = CliRunner().invoke(
        _transcriber_probe,
        [
            "--hf-token",
            "hf_fake_token",
            "--asr-model",
            "custom/asr",
            "--diarisation-model",
            "custom/diarisation",
        ],
    )

    assert result.exit_code == 0
    assert result.output == (
        "hf_token=hf_fake_token asr_model=custom/asr diarisation_model=custom/diarisation\n"
    )


def test_transcriber_options_decorator_defaults_hf_token_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    result = CliRunner().invoke(_transcriber_probe, [])

    assert result.exit_code == 0
    assert "hf_token=None" in result.output


def test_transcriber_options_decorator_reads_hf_token_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    result = CliRunner().invoke(_transcriber_probe, [])

    assert result.exit_code == 0
    assert "hf_token=hf_from_env" in result.output


# --- transcribe_merged_audio orchestration ----------------------------------


class FakeTranscriber:
    """Fake Transcriber that records calls instead of running real models."""

    def __init__(self, result: RawTranscript) -> None:
        self.result = result
        self.calls: list[Path] = []

    def transcribe(self, audio: Path) -> RawTranscript:
        self.calls.append(audio)
        return self.result


def _sample_raw_transcript() -> RawTranscript:
    return RawTranscript(
        clips=[
            SourceClip(path="/merged.m4a", offset_seconds=0.0, duration_seconds=4.0)
        ],
        utterances=[Utterance(start=0.0, end=4.0, speaker="SPEAKER_00", text="hello")],
        audio_sha256="abc123",
    )


def test_transcribe_merged_audio_transcribes_and_caches_on_a_cache_miss(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "merged.m4a"
    audio.write_bytes(b"fake audio")
    cache = RunCache(tmp_path / "cache")
    fake = FakeTranscriber(_sample_raw_transcript())

    result = transcribe_merged_audio(
        audio, run_id="run-1", transcriber=fake, cache=cache
    )

    assert result == fake.result
    assert fake.calls == [audio]
    assert cache.load("run-1", "raw_transcript", RawTranscript) == fake.result


def test_transcribe_merged_audio_returns_cached_result_without_calling_transcriber(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "merged.m4a"
    audio.write_bytes(b"fake audio")
    cache = RunCache(tmp_path / "cache")
    cached_result = _sample_raw_transcript()
    cache.store("run-1", "raw_transcript", cached_result)
    fake = FakeTranscriber(
        RawTranscript(clips=[], utterances=[], audio_sha256="should-not-be-used")
    )

    result = transcribe_merged_audio(
        audio, run_id="run-1", transcriber=fake, cache=cache
    )

    assert result == cached_result
    assert fake.calls == []


# --- CLI: `transcript asr` delegation ---------------------------------------


def test_asr_delegates_to_transcribe_merged_audio_and_prints_its_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}
    fake_result = _sample_raw_transcript()
    audio_path = tmp_path / "merged.m4a"
    audio_path.write_bytes(b"fake audio")

    def fake_transcribe_merged_audio(
        audio: Path, *, run_id: str, transcriber: object, cache: object
    ) -> RawTranscript:
        captured["audio"] = audio
        captured["run_id"] = run_id
        captured["transcriber"] = transcriber
        captured["cache"] = cache
        return fake_result

    monkeypatch.setattr(
        transcript_cli, "transcribe_merged_audio", fake_transcribe_merged_audio
    )
    runner = CliRunner()

    # No --hf-token/HF_TOKEN: the fake never actually calls the transcriber,
    # so a missing token must not stop delegation from happening.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    result = runner.invoke(
        main, ["transcript", "asr", str(audio_path), "--run-id", "meeting-abc123"]
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == json.loads(fake_result.model_dump_json())
    assert captured["audio"] == audio_path
    assert captured["run_id"] == "meeting-abc123"
    assert isinstance(captured["transcriber"], LocalTranscriber)


def test_transcript_asr_help_exits_zero() -> None:
    result = CliRunner().invoke(main, ["transcript", "asr", "--help"])

    assert result.exit_code == 0
    assert "asr" in result.output.lower() or "diaris" in result.output.lower()


def test_transcript_asr_requires_run_id(tmp_path: Path) -> None:
    audio_path = tmp_path / "merged.m4a"
    audio_path.write_bytes(b"fake audio")

    result = CliRunner().invoke(main, ["transcript", "asr", str(audio_path)])

    assert result.exit_code != 0
    assert "run-id" in result.output.lower()


# --- live: real LocalTranscriber --------------------------------------------


def _generate_speech_sample(tmp_path: Path) -> Path:
    """A short, real speech clip synthesised locally via macOS `say` + ffmpeg.

    Generated fresh each run rather than committed, since the whole
    transcription pipeline is already macOS/Apple-Silicon-only (MLX).
    """
    aiff = tmp_path / "sample.aiff"
    subprocess.run(
        [
            "say",
            "-o",
            str(aiff),
            "This is a short test recording for the jake tools transcription pipeline.",
        ],
        check=True,
        capture_output=True,
    )
    wav = tmp_path / "sample.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(aiff), "-ac", "1", "-ar", "16000", str(wav)],
        check=True,
        capture_output=True,
    )
    return wav


@pytest.mark.live
def test_local_transcriber_produces_nonempty_monotonic_utterances(
    tmp_path: Path,
) -> None:
    import os

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        pytest.fail(
            "HF_TOKEN must be set (and belong to an account that has "
            "accepted pyannote/speaker-diarization-3.1's licence) to run "
            "the live ASR+diarisation test."
        )

    audio = _generate_speech_sample(tmp_path)
    transcriber = LocalTranscriber(hf_token=hf_token)

    result = transcriber.transcribe(audio)

    assert result.utterances
    assert all(u.text.strip() for u in result.utterances)
    starts = [u.start for u in result.utterances]
    ends = [u.end for u in result.utterances]
    assert starts == sorted(starts)
    assert ends == sorted(ends)
    assert all(u.start <= u.end for u in result.utterances)
    assert result.audio_sha256
    assert result.clips and result.clips[0].path == str(audio)
