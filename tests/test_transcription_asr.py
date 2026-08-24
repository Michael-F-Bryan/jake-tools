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
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from jake_tools.cli import main
from jake_tools.cli.transcript_options import TranscriberOptions, transcriber_options
from jake_tools.transcription import asr as asr_module
from jake_tools.transcription.asr import (
    DEFAULT_ASR_CHUNK_DURATION,
    DEFAULT_ASR_CHUNK_OVERLAP,
    UNKNOWN_SPEAKER,
    AsrSegment,
    DiarisedTurn,
    LocalTranscriber,
    MissingHfTokenError,
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


# --- align(): genuine cross-talk (overlapping diarised turns) -----------


def test_align_permits_overlapping_utterances_for_genuine_cross_talk() -> None:
    """pyannote's overlap-aware segmentation reports genuinely overlapping
    turns for real cross-talk. align() must not clamp, truncate, or drop
    that overlap away — it's information the polish stage needs to untangle
    interleaved speech. Reviewer's repro shape: one segment 0-10s, a
    SPEAKER_00 turn 0-6s and a SPEAKER_01 turn 4-10s (both overlap the
    segment, and each other, from 4-6s)."""
    segments = [AsrSegment(start=0.0, end=10.0, text="a b c d e f g h i j")]
    turns = [
        DiarisedTurn(start=0.0, end=6.0, speaker="SPEAKER_00"),
        DiarisedTurn(start=4.0, end=10.0, speaker="SPEAKER_01"),
    ]

    result = align(segments, turns)

    # Both turns are 6s long (0-6 and 4-10), so an even word split is
    # expected — the point of this test is the overlapping timestamps, not
    # the exact word boundary.
    assert result == [
        Utterance(start=0.0, end=6.0, speaker="SPEAKER_00", text="a b c d e"),
        Utterance(start=4.0, end=10.0, speaker="SPEAKER_01", text="f g h i j"),
    ]
    # The 4-6s overlap is real and preserved, not a bug.
    assert result[0].end > result[1].start
    # The ordering contract is a total order by start (tiebroken by
    # (start, end, speaker)), not disjointness.
    starts = [u.start for u in result]
    assert starts == sorted(starts)


def test_align_is_deterministic_regardless_of_diarised_turn_input_order() -> None:
    segments = [AsrSegment(start=0.0, end=10.0, text="a b c d e f g h i j")]
    turns_forward = [
        DiarisedTurn(start=0.0, end=6.0, speaker="SPEAKER_00"),
        DiarisedTurn(start=4.0, end=10.0, speaker="SPEAKER_01"),
    ]
    turns_reversed = list(reversed(turns_forward))

    assert align(segments, turns_forward) == align(segments, turns_reversed)


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


def test_align_drops_a_split_share_that_rounds_down_to_zero_words() -> None:
    """When a segment's text is too short to give every genuine speaker a
    non-empty share, the share that would round to zero words is dropped
    entirely rather than emitted as an empty-text utterance — consistent
    with the module's no-empty-text intent (see `_apportion`'s docstring).
    Reviewer's repro: a 1-word segment split between a 9.9s turn and a
    0.1s turn — the 0.1s turn's share rounds to zero words."""
    segments = [AsrSegment(start=0.0, end=10.0, text="hi")]
    turns = [
        DiarisedTurn(start=0.0, end=9.9, speaker="SPEAKER_00"),
        DiarisedTurn(start=9.9, end=10.0, speaker="SPEAKER_01"),
    ]

    result = align(segments, turns)

    assert result == [Utterance(start=0.0, end=9.9, speaker="SPEAKER_00", text="hi")]
    assert all(u.text for u in result)  # no empty-text utterances


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


# --- LocalTranscriber: chunked ASR for long audio ---------------------------


class _FakeAlignedSentence:
    """Shaped like `parakeet_mlx.alignment.AlignedSentence` (start/end/text
    are the only attributes `_run_asr` reads)."""

    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text


class _FakeAlignedResult:
    def __init__(self, sentences: list[_FakeAlignedSentence]) -> None:
        self.sentences = sentences


class _FakeParakeetModel:
    """Fake parakeet-mlx model: records how `.transcribe()` was called
    instead of running real MLX inference. `chunk_progress` simulates two
    `chunk_callback` invocations (parakeet-mlx's own per-chunk progress
    hook — see `_run_asr`), so tests can assert it's wired to stage
    logging without a real chunked encode."""

    def __init__(
        self, sentences: list[_FakeAlignedSentence], *, sample_rate: int = 16000
    ) -> None:
        self._sentences = sentences
        self.preprocessor_config = SimpleNamespace(sample_rate=sample_rate)
        self.transcribe_calls: list[dict[str, object]] = []

    def transcribe(
        self,
        path: Path,
        *,
        chunk_duration: float | None = None,
        overlap_duration: float = 15.0,
        chunk_callback: Callable[[int, int], None] | None = None,
    ) -> _FakeAlignedResult:
        self.transcribe_calls.append(
            {
                "path": path,
                "chunk_duration": chunk_duration,
                "overlap_duration": overlap_duration,
            }
        )
        if chunk_callback is not None:
            sample_rate = self.preprocessor_config.sample_rate
            chunk_callback(sample_rate * 100, sample_rate * 200)
        return _FakeAlignedResult(self._sentences)


class _FakeDiarisationAnnotation:
    def __init__(self, turns: list[tuple[float, float, str]]) -> None:
        self._turns = turns

    def itertracks(self, yield_label: bool = True) -> object:
        for start, end, speaker in self._turns:
            yield SimpleNamespace(start=start, end=end), None, speaker


class _FakePyannotePipeline:
    """Fake pyannote pipeline: records what it was called with instead of
    running real diarisation inference. Shaped like pyannote-audio 4.x's
    `DiarizeOutput` — a `.speaker_diarization` attribute holding the
    overlap-inclusive `Annotation`, not a bare `Annotation` return value
    (that was pyannote-audio 3.x's shape)."""

    def __init__(self, turns: list[tuple[float, float, str]]) -> None:
        self._turns = turns
        self.call_args: list[object] = []

    def __call__(self, file: object) -> SimpleNamespace:
        self.call_args.append(file)
        return SimpleNamespace(
            speaker_diarization=_FakeDiarisationAnnotation(self._turns)
        )


class _FakeAudioSamples:
    def __init__(self, data: object, sample_rate: int) -> None:
        self.data = data
        self.sample_rate = sample_rate


class _FakeAudioDecoder:
    """Fake `torchcodec.decoders.AudioDecoder` (constructed fresh per call,
    same as the real one): always hands back the same canned samples
    instead of running real audio decoding."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def get_all_samples(self) -> _FakeAudioSamples:
        return _FakeAudioSamples(data="fake-waveform-tensor", sample_rate=48000)


def test_local_transcriber_enables_parakeets_own_chunking_for_long_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression test for the real-meeting failure: parakeet-mlx's Conformer
    encoder runs full self-attention over the *entire* input in one shot
    unless `chunk_duration` is passed to `model.transcribe(...)` — memory
    cost grows with the square of audio length. A real ~31-minute meeting
    turned an un-chunked encode into a single 34GB Metal allocation
    (`[metal::malloc] Attempting to allocate 34091191040 bytes...`). This
    asserts `LocalTranscriber` always passes `chunk_duration`/
    `overlap_duration` through to parakeet's own chunking mechanism, rather
    than silently falling back to whole-file mode — and that timestamps
    coming back from a (simulated) chunked/stitched result stay
    meeting-relative and monotonic once converted into `AsrSegment`s and
    diarised utterances."""
    # A sentence near 31 minutes in stands in for a chunk parakeet-mlx's
    # own stitching (`merge_longest_contiguous`) already re-based onto
    # file-relative time — well past what a single un-chunked window could
    # safely encode without this fix.
    fake_sentences = [
        _FakeAlignedSentence(0.0, 4.0, " Hello there."),
        _FakeAlignedSentence(1860.0, 1865.0, " Near the end of a long meeting."),
    ]
    fake_model = _FakeParakeetModel(fake_sentences)
    fake_pipeline = _FakePyannotePipeline(
        [(0.0, 4.0, "SPEAKER_00"), (1860.0, 1865.0, "SPEAKER_01")]
    )
    monkeypatch.setattr(
        asr_module, "parakeet_from_pretrained", lambda model_id: fake_model
    )
    monkeypatch.setattr(
        asr_module.Pipeline,
        "from_pretrained",
        lambda checkpoint, token=None: fake_pipeline,
    )
    monkeypatch.setattr(asr_module, "AudioDecoder", _FakeAudioDecoder)
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")

    transcriber = LocalTranscriber(hf_token="fake-token")
    result = transcriber.transcribe(audio)

    assert fake_model.transcribe_calls == [
        {
            "path": audio,
            "chunk_duration": DEFAULT_ASR_CHUNK_DURATION,
            "overlap_duration": DEFAULT_ASR_CHUNK_OVERLAP,
        }
    ]
    # Diarisation gets an in-memory waveform dict, not a bare path — see
    # _run_diarisation's docstring comment for why (the real-meeting
    # sample-count assertion failure this sidesteps).
    assert fake_pipeline.call_args == [
        {"waveform": "fake-waveform-tensor", "sample_rate": 48000}
    ]
    assert result.utterances == [
        Utterance(start=0.0, end=4.0, speaker="SPEAKER_00", text="Hello there."),
        Utterance(
            start=1860.0,
            end=1865.0,
            speaker="SPEAKER_01",
            text="Near the end of a long meeting.",
        ),
    ]
    starts = [u.start for u in result.utterances]
    assert starts == sorted(starts)  # meeting-relative, monotonic


# --- LocalTranscriber: memory hardening (caps, cleanup, stage logging) ------


def test_local_transcriber_caps_accelerator_memory_before_each_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`mx.set_memory_limit`/`torch.mps.set_per_process_memory_fraction` give
    MLX and torch's MPS allocator a fail-fast ceiling instead of letting an
    oversized allocation grow unbounded (see `DEFAULT_MPS_MEMORY_FRACTION`).
    Both must be set — the MLX one before ASR loads its model, the MPS one
    before diarisation loads its pipeline."""
    fake_model = _FakeParakeetModel([_FakeAlignedSentence(0.0, 1.0, "hi")])
    fake_pipeline = _FakePyannotePipeline([(0.0, 1.0, "SPEAKER_00")])
    monkeypatch.setattr(
        asr_module, "parakeet_from_pretrained", lambda model_id: fake_model
    )
    monkeypatch.setattr(
        asr_module.Pipeline,
        "from_pretrained",
        lambda checkpoint, token=None: fake_pipeline,
    )
    monkeypatch.setattr(asr_module, "AudioDecoder", _FakeAudioDecoder)
    mx_limit_calls: list[int] = []
    mps_fraction_calls: list[float] = []
    monkeypatch.setattr(asr_module.mx, "set_memory_limit", mx_limit_calls.append)
    monkeypatch.setattr(
        asr_module.torch.mps,
        "set_per_process_memory_fraction",
        mps_fraction_calls.append,
    )
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")

    LocalTranscriber(hf_token="fake-token").transcribe(audio)

    assert mx_limit_calls == [asr_module.default_memory_budget_bytes()]
    assert mps_fraction_calls == [asr_module.DEFAULT_MPS_MEMORY_FRACTION]


def test_local_transcriber_clears_mlx_cache_before_diarisation_pipeline_loads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sequential peaks: parakeet's model must be released and MLX's buffer
    cache cleared before the diarisation pipeline loads, not after — so the
    two models' memory footprints never have to coexist. Asserted via call
    order between `mx.clear_cache` and `Pipeline.from_pretrained`, both real
    collaborators of the method under test rather than its own internals."""
    order: list[str] = []
    fake_model = _FakeParakeetModel([_FakeAlignedSentence(0.0, 1.0, "hi")])
    fake_pipeline = _FakePyannotePipeline([(0.0, 1.0, "SPEAKER_00")])

    def _fake_from_pretrained(
        checkpoint: str, token: str | None = None
    ) -> _FakePyannotePipeline:
        order.append("Pipeline.from_pretrained")
        return fake_pipeline

    monkeypatch.setattr(
        asr_module, "parakeet_from_pretrained", lambda model_id: fake_model
    )
    monkeypatch.setattr(asr_module.Pipeline, "from_pretrained", _fake_from_pretrained)
    monkeypatch.setattr(asr_module, "AudioDecoder", _FakeAudioDecoder)
    monkeypatch.setattr(
        asr_module.mx, "clear_cache", lambda: order.append("mx.clear_cache")
    )
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")

    LocalTranscriber(hf_token="fake-token").transcribe(audio)

    assert order == ["mx.clear_cache", "Pipeline.from_pretrained"]


def test_local_transcriber_logs_flushed_stage_progress_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stage-progress lines are the only trail a run leaves if it dies
    mid-stage — the real incident's stderr log was completely empty. This
    asserts each stage's start/complete line lands on stderr, plus
    parakeet-mlx's own per-chunk progress hook (`chunk_callback`) being
    wired through to the same logging."""
    fake_model = _FakeParakeetModel([_FakeAlignedSentence(0.0, 1.0, "hi")])
    fake_pipeline = _FakePyannotePipeline([(0.0, 1.0, "SPEAKER_00")])
    monkeypatch.setattr(
        asr_module, "parakeet_from_pretrained", lambda model_id: fake_model
    )
    monkeypatch.setattr(
        asr_module.Pipeline,
        "from_pretrained",
        lambda checkpoint, token=None: fake_pipeline,
    )
    monkeypatch.setattr(asr_module, "AudioDecoder", _FakeAudioDecoder)
    audio = tmp_path / "meeting.m4a"
    audio.write_bytes(b"fake audio")

    LocalTranscriber(hf_token="fake-token").transcribe(audio)

    stderr = capsys.readouterr().err
    assert "[transcribe] ASR starting" in stderr
    assert "[transcribe] ASR progress:" in stderr
    assert "[transcribe] ASR complete in" in stderr
    assert "[transcribe] diarisation starting" in stderr
    assert "[transcribe] diarisation complete in" in stderr


# --- LocalTranscriber: missing HF_TOKEN -------------------------------------


def test_local_transcriber_transcribe_raises_missing_hf_token_error_without_a_token(
    tmp_path: Path,
) -> None:
    """The token check runs before any file or model access, so this is
    fully testable offline — a hard constraint the plan calls out
    explicitly, so it gets its own test rather than only living inside the
    live test's setup."""
    transcriber = LocalTranscriber(hf_token=None)

    with pytest.raises(MissingHfTokenError, match="HF_TOKEN"):
        transcriber.transcribe(tmp_path / "does-not-need-to-exist.m4a")


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
