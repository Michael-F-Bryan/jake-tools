from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from jake_tools.cli import main
from jake_tools.cli.transcript_options import TranscriberOptions
from jake_tools.transcription import asr as asr_module
from jake_tools.transcription.asr import (
    DEFAULT_DIARISATION_MODEL,
    AcceleratorOutOfMemoryError,
    AsrCheckpoint,
    AsrSegment,
    DiarisationCheckpoint,
    DiarisedTurn,
    LocalTranscriber,
    TranscriberError,
    select_diarisation_device,
    transcribe_merged_audio,
)
from jake_tools.transcription.cache import RunCache, sha256_of
from jake_tools.transcription.models import RawTranscript


class FakeAnnotation:
    def itertracks(self, yield_label: bool = True):
        yield SimpleNamespace(start=0.0, end=2.0), None, "SPEAKER_00"


class FakePipeline:
    def __init__(self) -> None:
        self.devices: list[object] = []
        self.calls: list[dict[str, object]] = []

    def to(self, device: object) -> FakePipeline:
        self.devices.append(device)
        return self

    def __call__(self, payload: object, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(speaker_diarization=FakeAnnotation())


def test_community_one_is_the_default_diarisation_checkpoint() -> None:
    assert DEFAULT_DIARISATION_MODEL == "pyannote/speaker-diarization-community-1"


def test_auto_selects_cpu_when_mps_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asr_module.torch.backends.mps, "is_available", lambda: False)

    assert select_diarisation_device("auto") == "cpu"


def test_explicit_cpu_does_not_probe_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        asr_module.torch.backends.mps,
        "is_available",
        lambda: pytest.fail("explicit CPU selection probed MPS"),
    )

    assert select_diarisation_device("cpu") == "cpu"


def test_explicit_mps_refuses_to_fallback_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asr_module.torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(TranscriberError, match="MPS is unavailable"):
        select_diarisation_device("mps")


def test_progress_hook_logs_named_steps_without_poll_spam(
    capsys: pytest.CaptureFixture[str],
) -> None:
    hook = asr_module._DiarisationProgressHook()
    hook("segmentation", None, completed=1, total=10)
    hook("segmentation", None, completed=2, total=10)
    hook("embeddings", None, completed=1, total=2)
    hook("clustering", None, completed=1, total=1)

    lines = capsys.readouterr().err.splitlines()
    assert sum("segmentation" in line for line in lines) == 1
    assert any("embeddings" in line for line in lines)
    assert any("clustering" in line for line in lines)


def test_transcriber_options_expose_device_and_num_speakers() -> None:
    options = TranscriberOptions(
        hf_token="token",
        asr_model="asr",
        diarisation_model=DEFAULT_DIARISATION_MODEL,
        diarisation_device="cpu",
        num_speakers=2,
        asr_chunk_duration=120.0,
        asr_chunk_overlap=15.0,
        memory_budget_bytes=None,
        diarisation_segmentation_batch_size=4,
        diarisation_embedding_batch_size=4,
    )

    transcriber = options.transcriber()

    assert isinstance(transcriber, LocalTranscriber)
    assert transcriber.diarisation_device == "cpu"
    assert transcriber.num_speakers == 2


def test_transcript_asr_help_documents_device_and_speaker_count() -> None:
    result = CliRunner().invoke(main, ["transcript", "asr", "--help"])

    assert result.exit_code == 0
    assert "--diarisation-device" in result.output
    assert "--num-speakers" in result.output


def test_local_transcriber_moves_pipeline_and_passes_known_speaker_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pipeline = FakePipeline()

    class FakeModel:
        preprocessor_config = SimpleNamespace(sample_rate=16000)

        def transcribe(self, audio: Path, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                sentences=[SimpleNamespace(start=0.0, end=2.0, text="hello")]
            )

    monkeypatch.setattr(asr_module, "parakeet_from_pretrained", lambda _: FakeModel())
    monkeypatch.setattr(
        asr_module.Pipeline, "from_pretrained", lambda *a, **k: pipeline
    )
    monkeypatch.setattr(
        asr_module,
        "AudioDecoder",
        lambda _, **kwargs: SimpleNamespace(
            get_all_samples=lambda: SimpleNamespace(data="waveform", sample_rate=16000)
        ),
    )
    monkeypatch.setattr(asr_module, "select_diarisation_device", lambda _: "cpu")
    monkeypatch.setattr(asr_module.mx, "set_memory_limit", lambda _: None)
    monkeypatch.setattr(asr_module.mx, "clear_cache", lambda: None)
    monkeypatch.setattr(
        asr_module.torch.mps, "set_per_process_memory_fraction", lambda _: None
    )
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"audio")

    LocalTranscriber(hf_token="token", diarisation_device="cpu").transcribe(
        audio, num_speakers=2
    )

    assert [str(device) for device in pipeline.devices] == ["cpu"]
    assert pipeline.calls == [{"num_speakers": 2, "hook": pipeline.calls[0]["hook"]}]


def test_local_transcriber_reuses_asr_checkpoint_but_recomputes_stale_diarisation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"audio")
    cache = RunCache(tmp_path / "cache")
    transcriber = LocalTranscriber(
        hf_token="token", diarisation_device="cpu", num_speakers=2
    )
    asr_calls = 0
    diarisation_calls = 0

    def run_asr(_: Path) -> list[AsrSegment]:
        nonlocal asr_calls
        asr_calls += 1
        return [AsrSegment(start=0.0, end=2.0, text="hello")]

    def run_diarisation(_: Path, *, device: str, num_speakers: int | None):
        nonlocal diarisation_calls
        diarisation_calls += 1
        return [DiarisedTurn(start=0.0, end=2.0, speaker="SPEAKER_00")]

    monkeypatch.setattr(transcriber, "_run_asr", run_asr)
    monkeypatch.setattr(transcriber, "_run_diarisation", run_diarisation)

    first = transcribe_merged_audio(
        audio, run_id="run", transcriber=transcriber, cache=cache
    )
    second = transcribe_merged_audio(
        audio, run_id="run", transcriber=transcriber, cache=cache
    )
    transcriber._num_speakers = 3
    third = transcribe_merged_audio(
        audio, run_id="run", transcriber=transcriber, cache=cache
    )

    assert first.model_copy(update={"timings": []}) == second.model_copy(
        update={"timings": []}
    )
    assert third.num_speakers == 3
    assert asr_calls == 1
    assert diarisation_calls == 2
    assert cache.load("run", "asr_checkpoint", AsrCheckpoint) is not None
    assert (
        cache.load("run", "diarisation_checkpoint", DiarisationCheckpoint) is not None
    )


def test_local_transcriber_reuses_asr_checkpoint_but_recomputes_diarisation_for_changed_batch_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A changed diarisation batch size is an output-affecting diarisation
    setting: it must invalidate the diarisation checkpoint (and the
    promoted raw-transcript fast path built from it) while leaving a valid
    ASR checkpoint untouched — the same contract already proven for
    `num_speakers` above, extended to the new batch-size knobs."""
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"audio")
    cache = RunCache(tmp_path / "cache")
    transcriber = LocalTranscriber(
        hf_token="token",
        diarisation_device="cpu",
        diarisation_segmentation_batch_size=4,
        diarisation_embedding_batch_size=4,
    )
    asr_calls = 0
    diarisation_calls = 0

    def run_asr(_: Path) -> list[AsrSegment]:
        nonlocal asr_calls
        asr_calls += 1
        return [AsrSegment(start=0.0, end=2.0, text="hello")]

    def run_diarisation(_: Path, *, device: str, num_speakers: int | None):
        nonlocal diarisation_calls
        diarisation_calls += 1
        return [DiarisedTurn(start=0.0, end=2.0, speaker="SPEAKER_00")]

    monkeypatch.setattr(transcriber, "_run_asr", run_asr)
    monkeypatch.setattr(transcriber, "_run_diarisation", run_diarisation)

    first = transcribe_merged_audio(
        audio, run_id="run", transcriber=transcriber, cache=cache
    )
    second = transcribe_merged_audio(
        audio, run_id="run", transcriber=transcriber, cache=cache
    )
    transcriber._diarisation_embedding_batch_size = 8
    third = transcribe_merged_audio(
        audio, run_id="run", transcriber=transcriber, cache=cache
    )

    assert first.model_copy(update={"timings": []}) == second.model_copy(
        update={"timings": []}
    )
    assert third.diarisation_embedding_batch_size == 8
    assert asr_calls == 1
    assert diarisation_calls == 2
    stored_diarisation = cache.load(
        "run", "diarisation_checkpoint", DiarisationCheckpoint
    )
    assert stored_diarisation is not None
    assert stored_diarisation.embedding_batch_size == 8
    assert stored_diarisation.segmentation_batch_size == 4
    assert stored_diarisation.sample_rate == asr_module.DIARISATION_SAMPLE_RATE
    assert stored_diarisation.channels == asr_module.DIARISATION_CHANNELS


def test_local_transcriber_recomputes_diarisation_for_a_checkpoint_that_predates_batch_size_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A diarisation checkpoint written before this change (no batch-size/
    sample-rate/channel fields at all) must not be silently trusted as
    still matching the current configuration — `DiarisationCheckpoint`'s
    new fields are required, so `load_resumable`'s existing
    malformed-cache handling (see the parametrized truncated-checkpoint
    test below) treats the legacy shape as a cache miss rather than a
    validated hit."""
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"audio")
    cache = RunCache(tmp_path / "cache")
    cache.run_dir("run").joinpath("diarisation_checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "audio_sha256": sha256_of(audio),
                "model": DEFAULT_DIARISATION_MODEL,
                "device": "cpu",
                "num_speakers": None,
                "turns": [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_OLD"}],
            }
        )
    )
    transcriber = LocalTranscriber(hf_token="token", diarisation_device="cpu")
    asr_calls = 0
    diarisation_calls = 0

    def run_asr(_: Path) -> list[AsrSegment]:
        nonlocal asr_calls
        asr_calls += 1
        return [AsrSegment(start=0.0, end=2.0, text="hello")]

    def run_diarisation(_: Path, *, device: str, num_speakers: int | None):
        nonlocal diarisation_calls
        diarisation_calls += 1
        return [DiarisedTurn(start=0.0, end=2.0, speaker="SPEAKER_00")]

    monkeypatch.setattr(transcriber, "_run_asr", run_asr)
    monkeypatch.setattr(transcriber, "_run_diarisation", run_diarisation)

    result = transcribe_merged_audio(
        audio, run_id="run", transcriber=transcriber, cache=cache
    )

    assert result.utterances[0].speaker == "SPEAKER_00"
    assert asr_calls == 1
    assert diarisation_calls == 1


@pytest.mark.parametrize(
    "malformed_name", ["asr_checkpoint", "diarisation_checkpoint", "raw_transcript"]
)
def test_local_transcriber_recomputes_after_a_truncated_checkpoint(
    malformed_name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"audio")
    cache = RunCache(tmp_path / "cache")
    cache.run_dir("run").joinpath(f"{malformed_name}.json").write_text("{")
    transcriber = LocalTranscriber(
        hf_token="token", diarisation_device="cpu", num_speakers=2
    )
    asr_calls = 0
    diarisation_calls = 0

    def run_asr(_: Path) -> list[AsrSegment]:
        nonlocal asr_calls
        asr_calls += 1
        return [AsrSegment(start=0.0, end=2.0, text="fresh")]

    def run_diarisation(_: Path, *, device: str, num_speakers: int | None):
        nonlocal diarisation_calls
        diarisation_calls += 1
        return [DiarisedTurn(start=0.0, end=2.0, speaker="SPEAKER_00")]

    monkeypatch.setattr(transcriber, "_run_asr", run_asr)
    monkeypatch.setattr(transcriber, "_run_diarisation", run_diarisation)

    result = transcribe_merged_audio(
        audio, run_id="run", transcriber=transcriber, cache=cache
    )

    assert result.utterances[0].text == "fresh"
    assert asr_calls == 1
    assert diarisation_calls == 1
    model_type = {
        "asr_checkpoint": AsrCheckpoint,
        "diarisation_checkpoint": DiarisationCheckpoint,
        "raw_transcript": RawTranscript,
    }[malformed_name]
    assert cache.load("run", malformed_name, model_type) is not None


def test_changed_asr_chunk_configuration_promotes_new_raw_transcript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"audio")
    cache = RunCache(tmp_path / "cache")
    transcriber = LocalTranscriber(
        hf_token="token",
        diarisation_device="cpu",
        asr_chunk_duration=120.0,
        asr_chunk_overlap=15.0,
    )

    def run_asr(_: Path) -> list[AsrSegment]:
        text = "old" if transcriber._asr_chunk_duration == 120.0 else "new"
        return [AsrSegment(start=0.0, end=2.0, text=text)]

    monkeypatch.setattr(transcriber, "_run_asr", run_asr)
    monkeypatch.setattr(
        transcriber,
        "_run_diarisation",
        lambda _, *, device, num_speakers: [
            DiarisedTurn(start=0.0, end=2.0, speaker="SPEAKER_00")
        ],
    )

    first = transcribe_merged_audio(
        audio, run_id="run", transcriber=transcriber, cache=cache
    )
    transcriber._asr_chunk_duration = 60.0
    second = transcribe_merged_audio(
        audio, run_id="run", transcriber=transcriber, cache=cache
    )

    assert first.utterances[0].text == "old"
    assert second.utterances[0].text == "new"
    assert second.asr_chunk_duration == 60.0
    assert second.asr_chunk_overlap == 15.0


def test_cpu_diarisation_does_not_touch_the_mps_allocator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pipeline = FakePipeline()
    monkeypatch.setattr(
        asr_module.Pipeline,
        "from_pretrained",
        lambda *args, **kwargs: pipeline,
    )
    monkeypatch.setattr(
        asr_module,
        "AudioDecoder",
        lambda _, **kwargs: SimpleNamespace(
            get_all_samples=lambda: SimpleNamespace(data="waveform", sample_rate=16000)
        ),
    )
    monkeypatch.setattr(
        asr_module.torch.mps,
        "set_per_process_memory_fraction",
        lambda _: pytest.fail("CPU diarisation touched the MPS allocator"),
    )

    transcriber = LocalTranscriber(hf_token="token", diarisation_device="cpu")
    result = transcriber._run_diarisation(
        tmp_path / "audio.m4a", device="cpu", num_speakers=None
    )

    assert result[0].speaker == "SPEAKER_00"


def test_diarisation_placement_runtime_error_preserves_domain_cause(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FailingPipeline(FakePipeline):
        def to(self, device: object) -> FakePipeline:
            raise RuntimeError("placement failed")

    monkeypatch.setattr(
        asr_module.Pipeline,
        "from_pretrained",
        lambda *args, **kwargs: FailingPipeline(),
    )

    transcriber = LocalTranscriber(hf_token="token", diarisation_device="cpu")
    with pytest.raises(AcceleratorOutOfMemoryError) as exc_info:
        transcriber._run_diarisation(
            tmp_path / "audio.m4a", device="cpu", num_speakers=None
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_stale_legacy_raw_transcript_is_not_reused(tmp_path: Path) -> None:
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"audio")
    cache = RunCache(tmp_path / "cache")
    cache.store(
        "run",
        "raw_transcript",
        RawTranscript(clips=[], utterances=[], audio_sha256="old"),
    )

    class FakeTranscriber:
        def transcribe(self, audio: Path) -> RawTranscript:
            return RawTranscript(clips=[], utterances=[], audio_sha256="new")

    result = transcribe_merged_audio(
        audio, run_id="run", transcriber=FakeTranscriber(), cache=cache
    )

    assert result.audio_sha256 == "new"


def test_local_stage_timing_is_durable_and_has_zero_api_cost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"audio")
    cache = RunCache(tmp_path / "cache")
    transcriber = LocalTranscriber(hf_token="token", diarisation_device="cpu")
    monkeypatch.setattr(
        transcriber,
        "_run_asr",
        lambda _: [AsrSegment(start=0.0, end=2.0, text="hello")],
    )
    monkeypatch.setattr(
        transcriber,
        "_run_diarisation",
        lambda _, device, num_speakers: [
            DiarisedTurn(start=0.0, end=2.0, speaker="SPEAKER_00")
        ],
    )

    result = transcribe_merged_audio(
        audio, run_id="run", transcriber=transcriber, cache=cache
    )
    timings = cache.load_timings("run")

    assert {item.stage for item in timings} >= {
        "ASR",
        "diarisation",
        "alignment/checkpoint promotion",
    }
    assert all(item.api_rate_cost == 0.0 for item in timings)
    assert result.timings


def test_local_stage_rtf_uses_full_recording_duration_with_trailing_silence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"audio")
    cache = RunCache(tmp_path / "cache")
    transcriber = LocalTranscriber(hf_token="token", diarisation_device="cpu")
    monkeypatch.setattr(
        transcriber,
        "_run_asr",
        lambda _: [AsrSegment(start=0.0, end=2.0, text="spoken")],
    )
    monkeypatch.setattr(
        transcriber,
        "_run_diarisation",
        lambda _, *, device, num_speakers: [
            DiarisedTurn(start=0.0, end=2.0, speaker="SPEAKER_00")
        ],
    )

    transcribe_merged_audio(
        audio,
        run_id="run",
        transcriber=transcriber,
        cache=cache,
        media_duration_seconds=10.0,
    )

    timings = {timing.stage: timing for timing in cache.load_timings("run")}
    assert timings["ASR"].media_duration_seconds == 10.0
    assert timings["diarisation"].media_duration_seconds == 10.0
    assert timings["ASR"].rtf is not None
    assert timings["ASR"].rtf < 1.0
