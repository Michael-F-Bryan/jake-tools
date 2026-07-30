"""Orchestration behaviour: real prepare stage (real ffmpeg), fake ASR/
diarisation stages injected at the callable seam `run_inference` exposes.

This is the M11 partial-failure proof: ASR success + diarisation failure
must retain the completed ASR artefact and record diarisation as
failed-retryable, while prepare/response assembly stay entirely real.
"""

from __future__ import annotations

from pathlib import Path

from inference_worker.models import (
    AsrOutput,
    AsrStageResult,
    AudioPreparationConfig,
    DiarisationStageResult,
    PrepareStageResult,
    StageError,
    StageObservations,
)
from inference_worker.orchestrator import run_inference


def _fake_completed_asr(
    wav_path: Path, wav_sha256: str, timeout_s: float
) -> AsrStageResult:
    return AsrStageResult(
        status="completed",
        config_hash="fake-asr-config",
        input_audio_sha256=wav_sha256,
        observations=StageObservations(wall_time_ms=5),
        output=AsrOutput(text="hello", tokens=[]),
    )


def _fake_gated_diarisation(
    wav_path, wav_sha256, constraints, timeout_s
) -> DiarisationStageResult:
    return DiarisationStageResult(
        status="failed",
        config_hash="fake-diar-config",
        input_audio_sha256=wav_sha256,
        observations=StageObservations(wall_time_ms=3),
        error=StageError(
            error_class="model-access-denied",
            message="pyannote/speaker-diarization-community-1 is gated",
            retryable=True,
        ),
    )


def test_asr_success_survives_diarisation_failure(
    request_factory, sine_wav_factory, tmp_path
):
    wav = sine_wav_factory(tmp_path / "source.wav")
    request = request_factory(audio_path=wav)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    response = run_inference(
        request,
        out_dir,
        lockfile_path=Path("uv.lock"),
        asr_fn=_fake_completed_asr,
        diarise_fn=_fake_gated_diarisation,
    )

    assert response.prepare.status == "completed"  # real ffmpeg ran
    assert response.asr.status == "completed"
    assert response.asr.output.text == "hello"
    assert response.diarisation.status == "failed"
    assert response.diarisation.error.error_class == "model-access-denied"
    assert response.diarisation.error.retryable is True


def test_prepare_failure_blocks_both_asr_and_diarisation(request_factory, tmp_path):
    missing_source = tmp_path / "does-not-exist.wav"
    # Building the request directly (not through request_factory's
    # sha256_file default) since the source file never exists.
    request = request_factory(audio_path=missing_source, audio_sha256="a" * 64)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def _asr_should_not_run(*args, **kwargs):
        raise AssertionError("asr_fn must not run when prepare fails")

    def _diarise_should_not_run(*args, **kwargs):
        raise AssertionError("diarise_fn must not run when prepare fails")

    response = run_inference(
        request,
        out_dir,
        lockfile_path=Path("uv.lock"),
        asr_fn=_asr_should_not_run,
        diarise_fn=_diarise_should_not_run,
    )

    assert response.prepare.status == "failed"
    assert response.asr.status == "failed"
    assert response.asr.error.error_class == "prepare-failed"
    assert response.diarisation.status == "failed"
    assert response.diarisation.error.error_class == "prepare-failed"


def test_prepare_fn_seam_is_honoured_for_deterministic_failure(
    request_factory, sine_wav_factory, tmp_path
):
    """A second, purely-injected seam test: even prepare itself can be
    faked, proving orchestration doesn't hardcode the real implementation."""
    wav = sine_wav_factory(tmp_path / "source.wav")
    request = request_factory(audio_path=wav)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def _fake_failed_prepare(
        source_path, out_dir, config: AudioPreparationConfig, timeout_s
    ):
        return PrepareStageResult(
            status="failed",
            config_hash="fake-prepare-config",
            observations=StageObservations(wall_time_ms=1),
            error=StageError(
                error_class="ffmpeg-failed", message="boom", retryable=False
            ),
        )

    response = run_inference(
        request,
        out_dir,
        lockfile_path=Path("uv.lock"),
        prepare_fn=_fake_failed_prepare,
        asr_fn=_fake_completed_asr,
        diarise_fn=_fake_gated_diarisation,
    )

    assert response.prepare.status == "failed"
    assert response.asr.status == "failed"
    assert response.asr.error.retryable is False  # propagated from prepare's error
    assert response.diarisation.status == "failed"
