"""Orchestration behaviour: real prepare stage (real ffmpeg), fake ASR/
diarisation stages injected at the callable seam `run_inference` exposes.

This is the M11 partial-failure proof: ASR success + diarisation failure
must retain the completed ASR artefact and record diarisation as
failed-retryable, while prepare/response assembly stay entirely real.

ASR/diarisation results are separate, individually-atomic JSON files
(m6) — `response.asr`/`.diarisation` are ArtefactRef pointers, so tests
that care about the actual stage content read the file the ref points at.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_worker import asr, diarise
from inference_worker.models import (
    AsrOutput,
    AsrStageResult,
    AudioPreparationConfig,
    DiarisationOutput,
    DiarisationStageResult,
    ModelIdentity,
    ModelProvenance,
    PrepareStageResult,
    SpeakerConstraints,
    StageError,
    StageObservations,
)
from inference_worker.orchestrator import (
    ASR_FILENAME,
    DIARISATION_FILENAME,
    RESPONSE_FILENAME,
    run_inference,
    worker_internal_error_response,
)


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
    wav_path,
    wav_sha256,
    constraints: SpeakerConstraints,
    duration_ms: int,
    timeout_s: float,
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
            # Mirrors run_diarisation's own behaviour (M5, N5): the
            # prepared wav it was given always survives a diarisation
            # failure, named as a bare filename relative to the out-dir.
            retained_artefacts=[Path(wav_path).name],
        ),
    )


def _read_artefact(out_dir: Path, filename: str) -> dict:
    return json.loads((out_dir / filename).read_text())


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
    assert response.diarisation.status == "failed"

    asr_artefact = _read_artefact(out_dir, response.asr.filename)
    assert asr_artefact["output"]["text"] == "hello"
    diarisation_artefact = _read_artefact(out_dir, response.diarisation.filename)
    assert diarisation_artefact["error"]["error_class"] == "model-access-denied"
    assert diarisation_artefact["error"]["retryable"] is True


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
    assert response.diarisation.status == "failed"
    asr_artefact = _read_artefact(out_dir, response.asr.filename)
    diarisation_artefact = _read_artefact(out_dir, response.diarisation.filename)
    # m4: blocked-downstream stages get their own error class and are
    # always retryable — retrying the whole request once prepare's own
    # (possibly non-retryable) problem is fixed always unblocks them.
    assert asr_artefact["error"]["error_class"] == "dependency-skipped"
    assert asr_artefact["error"]["retryable"] is True
    assert diarisation_artefact["error"]["error_class"] == "dependency-skipped"
    assert diarisation_artefact["error"]["retryable"] is True


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
        source_path,
        out_dir,
        config: AudioPreparationConfig,
        source_sha256: str,
        timeout_s,
    ):
        return PrepareStageResult(
            status="failed",
            config_hash="fake-prepare-config",
            source_sha256=source_sha256,
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
    assert response.prepare.source_sha256 == request.audio.sha256
    assert response.asr.status == "failed"
    assert response.diarisation.status == "failed"


# --- M3: source audio echoed on the response -----------------------------


def test_response_echoes_the_request_audio_ref(
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

    assert response.audio == request.audio


# --- m6: ASR/diarisation are separate, individually-atomic files ---------


def test_asr_and_diarisation_are_written_as_separate_files(
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

    assert response.asr.filename == ASR_FILENAME
    assert response.diarisation.filename == DIARISATION_FILENAME
    assert (out_dir / ASR_FILENAME).exists()
    assert (out_dir / DIARISATION_FILENAME).exists()
    # response.json never embeds the stage content directly (m6) — the
    # ArtefactRef's hash must match what's actually on disk.
    from inference_worker.provenance import sha256_file

    assert response.asr.sha256 == sha256_file(out_dir / ASR_FILENAME)
    assert response.diarisation.sha256 == sha256_file(out_dir / DIARISATION_FILENAME)
    # A caller must be able to load asr.json on its own, as a complete,
    # standalone AsrStageResult, without response.json.
    standalone = AsrStageResult.model_validate_json(
        (out_dir / ASR_FILENAME).read_text()
    )
    assert standalone.status == "completed"


# --- M5: retained_artefacts names the ASR artefact on a diarise failure --


def test_diarisation_failure_retains_the_completed_asr_artefact_ref(
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

    diarisation_artefact = _read_artefact(out_dir, response.diarisation.filename)
    retained = diarisation_artefact["error"]["retained_artefacts"]
    assert ASR_FILENAME in retained
    assert response.prepare.output is not None
    # N5: bare filename relative to the out-dir, consistent with
    # ArtefactRef's filenames — not the absolute prepared-wav path.
    assert Path(response.prepare.output.path).name in retained
    assert all(not p.startswith("/") for p in retained)


# --- B3: the worker-internal-error safety net -----------------------------


def test_worker_internal_error_response_is_valid_and_writes_artefacts(
    request_factory, sine_wav_factory, tmp_path
):
    wav = sine_wav_factory(tmp_path / "source.wav")
    request = request_factory(audio_path=wav)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    response = worker_internal_error_response(
        request, out_dir, Path("uv.lock"), RuntimeError("boom")
    )

    assert response.prepare.status == "failed"
    assert response.prepare.error.error_class == "worker-internal-error"
    assert response.asr.status == "failed"
    assert response.diarisation.status == "failed"
    assert (out_dir / response.asr.filename).exists()
    assert (out_dir / response.diarisation.filename).exists()
    asr_artefact = _read_artefact(out_dir, response.asr.filename)
    assert asr_artefact["error"]["error_class"] == "worker-internal-error"
    assert "boom" in asr_artefact["error"]["message"]


# --- N1: response.json is never a stale manifest for a different run -----


def test_a_crashed_run_leaves_no_stale_response_json_behind(
    request_factory, sine_wav_factory, tmp_path
):
    """N1: response.json + the artefacts it points at must be atomic AS A
    SET, not just individually. Seed the out-dir with a prior valid
    response.json, then crash partway through this run (after prepare's
    real artefact write, inside a faked asr_fn) — the stale manifest must
    be gone afterwards (it would otherwise describe THIS run's mismatched
    artefacts), while whatever this run did manage to produce survives:
    the honest state is "no manifest yet", never a lying one."""
    wav = sine_wav_factory(tmp_path / "source.wav")
    request = request_factory(audio_path=wav)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / RESPONSE_FILENAME).write_text(
        '{"stale": "manifest from a previous run"}'
    )

    def _crashing_asr(*args, **kwargs):
        raise RuntimeError("simulated crash mid-run")

    with pytest.raises(RuntimeError):
        run_inference(
            request,
            out_dir,
            lockfile_path=Path("uv.lock"),
            asr_fn=_crashing_asr,
            diarise_fn=_fake_gated_diarisation,
        )

    assert not (out_dir / RESPONSE_FILENAME).exists()
    assert (
        out_dir / "prepared.wav"
    ).exists()  # prepare's own artefact: the honest state


# --- N4: a declared model *version* mismatch is not refused, but visible -


def _fake_completed_asr_with_provenance(
    wav_path: Path, wav_sha256: str, timeout_s: float
) -> AsrStageResult:
    return AsrStageResult(
        status="completed",
        config_hash="fake-asr-config",
        input_audio_sha256=wav_sha256,
        observations=StageObservations(wall_time_ms=1),
        model_provenance=ModelProvenance(
            identity=ModelIdentity(
                name=asr.MODEL_ID, version="real-observed-asr-revision"
            ),
            package_versions={"parakeet-mlx": "0.5.2", "mlx": "0.32.0"},
        ),
        output=AsrOutput(text="hi", tokens=[]),
    )


def _fake_completed_diarisation_with_provenance(
    wav_path,
    wav_sha256,
    constraints: SpeakerConstraints,
    duration_ms: int,
    timeout_s: float,
) -> DiarisationStageResult:
    return DiarisationStageResult(
        status="completed",
        config_hash="fake-diar-config",
        input_audio_sha256=wav_sha256,
        observations=StageObservations(wall_time_ms=1),
        model_provenance=ModelProvenance(
            identity=ModelIdentity(
                name=diarise.PIPELINE_ID, version="real-observed-diarisation-revision"
            ),
            package_versions={"pyannote-audio": "4.0.7", "torch": "2.13.0"},
        ),
        output=DiarisationOutput(segments=[]),
    )


def test_declared_model_version_mismatch_completes_and_appears_in_the_delta(
    request_factory, sine_wav_factory, tmp_path
):
    """N4: _refuse_unpinned_models only checks the model *name* — a
    declared version like "deadbeefdeadbeef" (callers may not pin an
    exact revision) must not be refused, but the mismatch against what
    actually ran must still be visible, not silently dropped."""
    wav = sine_wav_factory(tmp_path / "source.wav")
    request = request_factory(audio_path=wav)
    request = request.model_copy(
        update={
            "asr_model": request.asr_model.model_copy(
                update={"version": "deadbeefdeadbeef"}
            ),
            "diarisation_model": request.diarisation_model.model_copy(
                update={"version": "deadbeefdeadbeef"}
            ),
        }
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    response = run_inference(
        request,
        out_dir,
        lockfile_path=Path("uv.lock"),
        asr_fn=_fake_completed_asr_with_provenance,
        diarise_fn=_fake_completed_diarisation_with_provenance,
    )

    assert response.asr.status == "completed"
    assert response.diarisation.status == "completed"
    assert response.runtime_provenance_delta["asr_model.version"] == (
        "declared=deadbeefdeadbeef observed=real-observed-asr-revision"
    )
    assert response.runtime_provenance_delta["diarisation_model.version"] == (
        "declared=deadbeefdeadbeef observed=real-observed-diarisation-revision"
    )


def test_matching_declared_model_version_leaves_no_delta_entry(
    request_factory, sine_wav_factory, tmp_path
):
    wav = sine_wav_factory(tmp_path / "source.wav")
    request = request_factory(audio_path=wav)
    request = request.model_copy(
        update={
            "asr_model": request.asr_model.model_copy(
                update={"version": "real-observed-asr-revision"}
            ),
        }
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    response = run_inference(
        request,
        out_dir,
        lockfile_path=Path("uv.lock"),
        asr_fn=_fake_completed_asr_with_provenance,
        diarise_fn=_fake_gated_diarisation,
    )

    assert "asr_model.version" not in response.runtime_provenance_delta
