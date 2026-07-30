"""CLI contract-level behaviour, plus the CLI's own success path (M7).

Exit code contract: 0 whenever a valid response.json was written, nonzero
only for contract-level failures (unreadable/invalid request, an
unpinned declared model name, audio hash mismatch, unwritable out-dir).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from inference_worker import asr, diarise
from inference_worker.__main__ import main
from inference_worker.models import (
    ArtefactRef,
    AsrStageResult,
    DiarisationStageResult,
    InferenceRequest,
    InferenceResponse,
    PreparedAudio,
    PrepareStageResult,
    RuntimeProvenance,
    StageError,
    StageObservations,
)


def test_run_rejects_missing_request_file(tmp_path):
    out_dir = tmp_path / "out"

    exit_code = main(
        ["run", str(tmp_path / "does-not-exist.json"), "--out-dir", str(out_dir)]
    )

    assert exit_code != 0
    assert not (out_dir / "response.json").exists()
    assert not out_dir.exists()


def test_run_rejects_malformed_request_json(tmp_path):
    request_path = tmp_path / "request.json"
    request_path.write_text("{not valid json")
    out_dir = tmp_path / "out"

    exit_code = main(["run", str(request_path), "--out-dir", str(out_dir)])

    assert exit_code != 0
    assert not (out_dir / "response.json").exists()
    assert not out_dir.exists()


def test_run_rejects_audio_hash_mismatch(request_factory, sine_wav_factory, tmp_path):
    wav = sine_wav_factory(tmp_path / "source.wav")
    request = request_factory(
        audio_path=wav, audio_sha256="0" * 64
    )  # deliberately wrong
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json())
    out_dir = tmp_path / "out"

    exit_code = main(["run", str(request_path), "--out-dir", str(out_dir)])

    assert exit_code != 0
    assert not (out_dir / "response.json").exists()
    # m5: out-dir must only be created once every contract-level check
    # upstream (including hash verification) has passed.
    assert not out_dir.exists()


def test_run_rejects_unpinned_asr_model_name(
    request_factory, sine_wav_factory, tmp_path
):
    """M4: a request declaring a model this worker doesn't run must be
    refused at the contract boundary, not silently satisfied with the
    pinned model and a misleading exit 0."""
    wav = sine_wav_factory(tmp_path / "source.wav")
    request = request_factory(audio_path=wav, asr_model_name="openai/whisper-large-v3")
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json())
    out_dir = tmp_path / "out"

    exit_code = main(["run", str(request_path), "--out-dir", str(out_dir)])

    assert exit_code != 0
    assert not (out_dir / "response.json").exists()
    assert not out_dir.exists()


def test_run_rejects_unpinned_diarisation_model_name(
    request_factory, sine_wav_factory, tmp_path
):
    wav = sine_wav_factory(tmp_path / "source.wav")
    request = request_factory(
        audio_path=wav, diarisation_model_name="pyannote/speaker-diarization-3.1"
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json())
    out_dir = tmp_path / "out"

    exit_code = main(["run", str(request_path), "--out-dir", str(out_dir)])

    assert exit_code != 0
    assert not out_dir.exists()


def test_run_accepts_the_exact_pinned_model_names(
    request_factory, sine_wav_factory, tmp_path
):
    """The refusal in M4 is name-based, not a rejection of every request
    — the pinned names (what request_factory declares by default) must
    pass this specific check. (Full success needs real stages / a fake
    run_inference_fn — see test_run_success_path_via_injected_run_inference.)"""
    wav = sine_wav_factory(tmp_path / "source.wav")
    request = request_factory(
        audio_path=wav,
        asr_model_name=asr.MODEL_ID,
        diarisation_model_name=diarise.PIPELINE_ID,
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json())
    out_dir = tmp_path / "out"

    def _fake_run_inference(
        request: InferenceRequest, out_dir: Path, lockfile_path: Path
    ) -> InferenceResponse:
        return _minimal_response(request, out_dir)

    exit_code = main(
        ["run", str(request_path), "--out-dir", str(out_dir)],
        run_inference_fn=_fake_run_inference,
    )

    assert exit_code == 0


def test_run_rejects_unwritable_out_dir(request_factory, sine_wav_factory, tmp_path):
    wav = sine_wav_factory(tmp_path / "source.wav")
    request = request_factory(audio_path=wav)
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json())

    read_only_parent = tmp_path / "locked"
    read_only_parent.mkdir()
    out_dir = read_only_parent / "out"
    os.chmod(read_only_parent, 0o500)
    try:
        exit_code = main(["run", str(request_path), "--out-dir", str(out_dir)])
    finally:
        os.chmod(read_only_parent, 0o700)  # restore so tmp_path cleanup can delete it

    assert exit_code != 0
    assert not out_dir.exists()


# --- M7: the CLI's own success path, via the injectable run_inference seam


def _minimal_response(request: InferenceRequest, out_dir: Path) -> InferenceResponse:
    """A fast, fully synthetic InferenceResponse — including a failed
    stage, per M7 — that still writes its own asr.json/diarisation.json
    (main() only writes response.json; the injected fn owns its own
    artefacts, same as the real run_inference does)."""
    asr_result = AsrStageResult(
        status="completed",
        config_hash="fake",
        input_audio_sha256="a" * 64,
        observations=StageObservations(wall_time_ms=1),
        output={"text": "hi", "tokens": []},
    )
    diarisation_result = DiarisationStageResult(
        status="failed",
        config_hash="fake",
        observations=StageObservations(wall_time_ms=1),
        error=StageError(
            error_class="model-access-denied", message="gated", retryable=True
        ),
    )
    (out_dir / "asr.json").write_text(asr_result.model_dump_json())
    (out_dir / "diarisation.json").write_text(diarisation_result.model_dump_json())
    from inference_worker.provenance import sha256_file

    return InferenceResponse(
        request_id=request.request_id,
        audio=request.audio,
        prepare=PrepareStageResult(
            status="completed",
            config_hash="fake",
            source_sha256=request.audio.sha256,
            observations=StageObservations(wall_time_ms=1),
            output=PreparedAudio(
                path=str(out_dir / "prepared.wav"), sha256="b" * 64, duration_ms=1000
            ),
        ),
        asr=ArtefactRef(
            filename="asr.json",
            sha256=sha256_file(out_dir / "asr.json"),
            status="completed",
        ),
        diarisation=ArtefactRef(
            filename="diarisation.json",
            sha256=sha256_file(out_dir / "diarisation.json"),
            status="failed",
        ),
        runtime=RuntimeProvenance(
            python_version="3.12.11",
            ml_framework_versions={},
            worker_package_version="0.1.0",
            dependency_lockfile_sha256="0" * 64,
        ),
    )


def test_run_success_path_via_injected_run_inference(
    request_factory, sine_wav_factory, tmp_path
):
    """M7: every other CLI test asserts nonzero exit — this is the missing
    success-path coverage: exit 0 + a parseable response.json, containing
    a failed stage (the M11 partial-failure case still means exit 0)."""
    wav = sine_wav_factory(tmp_path / "source.wav")
    request = request_factory(audio_path=wav)
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json())
    out_dir = tmp_path / "out"

    def _fake_run_inference(
        request: InferenceRequest, out_dir: Path, lockfile_path: Path
    ) -> InferenceResponse:
        out_dir.mkdir(parents=True, exist_ok=True)
        return _minimal_response(request, out_dir)

    exit_code = main(
        ["run", str(request_path), "--out-dir", str(out_dir)],
        run_inference_fn=_fake_run_inference,
    )

    assert exit_code == 0
    response_path = out_dir / "response.json"
    assert response_path.exists()
    response = json.loads(response_path.read_text())
    assert response["asr"]["status"] == "completed"
    assert response["diarisation"]["status"] == "failed"


def test_run_wraps_an_unexpected_run_inference_error_and_still_exits_0(
    request_factory, sine_wav_factory, tmp_path
):
    """B3, exercised at the CLI boundary: run_inference raising something
    completely unexpected must still produce a valid response.json (the
    worker-internal-error fallback), not an unhandled traceback and a
    nonzero exit with nothing written."""
    wav = sine_wav_factory(tmp_path / "source.wav")
    request = request_factory(audio_path=wav)
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json())
    out_dir = tmp_path / "out"

    def _broken_run_inference(
        request: InferenceRequest, out_dir: Path, lockfile_path: Path
    ) -> InferenceResponse:
        raise FileNotFoundError("uv.lock vanished mid-run")

    exit_code = main(
        ["run", str(request_path), "--out-dir", str(out_dir)],
        run_inference_fn=_broken_run_inference,
    )

    assert exit_code == 0
    response = json.loads((out_dir / "response.json").read_text())
    assert response["prepare"]["error"]["error_class"] == "worker-internal-error"
