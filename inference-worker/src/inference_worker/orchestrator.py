"""Serial stage orchestration: prepare -> asr -> diarise -> InferenceResponse.

Stage callables are injectable (default to the real implementations) so
tests can exercise real request handling, hash verification, and
response assembly while faking out heavyweight ASR/diarisation model
calls at this seam — see tests/test_orchestrator.py. ASR still runs even
if diarisation later fails: they're independent once the wav exists, and
the M11 contract requires a completed ASR artefact to survive a
diarisation failure.

ASR and diarisation results are written as separate, individually-atomic
JSON files (m6) — ``InferenceResponse.asr``/``diarisation`` are
``ArtefactRef`` pointers to them, not the embedded objects.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from inference_worker.asr import asr_stage_config_hash, run_asr
from inference_worker.atomic_io import atomic_write_text
from inference_worker.diarise import diarisation_stage_config_hash, run_diarisation
from inference_worker.models import (
    ArtefactRef,
    AsrStageResult,
    AudioPreparationConfig,
    DiarisationStageResult,
    InferenceRequest,
    InferenceResponse,
    PrepareStageResult,
    SpeakerConstraints,
    StageError,
    StageObservations,
)
from inference_worker.prepare import prepare_audio
from inference_worker.provenance import (
    compute_runtime_delta,
    observed_runtime_provenance,
    sha256_file,
)

ASR_FILENAME = "asr.json"
DIARISATION_FILENAME = "diarisation.json"

PrepareFn = Callable[
    [Path, Path, AudioPreparationConfig, str, float], PrepareStageResult
]
AsrFn = Callable[[Path, str, float], AsrStageResult]
DiariseFn = Callable[
    [Path, str, SpeakerConstraints, int, float], DiarisationStageResult
]
# M7: the CLI's own injection seam — lets tests exercise main()'s success
# path (exit 0, parseable response.json) without a real inference run.
RunInferenceFn = Callable[[InferenceRequest, Path, Path], InferenceResponse]


def run_inference(
    request: InferenceRequest,
    out_dir: Path,
    lockfile_path: Path,
    *,
    prepare_fn: PrepareFn = prepare_audio,
    asr_fn: AsrFn = run_asr,
    diarise_fn: DiariseFn = run_diarisation,
) -> InferenceResponse:
    prepare_result = prepare_fn(
        Path(request.audio.path),
        out_dir,
        request.audio_preparation,
        request.audio.sha256,
        request.stage_timeouts.prepare_s,
    )
    if prepare_result.status != "completed" or prepare_result.output is None:
        asr_result = _blocked_asr_result(prepare_result)
        diarisation_result = _blocked_diarisation_result(
            prepare_result, request.speaker_constraints
        )
    else:
        wav = prepare_result.output
        asr_result = asr_fn(Path(wav.path), wav.sha256, request.stage_timeouts.asr_s)
        diarisation_result = diarise_fn(
            Path(wav.path),
            wav.sha256,
            request.speaker_constraints,
            wav.duration_ms,
            request.stage_timeouts.diarise_s,
        )
        diarisation_result = _augment_with_asr_retained_ref(
            asr_result, diarisation_result
        )

    asr_ref = _write_stage_artefact(out_dir, ASR_FILENAME, asr_result)
    diarisation_ref = _write_stage_artefact(
        out_dir, DIARISATION_FILENAME, diarisation_result
    )
    observed_runtime = observed_runtime_provenance(lockfile_path)

    return InferenceResponse(
        request_id=request.request_id,
        audio=request.audio,
        prepare=prepare_result,
        asr=asr_ref,
        diarisation=diarisation_ref,
        runtime=observed_runtime,
        runtime_provenance_delta=compute_runtime_delta(
            request.runtime_provenance, observed_runtime
        ),
    )


def worker_internal_error_response(
    request: InferenceRequest, out_dir: Path, lockfile_path: Path, exc: Exception
) -> InferenceResponse:
    """B3 safety net: every stage already converts its own known failure
    modes into a clean typed result — this is for whatever's left. If
    something genuinely unexpected still escapes ``run_inference``, this
    still emits a valid, honestly labelled response instead of losing the
    whole run (and any already-completed stage work) to an unhandled
    traceback bubbling out of the CLI."""
    error = StageError(
        error_class="worker-internal-error",
        message=f"{type(exc).__name__}: {exc}",
        retryable=False,
    )
    zero_obs = StageObservations(wall_time_ms=0, process_peak_rss_bytes=None)
    prepare_result = PrepareStageResult(
        status="failed",
        config_hash="unresolved",
        source_sha256=request.audio.sha256,
        observations=zero_obs,
        error=error,
    )
    asr_result = AsrStageResult(
        status="failed", config_hash="unresolved", observations=zero_obs, error=error
    )
    diarisation_result = DiarisationStageResult(
        status="failed", config_hash="unresolved", observations=zero_obs, error=error
    )
    asr_ref = _write_stage_artefact(out_dir, ASR_FILENAME, asr_result)
    diarisation_ref = _write_stage_artefact(
        out_dir, DIARISATION_FILENAME, diarisation_result
    )
    return InferenceResponse(
        request_id=request.request_id,
        audio=request.audio,
        prepare=prepare_result,
        asr=asr_ref,
        diarisation=diarisation_ref,
        runtime=observed_runtime_provenance(lockfile_path),
        runtime_provenance_delta={},
    )


def _write_stage_artefact(
    out_dir: Path, filename: str, stage_result: BaseModel
) -> ArtefactRef:
    """m6: ASR/diarisation results are separate, individually-atomic JSON
    files (B2's atomic-write pattern) — response.json only points at
    them, so a caller can promote e.g. asr.json alone as a capability
    output without slicing response.json apart."""
    path = out_dir / filename
    atomic_write_text(path, stage_result.model_dump_json(indent=2))
    return ArtefactRef(
        filename=filename, sha256=sha256_file(path), status=stage_result.status
    )


def _augment_with_asr_retained_ref(
    asr_result: AsrStageResult, diarisation_result: DiarisationStageResult
) -> DiarisationStageResult:
    """M5: retained_artefacts is the flagship partial-failure record — a
    failed diarisation stage that ran alongside a completed ASR stage
    should name the ASR artefact that survives it, not just the wav."""
    if diarisation_result.status != "failed" or diarisation_result.error is None:
        return diarisation_result
    if asr_result.status != "completed":
        return diarisation_result
    augmented_error = diarisation_result.error.model_copy(
        update={
            "retained_artefacts": [
                *diarisation_result.error.retained_artefacts,
                ASR_FILENAME,
            ]
        }
    )
    return diarisation_result.model_copy(update={"error": augmented_error})


def _dependency_failed_error(prepare_result: PrepareStageResult) -> StageError:
    cause = (
        prepare_result.error.message
        if prepare_result.error
        else "no output was produced"
    )
    return StageError(
        error_class="dependency-skipped",
        message=f"skipped: audio preparation did not complete ({cause})",
        # m4: retrying the whole request once prepare's own problem is
        # fixed always unblocks these stages — this is not prepare's own
        # retryable flag (which describes prepare's failure, not theirs).
        retryable=True,
    )


def _blocked_asr_result(prepare_result: PrepareStageResult) -> AsrStageResult:
    return AsrStageResult(
        status="failed",
        config_hash=asr_stage_config_hash(),
        observations=StageObservations(wall_time_ms=0, process_peak_rss_bytes=None),
        error=_dependency_failed_error(prepare_result),
    )


def _blocked_diarisation_result(
    prepare_result: PrepareStageResult, constraints: SpeakerConstraints
) -> DiarisationStageResult:
    return DiarisationStageResult(
        status="failed",
        config_hash=diarisation_stage_config_hash(constraints),
        observations=StageObservations(wall_time_ms=0, process_peak_rss_bytes=None),
        error=_dependency_failed_error(prepare_result),
    )
