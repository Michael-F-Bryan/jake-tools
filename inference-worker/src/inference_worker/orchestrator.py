"""Serial stage orchestration: prepare -> asr -> diarise -> InferenceResponse.

Stage callables are injectable (default to the real implementations) so
tests can exercise real request handling, hash verification, and
response assembly while faking out heavyweight ASR/diarisation model
calls at this seam — see tests/test_orchestrator.py. ASR still runs even
if diarisation later fails: they're independent once the wav exists, and
the M11 contract requires a completed ASR artefact to survive a
diarisation failure.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from inference_worker.asr import asr_stage_config_hash, run_asr
from inference_worker.diarise import diarisation_stage_config_hash, run_diarisation
from inference_worker.models import (
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
from inference_worker.provenance import observed_runtime_provenance

PrepareFn = Callable[[Path, Path, AudioPreparationConfig, float], PrepareStageResult]
AsrFn = Callable[[Path, str, float], AsrStageResult]
DiariseFn = Callable[[Path, str, SpeakerConstraints, float], DiarisationStageResult]


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
        request.stage_timeouts.prepare_s,
    )
    if prepare_result.status != "completed" or prepare_result.output is None:
        return InferenceResponse(
            request_id=request.request_id,
            prepare=prepare_result,
            asr=_blocked_asr_result(prepare_result),
            diarisation=_blocked_diarisation_result(
                prepare_result, request.speaker_constraints
            ),
            runtime=observed_runtime_provenance(lockfile_path),
        )

    wav = prepare_result.output
    asr_result = asr_fn(Path(wav.path), wav.sha256, request.stage_timeouts.asr_s)
    diarisation_result = diarise_fn(
        Path(wav.path),
        wav.sha256,
        request.speaker_constraints,
        request.stage_timeouts.diarise_s,
    )
    return InferenceResponse(
        request_id=request.request_id,
        prepare=prepare_result,
        asr=asr_result,
        diarisation=diarisation_result,
        runtime=observed_runtime_provenance(lockfile_path),
    )


def _dependency_failed_error(prepare_result: PrepareStageResult) -> StageError:
    cause = (
        prepare_result.error.message
        if prepare_result.error
        else "no output was produced"
    )
    retryable = prepare_result.error.retryable if prepare_result.error else False
    return StageError(
        error_class="prepare-failed",
        message=f"skipped: audio preparation did not complete ({cause})",
        retryable=retryable,
    )


def _blocked_asr_result(prepare_result: PrepareStageResult) -> AsrStageResult:
    return AsrStageResult(
        status="failed",
        config_hash=asr_stage_config_hash(),
        observations=StageObservations(wall_time_ms=0, peak_rss_bytes=None),
        error=_dependency_failed_error(prepare_result),
    )


def _blocked_diarisation_result(
    prepare_result: PrepareStageResult, constraints: SpeakerConstraints
) -> DiarisationStageResult:
    return DiarisationStageResult(
        status="failed",
        config_hash=diarisation_stage_config_hash(constraints),
        observations=StageObservations(wall_time_ms=0, peak_rss_bytes=None),
        error=_dependency_failed_error(prepare_result),
    )
