"""Stage 3: Pyannote Community-1 diarisation on the prepared wav.

One pinned pipeline, no backend-selection flags (see models.py docstring).
Gated-model access denial is a clean, retryable stage failure — not a
crash and not a hosted fallback (M15 forbids hosted ASR/diarisation
outright).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from pydantic import ValidationError

from inference_worker.models import (
    DiarisationOutput,
    DiarisationSegment,
    DiarisationStageResult,
    ModelIdentity,
    ModelProvenance,
    SpeakerConstraints,
    StageError,
)
from inference_worker.provenance import (
    best_effort_package_version,
    config_hash,
    local_model_revision,
    stage_observations,
)
from inference_worker.timeouts import StageTimeoutError, enforce_timeout

# B1/N2: pyannote-audio 4.0.7 enables OpenTelemetry metrics BY DEFAULT
# (pyannote/audio/telemetry/metrics.py sets PYANNOTE_METRICS_ENABLED=true
# at import time if unset) and sends recording duration + the requested
# min/max/exact speaker constraints + a per-process session UUID to
# https://otel.pyannote.ai/v1/traces on every pipeline call — with the
# OpenTelemetry logger forced to CRITICAL, so a blocked/failed export is
# invisible. That's undeclared egress of source-derived metadata from a
# worker M15 requires to run local-only, and the corpus this worker
# processes carries its own privacy rules. This is a hard assignment, not
# `os.environ.setdefault` — an *inherited* PYANNOTE_METRICS_ENABLED=true
# in the ambient shell/CI environment reinstated the telemetry beacon in
# a verified test (the DNS target reappeared) when this used setdefault.
# The worker's own choice must win regardless of what the environment it
# was launched from happens to already have set; opt-in-via-environment
# is deliberately not honoured here. Must still run before pyannote.audio
# is imported anywhere below (it's only ever imported lazily, inside
# run_diarisation). Do not remove this without also verifying
# pyannote-audio has actually turned telemetry off by default upstream.
os.environ["PYANNOTE_METRICS_ENABLED"] = "false"

PIPELINE_ID = "pyannote/speaker-diarization-community-1"


def diarisation_stage_config_hash(constraints: SpeakerConstraints) -> str:
    # M2: fold in the resolved model revision + package versions, not just
    # the compile-time PIPELINE_ID constant — two runs against different
    # cached revisions must hash differently for M11/M12 hash-keyed reuse
    # to be sound. N3: best-effort, not strict package_version() — this
    # runs before run_diarisation's own try block even starts, so a
    # package missing on some platform must not raise here and degrade a
    # per-stage failure into a whole-run worker-internal-error.
    return config_hash(
        {
            "pipeline_id": PIPELINE_ID,
            "model_revision": local_model_revision(PIPELINE_ID) or "unresolved",
            "pyannote-audio": best_effort_package_version("pyannote-audio"),
            "torch": best_effort_package_version("torch"),
            **constraints.model_dump(),
        }
    )


def _model_provenance() -> ModelProvenance | None:
    """Best-effort model identity from whatever's locally cached right
    now — reachable even when the gated weights below aren't (a prior
    successful run cached them), and None (never a fabricated "unknown")
    if nothing has ever been cached."""
    revision = local_model_revision(PIPELINE_ID)
    if revision is None:
        return None
    return ModelProvenance(
        identity=ModelIdentity(name=PIPELINE_ID, version=revision),
        package_versions={
            "pyannote-audio": best_effort_package_version("pyannote-audio"),
            "torch": best_effort_package_version("torch"),
        },
    )


def run_diarisation(
    wav_path: Path,
    wav_sha256: str,
    constraints: SpeakerConstraints,
    duration_ms: int,
    timeout_s: float,
) -> DiarisationStageResult:
    stage_config_hash = diarisation_stage_config_hash(constraints)
    started = time.monotonic()
    try:
        with enforce_timeout(timeout_s):
            # Imported lazily so a CLI invocation that never reaches this
            # stage doesn't pay for torch/pyannote import time.
            import torch
            from huggingface_hub.errors import GatedRepoError
            from pyannote.audio import Pipeline

            try:
                pipeline = Pipeline.from_pretrained(PIPELINE_ID)
            except GatedRepoError as exc:
                return _failed(
                    stage_config_hash,
                    started,
                    "model-access-denied",
                    f"{PIPELINE_ID} is gated and the worker's Hugging Face "
                    f"credentials have not been granted access: {exc}",
                    retryable=True,
                    model_provenance=_model_provenance(),
                    retained_artefacts=[wav_path.name],
                )
            if pipeline is None:
                return _failed(
                    stage_config_hash,
                    started,
                    "diarisation-failed",
                    f"Pipeline.from_pretrained({PIPELINE_ID!r}) returned None",
                    retryable=False,
                    model_provenance=_model_provenance(),
                    retained_artefacts=[wav_path.name],
                )
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            pipeline.to(device)
            output = pipeline(str(wav_path), **_speaker_kwargs(constraints))
    except StageTimeoutError as exc:
        return _failed(
            stage_config_hash,
            started,
            "timeout",
            str(exc),
            retryable=True,
            model_provenance=_model_provenance(),
            retained_artefacts=[wav_path.name],
        )
    except Exception as exc:
        return _failed(
            stage_config_hash,
            started,
            "diarisation-failed",
            f"{type(exc).__name__}: {exc}",
            retryable=False,
            model_provenance=_model_provenance(),
            retained_artefacts=[wav_path.name],
        )

    # A successful call means the pipeline just loaded (and, if this was
    # the first-ever run, downloaded-then-cached) the model, so a cached
    # revision must now be resolvable. If it somehow isn't, that's a real
    # problem, not a silent "unknown" on a status="completed" stage (M1).
    model_provenance = _model_provenance()
    if model_provenance is None:
        return _failed(
            stage_config_hash,
            started,
            "diarisation-failed",
            f"diarisation completed but no cached revision for {PIPELINE_ID!r} could be resolved locally",
            retryable=False,
            retained_artefacts=[wav_path.name],
        )

    # Community-1 returns a DiarizeOutput (speaker_diarization /
    # exclusive_speaker_diarization / speaker_embeddings); older pyannote
    # pipelines return a plain Annotation directly. Use the overlap-aware
    # `speaker_diarization` — `exclusive_speaker_diarization` serialises
    # overlap away, which the M11 contract doesn't ask us to discard.
    annotation = getattr(output, "speaker_diarization", output)
    try:
        segments = _build_segments(annotation, duration_ms)
    except ValidationError as exc:
        # A reversed span survives the overshoot clamp only if pyannote
        # returned a segment starting after the file's own duration —
        # that's a modelling error, not evidence worth keeping.
        return _failed(
            stage_config_hash,
            started,
            "diarisation-invalid-output",
            f"pyannote returned a segment with an invalid span: {exc}",
            retryable=False,
            model_provenance=model_provenance,
            retained_artefacts=[wav_path.name],
        )

    return DiarisationStageResult(
        status="completed",
        config_hash=stage_config_hash,
        input_audio_sha256=wav_sha256,
        observations=stage_observations(started),
        model_provenance=model_provenance,
        output=DiarisationOutput(segments=segments),
    )


def _build_segments(annotation, duration_ms: int) -> list[DiarisationSegment]:
    """Pure mapping from pyannote's Annotation.itertracks() to typed
    DiarisationSegments, pulled out of run_diarisation so it's
    unit-testable (M6) with a fake annotation, without the real gated
    model: clamps end_ms overshoot past the file's own duration, and
    raises pydantic's ValidationError on a span that's still reversed
    after clamping (caught by the caller and converted into a clean
    stage failure)."""
    return [
        DiarisationSegment(
            start_ms=round(turn.start * 1000),
            end_ms=min(round(turn.end * 1000), duration_ms),
            speaker_label=str(speaker),
        )
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def _speaker_kwargs(constraints: SpeakerConstraints) -> dict[str, int]:
    if constraints.exact_speakers is not None:
        return {"num_speakers": constraints.exact_speakers}
    kwargs: dict[str, int] = {}
    if constraints.min_speakers is not None:
        kwargs["min_speakers"] = constraints.min_speakers
    if constraints.max_speakers is not None:
        kwargs["max_speakers"] = constraints.max_speakers
    return kwargs


def _failed(
    stage_config_hash: str,
    started: float,
    error_class: str,
    message: str,
    *,
    retryable: bool,
    model_provenance: ModelProvenance | None = None,
    retained_artefacts: list[str] | None = None,
) -> DiarisationStageResult:
    return DiarisationStageResult(
        status="failed",
        config_hash=stage_config_hash,
        model_provenance=model_provenance,
        observations=stage_observations(started),
        error=StageError(
            error_class=error_class,
            message=message,
            retryable=retryable,
            retained_artefacts=retained_artefacts or [],
        ),
    )
