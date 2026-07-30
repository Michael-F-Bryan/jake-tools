"""Stage 2: Parakeet ASR (parakeet-mlx) on the prepared wav.

One pinned model, no backend-selection flags (see models.py docstring).
"""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import ValidationError

from inference_worker.models import (
    AsrOutput,
    AsrStageResult,
    AsrToken,
    ModelIdentity,
    ModelProvenance,
    StageError,
)
from inference_worker.provenance import (
    config_hash,
    local_model_revision,
    package_version,
    stage_observations,
)
from inference_worker.timeouts import StageTimeoutError, enforce_timeout

MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v2"


def asr_stage_config_hash() -> str:
    # M2: fold in the resolved model revision + package versions, not just
    # the compile-time MODEL_ID constant — two runs against different
    # cached revisions must hash differently for M11/M12 hash-keyed reuse
    # to be sound.
    return config_hash(
        {
            "model_id": MODEL_ID,
            "model_revision": local_model_revision(MODEL_ID) or "unresolved",
            "parakeet-mlx": package_version("parakeet-mlx"),
            "mlx": package_version("mlx"),
        }
    )


def _model_provenance() -> ModelProvenance | None:
    """Best-effort model identity from whatever's locally cached right
    now. None if nothing is cached yet — never a fabricated "unknown"."""
    revision = local_model_revision(MODEL_ID)
    if revision is None:
        return None
    return ModelProvenance(
        identity=ModelIdentity(name=MODEL_ID, version=revision),
        package_versions={
            "parakeet-mlx": package_version("parakeet-mlx"),
            "mlx": package_version("mlx"),
        },
    )


def _build_tokens(raw_tokens) -> list[AsrToken]:
    """Pure mapping from parakeet-mlx's raw AlignedToken objects to typed
    AsrTokens, pulled out of run_asr so it's unit-testable (M6) without a
    real model: raises pydantic's ValidationError on a reversed span,
    caught by the caller and converted into a clean stage failure."""
    return [
        AsrToken(
            text=token.text,
            start_ms=round(token.start * 1000),
            end_ms=round(token.end * 1000),
            confidence=token.confidence,
        )
        for token in raw_tokens
    ]


def run_asr(wav_path: Path, wav_sha256: str, timeout_s: float) -> AsrStageResult:
    stage_config_hash = asr_stage_config_hash()
    started = time.monotonic()
    try:
        with enforce_timeout(timeout_s):
            # Imported lazily: importing parakeet_mlx eagerly at module load
            # would pull in mlx/Metal initialisation for every CLI
            # invocation, including ones that never reach this stage.
            from parakeet_mlx import from_pretrained

            model = from_pretrained(MODEL_ID)
            result = model.transcribe(wav_path)
    except StageTimeoutError as exc:
        return _failed(
            stage_config_hash,
            started,
            "timeout",
            str(exc),
            retryable=True,
            model_provenance=_model_provenance(),
            retained_artefacts=[str(wav_path)],
        )
    except Exception as exc:
        return _failed(
            stage_config_hash,
            started,
            "asr-failed",
            f"{type(exc).__name__}: {exc}",
            retryable=False,
            model_provenance=_model_provenance(),
            retained_artefacts=[str(wav_path)],
        )

    # A successful transcribe means `from_pretrained` just loaded (and, if
    # this was the first-ever run, downloaded-then-cached) the model, so a
    # cached revision must now be resolvable. If it somehow isn't, that's
    # a real problem, not a silent "unknown" on a status="completed" stage
    # (M1) — fail the stage instead of lying about its provenance.
    model_provenance = _model_provenance()
    if model_provenance is None:
        return _failed(
            stage_config_hash,
            started,
            "asr-failed",
            f"transcription completed but no cached revision for {MODEL_ID!r} could be resolved locally",
            retryable=False,
            retained_artefacts=[str(wav_path)],
        )

    try:
        tokens = _build_tokens(result.tokens)
    except ValidationError as exc:
        # M6: a reversed span (end_ms < start_ms) is a modelling error in
        # the raw ASR output, not evidence worth keeping — fail the stage
        # rather than silently accepting or discarding it. Zero-length
        # spans (start_ms == end_ms) are valid and pass through untouched.
        return _failed(
            stage_config_hash,
            started,
            "asr-invalid-output",
            f"parakeet returned a token with an invalid span: {exc}",
            retryable=False,
            model_provenance=model_provenance,
            retained_artefacts=[str(wav_path)],
        )

    return AsrStageResult(
        status="completed",
        config_hash=stage_config_hash,
        input_audio_sha256=wav_sha256,
        observations=stage_observations(started),
        model_provenance=model_provenance,
        output=AsrOutput(text=result.text, tokens=tokens),
    )


def _failed(
    stage_config_hash: str,
    started: float,
    error_class: str,
    message: str,
    *,
    retryable: bool,
    model_provenance: ModelProvenance | None = None,
    retained_artefacts: list[str] | None = None,
) -> AsrStageResult:
    return AsrStageResult(
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
