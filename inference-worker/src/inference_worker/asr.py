"""Stage 2: Parakeet ASR (parakeet-mlx) on the prepared wav.

One pinned model, no backend-selection flags (see models.py docstring).
"""

from __future__ import annotations

import time
from pathlib import Path

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
    hf_repo_revision,
    package_version,
    stage_observations,
)
from inference_worker.timeouts import StageTimeoutError, enforce_timeout

MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v2"


def asr_stage_config_hash() -> str:
    return config_hash({"model_id": MODEL_ID})


def run_asr(wav_path: Path, wav_sha256: str, timeout_s: float) -> AsrStageResult:
    stage_config_hash = asr_stage_config_hash()
    # Public repo metadata (the pinned revision) doesn't require the model
    # weights, so provenance is available whether or not transcription
    # itself succeeds — mirrors diarise.py's model_provenance handling.
    model_provenance = ModelProvenance(
        identity=ModelIdentity(name=MODEL_ID, version=hf_repo_revision(MODEL_ID)),
        package_versions={
            "parakeet-mlx": package_version("parakeet-mlx"),
            "mlx": package_version("mlx"),
        },
    )
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
            model_provenance=model_provenance,
        )
    except Exception as exc:
        return _failed(
            stage_config_hash,
            started,
            "asr-failed",
            f"{type(exc).__name__}: {exc}",
            retryable=False,
            model_provenance=model_provenance,
        )

    tokens = [
        AsrToken(
            text=token.text,
            start_ms=round(token.start * 1000),
            end_ms=round(token.end * 1000),
            confidence=token.confidence,
        )
        for token in result.tokens
    ]
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
) -> AsrStageResult:
    return AsrStageResult(
        status="failed",
        config_hash=stage_config_hash,
        model_provenance=model_provenance,
        observations=stage_observations(started),
        error=StageError(error_class=error_class, message=message, retryable=retryable),
    )
