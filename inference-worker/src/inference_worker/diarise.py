"""Stage 3: Pyannote Community-1 diarisation on the prepared wav.

One pinned pipeline, no backend-selection flags (see models.py docstring).
Gated-model access denial is a clean, retryable stage failure — not a
crash and not a hosted fallback (M15 forbids hosted ASR/diarisation
outright).
"""

from __future__ import annotations

import time
from pathlib import Path

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
    config_hash,
    hf_repo_revision,
    package_version,
    stage_observations,
)
from inference_worker.timeouts import StageTimeoutError, enforce_timeout

PIPELINE_ID = "pyannote/speaker-diarization-community-1"


def diarisation_stage_config_hash(constraints: SpeakerConstraints) -> str:
    return config_hash({"pipeline_id": PIPELINE_ID, **constraints.model_dump()})


def run_diarisation(
    wav_path: Path,
    wav_sha256: str,
    constraints: SpeakerConstraints,
    timeout_s: float,
) -> DiarisationStageResult:
    stage_config_hash = diarisation_stage_config_hash(constraints)
    # Public repo metadata (the pinned revision) is reachable even when the
    # gated weights below are not, so provenance is available in both the
    # completed and the model-access-denied outcome.
    model_provenance = ModelProvenance(
        identity=ModelIdentity(name=PIPELINE_ID, version=hf_repo_revision(PIPELINE_ID)),
        package_versions={
            "pyannote-audio": package_version("pyannote-audio"),
            "torch": package_version("torch"),
        },
    )
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
                    model_provenance=model_provenance,
                )
            if pipeline is None:
                return _failed(
                    stage_config_hash,
                    started,
                    "diarisation-failed",
                    f"Pipeline.from_pretrained({PIPELINE_ID!r}) returned None",
                    retryable=False,
                    model_provenance=model_provenance,
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
            model_provenance=model_provenance,
        )
    except Exception as exc:
        return _failed(
            stage_config_hash,
            started,
            "diarisation-failed",
            f"{type(exc).__name__}: {exc}",
            retryable=False,
            model_provenance=model_provenance,
        )

    # Community-1 returns a DiarizeOutput (speaker_diarization /
    # exclusive_speaker_diarization / speaker_embeddings); older pyannote
    # pipelines return a plain Annotation directly. Use the overlap-aware
    # `speaker_diarization` — `exclusive_speaker_diarization` serialises
    # overlap away, which the M11 contract doesn't ask us to discard.
    annotation = getattr(output, "speaker_diarization", output)
    segments = [
        DiarisationSegment(
            start_ms=round(turn.start * 1000),
            end_ms=round(turn.end * 1000),
            speaker_label=str(speaker),
        )
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    return DiarisationStageResult(
        status="completed",
        config_hash=stage_config_hash,
        input_audio_sha256=wav_sha256,
        observations=stage_observations(started),
        model_provenance=model_provenance,
        output=DiarisationOutput(segments=segments),
    )


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
) -> DiarisationStageResult:
    return DiarisationStageResult(
        status="failed",
        config_hash=stage_config_hash,
        model_provenance=model_provenance,
        observations=stage_observations(started),
        error=StageError(error_class=error_class, message=message, retryable=retryable),
    )
