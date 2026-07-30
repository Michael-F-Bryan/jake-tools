"""Typed request/response contract for the inference worker (spec M11).

All output timestamps are integer milliseconds in the source media's own
coordinate domain (media start = 0), half-open spans ``[start_ms, end_ms)``.

``InferenceRequest.asr_model``, ``diarisation_model`` and
``runtime_provenance`` are the *caller's declared expectations* for the
worker's pinned model/runtime identity. The worker does not use them to
select behaviour — it runs exactly one pinned ASR model and one pinned
diarisation model (no backend-selection flags) — they are recorded purely
as typed, validated data for the caller's own audit trail. The worker's
*actual observed* provenance is what ships in the response
(``AsrStageResult.model_provenance``, ``DiarisationStageResult
.model_provenance``, ``InferenceResponse.runtime``).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# M1 ID format: "<prefix>_<uuid7>". The uuid7 half enforces RFC 9562
# version nibble '7' and variant nibbles {8,9,a,b}.
_REQUEST_ID_RE = re.compile(
    r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*_"
    r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_SHA256_RE = r"^[0-9a-f]{64}$"


# --- shared value objects -------------------------------------------------


class AudioArtefact(BaseModel):
    """A reference to a single audio/media file plus its content hash."""

    path: str
    sha256: str = Field(pattern=_SHA256_RE)


class AudioPreparationConfig(BaseModel):
    """ffmpeg's normalisation target. Fixed by contract, run once."""

    sample_rate_hz: Literal[16000] = 16000
    channels: Literal[1] = 1
    sample_format: Literal["s16le"] = "s16le"


class ModelIdentity(BaseModel):
    """A pinned model checkpoint: repo id plus exact revision."""

    name: str
    version: str


class ModelProvenance(BaseModel):
    """Observed identity of a model plus the library versions that ran it."""

    identity: ModelIdentity
    package_versions: dict[str, str]


class RuntimeProvenance(BaseModel):
    """Pinned runtime identity: declared by the caller, observed by the worker."""

    python_version: str
    ml_framework_versions: dict[str, str]
    worker_package_version: str
    dependency_lockfile_sha256: str


class SpeakerConstraints(BaseModel):
    min_speakers: int | None = Field(default=None, ge=1)
    max_speakers: int | None = Field(default=None, ge=1)
    exact_speakers: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_consistency(self) -> SpeakerConstraints:
        if self.exact_speakers is not None and (
            self.min_speakers is not None or self.max_speakers is not None
        ):
            raise ValueError(
                "exact_speakers is mutually exclusive with min_speakers/max_speakers"
            )
        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers > self.max_speakers
        ):
            raise ValueError("min_speakers must be <= max_speakers")
        return self


class StageTimeouts(BaseModel):
    prepare_s: float = Field(gt=0)
    asr_s: float = Field(gt=0)
    diarise_s: float = Field(gt=0)


# --- request ---------------------------------------------------------------


class InferenceRequest(BaseModel):
    request_id: str
    audio: AudioArtefact
    audio_preparation: AudioPreparationConfig = AudioPreparationConfig()
    asr_model: ModelIdentity
    diarisation_model: ModelIdentity
    runtime_provenance: RuntimeProvenance
    speaker_constraints: SpeakerConstraints = SpeakerConstraints()
    stage_timeouts: StageTimeouts

    @model_validator(mode="after")
    def _check_request_id(self) -> InferenceRequest:
        if not _REQUEST_ID_RE.match(self.request_id):
            raise ValueError(
                f"request_id {self.request_id!r} does not match the M1 "
                "'<prefix>_<uuid7>' ID format"
            )
        return self


# --- response ----------------------------------------------------------


class StageError(BaseModel):
    error_class: str
    message: str
    retryable: bool
    retained_artefacts: list[str] = Field(default_factory=list)


class StageObservations(BaseModel):
    wall_time_ms: int
    peak_rss_bytes: int | None = None


class StageResultBase(BaseModel):
    """Shared shape for the three per-stage results.

    Subclasses add their own ``output`` field. A single invariant lives
    here (rather than duplicated per subclass): completed stages carry
    output and no error; failed stages carry an error and no output.
    """

    status: Literal["completed", "failed"]
    config_hash: str
    observations: StageObservations
    error: StageError | None = None

    @model_validator(mode="after")
    def _check_status_invariants(self) -> StageResultBase:
        if self.status == "failed" and self.error is None:
            raise ValueError("a failed stage result requires an error")
        if self.status == "completed":
            if self.error is not None:
                raise ValueError("a completed stage result must not carry an error")
            if getattr(self, "output", None) is None:
                raise ValueError("a completed stage result requires output")
        return self


class PreparedAudio(BaseModel):
    path: str
    sha256: str = Field(pattern=_SHA256_RE)
    duration_ms: int


class PrepareStageResult(StageResultBase):
    output: PreparedAudio | None = None


class AsrToken(BaseModel):
    """One ASR token/word with its half-open millisecond span."""

    text: str
    start_ms: int
    end_ms: int
    confidence: float


class AsrOutput(BaseModel):
    text: str
    tokens: list[AsrToken]


class AsrStageResult(StageResultBase):
    input_audio_sha256: str | None = None
    model_provenance: ModelProvenance | None = None
    output: AsrOutput | None = None


class DiarisationSegment(BaseModel):
    """One diarised speech turn. ``speaker_label`` is a local cluster label
    (e.g. ``SPEAKER_00``), not a participant identity."""

    start_ms: int
    end_ms: int
    speaker_label: str


class DiarisationOutput(BaseModel):
    segments: list[DiarisationSegment]


class DiarisationStageResult(StageResultBase):
    input_audio_sha256: str | None = None
    model_provenance: ModelProvenance | None = None
    output: DiarisationOutput | None = None


class InferenceResponse(BaseModel):
    request_id: str
    prepare: PrepareStageResult
    asr: AsrStageResult
    diarisation: DiarisationStageResult
    runtime: RuntimeProvenance
