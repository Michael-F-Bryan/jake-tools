"""Typed request/response contract for the inference worker (spec M11).

All output timestamps are integer milliseconds in the source media's own
coordinate domain (media start = 0), half-open spans ``[start_ms, end_ms)``.

``InferenceRequest.asr_model``, ``diarisation_model`` and
``runtime_provenance`` are the *caller's declared expectations* for the
worker's pinned model/runtime identity. The worker runs exactly one pinned
ASR model and one pinned diarisation model (no backend-selection flags):
a declared model *name* that doesn't match the pinned one is refused at
the CLI's contract boundary (M4, see ``__main__.py``); a declared runtime
that doesn't match the observed one is not refused, but the mismatch is
recorded in ``InferenceResponse.runtime_provenance_delta`` rather than
silently dropped. The worker's *actual observed* provenance is what ships
in the response (``AsrStageResult.model_provenance``,
``DiarisationStageResult.model_provenance``, ``InferenceResponse.runtime``).

M11 calls for the ASR and diarisation artefacts to be *separate* from
each other. In practice they're separate on-disk JSON files too (m6):
``InferenceResponse.asr``/``diarisation`` are ``ArtefactRef`` pointers
(filename + hash + status) to ``asr.json``/``diarisation.json``, each
written independently and atomically — a caller can hand ``asr.json``
alone to something that only cares about the ASR capability, without
slicing ``response.json`` apart.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# M1 ID format: "<prefix>_<uuid7>", prefix drawn from M1's fixed set.
# The uuid7 half enforces RFC 9562 version nibble '7' and variant
# nibbles {8,9,a,b}.
# The worker is invoked per operation attempt (M11); jake-tools mints
# attempt_/run_ IDs per M1. "attempt" is the expected caller identity;
# "run" is tolerated for manual diagnostic invocations.
_VALID_ID_PREFIXES = frozenset({"attempt", "run"})
_REQUEST_ID_RE = re.compile(
    r"^(?P<prefix>[a-z][a-z0-9]*)_"
    r"(?P<uuid7>[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})$"
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
        match = _REQUEST_ID_RE.match(self.request_id)
        if not match:
            raise ValueError(
                f"request_id {self.request_id!r} does not match the M1 "
                "'<prefix>_<uuid7>' ID format"
            )
        if match.group("prefix") not in _VALID_ID_PREFIXES:
            raise ValueError(
                f"request_id prefix {match.group('prefix')!r} is not in the "
                f"M1 prefix set {sorted(_VALID_ID_PREFIXES)!r} for inference requests"
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
    process_peak_rss_bytes: int | None = None
    """Best-effort ``RUSAGE_SELF.ru_maxrss`` sampled when this stage
    finished. This is the *whole process's* cumulative peak RSS since the
    worker started — not a per-stage delta — so later stages will report
    values that already include memory retained by earlier stages."""


class TimeSpan(BaseModel):
    """Shared half-open millisecond span invariant for ``AsrToken`` and
    ``DiarisationSegment``: non-negative, never reversed. Zero-length
    spans (``start_ms == end_ms``) are valid and preserved as evidence —
    e.g. Parakeet's duration head structurally emits some zero-length
    tokens. A reversed span is a modelling error, not raw evidence, so it
    raises rather than being silently accepted (M6)."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_span(self) -> TimeSpan:
        if self.end_ms < self.start_ms:
            raise ValueError(
                f"end_ms ({self.end_ms}) must be >= start_ms ({self.start_ms})"
            )
        return self


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
    duration_ms: int = Field(ge=0)


class PrepareStageResult(StageResultBase):
    # M3: always known (verified against the request's declared hash
    # before this stage ever runs), independent of whether ffmpeg itself
    # succeeds — resume-by-hash needs the *source* hash, not just the
    # prepared wav's.
    source_sha256: str = Field(pattern=_SHA256_RE)
    output: PreparedAudio | None = None


class AsrToken(TimeSpan):
    """One ASR token/word with its half-open millisecond span."""

    text: str
    confidence: float


class AsrOutput(BaseModel):
    text: str
    tokens: list[AsrToken]


class AsrStageResult(StageResultBase):
    input_audio_sha256: str | None = None
    model_provenance: ModelProvenance | None = None
    output: AsrOutput | None = None


class DiarisationSegment(TimeSpan):
    """One diarised speech turn. ``speaker_label`` is a local cluster label
    (e.g. ``SPEAKER_00``), not a participant identity."""

    speaker_label: str


class DiarisationOutput(BaseModel):
    segments: list[DiarisationSegment]


class DiarisationStageResult(StageResultBase):
    input_audio_sha256: str | None = None
    model_provenance: ModelProvenance | None = None
    output: DiarisationOutput | None = None


class ArtefactRef(BaseModel):
    """Pointer to a stage's separately-written, individually-atomic JSON
    artefact (B2, m6). The file itself is a complete, standalone
    ``AsrStageResult``/``DiarisationStageResult`` — the capability output
    a caller can promote on its own — this is just enough to locate and
    verify it without opening it."""

    filename: str
    sha256: str = Field(pattern=_SHA256_RE)
    status: Literal["completed", "failed"]


class InferenceResponse(BaseModel):
    request_id: str
    audio: AudioArtefact  # M3: echo of the request's declared source ref
    prepare: PrepareStageResult
    asr: ArtefactRef
    diarisation: ArtefactRef
    runtime: RuntimeProvenance
    # M4: declared (request.runtime_provenance) vs observed (runtime)
    # differences only; empty when they fully agree.
    runtime_provenance_delta: dict[str, str] = Field(default_factory=dict)
