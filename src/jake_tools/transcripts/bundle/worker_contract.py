"""M11: the frozen wire contract with the standalone ``inference-worker``
project.

``inference-worker`` has its own ``pyproject.toml``, pinned to Python
3.12 (jake-tools stays on 3.14) -- it is never a jake-tools dependency
(AGENTS.md: "no new dependencies") and is never installed into this
project's own virtualenv. Its request/response schema
(``inference-worker/src/inference_worker/models.py``) is frozen per
CONTRACTS.md's scope fence: a change there is a STOP-and-report, not
something this slice edits. The types below *mirror* just the wire
fields ``transcribe.py`` constructs/parses -- a deliberate, small
duplication of a frozen contract, not a live import of the worker's own
pydantic models (a cross-project import would need either an unsupported
dependency or a ``sys.path`` shim that a static type checker cannot
resolve; see ``tests/inference_worker_shim.py`` for the test-only version
that *does* import the real models, specifically so contract drift
breaks tests rather than this file silently disagreeing with them).

The worker's own *pinned model identities*
(``inference_worker.asr.MODEL_ID`` / ``inference_worker.diarise.
PIPELINE_ID``) are read directly from its source files via ``ast`` -- a
"small shared read", not a hand-copied string -- so a future pinned-model
change there is picked up automatically the next time this runs, never
silently drifting out of sync.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..errors import TranscriptError

REPO_ROOT = Path(__file__).resolve().parents[4]
INFERENCE_WORKER_ROOT = REPO_ROOT / "inference-worker"
_ASR_SOURCE = INFERENCE_WORKER_ROOT / "src" / "inference_worker" / "asr.py"
_DIARISE_SOURCE = INFERENCE_WORKER_ROOT / "src" / "inference_worker" / "diarise.py"


class WorkerContractError(TranscriptError):
    """Base class for every error this module raises."""


class PinnedModelConstantNotFoundError(WorkerContractError):
    """The worker's own source no longer defines the expected module-level
    string constant -- a genuine contract change (STOP and report per
    CONTRACTS.md's scope fence), not something to guess a fallback for.
    """


def _read_module_level_str_constant(source_path: Path, constant_name: str) -> str:
    try:
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"), filename=str(source_path)
        )
    except OSError as exc:
        raise PinnedModelConstantNotFoundError(
            f"cannot read {source_path} to find {constant_name!r}: {exc}"
        ) from exc
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == constant_name
            for target in node.targets
        ):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    raise PinnedModelConstantNotFoundError(
        f"{constant_name!r} is not a module-level string constant in {source_path} -- "
        "the worker's pinned-model contract may have changed (STOP and report)."
    )


def worker_asr_model_id() -> str:
    """The worker's own pinned ASR model repo id (``inference_worker.asr.
    MODEL_ID``), read from source -- never hand-copied."""
    return _read_module_level_str_constant(_ASR_SOURCE, "MODEL_ID")


def worker_diarisation_model_id() -> str:
    """The worker's own pinned diarisation pipeline id
    (``inference_worker.diarise.PIPELINE_ID``), read from source."""
    return _read_module_level_str_constant(_DIARISE_SOURCE, "PIPELINE_ID")


def build_uv_run_command(request_path: Path, out_dir: Path) -> list[str]:
    """M11: ``uv run --project <inference-worker> python -m inference_worker
    run REQUEST --out-dir OUT`` -- the exact invocation CONTRACTS.md M11
    specifies. A plain list, built once here so every caller (production
    and the fake-worker test seam alike) agrees on argument order.
    """
    return [
        "uv",
        "run",
        "--project",
        str(INFERENCE_WORKER_ROOT),
        "python",
        "-m",
        "inference_worker",
        "run",
        str(request_path),
        "--out-dir",
        str(out_dir),
    ]


# -- mirrored wire types: request side ----------------------------------

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class WireAudioArtefact(BaseModel):
    path: str
    sha256: str = Field(pattern=_SHA256_PATTERN)


class WireAudioPreparationConfig(BaseModel):
    sample_rate_hz: Literal[16000] = 16000
    channels: Literal[1] = 1
    sample_format: Literal["s16le"] = "s16le"


class WireModelIdentity(BaseModel):
    name: str
    version: str


class WireRuntimeProvenance(BaseModel):
    python_version: str
    ml_framework_versions: dict[str, str] = Field(default_factory=dict)
    worker_package_version: str
    dependency_lockfile_sha256: str


class WireSpeakerConstraints(BaseModel):
    min_speakers: int | None = Field(default=None, ge=1)
    max_speakers: int | None = Field(default=None, ge=1)
    exact_speakers: int | None = Field(default=None, ge=1)


class WireStageTimeouts(BaseModel):
    prepare_s: float = Field(gt=0)
    asr_s: float = Field(gt=0)
    diarise_s: float = Field(gt=0)


class WireInferenceRequest(BaseModel):
    request_id: str
    audio: WireAudioArtefact
    audio_preparation: WireAudioPreparationConfig = WireAudioPreparationConfig()
    asr_model: WireModelIdentity
    diarisation_model: WireModelIdentity
    runtime_provenance: WireRuntimeProvenance
    speaker_constraints: WireSpeakerConstraints = WireSpeakerConstraints()
    stage_timeouts: WireStageTimeouts


# -- mirrored wire types: response side (only what transcribe.py reads) -----


class WireArtefactRef(BaseModel):
    filename: str
    sha256: str = Field(pattern=_SHA256_PATTERN)
    status: Literal["completed", "failed"]


class WireInferenceResponse(BaseModel):
    request_id: str
    audio: WireAudioArtefact
    asr: WireArtefactRef
    diarisation: WireArtefactRef


class WireModelProvenanceIdentity(BaseModel):
    name: str
    version: str


class WireModelProvenance(BaseModel):
    identity: WireModelProvenanceIdentity


class WireStageError(BaseModel):
    error_class: str
    message: str
    retryable: bool = False


class WireStageResult(BaseModel):
    """The shared top-level shape of ``asr.json``/``diarisation.json``
    (``AsrStageResult``/``DiarisationStageResult`` in the real contract) --
    only the fields ``transcribe.py`` actually reads."""

    status: Literal["completed", "failed"]
    config_hash: str
    model_provenance: WireModelProvenance | None = None
    error: WireStageError | None = None


class WireAsrToken(BaseModel):
    """One raw ASR token. ``text`` carries its own leading whitespace --
    the worker builds its full transcript with ``"".join(token.text)``, so
    normalisation must join the same way rather than inserting separators
    the model never emitted."""

    start_ms: int
    end_ms: int
    text: str
    confidence: float


class WireAsrOutput(BaseModel):
    text: str
    tokens: list[WireAsrToken] = Field(default_factory=list)
    chunk_boundaries_ms: list[int] = Field(default_factory=list)


class WireAsrStageResult(WireStageResult):
    """``asr.json`` as normalisation reads it: the stage envelope plus the
    raw token stream. Raw output is faithful to the model (M11) --
    duplicates and zero-length tokens are legal here and are cleaned, with
    lineage, only when a canonical turn set is built."""

    output: WireAsrOutput | None = None


class WireDiarisationSegment(BaseModel):
    """One diarised span. ``speaker_label`` is a *local cluster* label
    (``SPEAKER_00``), never a participant identity (F13)."""

    start_ms: int
    end_ms: int
    speaker_label: str


class WireDiarisationOutput(BaseModel):
    segments: list[WireDiarisationSegment] = Field(default_factory=list)


class WireDiarisationStageResult(WireStageResult):
    output: WireDiarisationOutput | None = None
