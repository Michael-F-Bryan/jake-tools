"""M11: the worker-invocation transform.

Per media artefact: build a request (declared models read from the
worker's own pinned constants, audio hash from the artefact record,
speaker constraints from participant evidence), invoke the pinned
inference worker as a subprocess, verify the response's artefact-ref
hashes, ingest ``asr.json``/``diarisation.json`` as derived artefacts,
and promote ``inference.asr``/``inference.diarisation`` (M4) via new
:class:`~.components.AsrResultComponent`/:class:`~.components.
DiarisationResultComponent` components -- one only ever exists for a
*completed* stage (see their docstrings), which is what makes M11's
partial-failure rule ("completed stages promote even when siblings
fail") hold structurally rather than by convention.

Resume discipline (M11: "resume reuses ... completed stages by input+
config hash match only"): before invoking, each media member's ASR and
diarisation needs are decided independently against a
*caller-computed* ``request_fingerprint`` (audio hash + declared model
identity + relevant config, all known before the worker ever runs --
never the worker's own reported ``config_hash``, which depends on
runtime-resolved details like a locally-cached model revision this
process cannot know ahead of time). If neither stage needs a fresh run
for a given medium, the worker is never invoked for it at all -- this is
what makes a second ``transcribe_media()`` call over unchanged inputs a
zero-invocation no-op. Because the worker's own CLI cannot run "only
diarisation" or "only ASR" (see ``orchestrator.run_inference``), a run
still invoked for one needed stage will also silently recompute the
other's output when it doesn't need re-committing -- that recomputed
half is simply not turned into a new component (never a duplicate, never
a wasted write).

The `attempts/` directory M16 reserves for this exists on disk (every
bundle gets one), but `BundleStore` exposes no public method to write to
it, and `records.AttemptRecord` is a documented M11 stub with no
input/config-hash fields (`records.py`/`store.py` are forbidden to touch
in this slice -- see the scope fence). Per-stage attempt identity is
still genuine (`attempt_<uuid7>` IDs, matching the worker's own request-
ID contract) and resume-by-hash is fully functional; it lives on the
`AsrResultComponent`/`DiarisationResultComponent` records themselves
rather than in a separate `attempts/*.json` audit trail. See the final
report for the follow-up this implies.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..errors import TranscriptError
from .components import (
    AsrResultComponent,
    AsrResultComponentBody,
    DiarisationResultComponent,
    DiarisationResultComponentBody,
    MediaRecordingComponent,
    ParticipantSetComponent,
)
from .document import NoDocumentYet, TranscriptDocumentV1, project_head
from .ids import ArtefactId, RunId, mint_id
from .records import ArtefactRecord, OperationRef, RevisionRecord
from .store import BundleStore
from .worker_contract import (
    INFERENCE_WORKER_ROOT,
    WireArtefactRef,
    WireAudioArtefact,
    WireAudioPreparationConfig,
    WireInferenceRequest,
    WireInferenceResponse,
    WireModelIdentity,
    WireRuntimeProvenance,
    WireSpeakerConstraints,
    WireStageResult,
    WireStageTimeouts,
    build_uv_run_command,
    worker_asr_model_id,
    worker_diarisation_model_id,
)


class TranscribeError(TranscriptError):
    """Base class for every error this module raises."""


class NoDocumentToTranscribeError(TranscribeError):
    """The bundle has no assembled document yet (M18: assemble first)."""


class NoTranscribableMediaError(TranscribeError):
    """No :class:`~.components.MediaRecordingComponent` exists in the
    current document to transcribe."""


class WorkerInvocationError(TranscribeError):
    """The worker subprocess did not produce a ``response.json`` (a
    contract-level failure -- see ``inference_worker.__main__``'s own
    docstring for what that covers)."""


class WorkerResponseHashMismatchError(TranscribeError):
    """A hash the worker's own response declared does not match the
    actual bytes on disk (M11: "hash verification")."""


SubprocessRunner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]

# v1's single unpinned sentinel: the exact worker-resolved model revision
# is not knowable from this process ahead of time (it depends on
# whatever is cached in the worker's own, separately-managed environment)
# and is not enforced by the worker's own contract (only the model
# *name* is refused on mismatch; a *version* mismatch is recorded via
# InferenceResponse.runtime_provenance_delta, never refused) -- so this
# fixed string participates consistently in every fingerprint computed
# from it, rather than a moving target this process cannot observe.
_UNPINNED_MODEL_VERSION = "unpinned"

_DEFAULT_PREPARE_TIMEOUT_S = 300.0
_DEFAULT_ASR_TIMEOUT_S = 1800.0
_DEFAULT_DIARISE_TIMEOUT_S = 1800.0


def default_subprocess_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    """The real seam: an actual subprocess (``uv run --project
    inference-worker ...``). Public (not a leading-underscore private
    name) so ``control.py``'s orchestration layer can reference the same
    default explicitly when threading an optional override through from
    the CLI, rather than depending on this function's own default
    parameter value implicitly."""
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fingerprint(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _sha256_hex(canonical.encode("utf-8"))


def _request_fingerprint(
    *,
    audio_sha256: str,
    model_name: str,
    audio_preparation: WireAudioPreparationConfig,
    extra: dict[str, object] | None = None,
) -> str:
    payload: dict[str, object] = {
        "audio_sha256": audio_sha256,
        "model_name": model_name,
        "model_version": _UNPINNED_MODEL_VERSION,
        "audio_preparation": audio_preparation.model_dump(mode="json"),
    }
    if extra:
        payload.update(extra)
    return _fingerprint(payload)


def _worker_package_version() -> str:
    pyproject = INFERENCE_WORKER_ROOT / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        return str(data["project"]["version"])
    except OSError, KeyError, tomllib.TOMLDecodeError:
        return "unresolved"


def _declared_runtime_provenance() -> WireRuntimeProvenance:
    """M11: the caller's *declared* runtime expectations. Honestly
    partial where this process cannot know the worker's separately
    managed venv contents (a different, separately pinned Python; see
    AGENTS.md's "no new dependencies" -- jake-tools never installs the
    worker's own ML packages to inspect them). Not enforced by the
    worker's own contract (only the audio hash is); a real mismatch
    surfaces via ``InferenceResponse.runtime_provenance_delta`` instead
    of failing the request. ``dependency_lockfile_sha256`` *is* something
    this process can know for certain (the worker's own ``uv.lock`` on
    disk right now) and is computed for real.
    """
    lockfile = INFERENCE_WORKER_ROOT / "uv.lock"
    lockfile_hash = (
        _sha256_hex(lockfile.read_bytes()) if lockfile.exists() else "unresolved"
    )
    return WireRuntimeProvenance(
        python_version=_UNPINNED_MODEL_VERSION,
        ml_framework_versions={},
        worker_package_version=_worker_package_version(),
        dependency_lockfile_sha256=lockfile_hash,
    )


def _speaker_constraints(document: TranscriptDocumentV1) -> WireSpeakerConstraints:
    """M11: "min 1, max = declared attendee count when known." Known
    means exactly one participant set is present in the closure (more
    than one is already an ambiguous ``participants.declared`` failure
    the registry itself reports elsewhere); none present leaves ``max``
    unconstrained rather than guessed.
    """
    participant_sets = [
        component
        for component in document.components.values()
        if isinstance(component, ParticipantSetComponent)
    ]
    max_speakers = (
        len(participant_sets[0].participants) if len(participant_sets) == 1 else None
    )
    return WireSpeakerConstraints(min_speakers=1, max_speakers=max_speakers)


@dataclass(frozen=True)
class TranscribeOutcome:
    """Everything :func:`transcribe_media` did, for the caller/executor
    to report and for tests to assert on -- ``worker_invocations`` is
    what the resume-reuse test checks (0 means every media member's ASR
    and diarisation were both already satisfied by matching
    fingerprints)."""

    revision: RevisionRecord | None
    completed_asr_media_ids: tuple[ArtefactId, ...] = ()
    completed_diarisation_media_ids: tuple[ArtefactId, ...] = ()
    reused_asr_media_ids: tuple[ArtefactId, ...] = ()
    reused_diarisation_media_ids: tuple[ArtefactId, ...] = ()
    failed_asr_media_ids: tuple[ArtefactId, ...] = ()
    failed_diarisation_media_ids: tuple[ArtefactId, ...] = ()
    worker_invocations: int = 0


def _invoke_worker(
    request: WireInferenceRequest, out_dir: Path, subprocess_runner: SubprocessRunner
) -> WireInferenceResponse:
    request_path = out_dir.parent / "request.json"
    request_path.write_text(request.model_dump_json(indent=2), encoding="utf-8")
    command = build_uv_run_command(request_path, out_dir)
    result = subprocess_runner(command)

    response_path = out_dir / "response.json"
    if not response_path.exists():
        raise WorkerInvocationError(
            f"the inference worker did not produce {response_path} (exit code "
            f"{result.returncode}): {result.stderr or result.stdout}"
        )
    response = WireInferenceResponse.model_validate_json(
        response_path.read_text(encoding="utf-8")
    )
    if response.audio.sha256 != request.audio.sha256:
        raise WorkerResponseHashMismatchError(
            f"worker response echoed audio sha256={response.audio.sha256}, but the "
            f"request declared {request.audio.sha256}."
        )
    return response


def _ingest_stage_result(
    store: BundleStore,
    *,
    media_artefact_id: ArtefactId,
    source_id: str,
    out_dir: Path,
    filename: str,
    artefact_ref: WireArtefactRef,
    fingerprint: str,
) -> AsrResultComponentBody | DiarisationResultComponentBody:
    """Verify a completed stage artefact's hash, ingest it as a derived
    artefact (``derived_from`` the media artefact, M11), and build the
    matching Result component body -- never adds it to the store itself
    (the caller batches every media member's components into one
    ``append_revision`` call, mirroring ``assemble()``'s shape)."""
    path = out_dir / filename
    content = path.read_bytes()
    actual_hash = _sha256_hex(content)
    if actual_hash != artefact_ref.sha256:
        raise WorkerResponseHashMismatchError(
            f"{filename} content hash does not match the response's own declared "
            f"hash (declared {artefact_ref.sha256}, actual {actual_hash})."
        )
    stage_result = WireStageResult.model_validate_json(content)

    ingested = store.ingest_artefact(
        source_id=source_id,
        content=content,
        kind=filename.removesuffix(".json"),
        producer="inference-worker",
        acquisition_locator=str(path),
        derived_from=(media_artefact_id,),
    )
    if stage_result.model_provenance is not None:
        model_name = stage_result.model_provenance.identity.name
        model_version = stage_result.model_provenance.identity.version
    else:
        model_name = model_version = "unresolved"

    attempt_id = mint_id("attempt")
    if filename == "asr.json":
        return AsrResultComponentBody(
            media_artefact_id=media_artefact_id,
            result_artefact_id=ingested.artefact_id,
            attempt_id=attempt_id,
            request_fingerprint=fingerprint,
            worker_config_hash=stage_result.config_hash,
            model_name=model_name,
            model_version=model_version,
        )
    return DiarisationResultComponentBody(
        media_artefact_id=media_artefact_id,
        result_artefact_id=ingested.artefact_id,
        attempt_id=attempt_id,
        request_fingerprint=fingerprint,
        worker_config_hash=stage_result.config_hash,
        model_name=model_name,
        model_version=model_version,
    )


def transcribe_media(
    store: BundleStore,
    *,
    run_id: RunId,
    subprocess_runner: SubprocessRunner = default_subprocess_runner,
    force: bool = False,
    stage_timeouts: WireStageTimeouts | None = None,
) -> TranscribeOutcome:
    """M11: transcribe every :class:`~.components.MediaRecordingComponent`
    in the current head document.

    ``force=True`` re-invokes the worker for every media member
    regardless of a matching ``request_fingerprint`` -- recorded in the
    appended revision's own ``OperationRef.rationale`` (M11: "a `force`
    flag may override, recorded"). Raises :class:`NoDocumentToTranscribeError`
    if nothing has been assembled yet, or :class:`NoTranscribableMediaError`
    if the document has no ingested media at all.
    """
    document = project_head(store)
    if isinstance(document, NoDocumentYet):
        raise NoDocumentToTranscribeError(
            "this bundle has no assembled document yet (M18: run `transform "
            "assemble` first)."
        )

    media_components = [
        component
        for component in document.components.values()
        if isinstance(component, MediaRecordingComponent)
    ]
    if not media_components:
        raise NoTranscribableMediaError(
            "no media.recording evidence exists in this document to transcribe."
        )

    existing_asr = {
        component.media_artefact_id: component
        for component in document.components.values()
        if isinstance(component, AsrResultComponent)
    }
    existing_diarisation = {
        component.media_artefact_id: component
        for component in document.components.values()
        if isinstance(component, DiarisationResultComponent)
    }

    audio_preparation = WireAudioPreparationConfig()
    constraints = _speaker_constraints(document)
    timeouts = stage_timeouts or WireStageTimeouts(
        prepare_s=_DEFAULT_PREPARE_TIMEOUT_S,
        asr_s=_DEFAULT_ASR_TIMEOUT_S,
        diarise_s=_DEFAULT_DIARISE_TIMEOUT_S,
    )
    runtime_provenance = _declared_runtime_provenance()
    asr_model_id = worker_asr_model_id()
    diarisation_model_id = worker_diarisation_model_id()

    new_bodies: list[AsrResultComponentBody | DiarisationResultComponentBody] = []
    completed_asr: list[ArtefactId] = []
    completed_diarisation: list[ArtefactId] = []
    reused_asr: list[ArtefactId] = []
    reused_diarisation: list[ArtefactId] = []
    failed_asr: list[ArtefactId] = []
    failed_diarisation: list[ArtefactId] = []
    invocations = 0

    for media in media_components:
        artefact: ArtefactRecord = store.load_artefact(media.source_artefact_id)
        asr_fingerprint = _request_fingerprint(
            audio_sha256=artefact.sha256,
            model_name=asr_model_id,
            audio_preparation=audio_preparation,
        )
        diarisation_fingerprint = _request_fingerprint(
            audio_sha256=artefact.sha256,
            model_name=diarisation_model_id,
            audio_preparation=audio_preparation,
            extra={"speaker_constraints": constraints.model_dump(mode="json")},
        )

        prior_asr = existing_asr.get(media.source_artefact_id)
        prior_diarisation = existing_diarisation.get(media.source_artefact_id)
        need_asr = (
            force
            or prior_asr is None
            or prior_asr.request_fingerprint != asr_fingerprint
        )
        need_diarisation = (
            force
            or prior_diarisation is None
            or prior_diarisation.request_fingerprint != diarisation_fingerprint
        )

        if not need_asr and not need_diarisation:
            reused_asr.append(media.source_artefact_id)
            reused_diarisation.append(media.source_artefact_id)
            continue

        invocations += 1
        request = WireInferenceRequest(
            request_id=mint_id("attempt"),
            audio=WireAudioArtefact(
                path=str(store.root / "blobs" / artefact.sha256), sha256=artefact.sha256
            ),
            audio_preparation=audio_preparation,
            asr_model=WireModelIdentity(
                name=asr_model_id, version=_UNPINNED_MODEL_VERSION
            ),
            diarisation_model=WireModelIdentity(
                name=diarisation_model_id, version=_UNPINNED_MODEL_VERSION
            ),
            runtime_provenance=runtime_provenance,
            speaker_constraints=constraints,
            stage_timeouts=timeouts,
        )
        with tempfile.TemporaryDirectory(prefix="jake-tools-transcribe-") as work_dir:
            out_dir = Path(work_dir) / "out"
            response = _invoke_worker(request, out_dir, subprocess_runner)

            if need_asr:
                if response.asr.status == "completed":
                    new_bodies.append(
                        _ingest_stage_result(
                            store,
                            media_artefact_id=media.source_artefact_id,
                            source_id=artefact.source_id,
                            out_dir=out_dir,
                            filename="asr.json",
                            artefact_ref=response.asr,
                            fingerprint=asr_fingerprint,
                        )
                    )
                    completed_asr.append(media.source_artefact_id)
                else:
                    failed_asr.append(media.source_artefact_id)
            else:
                reused_asr.append(media.source_artefact_id)

            if need_diarisation:
                if response.diarisation.status == "completed":
                    new_bodies.append(
                        _ingest_stage_result(
                            store,
                            media_artefact_id=media.source_artefact_id,
                            source_id=artefact.source_id,
                            out_dir=out_dir,
                            filename="diarisation.json",
                            artefact_ref=response.diarisation,
                            fingerprint=diarisation_fingerprint,
                        )
                    )
                    completed_diarisation.append(media.source_artefact_id)
                else:
                    failed_diarisation.append(media.source_artefact_id)
            else:
                reused_diarisation.append(media.source_artefact_id)

    if not new_bodies:
        return TranscribeOutcome(
            revision=None,
            reused_asr_media_ids=tuple(reused_asr),
            reused_diarisation_media_ids=tuple(reused_diarisation),
            failed_asr_media_ids=tuple(failed_asr),
            failed_diarisation_media_ids=tuple(failed_diarisation),
            worker_invocations=invocations,
        )

    added: list[AsrResultComponent | DiarisationResultComponent] = []
    for body in new_bodies:
        stored = store.add_component(body)
        assert isinstance(stored, AsrResultComponent | DiarisationResultComponent)
        added.append(stored)
    # F1/M16 coherent-snapshot rule: each Asr/DiarisationResultComponent
    # declares its own `result_artefact_id` (the freshly-ingested
    # asr.json/diarisation.json) as an input ref (component_input_refs,
    # components.py) -- a brand new artefact no ancestor revision ever
    # declared, so *this* revision must claim it in its own `artefact_ids`
    # or `update_head`'s structural closure check refuses it as dangling.
    new_result_artefact_ids = tuple(component.result_artefact_id for component in added)
    force_note = (
        "force=True: re-running despite matching fingerprints; " if force else ""
    )
    revision = store.append_revision(
        operation=OperationRef(
            kind="transcribe",
            input_ids=tuple(
                component.source_artefact_id for component in media_components
            ),
            rationale=f"{force_note}ASR/diarisation via the pinned inference worker (M11)",
        ),
        parent_revision_ids=(document.revision_id,),
        artefact_ids=new_result_artefact_ids,
        component_ids=tuple(component.component_id for component in added),
    )
    store.update_head(run_id=run_id, revision_id=revision.revision_id)
    return TranscribeOutcome(
        revision=revision,
        completed_asr_media_ids=tuple(completed_asr),
        completed_diarisation_media_ids=tuple(completed_diarisation),
        reused_asr_media_ids=tuple(reused_asr),
        reused_diarisation_media_ids=tuple(reused_diarisation),
        failed_asr_media_ids=tuple(failed_asr),
        failed_diarisation_media_ids=tuple(failed_diarisation),
        worker_invocations=invocations,
    )
