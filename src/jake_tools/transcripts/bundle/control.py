"""Bundle control-plane orchestration (Phase 2).

The operator-facing CLI (``jake_tools.cli.transcript_bundle``) stays thin;
every multi-step bundle operation -- create, ingest, inspect, validate,
status, resume -- lives here instead. Nothing in this module mints IDs,
writes to the bundle directory directly, or bypasses ``BundleStore``'s
locking: it is plain orchestration over ``store``/``document``/
``registry``'s own public surface (M1, M2, M3, M4, M16).
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

from ..errors import TranscriptError
from ..stages import StructuredAgent
from .assemble import assemble
from .components import (
    ArtefactSelection,
    ComponentId,
    Disposition,
    SpeakerHypothesisSetComponent,
    SpeakerReviewComponent,
    TextEditMode,
    TimedTurnSetComponent,
    TimelineCombinedComponent,
    component_input_refs,
)
from .document import TranscriptDocumentV1, project_head, project_revision
from .ids import ArtefactId, RevisionId, RunId, mint_id
from .normalise import normalise_transcript
from .products import (
    ChapterOutcome,
    MinutesOutcome,
    transform_chapters,
    transform_minutes,
)
from .records import (
    DURABLE_RUN_STATES,
    BundleManifest,
    Lease,
    NoDocumentYet,
    OperationRef,
    RenderRecord,
    ReviewRecord,
    RevisionRecord,
    RunRecord,
    RunState,
    SourceAssociation,
    SourceMembershipRecord,
)
from .registry import CapabilityKey, CapabilityStatus
from .render import (
    DUMC_VARIANT,
    MEETING_NOTE_PROFILE,
    MEETING_NOTE_PROFILE_VERSION,
    RENDERER_VERSION,
    destination_of,
    render_meeting_note,
    template_sha256,
)
from .review import (
    ApplyReviewOutcome,
    commit_review,
    prepare_review_application,
)
from .speakers import propose_speakers
from .store import ArtefactRecord, BundleStore, UnknownSourceError
from .text import TextTransformOutcome, transform_text
from .timeline import transform_timeline
from .transcribe import SubprocessRunner, default_subprocess_runner, transcribe_media

#: M16 default bundle root, relative to the CLI's working directory --
#: gitignored, overridable per command via ``--root``/``--bundle``.
DEFAULT_BUNDLES_ROOT = Path("_working/transcripts/bundles")


class BundleControlError(TranscriptError):
    """Base class for every error this module raises."""


class BundleStagingError(BundleControlError):
    """``bundle create`` could not rename its staged directory into place."""


class NoResumableRunError(BundleControlError):
    """No run in the bundle is in a durable state (M2)."""


class AmbiguousResumeTargetError(BundleControlError):
    """More than one run is resumable; ``--run`` is required to disambiguate."""


class RunNotResumableError(BundleControlError):
    """``--run`` named a run that is not currently in a durable state."""


class NoExecutorRegisteredError(BundleControlError):
    """``run.next_action.kind`` has no registered executor (v1 ships none)."""


class ExecutorFailedError(BundleControlError):
    """An executor raised while dispatching a run's ``next_action`` (M2).

    The run has already been moved to ``failed`` (durable, resumable)
    with the exception recorded on its own ``next_action.rationale`` and
    the lease released by the time this is raised -- never left stuck in
    ``running`` with the lease held, which would make it permanently
    unresumable through the public API.
    """


class NoTakeOverTargetError(BundleControlError):
    """``--take-over``'s precondition (a lease held by a dead PID on a
    ``running`` run, M2) did not hold."""


# -- bundle create ------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CreatedBundle:
    store: BundleStore
    manifest: BundleManifest


def create_bundle(
    root: Path | None = None,
    *,
    rename: Callable[[Path, Path], None] | None = None,
) -> CreatedBundle:
    """Create a fresh, empty bundle under ``root`` (M16 default:
    ``_working/transcripts/bundles/``).

    ``BundleStore.create_bundle`` mints ``bundle_id`` as part of creating
    the bundle itself (M1: IDs are minted by the store, never chosen by a
    caller ahead of time) -- so the bundle's own directory name (M16:
    ``<root>/<bundle_id>/``) cannot be picked before that call returns.
    This stages the bundle in a throwaway sibling directory under
    ``root``, creates it there, then renames into the now-known
    ``<bundle_id>`` directory once the store has minted it; the store
    itself is never asked to accept, or told to reconcile with, a
    caller-chosen bundle_id.

    ``rename`` is an injectable seam (default ``os.rename``, resolved at
    call time rather than bound as a default so patching the real
    ``os.rename`` -- a genuine filesystem boundary -- is enough to
    exercise this without a bespoke test-only hook), mirroring
    ``BundleStore``'s own ``replace``/``link`` seams. On failure the
    staging directory is removed -- never left behind for a later
    ``bundle create`` to silently ignore -- and a typed
    ``BundleStagingError`` is raised naming both paths; if the cleanup
    itself also fails, the raised error names the leaked staging
    directory so it is at least diagnosable rather than silently lost.
    """
    rename = rename or os.rename
    parent = root if root is not None else DEFAULT_BUNDLES_ROOT
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=parent, prefix=".bundle-creating-"))
    manifest = BundleStore(staging).create_bundle()
    final_root = parent / manifest.bundle_id
    try:
        rename(staging, final_root)
    except OSError as exc:
        try:
            shutil.rmtree(staging)
        except OSError as cleanup_exc:
            raise BundleStagingError(
                f"failed to move staged bundle {staging} to {final_root} "
                f"({exc}), and failed to remove the staging directory too "
                f"({cleanup_exc}); it must be cleaned up by hand: {staging}"
            ) from cleanup_exc
        raise BundleStagingError(
            f"failed to move staged bundle {staging} to {final_root}: {exc}"
        ) from exc
    return CreatedBundle(store=BundleStore(final_root), manifest=manifest)


# -- source ingest --------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class IngestOutcome:
    membership: SourceMembershipRecord
    artefact: ArtefactRecord
    already_ingested: bool


def _known_artefact_ids(store: BundleStore) -> frozenset[str]:
    """Every artefact ID visible via the public API *before* an ingest
    call, for honest idempotency reporting (M16: "re-ingesting identical
    bytes ... is a no-op returning the existing artefact record").

    Only exact pre-assembly: ``document_head()``'s ``NoDocumentYet`` state
    is the only public query that lists "every ingested artefact" --
    once a bundle has a head, that candidate list is gone from the public
    surface, and only the document's own closure (artefacts actually
    assembled into it) remains visible. No command in this phase ever
    moves a bundle's head, so this limitation is latent, not live, here.
    """
    head = store.document_head()
    if isinstance(head, NoDocumentYet):
        return frozenset(head.candidate_artefact_ids)
    return frozenset()


def ingest_source(
    store: BundleStore,
    *,
    source_id: str | None,
    association: SourceAssociation,
    evidence: str,
    content: bytes,
    kind: str,
    producer: str,
    acquisition_locator: str,
) -> IngestOutcome:
    """Register a source membership (M3) if needed, then ingest ``content``
    as an artefact under it (M16).

    ``source_id`` reuses an existing membership; ``association``/
    ``evidence`` are ignored in that case (a membership's association is
    immutable once recorded) and are only used to register a *new* one.
    ``evidence`` for the default ``operator-assertion`` case is the CLI
    boundary's concern (the genuine invocation, M3) -- this function only
    stores whatever string it is given.
    """
    manifest = store.load_manifest()
    if source_id is not None:
        membership = next(
            (m for m in manifest.source_memberships if m.source_id == source_id),
            None,
        )
        if membership is None:
            raise UnknownSourceError(
                f"source {source_id!r} is not a member of this bundle; omit "
                "--source to register a new one."
            )
    else:
        membership = store.register_source(association=association, evidence=evidence)

    known_before = _known_artefact_ids(store)
    artefact = store.ingest_artefact(
        source_id=membership.source_id,
        content=content,
        kind=kind,
        producer=producer,
        acquisition_locator=acquisition_locator,
    )
    return IngestOutcome(
        membership=membership,
        artefact=artefact,
        already_ingested=artefact.artefact_id in known_before,
    )


# -- inspect ------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BundleOverview:
    """Everything ``inspect`` reports. ``document`` carries the head
    projection (or the explicit no-document-yet state, M16) and is the
    source of components and capability statuses; the remaining fields
    are bundle-wide facts the projection itself does not carry.
    """

    bundle_id: str
    document_id: str
    root: Path
    document: TranscriptDocumentV1 | NoDocumentYet
    sources: tuple[SourceMembershipRecord, ...]
    artefact_ids: tuple[str, ...]
    revision_count: int
    runs: tuple[RunRecord, ...]
    components_by_kind: Mapping[str, int]


def inspect_bundle(store: BundleStore) -> BundleOverview:
    """M16 bundle overview: IDs, head, sources, artefacts, revisions,
    runs, components, and capability statuses (read off the head
    projection's own ``capabilities`` mapping -- no separate validation
    pass needed, ``project_head``/``project_revision`` already ran one).

    ``artefact_ids``/``revision_count``/``components_by_kind`` read
    differently depending on whether the head is null, mirroring
    ``_known_artefact_ids``'s limitation: pre-assembly they cover every
    ingested artefact (there is nothing else yet); post-assembly they
    cover what the document's own revision closure actually references,
    which is the only thing left visible through the public API.
    """
    manifest = store.load_manifest()
    document = project_head(store)
    components_by_kind: dict[str, int]
    if isinstance(document, NoDocumentYet):
        artefact_ids = document.candidate_artefact_ids
        revision_count = 0
        components_by_kind = {}
    else:
        closure = store.resolve_revision_closure(document.revision_id)
        artefact_ids = tuple(
            sorted(
                {
                    artefact_id
                    for revision in closure.ancestors.values()
                    for artefact_id in revision.artefact_ids
                }
            )
        )
        revision_count = len(closure.ancestors)
        components_by_kind = dict(
            Counter(
                component.component_kind.value
                for component in document.components.values()
            )
        )

    runs = tuple(store.load_run(run_id) for run_id in manifest.run_ids)
    return BundleOverview(
        bundle_id=manifest.bundle_id,
        document_id=manifest.document_id,
        root=store.root,
        document=document,
        sources=manifest.source_memberships,
        artefact_ids=artefact_ids,
        revision_count=revision_count,
        runs=runs,
        components_by_kind=components_by_kind,
    )


# -- validate ------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CapabilityFailure:
    """One genuine capability failure (M4): either a many-cardinality
    key's failed member (``member_id`` set), or a one-cardinality key's
    own failed top-level status, which has no members to report
    (``member_id`` is ``None`` -- e.g. ``participants.declared`` with
    more than one candidate component present).
    """

    capability: CapabilityKey
    member_id: str | None
    detail: str


def project_document(
    store: BundleStore, *, revision_id: str | None
) -> TranscriptDocumentV1 | NoDocumentYet:
    """Project ``revision_id``, or the current head when omitted (M16)."""
    if revision_id is not None:
        return project_revision(store, revision_id)
    return project_head(store)


def capability_failures(
    document: TranscriptDocumentV1,
) -> tuple[CapabilityFailure, ...]:
    """Every genuinely-failed capability (M4) -- member-level AND
    top-level -- across every key.

    A one-cardinality key's own ``failed`` status carries no ``members``
    at all, so a scan limited to ``record.members`` never sees it: two
    participant-set components on one revision (``participants.declared``
    is one-cardinality) reproduces this exactly, and is the scenario the
    regression test below is built from. Both shapes are genuine
    failures a validator actually computed, not merely a non-present
    status, so both belong in this exit-code signal. M4's "two validated
    recordings and one reference-only embed" degrade-without-failing
    example is unaffected either way: that only degrades a many-key's
    *aggregate*, with no member and no top-level status ever reaching
    ``failed``.

    A one-cardinality key's own top-level ``failed`` status already
    blocks ``BundleStore.update_head`` from ever making that revision the
    head (``document.capability_validating_seam``), so a projected
    *head* is guarded against this by construction; ``--revision`` can
    still target an orphan revision that never went through that gate,
    which is exactly where this must still fire.
    """
    failures: list[CapabilityFailure] = []
    for key, record in document.capabilities.items():
        if record.members:
            failures.extend(
                CapabilityFailure(
                    capability=key, member_id=member.member_id, detail=member.detail
                )
                for member in record.members
                if member.status == CapabilityStatus.FAILED
            )
        elif record.status == CapabilityStatus.FAILED:
            failures.append(
                CapabilityFailure(
                    capability=key, member_id=None, detail=record.failure_detail
                )
            )
    return tuple(failures)


# -- status ------------------------------------------------------------


def _pid_is_alive(pid: int) -> bool:
    """Same liveness check ``BundleStore`` uses internally for lease
    take-over (``store._pid_is_alive`` is private); duplicated here since
    it is a genuine OS boundary primitive, not bundle-store domain logic
    worth reaching into a private helper for.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclasses.dataclass(frozen=True)
class BundleStatus:
    bundle_id: str
    lease: Lease | None
    lease_holder_alive: bool | None
    runs: tuple[RunRecord, ...]


def status_bundle(store: BundleStore) -> BundleStatus:
    manifest = store.load_manifest()
    lease = store.load_lease()
    alive = _pid_is_alive(lease.pid) if lease is not None else None
    runs = tuple(store.load_run(run_id) for run_id in manifest.run_ids)
    return BundleStatus(
        bundle_id=manifest.bundle_id, lease=lease, lease_holder_alive=alive, runs=runs
    )


# -- resume ------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ExecutorOutcome:
    """What an operation executor reports back to :func:`resume_run` (M2).

    Mirrors ``BundleStore.release_lease``'s own precondition: ``state``
    must be ``RunState.COMPLETED`` (``revision_id`` set when the executor
    appended one that should become the new head; ``next_action`` unset)
    or one of ``DURABLE_RUN_STATES`` (``next_action`` set, so a later
    resume replays it exactly). :func:`resume_run` is the only caller of
    ``update_head``/``release_lease``; an executor reports an outcome and
    never touches the lease or head itself, so every run's lifecycle
    bookkeeping goes through one place.
    """

    state: RunState
    revision_id: RevisionId | None = None
    next_action: OperationRef | None = None


BundleExecutor = Callable[[BundleStore, RunRecord], ExecutorOutcome]
AsyncBundleExecutor = Callable[[BundleStore, RunRecord], Awaitable[ExecutorOutcome]]


def _single_resumable_run(store: BundleStore, manifest: BundleManifest) -> str:
    resumable = [
        run_id
        for run_id in manifest.run_ids
        if store.load_run(run_id).state in DURABLE_RUN_STATES
    ]
    if not resumable:
        raise NoResumableRunError(
            "no run in this bundle is in a durable state (review_required, "
            "refused, or failed) to resume."
        )
    if len(resumable) > 1:
        raise AmbiguousResumeTargetError(
            f"{len(resumable)} runs are resumable ({', '.join(sorted(resumable))}); "
            "pass --run to choose one."
        )
    return resumable[0]


def _annotate_failure(next_action: OperationRef, exc: Exception) -> OperationRef:
    """Record an executor crash's type+message onto ``next_action``'s own
    ``rationale`` (BLOCKER 1). ``kind``/``input_ids``/``config_hash`` --
    the operation's actual identity -- stay byte-for-byte unchanged, so a
    later resume still dispatches the exact same recorded action (M2);
    ``rationale`` is free-text annotation, the only place this
    diagnostic detail can live without inventing a new ``RunRecord``
    field (``records.py`` is read-only in this phase).
    """
    detail = f"{type(exc).__name__}: {exc}"
    prefix = f"{next_action.rationale}\n" if next_action.rationale else ""
    return next_action.model_copy(
        update={"rationale": f"{prefix}executor failed: {detail}"}
    )


def _apply_outcome(
    store: BundleStore, run_id: str, outcome: ExecutorOutcome
) -> RunRecord:
    if outcome.state == RunState.COMPLETED:
        if outcome.revision_id is not None:
            store.update_head(run_id=run_id, revision_id=outcome.revision_id)
        return store.release_lease(run_id=run_id, new_state=RunState.COMPLETED)
    return store.release_lease(
        run_id=run_id, new_state=outcome.state, next_action=outcome.next_action
    )


def _run_executor_or_fail(
    store: BundleStore,
    run: RunRecord,
    executors: Mapping[str, BundleExecutor],
    *,
    on_missing_executor: Callable[[OperationRef], RunRecord],
) -> RunRecord:
    """Dispatch ``run.next_action`` and apply its outcome (M2), or
    recover -- shared by the ordinary durable-resume path and the
    ``--take-over`` crash-recovery path, which reach this only after
    successfully acquiring ``run``'s lease.

    Two distinct failure modes, handled differently:

    - **No executor registered** for the kind: not a crash, so
      ``on_missing_executor`` decides the run's fate (an ordinary resume
      reverts to the durable state it was resumed from; a take-over,
      having no such prior state, fails the run outright) and is
      expected to both apply that and raise ``NoExecutorRegisteredError``
      -- propagated here unchanged.
    - **The executor itself raises** (BLOCKER 1): always moves the run
      to ``failed`` (M2: durable, resumable) with the exception recorded
      via :func:`_annotate_failure` and the lease released, then
      re-raises as :class:`ExecutorFailedError` -- never left stuck in
      ``running`` with the lease held, which would make it permanently
      unresumable through the public API (``RunNotResumableError``
      forever).
    """
    next_action = run.next_action
    if next_action is None:
        # RunRecord's own invariant guarantees a non-terminal state
        # (running, which both resume paths just set) carries a
        # next_action; unreachable in practice, kept only so this stays
        # a typed error rather than an AttributeError if that invariant
        # is ever loosened.
        raise BundleControlError(f"run {run.run_id} has no next_action while running.")

    executor = executors.get(next_action.kind)
    if executor is None:
        return on_missing_executor(next_action)

    try:
        outcome = executor(store, run)
    except Exception as exc:
        failed_next_action = _annotate_failure(next_action, exc)
        store.release_lease(
            run_id=run.run_id, new_state=RunState.FAILED, next_action=failed_next_action
        )
        raise ExecutorFailedError(
            f"executor for operation kind {next_action.kind!r} raised "
            f"{type(exc).__name__}: {exc}; run {run.run_id} moved to 'failed' "
            "(resumable -- the same next_action will be retried)."
        ) from exc

    return _apply_outcome(store, run.run_id, outcome)


def resume_run(
    store: BundleStore,
    *,
    run_id: str | None,
    executors: Mapping[str, BundleExecutor],
    take_over: bool = False,
) -> RunRecord:
    """Re-acquire the lease for a durable-state run and dispatch its
    recorded ``next_action`` (M2). Never re-plans: the operation kind,
    input IDs, and config hash recorded when the run entered its durable
    state are replayed exactly, never re-derived from whatever the bundle
    directory happens to contain now.

    An operation kind with no registered executor is an explicit,
    abortable error (v1 ships none): the lease is released again and the
    run returns to precisely the durable state it was resumed from, so
    resuming an unimplemented operation costs nothing and stays safe to
    retry once a real executor exists. An executor that raises moves the
    run to ``failed`` instead (M2's durable, resumable terminal-ish
    state) rather than leaving it stuck ``running`` with the lease held.

    ``take_over=True`` is the true crash-recovery path (M2): a lease
    held by a dead PID on a run stuck in ``running`` (its process was
    killed mid-dispatch, so it never reached a durable state and
    released the lease itself). It creates a *new* run record
    (``takeover_of_run_id`` naming the stale one, per
    ``BundleStore.acquire_lease``'s own enforcement) and dispatches the
    stale run's exact recorded ``next_action`` through it -- the stale
    run record itself is never rewritten (M2: "a run never edits another
    run's record").
    """
    if take_over:
        return _resume_via_takeover(store, run_id=run_id, executors=executors)

    manifest = store.load_manifest()
    target_run_id = (
        run_id if run_id is not None else _single_resumable_run(store, manifest)
    )

    pending = store.load_run(target_run_id)
    if pending.state not in DURABLE_RUN_STATES:
        durable = ", ".join(state.value for state in DURABLE_RUN_STATES)
        hint = (
            " This run is 'running' with its lease still held; if its process "
            "crashed, use --take-over to recover it."
            if pending.state == RunState.RUNNING
            else ""
        )
        raise RunNotResumableError(
            f"run {target_run_id} is in state {pending.state.value!r}; only a "
            f"durable-state run ({durable}) can be resumed.{hint}"
        )
    original_state = pending.state
    original_next_action = pending.next_action

    store.acquire_lease(run_id=target_run_id, pid=os.getpid())
    run = store.load_run(target_run_id)

    def _revert_on_missing_executor(next_action: OperationRef) -> RunRecord:
        store.release_lease(
            run_id=target_run_id,
            new_state=original_state,
            next_action=original_next_action,
        )
        raise NoExecutorRegisteredError(
            f"no executor registered for operation kind {next_action.kind!r} "
            "(transforms arrive in later slices)."
        )

    return _run_executor_or_fail(
        store, run, executors, on_missing_executor=_revert_on_missing_executor
    )


def _resume_via_takeover(
    store: BundleStore, *, run_id: str | None, executors: Mapping[str, BundleExecutor]
) -> RunRecord:
    """The ``--take-over`` crash-recovery path (M2), split out of
    :func:`resume_run` for readability: find the stale lease holder,
    validate the take-over precondition, create the recovering run
    record, acquire the lease with ``take_over=True``, dispatch.
    """
    lease = store.load_lease()
    if lease is None:
        raise NoTakeOverTargetError(
            "no lease is held on this bundle; there is nothing to take over."
        )
    stale_run_id = run_id if run_id is not None else lease.run_id
    if stale_run_id != lease.run_id:
        raise NoTakeOverTargetError(
            f"run {stale_run_id} does not hold the bundle's lease (held by "
            f"{lease.run_id}); --take-over only recovers the run currently "
            "holding a stale lease."
        )
    stale_run = store.load_run(stale_run_id)
    if stale_run.state != RunState.RUNNING:
        raise NoTakeOverTargetError(
            f"run {stale_run_id} is in state {stale_run.state.value!r}, not "
            "'running'; --take-over only recovers a crashed run (running, "
            "with its lease held by a dead process)."
        )
    if _pid_is_alive(lease.pid):
        raise NoTakeOverTargetError(
            f"run {stale_run_id}'s lease is held by live process {lease.pid}; "
            "--take-over only recovers a lease held by a dead process."
        )
    stale_next_action = stale_run.next_action
    if stale_next_action is None:
        raise BundleControlError(
            f"run {stale_run_id} has no next_action while running."
        )

    new_run = store.create_run(
        next_action=stale_next_action, takeover_of_run_id=stale_run_id
    )
    store.acquire_lease(run_id=new_run.run_id, pid=os.getpid(), take_over=True)
    run = store.load_run(new_run.run_id)

    def _fail_on_missing_executor(next_action: OperationRef) -> RunRecord:
        # No prior durable state to revert to -- this run was created
        # fresh for the take-over -- so an unimplemented kind fails it
        # outright, same as a genuine executor crash (both are always a
        # legal release_lease target, unlike reverting to a state this
        # run never had).
        store.release_lease(
            run_id=new_run.run_id, new_state=RunState.FAILED, next_action=next_action
        )
        raise NoExecutorRegisteredError(
            f"no executor registered for operation kind {next_action.kind!r} "
            "(transforms arrive in later slices)."
        )

    return _run_executor_or_fail(
        store, run, executors, on_missing_executor=_fail_on_missing_executor
    )


# -- transform: timeline / transcribe (M6 / M11) --------------------------


def timeline_executor(store: BundleStore, run: RunRecord) -> ExecutorOutcome:
    """Dispatches M6's timeline transform for `resume` (M2's executor
    seam). Reads any explicit member order back from the run's own
    recorded ``next_action.input_ids`` -- an empty tuple means "use the
    transform's own note-embed default", exactly what
    :func:`run_timeline_transform` records when no explicit order was
    given.

    Returns ``revision_id=None`` -- unlike a plain executor that only
    *proposes* a revision, :func:`transform_timeline` already moved the
    head itself (the same "the operation both appends and moves the
    head" shape ``assemble()`` established); a second ``update_head``
    from :func:`_apply_outcome` would be a redundant, harmless no-op
    (the revision is already its own ancestor), so this simply omits it.
    """
    next_action = run.next_action
    if next_action is None:
        raise BundleControlError(f"run {run.run_id} has no next_action while running.")
    order = next_action.input_ids or None
    transform_timeline(store, run_id=run.run_id, order=order)
    return ExecutorOutcome(state=RunState.COMPLETED, revision_id=None)


def transcribe_executor(store: BundleStore, run: RunRecord) -> ExecutorOutcome:
    """Dispatches M11's transcribe transform for `resume` (M2's executor
    seam). Always ``force=False`` -- resuming a crashed or interrupted
    run must never silently redo already-completed work; ``--force`` is
    only ever honoured on the CLI's direct, first-attempt path (see
    :func:`run_transcribe_transform`'s own executor closure).

    ``revision_id=None`` for the same reason :func:`timeline_executor`
    returns it: :func:`transcribe_media` already moved the head itself
    when it had a new revision to move to.
    """
    transcribe_media(store, run_id=run.run_id)
    return ExecutorOutcome(state=RunState.COMPLETED, revision_id=None)


#: How ``transform assemble`` reads an ingested artefact's role from the
#: ``--kind`` the operator declared at ingest (M3: naming the input on the
#: command line *is* the assertion). Anything not listed is
#: ``evidence-only`` -- present in the document and cited by nothing,
#: which is the honest default for a source whose role was never stated.
_DISPOSITIONS_BY_KIND: Mapping[str, tuple[Disposition, ...]] = MappingProxyType(
    {
        "obsidian-note": (Disposition.DESTINATION, Disposition.NOTES),
        "local-media": (Disposition.MEDIA,),
        "audio": (Disposition.MEDIA,),
        "markdown": (Disposition.SELECTED_TRANSCRIPT,),
        "vtt": (Disposition.SELECTED_TRANSCRIPT,),
        "teams-transcript": (Disposition.SELECTED_TRANSCRIPT,),
        "gemini-notes": (Disposition.NOTES,),
    }
)


def _unassembled_components(
    store: BundleStore, *, selected_artefact_ids: frozenset[str]
) -> tuple[ComponentId, ...]:
    """The adapter-produced components these artefacts brought with them.

    A component qualifies when every artefact it declares as an input is
    one this assembly is selecting -- that is exactly M16's coherent-
    snapshot rule stated forwards: a component may only enter a document
    alongside the evidence it was derived from. Components already in the
    head's closure are left alone, so re-assembling after a second ingest
    adds the new ones without duplicating the old.
    """
    head = project_head(store)
    already_present = (
        frozenset() if isinstance(head, NoDocumentYet) else frozenset(head.components)
    )
    selected: list[ComponentId] = []
    for component in store.iter_components():
        if component.component_id in already_present:
            continue
        refs = component_input_refs(component)
        if refs.artefact_ids and not set(refs.artefact_ids) <= selected_artefact_ids:
            continue
        selected.append(component.component_id)
    return tuple(sorted(selected))


def assemble_candidates(
    store: BundleStore, *, run_id: RunId, rationale: str = ""
) -> RevisionRecord:
    """M18: bring every ingested candidate into the document.

    Dispositions come from each artefact's declared ``kind``
    (:data:`_DISPOSITIONS_BY_KIND`), never inferred from filenames or
    content -- M3 forbids association by resemblance, and the ``--kind``
    the operator typed at ingest is the assertion this reads back.
    """
    head = store.document_head()
    if isinstance(head, NoDocumentYet):
        candidate_ids = head.candidate_artefact_ids
    else:
        closure = store.resolve_revision_closure(head.revision_id)
        assembled = {
            artefact_id
            for revision in closure.ancestors.values()
            for artefact_id in revision.artefact_ids
        }
        candidate_ids = tuple(
            artefact.artefact_id
            for artefact in _iter_bundle_artefacts(store)
            if artefact.artefact_id not in assembled
        )
    if not candidate_ids:
        raise BundleControlError(
            "there are no unassembled artefacts in this bundle; ingest a source "
            "first (`source ingest`)."
        )
    selections = tuple(
        ArtefactSelection(
            artefact_id=artefact_id,
            dispositions=_DISPOSITIONS_BY_KIND.get(
                store.load_artefact(artefact_id).kind, (Disposition.EVIDENCE_ONLY,)
            ),
        )
        for artefact_id in candidate_ids
    )
    return assemble(
        store,
        run_id=run_id,
        selections=selections,
        rationale=rationale
        or "assemble every ingested candidate, dispositions from declared kinds (M18)",
        component_ids=_unassembled_components(
            store, selected_artefact_ids=frozenset(candidate_ids)
        ),
    )


def _iter_bundle_artefacts(store: BundleStore):
    """Every artefact record in the bundle, assembled or not.

    Only reached on a re-assembly (the bundle already has a head, so
    ``NoDocumentYet``'s candidate list is gone -- open item 1 in the
    handoff). Reads the store's own artefact directory through its public
    loader rather than reaching into private state.
    """
    directory = store.root / "artefacts"
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("artefact_*.json")):
        yield store.load_artefact(path.stem)


def assemble_executor(store: BundleStore, run: RunRecord) -> ExecutorOutcome:
    """Dispatches M18's assembly for `resume`. ``assemble()`` moves the
    head itself, so no second ``update_head`` is proposed here."""
    assemble_candidates(store, run_id=run.run_id)
    return ExecutorOutcome(state=RunState.COMPLETED, revision_id=None)


def run_assemble_transform(store: BundleStore) -> RunRecord:
    """CLI entry point for `transform assemble` (M18)."""
    return run_bundle_operation(
        store,
        next_action=OperationRef(
            kind="assemble", rationale="transcript transform assemble"
        ),
        executor=assemble_executor,
    )


def normalise_executor(store: BundleStore, run: RunRecord) -> ExecutorOutcome:
    """Dispatches M7's normalisation for `resume`. Like the timeline and
    transcribe executors, :func:`normalise_transcript` already moved the
    head itself, so no second ``update_head`` is proposed here."""
    normalise_transcript(store, run_id=run.run_id)
    return ExecutorOutcome(state=RunState.COMPLETED, revision_id=None)


def default_bundle_executors() -> Mapping[str, BundleExecutor]:
    """The production executor mapping (M2's resume seam): every
    operation kind ``transform timeline``/``transform transcribe`` may
    leave recorded as a durable run's ``next_action``. Wired as
    ``AppContext.bundle_executors``'s own default
    (``cli/context.py``) -- injected explicitly only by tests exercising
    the dispatch mechanism itself.
    """
    return MappingProxyType(
        {
            "assemble": assemble_executor,
            "timeline": timeline_executor,
            "transcribe": transcribe_executor,
            "normalise": normalise_executor,
        }
    )


def bundle_executors_with_agent(
    agent: StructuredAgent, *, model: str, context_note: str = ""
) -> Mapping[str, BundleExecutor]:
    """The production executor mapping plus the LLM-backed operations.

    ``speakers-propose`` and ``products`` need a ``ClaudeAgent``, which
    :func:`default_bundle_executors` has no way to build (it is imported by
    ``cli/context.py``'s own module-level default, long before any command
    has resolved ``--model``). Binding them here, at the point where a
    command already holds an agent, keeps the LLM seam out of the default
    mapping while still letting `resume` dispatch a durable run recorded
    against either kind.

    ``asyncio.run`` appears here because M2's executor seam is
    synchronous and `resume` is a synchronous command; the async
    orchestration functions below are called directly (never through this
    mapping) by the commands that are already inside an event loop.
    """
    executors = dict(default_bundle_executors())

    def _propose(store: BundleStore, run: RunRecord) -> ExecutorOutcome:
        asyncio.run(
            propose_speakers(store, run_id=run.run_id, agent=agent, model=model)
        )
        return ExecutorOutcome(state=RunState.COMPLETED, revision_id=None)

    def _products(store: BundleStore, run: RunRecord) -> ExecutorOutcome:
        asyncio.run(
            build_products(
                store,
                run_id=run.run_id,
                agent=agent,
                model=model,
                context_note=context_note,
            )
        )
        return ExecutorOutcome(state=RunState.COMPLETED, revision_id=None)

    executors["speakers-propose"] = _propose
    executors["products"] = _products
    return MappingProxyType(executors)


async def run_async_bundle_operation(
    store: BundleStore, *, next_action: OperationRef, executor: AsyncBundleExecutor
) -> RunRecord:
    """:func:`run_bundle_operation` for the LLM-backed transforms.

    A deliberate near-duplicate of the synchronous path: the only
    difference is ``await executor(...)``, and the alternative -- driving
    the sync path with ``asyncio.run`` from inside a command that is
    already in an event loop -- does not work at all. Failure handling is
    kept identical on purpose, so a crashed async transform lands in
    exactly the same durable, resumable ``failed`` state a crashed
    synchronous one does.
    """
    run = store.create_run(next_action=next_action)
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    run = store.load_run(run.run_id)
    try:
        outcome = await executor(store, run)
    except Exception as exc:
        store.release_lease(
            run_id=run.run_id,
            new_state=RunState.FAILED,
            next_action=_annotate_failure(next_action, exc),
        )
        raise ExecutorFailedError(
            f"executor for operation kind {next_action.kind!r} raised "
            f"{type(exc).__name__}: {exc}; run {run.run_id} moved to 'failed' "
            "(resumable -- the same next_action will be retried)."
        ) from exc
    return _apply_outcome(store, run.run_id, outcome)


def run_normalise_transform(store: BundleStore) -> RunRecord:
    """CLI entry point for `transform normalise` (M7)."""
    return run_bundle_operation(
        store,
        next_action=OperationRef(
            kind="normalise", rationale="transcript transform normalise"
        ),
        executor=normalise_executor,
    )


async def run_speakers_propose(
    store: BundleStore,
    *,
    agent: StructuredAgent,
    model: str,
    stop_for_review: bool = True,
    context_note: str = "",
) -> RunRecord:
    """M8: propose speakers, then stop at the durable review checkpoint.

    ``stop_for_review=True`` is the whole point of the overhaul: the run
    ends in ``review_required`` with the lease *released* and a recorded
    ``next_action`` of ``products``, so the operator can walk away, export
    a pack, decide the speakers at their leisure, apply it, and `resume`
    -- on another day, from another shell -- without repeating any of the
    stable upstream work.
    """

    async def _executor(store: BundleStore, run: RunRecord) -> ExecutorOutcome:
        await propose_speakers(store, run_id=run.run_id, agent=agent, model=model)
        if not stop_for_review:
            return ExecutorOutcome(state=RunState.COMPLETED, revision_id=None)
        return ExecutorOutcome(
            state=RunState.REVIEW_REQUIRED,
            next_action=OperationRef(
                kind="products",
                config_hash=None,
                rationale=(
                    "speaker hypotheses are proposals only (M8 rung 7); export a "
                    "review pack, decide each cluster, apply it, then resume to "
                    "build the polished transcript, chapters, and minutes"
                ),
            ),
        )

    return await run_async_bundle_operation(
        store,
        next_action=OperationRef(
            kind="speakers-propose", rationale="transcript speakers propose"
        ),
        executor=_executor,
    )


def run_review_apply(store: BundleStore, *, pack_path: Path) -> ApplyReviewOutcome:
    """M8: apply a filled review pack under its own run.

    A separate run from the one waiting in ``review_required``: that run
    released its lease precisely so this could happen later, and M2 allows
    exactly one *active* run, not one run per bundle lifetime.

    Validation happens *before* the run exists, so a refusal -- a stale
    pack, an anonymous reviewer, a decision naming an invented cluster --
    surfaces as that specific error and leaves no failed run behind.
    Re-applying an already-applied pack takes no lease at all.
    """
    prepared = prepare_review_application(store, pack_path=pack_path)
    if isinstance(prepared, ReviewRecord):
        return ApplyReviewOutcome(
            review=prepared,
            revision=None,
            component=None,
            already_applied=True,
            addressed_item_count=0,
            total_item_count=0,
        )

    outcome: list[ApplyReviewOutcome] = []

    def _executor(store: BundleStore, run: RunRecord) -> ExecutorOutcome:
        outcome.append(commit_review(store, run_id=run.run_id, prepared=prepared))
        return ExecutorOutcome(state=RunState.COMPLETED, revision_id=None)

    run_bundle_operation(
        store,
        next_action=OperationRef(
            kind="review-apply",
            input_ids=(str(pack_path),),
            rationale="apply a filled speaker review pack (M8)",
        ),
        executor=_executor,
    )
    return outcome[0]


async def _run_single_product_stage[T](
    store: BundleStore,
    *,
    kind: str,
    rationale: str,
    stage: Callable[[RunId], Awaitable[T]],
) -> T:
    """Run one LLM-backed transform under its own run and lease (M2).

    The three single-stage CLI entry points below differ only in which
    coroutine they await, so the run/lease/failure bookkeeping lives here
    once rather than three times -- and a failure in any of them lands in
    the same durable, resumable ``failed`` state.
    """
    produced: list[T] = []

    async def _executor(store: BundleStore, run: RunRecord) -> ExecutorOutcome:
        produced.append(await stage(run.run_id))
        return ExecutorOutcome(state=RunState.COMPLETED, revision_id=None)

    await run_async_bundle_operation(
        store,
        next_action=OperationRef(kind=kind, rationale=rationale),
        executor=_executor,
    )
    return produced[0]


async def run_text_transform(
    store: BundleStore,
    *,
    mode: TextEditMode,
    agent: StructuredAgent,
    model: str,
    context_note: str = "",
) -> TextTransformOutcome:
    """CLI entry point for `transform text --mode correct|polish` (M9)."""
    return await _run_single_product_stage(
        store,
        kind=f"text-{mode.value}",
        rationale=f"transcript transform text --mode {mode.value}",
        stage=lambda run_id: transform_text(
            store,
            run_id=run_id,
            mode=mode,
            agent=agent,
            model=model,
            context_note=context_note,
        ),
    )


async def run_chapter_transform(
    store: BundleStore,
    *,
    agent: StructuredAgent,
    model: str,
    context_note: str = "",
) -> ChapterOutcome:
    """CLI entry point for `transform chapter` (M10)."""
    return await _run_single_product_stage(
        store,
        kind="chapter",
        rationale="transcript transform chapter",
        stage=lambda run_id: transform_chapters(
            store,
            run_id=run_id,
            agent=agent,
            model=model,
            context_note=context_note,
        ),
    )


async def run_minutes_transform(
    store: BundleStore,
    *,
    agent: StructuredAgent,
    model: str,
    context_note: str = "",
) -> MinutesOutcome:
    """CLI entry point for `transform minutes` (M10)."""
    return await _run_single_product_stage(
        store,
        kind="minutes",
        rationale="transcript transform minutes",
        stage=lambda run_id: transform_minutes(
            store,
            run_id=run_id,
            agent=agent,
            model=model,
            context_note=context_note,
        ),
    )


@dataclasses.dataclass(frozen=True)
class ProductsOutcome:
    """Everything one `products` pass produced, for reporting."""

    corrected: TextTransformOutcome | None
    polished: TextTransformOutcome | None
    chapters: ChapterOutcome
    minutes: MinutesOutcome
    render: RenderRecord
    render_body: str


async def build_products(
    store: BundleStore,
    *,
    run_id: RunId,
    agent: StructuredAgent,
    model: str,
    context_note: str = "",
    correct: bool = True,
    polish: bool = True,
    include_transcript: bool = True,
) -> ProductsOutcome:
    """M9/M10/M17: correct, polish, chapter, minute, and render.

    Ordered deliberately. Correction runs before polish so the polish pass
    reads already-correct names; both run before chaptering and minuting so
    those stages quote the text a reader will actually see; rendering runs
    last, over a head that already carries everything. Each step appends
    its own revision, so a failure part-way leaves the completed steps
    durable and the run resumable rather than discarding the lot -- and a
    resumed run skips the text passes whose proofs the head already
    carries rather than redoing them over their own output.
    """

    async def _text_pass(mode: TextEditMode) -> TextTransformOutcome | None:
        """Run one text pass, unless the head already carries its proof.

        A resumed `products` run replays from the top (M2: resume executes
        the recorded next_action, it never re-plans), so a pass that
        already landed would otherwise run a second time over its own
        output -- re-polishing polished text, which is both wasted model
        spend and a real way to fail the fidelity gates on a diff that has
        nothing left to remove.
        """
        key = (
            CapabilityKey.TEXT_CORRECTED
            if mode == TextEditMode.CORRECT
            else CapabilityKey.TEXT_POLISHED
        )
        document = project_head(store)
        if (
            not isinstance(document, NoDocumentYet)
            and document.capability_status(key) == CapabilityStatus.PRESENT_VALIDATED
        ):
            return None
        return await transform_text(
            store,
            run_id=run_id,
            mode=mode,
            agent=agent,
            model=model,
            context_note=context_note,
        )

    corrected = await _text_pass(TextEditMode.CORRECT) if correct else None
    polished = await _text_pass(TextEditMode.POLISH) if polish else None
    chapters = await transform_chapters(
        store, run_id=run_id, agent=agent, model=model, context_note=context_note
    )
    minutes = await transform_minutes(
        store, run_id=run_id, agent=agent, model=model, context_note=context_note
    )
    render = render_document(store, include_transcript=include_transcript)
    return ProductsOutcome(
        corrected=corrected,
        polished=polished,
        chapters=chapters,
        minutes=minutes,
        render=render,
        render_body=store.load_render_output(render.render_id).decode("utf-8"),
    )


@dataclasses.dataclass(frozen=True)
class RecipeOutcome:
    """What one `recipe run obsidian-recording` invocation reached.

    ``review_required`` is a *success*, not a failure: it is the durable
    checkpoint the whole overhaul exists for. ``products`` is only present
    when the run got past it (because a review was already applied, or
    because the operator explicitly opted out of stopping).
    """

    run: RunRecord
    stopped_for_review: bool
    products: ProductsOutcome | None


async def run_obsidian_recording_recipe(
    store: BundleStore,
    *,
    agent: StructuredAgent,
    model: str,
    context_note: str = "",
    stop_for_review: bool = True,
    subprocess_runner: SubprocessRunner = default_subprocess_runner,
) -> RecipeOutcome:
    """The composed path for an Obsidian note with one or more recordings.

    Assumes ingestion and assembly already happened (`source ingest`,
    which is where the operator asserts what belongs to this bundle, M3).
    From there: timeline -> transcribe -> normalise -> propose -> stop.

    Every step is skipped when its output already exists, so re-running
    after a review costs nothing upstream -- that is the corpus's
    "regeneration after a speaker correction must not rerun acquisition or
    ASR" requirement, satisfied by asking the document what it already has
    rather than by a cache.
    """
    document = project_head(store)
    if isinstance(document, NoDocumentYet):
        raise BundleControlError(
            "this bundle has no assembled document yet; ingest the note and its "
            "recordings first (`source ingest --kind obsidian-note|local-media`)."
        )
    if not document.components_of(TimelineCombinedComponent):
        run_timeline_transform(store)
    if document.capability_status(CapabilityKey.INFERENCE_ASR) != (
        CapabilityStatus.PRESENT_VALIDATED
    ):
        run_transcribe_transform(store, subprocess_runner=subprocess_runner)
    if not project_head_components(store, TimedTurnSetComponent):
        run_normalise_transform(store)

    reviewed = bool(project_head_components(store, SpeakerReviewComponent))
    if not reviewed:
        if not project_head_components(store, SpeakerHypothesisSetComponent):
            run = await run_speakers_propose(
                store,
                agent=agent,
                model=model,
                stop_for_review=stop_for_review,
                context_note=context_note,
            )
            if stop_for_review:
                return RecipeOutcome(run=run, stopped_for_review=True, products=None)
        elif stop_for_review:
            return RecipeOutcome(
                run=_review_required_run(store),
                stopped_for_review=True,
                products=None,
            )

    products: list[ProductsOutcome] = []

    async def _executor(store: BundleStore, run: RunRecord) -> ExecutorOutcome:
        products.append(
            await build_products(
                store,
                run_id=run.run_id,
                agent=agent,
                model=model,
                context_note=context_note,
            )
        )
        return ExecutorOutcome(state=RunState.COMPLETED, revision_id=None)

    run = await run_async_bundle_operation(
        store,
        next_action=OperationRef(
            kind="products",
            rationale="polished transcript, chapters, minutes, and render",
        ),
        executor=_executor,
    )
    return RecipeOutcome(run=run, stopped_for_review=False, products=products[0])


def _review_required_run(store: BundleStore) -> RunRecord:
    """The run already waiting at the review checkpoint.

    Reached when hypotheses exist but no review has been applied: the
    recipe must report the *existing* durable run rather than creating a
    second one that proposes again, which would mint fresh hypotheses and
    invalidate a pack the operator may already be filling in.
    """
    manifest = store.load_manifest()
    waiting = [
        run
        for run in (store.load_run(run_id) for run_id in manifest.run_ids)
        if run.state == RunState.REVIEW_REQUIRED
    ]
    if not waiting:
        raise BundleControlError(
            "speaker hypotheses exist but no review has been applied and no run is "
            "waiting at the review checkpoint; apply a review pack, or re-run "
            "`transform speakers-propose` to create the checkpoint again."
        )
    return waiting[-1]


def project_head_components[T](store: BundleStore, kind: type[T]) -> tuple[T, ...]:
    """Components of ``kind`` in the *current* head -- re-read, not cached.

    The recipe moves the head between steps, so every "has this already
    happened?" question must look at the head as it is now; a projection
    captured at the top would answer about a document three revisions old.
    """
    document = project_head(store)
    if isinstance(document, NoDocumentYet):
        return ()
    return document.components_of(kind)  # pyright: ignore[reportArgumentType, reportReturnType]


def render_document(
    store: BundleStore,
    *,
    revision_id: str | None = None,
    include_transcript: bool = True,
    timezone: str = "Australia/Perth",
    locale: str = "en-AU",
) -> RenderRecord:
    """M17: render one revision and store the record plus its bytes.

    Every field M17 requires for a complete render identity is recorded,
    including the destination snapshot the apply step later checks the
    target against (M13). Rendering never moves the head and takes no
    lease: it is pure, and a render of a superseded revision is a
    perfectly legitimate thing to have -- applying one is what needs a
    deliberate override.
    """
    document = project_document(store, revision_id=revision_id)
    if isinstance(document, NoDocumentYet):
        raise BundleControlError(
            "this bundle has no assembled document yet (M18: run `transform "
            "assemble` first)."
        )
    result = render_meeting_note(document, include_transcript=include_transcript)
    destination = destination_of(document)
    record = RenderRecord(
        render_id=mint_id("render"),
        bundle_id=document.bundle_id,
        revision_id=document.revision_id,
        profile=MEETING_NOTE_PROFILE,
        profile_version=MEETING_NOTE_PROFILE_VERSION,
        renderer_version=RENDERER_VERSION,
        template_sha256=template_sha256(),
        parameters={
            "variant": DUMC_VARIANT,
            "timezone": timezone,
            "locale": locale,
            "include_transcript": str(include_transcript).lower(),
        },
        destination_snapshot_artefact_id=(
            None if destination is None else destination.note_artefact_id
        ),
        input_capability_keys=result.input_capability_keys,
        output_sha256=result.sha256,
        created_at=datetime.now(UTC),
    )
    return store.add_render(record, output=result.body.encode("utf-8"))


def run_bundle_operation(
    store: BundleStore, *, next_action: OperationRef, executor: BundleExecutor
) -> RunRecord:
    """Create a fresh run, acquire its lease, and dispatch ``executor``
    directly -- the CLI's "first attempt" path for `transform timeline`/
    `transform transcribe`, sharing :func:`resume_run`'s own success/
    failure/lease-release handling (:func:`_run_executor_or_fail`) so a
    failure here leaves the run in exactly the same safe, durable,
    resumable ``failed`` state a crashed `resume` would produce, rather
    than a bespoke duplicate of that logic living in the CLI layer.
    """
    run = store.create_run(next_action=next_action)
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    run = store.load_run(run.run_id)

    def _fail_on_missing_executor(missing_next_action: OperationRef) -> RunRecord:
        # Unreachable in practice: `executor` is supplied directly here,
        # never looked up by kind, so the single-entry map below always
        # resolves it. Kept only because `_run_executor_or_fail`'s shared
        # dispatch shape always requires a "no executor" branch.
        store.release_lease(
            run_id=run.run_id,
            new_state=RunState.FAILED,
            next_action=missing_next_action,
        )
        raise NoExecutorRegisteredError(
            f"no executor registered for operation kind {missing_next_action.kind!r}."
        )

    return _run_executor_or_fail(
        store,
        run,
        {next_action.kind: executor},
        on_missing_executor=_fail_on_missing_executor,
    )


def run_timeline_transform(
    store: BundleStore, *, order: tuple[ArtefactId, ...] | None = None
) -> RunRecord:
    """CLI entry point for `transform timeline` (M6): create+acquire+
    dispatch in one call. ``order`` is recorded on the run's own
    ``next_action.input_ids`` so a later `resume` (if this crashes before
    completing) replays the exact same member order (M2)."""
    return run_bundle_operation(
        store,
        next_action=OperationRef(
            kind="timeline",
            input_ids=order or (),
            rationale="transcript transform timeline",
        ),
        executor=timeline_executor,
    )


def run_transcribe_transform(
    store: BundleStore,
    *,
    force: bool = False,
    subprocess_runner: SubprocessRunner = default_subprocess_runner,
) -> RunRecord:
    """CLI entry point for `transform transcribe` (M11): create+acquire+
    dispatch in one call. ``force`` and ``subprocess_runner`` are only
    ever honoured on this direct path -- a later `resume` of a crashed
    run always dispatches through the registered, force=False
    :func:`transcribe_executor` instead (M2: resuming must never widen
    what a crashed run was about to do).
    """

    def _executor(store: BundleStore, run: RunRecord) -> ExecutorOutcome:
        # revision_id=None: transcribe_media() already moved the head
        # itself when it had a new revision to move to (see
        # transcribe_executor's own docstring for why).
        transcribe_media(
            store, run_id=run.run_id, force=force, subprocess_runner=subprocess_runner
        )
        return ExecutorOutcome(state=RunState.COMPLETED, revision_id=None)

    return run_bundle_operation(
        store,
        next_action=OperationRef(
            kind="transcribe",
            rationale=f"transcript transform transcribe (force={force})",
        ),
        executor=_executor,
    )
