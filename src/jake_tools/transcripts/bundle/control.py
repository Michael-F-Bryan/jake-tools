"""Bundle control-plane orchestration (Phase 2).

The operator-facing CLI (``jake_tools.cli.transcript_bundle``) stays thin;
every multi-step bundle operation -- create, ingest, inspect, validate,
status, resume -- lives here instead. Nothing in this module mints IDs,
writes to the bundle directory directly, or bypasses ``BundleStore``'s
locking: it is plain orchestration over ``store``/``document``/
``registry``'s own public surface (M1, M2, M3, M4, M16).
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path

from ..errors import TranscriptError
from .document import TranscriptDocumentV1, project_head, project_revision
from .ids import RevisionId
from .records import (
    DURABLE_RUN_STATES,
    BundleManifest,
    Lease,
    NoDocumentYet,
    OperationRef,
    RunRecord,
    RunState,
    SourceAssociation,
    SourceMembershipRecord,
)
from .registry import CapabilityKey, CapabilityStatus
from .store import ArtefactRecord, BundleStore, UnknownSourceError

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
