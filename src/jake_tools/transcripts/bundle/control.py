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


class NoResumableRunError(BundleControlError):
    """No run in the bundle is in a durable state (M2)."""


class AmbiguousResumeTargetError(BundleControlError):
    """More than one run is resumable; ``--run`` is required to disambiguate."""


class RunNotResumableError(BundleControlError):
    """``--run`` named a run that is not currently in a durable state."""


class NoExecutorRegisteredError(BundleControlError):
    """``run.next_action.kind`` has no registered executor (v1 ships none)."""


# -- bundle create ------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CreatedBundle:
    store: BundleStore
    manifest: BundleManifest


def create_bundle(root: Path | None = None) -> CreatedBundle:
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
    """
    parent = root if root is not None else DEFAULT_BUNDLES_ROOT
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=parent, prefix=".bundle-creating-"))
    manifest = BundleStore(staging).create_bundle()
    final_root = parent / manifest.bundle_id
    os.rename(staging, final_root)
    return CreatedBundle(store=BundleStore(final_root), manifest=manifest)


# -- source ingest --------------------------------------------------------


def default_operator_assertion_evidence(
    *, kind: str, producer: str, acquisition_locator: str
) -> str:
    """M3's default evidence text: "naming inputs on the command line IS
    the operator assertion, and is recorded as such." A plain record of
    what the operator named, so a later reader sees exactly what was
    asserted -- never inferred from title, date, or filename similarity.
    """
    return (
        f"transcript source ingest --kind {kind} --producer {producer} "
        f"{acquisition_locator}"
    )


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
class MemberFailure:
    capability: CapabilityKey
    member_id: str
    detail: str


def project_document(
    store: BundleStore, *, revision_id: str | None
) -> TranscriptDocumentV1 | NoDocumentYet:
    """Project ``revision_id``, or the current head when omitted (M16)."""
    if revision_id is not None:
        return project_revision(store, revision_id)
    return project_head(store)


def failed_members(document: TranscriptDocumentV1) -> tuple[MemberFailure, ...]:
    """Every genuinely-failed capability member (M4), across every key.

    Scoped to member-level failures on purpose, matching ``validate``'s
    exit-code contract: a many-cardinality capability's aggregate status
    can legitimately degrade to something other than present-validated
    with no member having failed at all (M4's own worked example -- two
    validated recordings plus one reference-only embed) -- aggregate
    status alone is not a trustworthy failure signal. A one-cardinality
    key's own top-level ``failed`` status (no members) already blocks
    ``BundleStore.update_head`` from ever making that revision the head
    in the first place (``document.capability_validating_seam``), so a
    projected *head* is guarded against that class of failure by
    construction; ``--revision`` can still target an orphan revision that
    never went through that gate, but member-level failure remains this
    command's own exit-code signal, by design, in v1.
    """
    return tuple(
        MemberFailure(capability=key, member_id=member.member_id, detail=member.detail)
        for key, record in document.capabilities.items()
        for member in record.members
        if member.status == CapabilityStatus.FAILED
    )


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


def resume_run(
    store: BundleStore, *, run_id: str | None, executors: Mapping[str, BundleExecutor]
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
    retry once a real executor exists.
    """
    manifest = store.load_manifest()
    target_run_id = (
        run_id if run_id is not None else _single_resumable_run(store, manifest)
    )

    pending = store.load_run(target_run_id)
    if pending.state not in DURABLE_RUN_STATES:
        durable = ", ".join(state.value for state in DURABLE_RUN_STATES)
        raise RunNotResumableError(
            f"run {target_run_id} is in state {pending.state.value!r}; only a "
            f"durable-state run ({durable}) can be resumed."
        )
    original_state = pending.state
    original_next_action = pending.next_action

    store.acquire_lease(run_id=target_run_id, pid=os.getpid())
    run = store.load_run(target_run_id)
    next_action = run.next_action
    if next_action is None:
        # RunRecord's own invariant guarantees a non-terminal state
        # (running, which acquire_lease just set) carries a next_action;
        # unreachable in practice, kept only so this stays a typed error
        # rather than an AttributeError if that invariant is ever loosened.
        raise BundleControlError(
            f"run {target_run_id} has no next_action while running."
        )

    executor = executors.get(next_action.kind)
    if executor is None:
        store.release_lease(
            run_id=target_run_id,
            new_state=original_state,
            next_action=original_next_action,
        )
        raise NoExecutorRegisteredError(
            f"no executor registered for operation kind {next_action.kind!r} "
            "(transforms arrive in later slices)."
        )

    outcome = executor(store, run)
    if outcome.state == RunState.COMPLETED:
        if outcome.revision_id is not None:
            store.update_head(run_id=target_run_id, revision_id=outcome.revision_id)
        return store.release_lease(run_id=target_run_id, new_state=RunState.COMPLETED)
    return store.release_lease(
        run_id=target_run_id, new_state=outcome.state, next_action=outcome.next_action
    )
