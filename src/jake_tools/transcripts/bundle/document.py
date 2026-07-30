"""``TranscriptDocumentV1``: the revision-scoped document projection (M16).

The projection is the only sanctioned way to "see" a bundle's document
state. It loads exactly one revision through the store, resolves that
revision's complete component graph (:meth:`.store.BundleStore.
resolve_revision_closure`), runs every registered capability validator
(:func:`.registry.validate`) over the result, and returns an immutable
snapshot -- typed components by kind, capability records with statuses,
and per-member queries. Nothing here retains a store handle or re-reads
anything after construction (F1): a `TranscriptDocumentV1` built from
revision A cannot be affected by a later head move to revision B, because
it never looks at the store again.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import TypeVar

from ..errors import TranscriptError
from .components import ComponentRecord
from .ids import BundleId, ComponentId, DocumentId, RevisionId
from .records import NoDocumentYet
from .registry import (
    CapabilityKey,
    CapabilityMemberStatus,
    CapabilityRecord,
    CapabilityStatus,
)
from .registry import validate as _run_registry_validation
from .store import BundleStore, CapabilityValidator, RevisionRecord

# Bounded to v1's whole closed component-record union, so
# `components_of(object)` is still a type error while every real kind --
# not just the two Phase 2 had -- narrows correctly.
_ComponentT = TypeVar("_ComponentT", bound=ComponentRecord)


class TranscriptProjectionError(TranscriptError):
    """Base class for every error this module raises."""


class CapabilityValidationFailedError(TranscriptProjectionError):
    """A revision's projected capabilities include a genuine `failed` proof
    (M16): a one-cardinality key's own status, or a many-cardinality key's
    per-member status.

    Raised by the ``validate_capabilities`` seam built by
    :func:`capability_validating_seam`; ``BundleStore.update_head`` lets
    this propagate, so the head never moves onto a revision whose
    component graph a registered validator could actually inspect and
    reject. A many key with a merely absent/not-available-from-source
    member never raises this (MAJOR 4) -- only a member that genuinely
    failed validation does.
    """


class DirectConstructionRefusedError(TranscriptProjectionError):
    """``TranscriptDocumentV1`` was constructed (or replaced) outside
    :func:`project_revision`'s own construction window.

    M16 requires closure validation to run before any consumer sees the
    document; a hand-built or ``dataclasses.replace``/``copy.replace``-d
    instance would let a caller fabricate a "validated" document (fake
    ``present-validated`` records) or splice fields from two different
    revisions into one object (exactly the "structurally impossible"
    mixed-revision assembly F1 requires) -- both are refused here.
    """


# Module-private construction gate (MAJOR 2): set only for the duration of
# `project_revision`'s own call to the dataclass constructor. A field-based
# sentinel does not work here -- `dataclasses.replace()` copies every
# unspecified field (including a sentinel field) forward from the original,
# valid instance, so a naive `__post_init__` check on a stored field would
# wave a replaced-and-mutated copy straight through. Gating on ambient
# construction-in-progress state instead means neither `dataclasses.replace()`
# nor a bare `TranscriptDocumentV1(...)` call is ever inside the window.
_constructing: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_constructing_transcript_document_v1", default=False
)


@contextlib.contextmanager
def _construction_window() -> Iterator[None]:
    token = _constructing.set(True)
    try:
        yield
    finally:
        _constructing.reset(token)


@dataclasses.dataclass(frozen=True)
class TranscriptDocumentV1:
    """One revision's complete, validated document state.

    Construct only via :func:`project_revision`/:func:`project_head` --
    enforced, not just documented: :meth:`__post_init__` refuses any
    construction outside :func:`project_revision`'s own window, and
    :meth:`__replace__` refuses ``copy.replace()``/``dataclasses.replace()``
    outright (the latter does not consult ``__replace__`` on this Python
    version and instead reconstructs via ``__init__`` with every
    unspecified field carried forward, which is exactly why the
    construction-window check -- not a stored sentinel field -- is what
    actually stops it). Both close off F1's "mixed-revision assembly must
    be structurally impossible": a caller cannot hand-build a document
    with fabricated ``present-validated`` records, and cannot splice one
    revision's fields onto another's document via a field-level replace.
    """

    document_id: DocumentId
    bundle_id: BundleId
    revision_id: RevisionId
    parent_revision_ids: tuple[RevisionId, ...]
    components: Mapping[ComponentId, ComponentRecord]
    capabilities: Mapping[CapabilityKey, CapabilityRecord]

    def __post_init__(self) -> None:
        if not _constructing.get():
            raise DirectConstructionRefusedError(
                "TranscriptDocumentV1 must be constructed via project_revision/"
                "project_head -- direct construction bypasses M16's "
                "closure-validate-before-any-consumer-sees-it rule."
            )

    def __replace__(self, **_changes: object) -> TranscriptDocumentV1:
        raise DirectConstructionRefusedError(
            "TranscriptDocumentV1 is immutable and revision-scoped (F1): "
            "replacing fields on an existing instance would let a caller mix "
            "fields from different revisions into one document, which M16 "
            "requires to be structurally impossible. Construct a fresh "
            "document via project_revision instead."
        )

    def capability(self, key: CapabilityKey) -> CapabilityRecord:
        return self.capabilities[key]

    def capability_status(self, key: CapabilityKey) -> CapabilityStatus:
        return self.capabilities[key].status

    def capability_members(
        self, key: CapabilityKey
    ) -> tuple[CapabilityMemberStatus, ...]:
        """M4: "consumers query many keys per member, never by aggregate
        status alone." The members list is exactly what the aggregate
        status alone cannot tell a caller. This -- not :meth:`components_of`
        -- is the capability consumer path.
        """
        return self.capabilities[key].members

    def components_of(self, kind: type[_ComponentT]) -> tuple[_ComponentT, ...]:
        """Every resolved component that is an instance of ``kind`` (e.g.
        ``NotesComponent``), in no particular order -- callers that care
        about order (M6 canonical turn order, M20 section order) get it
        from the component's own fields, never re-derived here.

        This is *evidence-level* access to the raw, resolved component
        graph -- it says nothing about whether that component's own
        capability actually validated (a broken ``NotesComponent`` still
        appears here even when ``capability(NOTES_PROVIDER).status`` is
        ``failed``). A caller that needs a *proof*, not raw evidence,
        wants :meth:`capability`/:meth:`capability_members` instead.
        ``kind`` is constrained to v1's closed component-record union, so
        a call like ``components_of(object)`` is a type error rather than
        an always-empty runtime no-op.
        """
        return tuple(
            component
            for component in self.components.values()
            if isinstance(component, kind)
        )


def project_revision(
    store: BundleStore, revision_id: RevisionId
) -> TranscriptDocumentV1:
    """Project exactly one revision (M16).

    Resolves the revision's complete component graph via the store
    (fail-closed on any dangling artefact/component ref, unknown
    component kind, or invalid M21 supersession claim -- see
    ``BundleStore._validate_structural_closure``), runs the capability
    registry over the result, and copies everything the returned document
    exposes into plain, frozen mappings before returning. The copy is
    what makes F1 hold: nothing downstream of this call can observe a
    later mutation to the store's on-disk state through an
    already-returned document.
    """
    manifest = store.load_manifest()
    closure = store.resolve_revision_closure(revision_id)
    capabilities = _run_registry_validation(revision_id, closure.components)
    with _construction_window():
        return TranscriptDocumentV1(
            document_id=manifest.document_id,
            bundle_id=manifest.bundle_id,
            revision_id=closure.target.revision_id,
            parent_revision_ids=closure.target.parent_revision_ids,
            components=MappingProxyType(dict(closure.components)),
            capabilities=MappingProxyType(dict(capabilities)),
        )


def project_head(store: BundleStore) -> TranscriptDocumentV1 | NoDocumentYet:
    """Project the bundle's current head (M16).

    A bundle whose head is still null has no document to project: this
    returns the store's own :class:`.records.NoDocumentYet` state
    unchanged -- an explicit typed state, never a crash and never a
    fabricated empty document.
    """
    head = store.document_head()
    if isinstance(head, NoDocumentYet):
        return head
    return project_revision(store, head.revision_id)


def _blocking_capability_keys(
    capabilities: Mapping[CapabilityKey, CapabilityRecord],
) -> Mapping[CapabilityKey, tuple[str, ...]]:
    """MAJOR 4: which capabilities should block a head move, and why.

    A ``many``-cardinality key (``members`` populated) blocks only on a
    *member* whose own status is ``failed`` -- M4's own worked example
    ("two validated recordings and one reference-only embed") requires a
    merely absent/not-available-from-source member to stay usable, so the
    aggregate top-level status is never consulted for these. A
    ``one``-cardinality key (no members) blocks on its own top-level
    ``failed`` status directly. The returned mapping's values name the
    failed member IDs (empty for a one-cardinality key), so a caller can
    build a precise error message without re-deriving this logic.
    """
    blocking: dict[CapabilityKey, tuple[str, ...]] = {}
    for key, record in capabilities.items():
        if record.members:
            failed_member_ids = tuple(
                member.member_id
                for member in record.members
                if member.status == CapabilityStatus.FAILED
            )
            if failed_member_ids:
                blocking[key] = failed_member_ids
        elif record.status == CapabilityStatus.FAILED:
            blocking[key] = ()
    return blocking


def capability_validating_seam(store: BundleStore) -> CapabilityValidator:
    """Build a ``validate_capabilities`` seam bound to ``store``'s own root.

    Projects the candidate revision through a *fresh* ``BundleStore`` for
    ``store.root`` -- taken directly from the passed-in store, never an
    independently-supplied second root that could silently disagree with
    the store actually performing the head move (MAJOR 5). Constructing a
    fresh instance (rather than reusing ``store`` itself) is exactly the
    shape ``BundleStore._locked``'s reentrancy is designed for (see
    ``test_capability_validator_with_its_own_store_instance_does_not_deadlock``
    in ``test_transcript_bundle_store.py``). Raises
    :class:`CapabilityValidationFailedError` if any capability the
    registry could actually validate came back genuinely ``failed`` (per
    :func:`_blocking_capability_keys` -- member-level for many keys,
    top-level for one keys) -- so ``update_head`` refuses to move onto
    that revision.
    """
    root = store.root

    def _validate(revision: RevisionRecord) -> None:
        own_store = BundleStore(root)
        document = project_revision(own_store, revision.revision_id)
        blocking = _blocking_capability_keys(document.capabilities)
        if blocking:
            details = ", ".join(
                f"{key.value}"
                + (f" (members: {sorted(member_ids)})" if member_ids else "")
                for key, member_ids in sorted(blocking.items())
            )
            raise CapabilityValidationFailedError(
                f"revision {revision.revision_id} failed capability validation: "
                f"{details}"
            )

    return _validate
