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

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

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


class TranscriptProjectionError(TranscriptError):
    """Base class for every error this module raises."""


class CapabilityValidationFailedError(TranscriptProjectionError):
    """A revision's projected capabilities include a `failed` proof (M16).

    Raised by the ``validate_capabilities`` seam built by
    :func:`capability_validating_seam`; ``BundleStore.update_head`` lets
    this propagate, so the head never moves onto a revision whose
    component graph a registered validator could actually inspect and
    reject.
    """


@dataclasses.dataclass(frozen=True)
class TranscriptDocumentV1:
    """One revision's complete, validated document state.

    Construct only via :func:`project_revision`/:func:`project_head` --
    never by hand -- so "closure validation runs before any consumer sees
    the document" (M16) stays true by construction: there is no code path
    that produces one of these without first resolving and validating the
    closure it describes.
    """

    document_id: DocumentId
    bundle_id: BundleId
    revision_id: RevisionId
    parent_revision_ids: tuple[RevisionId, ...]
    components: Mapping[ComponentId, ComponentRecord]
    capabilities: Mapping[CapabilityKey, CapabilityRecord]

    def capability(self, key: CapabilityKey) -> CapabilityRecord:
        return self.capabilities[key]

    def capability_status(self, key: CapabilityKey) -> CapabilityStatus:
        return self.capabilities[key].status

    def capability_members(
        self, key: CapabilityKey
    ) -> tuple[CapabilityMemberStatus, ...]:
        """M4: "consumers query many keys per member, never by aggregate
        status alone." The members list is exactly what the aggregate
        status alone cannot tell a caller.
        """
        return self.capabilities[key].members

    def components_of(self, kind: type) -> tuple[ComponentRecord, ...]:
        """Every resolved component that is an instance of ``kind`` (e.g.
        ``NotesComponent``), in no particular order -- callers that care
        about order (M6 canonical turn order, M20 section order) get it
        from the component's own fields, never re-derived here.
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
    (fail-closed on any dangling artefact/component ref or unknown
    component kind -- see ``BundleStore._validate_structural_closure``),
    runs the capability registry over the result, and copies everything
    the returned document exposes into plain, frozen mappings before
    returning. The copy is what makes F1 hold: nothing downstream of this
    call can observe a later mutation to the store's on-disk state through
    an already-returned document.
    """
    manifest = store.load_manifest()
    closure = store.resolve_revision_closure(revision_id)
    capabilities = _run_registry_validation(revision_id, closure.components)
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


def capability_validating_seam(root: Path) -> CapabilityValidator:
    """Build a ``validate_capabilities`` seam for ``BundleStore(root, ...)``.

    Projects the candidate revision through a *fresh* ``BundleStore`` for
    the same root -- the seam callback receives only the revision, no
    store handle, and constructing its own instance is exactly the shape
    ``BundleStore._locked``'s reentrancy is designed for (see
    ``test_capability_validator_with_its_own_store_instance_does_not_deadlock``
    in ``test_transcript_bundle_store.py``). Raises
    :class:`CapabilityValidationFailedError` if any capability the
    registry could actually validate came back ``failed`` -- so
    ``update_head`` refuses to move onto that revision.
    """

    def _validate(revision: RevisionRecord) -> None:
        own_store = BundleStore(root)
        document = project_revision(own_store, revision.revision_id)
        failed = {
            key: record
            for key, record in document.capabilities.items()
            if record.status == CapabilityStatus.FAILED
        }
        if failed:
            raise CapabilityValidationFailedError(
                f"revision {revision.revision_id} failed capability validation for: "
                f"{sorted(key.value for key in failed)}"
            )

    return _validate
