from __future__ import annotations

import os
from pathlib import Path

import pytest

from jake_tools.transcripts.bundle.components import (
    NotesComponentBody,
    NotesKind,
    NotesSection,
    ParticipantSetComponentBody,
)
from jake_tools.transcripts.bundle.document import (
    CapabilityValidationFailedError,
    TranscriptDocumentV1,
    capability_validating_seam,
    project_head,
    project_revision,
)
from jake_tools.transcripts.bundle.ids import mint_id
from jake_tools.transcripts.bundle.records import (
    NoDocumentYet,
    OperationRef,
    SourceAssociation,
)
from jake_tools.transcripts.bundle.registry import CapabilityKey, CapabilityStatus
from jake_tools.transcripts.bundle.store import BundleStore, CapabilityValidator


def _store(
    root: Path, *, validate_capabilities: CapabilityValidator | None = None
) -> BundleStore:
    store = (
        BundleStore(root, validate_capabilities=validate_capabilities)
        if validate_capabilities is not None
        else BundleStore(root)
    )
    store.create_bundle()
    return store


def _notes_body(artefact_id: str) -> NotesComponentBody:
    return NotesComponentBody(
        notes_kind=NotesKind.PROVIDER_SUMMARY,
        source_artefact_id=artefact_id,
        authored=False,
        sections=(NotesSection(section_id=mint_id("seg"), title="T", text="text"),),
    )


def _bundle_with_notes_revision(store: BundleStore) -> tuple[str, str]:
    """Registers a source, ingests an artefact, adds a valid notes
    component, appends a revision over both, and moves the head to it.
    Returns (run_id, revision_id)."""
    membership = store.register_source(
        association=SourceAssociation.OPERATOR_ASSERTION, evidence="cli"
    )
    artefact = store.ingest_artefact(
        source_id=membership.source_id,
        content=b"notes bytes",
        kind="notes",
        producer="test",
        acquisition_locator="/tmp/notes.md",
    )
    component = store.add_component(_notes_body(artefact.artefact_id))
    revision = store.append_revision(
        operation=OperationRef(kind="assemble"),
        artefact_ids=(artefact.artefact_id,),
        component_ids=(component.component_id,),
    )
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    store.update_head(run_id=run.run_id, revision_id=revision.revision_id)
    return run.run_id, revision.revision_id


# -- project_head / project_revision ------------------------------------------


def test_project_head_returns_no_document_yet_typed_state_on_a_fresh_bundle(
    tmp_path: Path,
) -> None:
    """M16: a bundle whose head is still null projects to the explicit
    NoDocumentYet state -- never a crash, never a fabricated document."""
    store = _store(tmp_path / "bundle")

    state = project_head(store)

    assert isinstance(state, NoDocumentYet)
    assert not isinstance(state, TranscriptDocumentV1)


def test_project_head_returns_a_document_once_a_head_revision_exists(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "bundle")
    _run_id, revision_id = _bundle_with_notes_revision(store)

    document = project_head(store)

    assert isinstance(document, TranscriptDocumentV1)
    assert document.revision_id == revision_id
    assert document.capability_status(CapabilityKey.NOTES_PROVIDER) == (
        CapabilityStatus.PRESENT_VALIDATED
    )


def test_projection_is_immutable_across_a_later_head_move(tmp_path: Path) -> None:
    """F1: constructing a document from revision A must be unaffected by a
    later head move to revision B -- the projection retains no store
    handle and re-reads nothing after construction."""
    store = _store(tmp_path / "bundle")
    run_id, revision_a_id = _bundle_with_notes_revision(store)
    document_a = project_revision(store, revision_a_id)

    # Move the head forward to a second revision carrying a *different*
    # component graph (a participant set, no notes component at all).
    participant_component = store.add_component(
        ParticipantSetComponentBody(participants=())
    )
    revision_b = store.append_revision(
        operation=OperationRef(kind="review-apply"),
        parent_revision_ids=(revision_a_id,),
        component_ids=(participant_component.component_id,),
    )
    store.update_head(run_id=run_id, revision_id=revision_b.revision_id)

    # The earlier snapshot still reflects revision A, untouched.
    assert document_a.revision_id == revision_a_id
    assert document_a.capability_status(CapabilityKey.NOTES_PROVIDER) == (
        CapabilityStatus.PRESENT_VALIDATED
    )
    assert participant_component.component_id not in document_a.components
    # A fresh projection of A is unaffected by B's existence too.
    document_a_again = project_revision(store, revision_a_id)
    assert document_a_again.components.keys() == document_a.components.keys()


def test_capability_members_query_on_a_many_key(tmp_path: Path) -> None:
    store = _store(tmp_path / "bundle")
    _run_id, revision_id = _bundle_with_notes_revision(store)

    document = project_revision(store, revision_id)

    members = document.capability_members(CapabilityKey.NOTES_PROVIDER)
    assert len(members) == 1
    assert members[0].status == CapabilityStatus.PRESENT_VALIDATED


# -- capability_validating_seam / update_head wiring --------------------------


def test_update_head_accepts_a_revision_whose_projection_validates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    store = _store(root, validate_capabilities=capability_validating_seam(root))

    _run_id, revision_id = _bundle_with_notes_revision(store)

    assert store.load_manifest().head_revision_id == revision_id


def test_update_head_rejects_a_revision_whose_projection_fails_validation(
    tmp_path: Path,
) -> None:
    """The seam projects the candidate revision and raises if any
    capability the registry can actually validate comes back failed --
    here, an empty participant-set component fails participants.declared."""
    root = tmp_path / "bundle"
    store = _store(root, validate_capabilities=capability_validating_seam(root))
    broken_component = store.add_component(ParticipantSetComponentBody(participants=()))
    revision = store.append_revision(
        operation=OperationRef(kind="assemble"),
        component_ids=(broken_component.component_id,),
    )
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())

    with pytest.raises(CapabilityValidationFailedError):
        store.update_head(run_id=run.run_id, revision_id=revision.revision_id)

    assert store.load_manifest().head_revision_id is None


def test_capability_validating_seam_constructs_its_own_store_without_deadlocking(
    tmp_path: Path,
) -> None:
    """The seam receives only the revision, not a store handle -- it must
    build its own BundleStore for the same root and still complete under
    update_head's own lock (the reentrancy the store's lock registry
    exists for)."""
    root = tmp_path / "bundle"
    store = _store(root, validate_capabilities=capability_validating_seam(root))

    run_id, revision_id = _bundle_with_notes_revision(store)

    assert store.load_manifest().head_revision_id == revision_id
