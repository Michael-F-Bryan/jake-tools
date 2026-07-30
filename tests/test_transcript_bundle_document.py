from __future__ import annotations

import copy
import dataclasses
import os
from pathlib import Path

import pytest

import jake_tools.transcripts.bundle.store as store_module
from jake_tools.transcripts.bundle.components import (
    NotesComponent,
    NotesComponentBody,
    NotesKind,
    NotesSection,
    NotesSectionBody,
    ParticipantDeclarationSource,
    ParticipantRecord,
    ParticipantSetComponentBody,
    ParticipantStatus,
)
from jake_tools.transcripts.bundle.document import (
    CapabilityValidationFailedError,
    DirectConstructionRefusedError,
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
from jake_tools.transcripts.bundle.store import BundleStore


def _store(root: Path) -> BundleStore:
    """Always uses BundleStore's own default ``validate_capabilities``
    (the real M4 registry, on by default since MAJOR 5). Tests that need
    the explicit ``None`` escape hatch construct ``BundleStore`` directly."""
    store = BundleStore(root)
    store.create_bundle()
    return store


def _notes_body(artefact_id: str) -> NotesComponentBody:
    return NotesComponentBody(
        notes_kind=NotesKind.PROVIDER_SUMMARY,
        source_artefact_id=artefact_id,
        authored=False,
        sections=(NotesSectionBody(title="T", text="text"),),
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


def _participant_body(name: str = "Jake") -> ParticipantSetComponentBody:
    return ParticipantSetComponentBody(
        participants=(
            ParticipantRecord(
                participant_id=mint_id("participant"),
                declaration_source=ParticipantDeclarationSource.OPERATOR,
                declaration_evidence=f"cli: --participant '{name}'",
                display_names=(name,),
                status=ParticipantStatus.DECLARED,
            ),
        )
    )


# -- BLOCKER 1 / M21: supersession as seen through the projection -----------


def test_correcting_a_participant_set_makes_the_correction_present_validated(
    tmp_path: Path,
) -> None:
    """The verifier's exact scenario: correcting a one-cardinality
    component (a wrong participant set) by superseding it must not brick
    the bundle. Revision B's projection must show participants.declared
    as present-validated, with exactly the new component -- not
    permanently failed, and not blocked by update_head's own
    ancestor-of-head check."""
    store = _store(tmp_path / "bundle")
    old_component = store.add_component(_participant_body("Wrong Name"))
    revision_a = store.append_revision(
        operation=OperationRef(kind="assemble"),
        component_ids=(old_component.component_id,),
    )
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    store.update_head(run_id=run.run_id, revision_id=revision_a.revision_id)

    document_a = project_revision(store, revision_a.revision_id)
    assert document_a.capability_status(CapabilityKey.PARTICIPANTS_DECLARED) == (
        CapabilityStatus.PRESENT_VALIDATED
    )

    new_component = store.add_component(_participant_body("Correct Name"))
    revision_b = store.append_revision(
        operation=OperationRef(kind="review-apply"),
        parent_revision_ids=(revision_a.revision_id,),
        component_ids=(new_component.component_id,),
        superseded_component_ids=(old_component.component_id,),
    )
    store.update_head(run_id=run.run_id, revision_id=revision_b.revision_id)

    document_b = project_revision(store, revision_b.revision_id)
    record = document_b.capability(CapabilityKey.PARTICIPANTS_DECLARED)
    assert record.status == CapabilityStatus.PRESENT_VALIDATED
    assert record.component_ids == (new_component.component_id,)
    assert old_component.component_id not in document_b.components
    assert new_component.component_id in document_b.components


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
    handle and re-reads nothing after construction.

    Revision B carries a genuinely *valid* participant-set component (not
    an empty one): since MAJOR 5 made the real M4 registry the store's
    default, an empty/broken component would now be refused by
    update_head itself -- this fixture must move the head onto a revision
    the registry actually accepts, per the coordinator's note."""
    store = _store(tmp_path / "bundle")
    run_id, revision_a_id = _bundle_with_notes_revision(store)
    document_a = project_revision(store, revision_a_id)

    participant_component = store.add_component(_participant_body())
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


# -- MAJOR 2: TranscriptDocumentV1 cannot be forged --------------------------


def test_direct_construction_of_transcript_document_v1_is_refused(
    tmp_path: Path,
) -> None:
    """A hand-built document (e.g. with fabricated present-validated
    records) must be impossible, not just discouraged by docstring."""
    with pytest.raises(DirectConstructionRefusedError):
        TranscriptDocumentV1(
            document_id=mint_id("doc"),
            bundle_id=mint_id("bundle"),
            revision_id=mint_id("rev"),
            parent_revision_ids=(),
            components={},
            capabilities={},
        )


def test_dataclasses_replace_on_transcript_document_v1_is_refused(
    tmp_path: Path,
) -> None:
    """dataclasses.replace() does not consult __replace__ on this Python
    version and instead reconstructs via __init__ with every unspecified
    field (including any stored sentinel) carried forward from the
    original -- the construction-window gate must catch this case too,
    not just a bare hand-built construction."""
    store = _store(tmp_path / "bundle")
    _run_id, revision_id = _bundle_with_notes_revision(store)
    document = project_revision(store, revision_id)

    with pytest.raises(DirectConstructionRefusedError):
        dataclasses.replace(document, revision_id=mint_id("rev"))


def test_copy_replace_on_transcript_document_v1_is_refused(tmp_path: Path) -> None:
    store = _store(tmp_path / "bundle")
    _run_id, revision_id = _bundle_with_notes_revision(store)
    document = project_revision(store, revision_id)

    with pytest.raises(DirectConstructionRefusedError):
        copy.replace(document, revision_id=mint_id("rev"))


def test_project_revision_can_still_construct_documents_after_a_refused_replace(
    tmp_path: Path,
) -> None:
    """The construction-window gate must not leak 'stuck open' or 'stuck
    closed' state across calls."""
    store = _store(tmp_path / "bundle")
    _run_id, revision_id = _bundle_with_notes_revision(store)
    document = project_revision(store, revision_id)

    with pytest.raises(DirectConstructionRefusedError):
        dataclasses.replace(document, revision_id=mint_id("rev"))

    second = project_revision(store, revision_id)
    assert second.revision_id == revision_id


# -- capability_validating_seam / update_head wiring (MAJOR 5) ----------------


def test_update_head_default_accepts_a_revision_whose_projection_validates(
    tmp_path: Path,
) -> None:
    """MAJOR 5: the registry seam is now BundleStore's default -- no
    explicit wiring required."""
    store = _store(tmp_path / "bundle")

    _run_id, revision_id = _bundle_with_notes_revision(store)

    assert store.load_manifest().head_revision_id == revision_id


def test_update_head_default_rejects_a_revision_whose_projection_fails_validation(
    tmp_path: Path,
) -> None:
    """The default seam projects the candidate revision and raises if any
    capability the registry can actually validate comes back genuinely
    failed -- here, two ambiguous participant-set components."""
    store = _store(tmp_path / "bundle")
    first = store.add_component(_participant_body("A"))
    second = store.add_component(_participant_body("B"))
    revision = store.append_revision(
        operation=OperationRef(kind="assemble"),
        component_ids=(first.component_id, second.component_id),
    )
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())

    with pytest.raises(CapabilityValidationFailedError):
        store.update_head(run_id=run.run_id, revision_id=revision.revision_id)

    assert store.load_manifest().head_revision_id is None


def test_update_head_escape_hatch_allows_a_structurally_valid_but_capability_broken_revision(
    tmp_path: Path,
) -> None:
    """MAJOR 5: validate_capabilities=None is the explicit, deliberate
    opt-out to structural-only validation -- a genuinely broken capability
    is still accepted (a test-only "diagnostic script" shape)."""
    store = BundleStore(tmp_path / "bundle", validate_capabilities=None)
    store.create_bundle()
    first = store.add_component(_participant_body("A"))
    second = store.add_component(_participant_body("B"))
    revision = store.append_revision(
        operation=OperationRef(kind="assemble"),
        component_ids=(first.component_id, second.component_id),
    )
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())

    updated = store.update_head(run_id=run.run_id, revision_id=revision.revision_id)

    assert updated.head_revision_id == revision.revision_id


def test_capability_validating_seam_takes_a_store_not_a_root(tmp_path: Path) -> None:
    """MAJOR 5: capability_validating_seam is bound to a BundleStore
    instance -- its root is always store.root, so there is no longer any
    way to construct a seam whose root silently disagrees with the store
    it is protecting."""
    root = tmp_path / "bundle"
    store = BundleStore(root, validate_capabilities=None)
    store.create_bundle()
    seam = capability_validating_seam(store)

    _run_id, revision_id = _bundle_with_notes_revision(store)
    revision = store.load_revision(revision_id)

    # Does not raise: the seam, built from `store`, validates against the
    # exact same bundle `store` itself just wrote to.
    seam(revision)


def test_capability_validating_seam_constructs_its_own_store_without_deadlocking(
    tmp_path: Path,
) -> None:
    """The seam builds its own BundleStore for store.root and still
    completes under update_head's own lock (the reentrancy the store's
    lock registry exists for)."""
    root = tmp_path / "bundle"
    store = _store(root)

    run_id, revision_id = _bundle_with_notes_revision(store)

    assert store.load_manifest().head_revision_id == revision_id


# -- MAJOR 4: seam blocks on genuine member-level failure only ---------------


def test_update_head_blocks_on_a_many_key_with_a_genuinely_failed_member(
    tmp_path: Path,
) -> None:
    """A many-cardinality key (notes.provider) with one valid and one
    genuinely broken (duplicate section IDs) member must still block the
    head move -- MAJOR 4 narrows blocking to genuine member failures, it
    does not remove blocking for them."""
    store = _store(tmp_path / "bundle")
    membership = store.register_source(
        association=SourceAssociation.OPERATOR_ASSERTION, evidence="cli"
    )
    artefact = store.ingest_artefact(
        source_id=membership.source_id,
        content=b"x",
        kind="notes",
        producer="test",
        acquisition_locator="/tmp/x",
    )
    good = store.add_component(_notes_body(artefact.artefact_id))

    shared_id = mint_id("seg")
    broken_body_record = NotesComponent(
        notes_kind=NotesKind.PROVIDER_SUMMARY,
        source_artefact_id=artefact.artefact_id,
        authored=False,
        sections=(
            NotesSection(section_id=shared_id, title="A", text="a"),
            NotesSection(section_id=shared_id, title="B", text="b"),
        ),
        component_id=mint_id("component"),
        content_hash=store_module._component_content_hash(
            # A placeholder content_hash value has no bearing on the
            # recomputed hash (MINOR C strips it entirely) -- any instance
            # of the intended content computes the real one.
            NotesComponentBody(
                notes_kind=NotesKind.PROVIDER_SUMMARY,
                source_artefact_id=artefact.artefact_id,
                authored=False,
                sections=(
                    NotesSectionBody(title="A", text="a"),
                    NotesSectionBody(title="B", text="b"),
                ),
            )
        ),
        created_at=good.created_at,
    )
    # Write the deliberately-broken record directly (bypassing
    # add_component's normal minting path -- this simulates a corrupted
    # or adversarially-crafted component file, but with a *correct*
    # content_hash: MINOR C's own on-load hash verification must not be
    # what blocks this revision, its many-key member validation must).
    (store.root / "components" / f"{broken_body_record.component_id}.json").write_text(
        broken_body_record.model_dump_json(), encoding="utf-8"
    )

    revision = store.append_revision(
        operation=OperationRef(kind="assemble"),
        artefact_ids=(artefact.artefact_id,),
        component_ids=(good.component_id, broken_body_record.component_id),
    )
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())

    with pytest.raises(CapabilityValidationFailedError, match="notes.provider"):
        store.update_head(run_id=run.run_id, revision_id=revision.revision_id)
