from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from jake_tools.transcripts.bundle.records import (
    NoDocumentYet,
    OperationRef,
    RunState,
    SourceAssociation,
)
from jake_tools.transcripts.bundle.store import (
    BundleAlreadyExistsError,
    BundleStore,
    LeaseHeldError,
    NotLeaseHolderError,
    RecordIdCollisionError,
    TakeOverRefusedError,
    UnknownRevisionError,
    UnknownSourceError,
    UnresolvedClosureError,
)


def _store(tmp_path: Path) -> BundleStore:
    store = BundleStore(tmp_path / "bundle")
    store.create_bundle()
    return store


def _registered_source(store: BundleStore) -> str:
    membership = store.register_source(
        association=SourceAssociation.OPERATOR_ASSERTION,
        evidence="jake-tools transcript obsidian-recording NOTE.md",
    )
    return membership.source_id


def _bogus_id(prefix: str) -> str:
    """A syntactically valid but never-minted ID, for closure-resolution tests."""
    return f"{prefix}_00000000-0000-7000-8000-000000000000"


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


# -- bundle lifecycle ---------------------------------------------------------


def test_create_bundle_mints_distinct_bundle_and_document_ids_and_records_the_edge(
    tmp_path: Path,
) -> None:
    store = BundleStore(tmp_path / "bundle")

    manifest = store.create_bundle()

    assert manifest.bundle_id != manifest.document_id
    assert manifest.bundle_id.startswith("bundle_")
    assert manifest.document_id.startswith("doc_")
    assert manifest.head_revision_id is None
    assert store.load_manifest().document_id == manifest.document_id


def test_create_bundle_lays_out_every_m16_directory(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    BundleStore(root).create_bundle()

    for subdirectory in (
        "blobs",
        "artefacts",
        "revisions",
        "runs",
        "attempts",
        "reviews",
        "renders",
        "applies",
    ):
        assert (root / subdirectory).is_dir()
    assert (root / "manifest.json").is_file()


def test_create_bundle_refuses_to_reinitialise_an_existing_bundle(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    with pytest.raises(BundleAlreadyExistsError):
        store.create_bundle()


def test_register_source_appends_a_membership_record_naming_the_bundle(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    membership = store.register_source(
        association=SourceAssociation.PROVIDER_ID, evidence="teams-event:abc123"
    )

    manifest = store.load_manifest()
    assert manifest.source_memberships == (membership,)
    assert membership.bundle_id == manifest.bundle_id
    assert membership.source_id.startswith("source_")


# -- ingestion ------------------------------------------------------------


def test_ingest_artefact_is_idempotent_for_identical_bytes_from_the_same_source(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    source_id = _registered_source(store)
    kwargs = {
        "source_id": source_id,
        "content": b"same bytes",
        "kind": "audio",
        "producer": "ingest:test",
        "acquisition_locator": "/tmp/a.wav",
    }

    first = store.ingest_artefact(**kwargs)
    second = store.ingest_artefact(**kwargs)

    assert first == second
    assert len(list((store.root / "artefacts").glob("*.json"))) == 1


def test_ingest_artefact_from_a_different_source_with_identical_bytes_is_distinct(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    source_a = _registered_source(store)
    source_b = _registered_source(store)

    from_a = store.ingest_artefact(
        source_id=source_a,
        content=b"same bytes",
        kind="audio",
        producer="ingest:test",
        acquisition_locator="/tmp/a.wav",
    )
    from_b = store.ingest_artefact(
        source_id=source_b,
        content=b"same bytes",
        kind="audio",
        producer="ingest:test",
        acquisition_locator="/tmp/a.wav",
    )

    assert from_a.artefact_id != from_b.artefact_id
    # The underlying bytes still dedupe to a single blob (content-addressed).
    assert from_a.blob_ref == from_b.blob_ref
    assert len(list((store.root / "blobs").iterdir())) == 1


def test_ingest_artefact_refuses_a_source_with_no_membership_record(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    with pytest.raises(UnknownSourceError):
        store.ingest_artefact(
            source_id="source_nonexistent",
            content=b"bytes",
            kind="audio",
            producer="ingest:test",
            acquisition_locator="/tmp/a.wav",
        )


def test_ingest_artefact_never_moves_the_head(tmp_path: Path) -> None:
    store = _store(tmp_path)
    source_id = _registered_source(store)

    store.ingest_artefact(
        source_id=source_id,
        content=b"bytes",
        kind="audio",
        producer="ingest:test",
        acquisition_locator="/tmp/a.wav",
    )

    assert store.load_manifest().head_revision_id is None


# -- revisions --------------------------------------------------------------


def test_append_revision_refuses_an_unresolvable_parent(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(UnknownRevisionError):
        store.append_revision(
            operation=OperationRef(kind="assemble"),
            parent_revision_ids=("rev_nonexistent",),
        )


def test_append_revision_accepts_a_real_parent_and_records_it(tmp_path: Path) -> None:
    store = _store(tmp_path)
    root_revision = store.append_revision(operation=OperationRef(kind="assemble"))

    child = store.append_revision(
        operation=OperationRef(kind="review-apply"),
        parent_revision_ids=(root_revision.revision_id,),
    )

    assert child.parent_revision_ids == (root_revision.revision_id,)
    assert store.load_revision(child.revision_id) == child


def test_revision_files_are_never_rewritten_by_later_store_operations(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    source_id = _registered_source(store)
    revision = store.append_revision(operation=OperationRef(kind="assemble"))
    path = store.root / "revisions" / f"{revision.revision_id}.json"
    written_bytes = path.read_bytes()

    store.ingest_artefact(
        source_id=source_id,
        content=b"x",
        kind="audio",
        producer="p",
        acquisition_locator="/tmp/x",
    )
    store.append_revision(
        operation=OperationRef(kind="assemble"),
        parent_revision_ids=(revision.revision_id,),
    )

    assert path.read_bytes() == written_bytes


def test_append_only_write_primitive_refuses_to_overwrite_an_existing_record(
    tmp_path: Path,
) -> None:
    """The shared write primitive behind revisions/artefacts/runs (M1, M16):
    a second write to a path that already holds a record is refused
    outright, never merged or silently replaced."""
    store = _store(tmp_path)
    revision = store.append_revision(operation=OperationRef(kind="assemble"))
    path = store.root / "revisions" / f"{revision.revision_id}.json"

    with pytest.raises(RecordIdCollisionError):
        store._write_json_exclusive(
            path, revision, conflict_error=RecordIdCollisionError
        )


# -- transactional head -------------------------------------------------------


def test_update_head_moves_to_a_structurally_valid_revision(tmp_path: Path) -> None:
    store = _store(tmp_path)
    source_id = _registered_source(store)
    artefact = store.ingest_artefact(
        source_id=source_id,
        content=b"x",
        kind="audio",
        producer="p",
        acquisition_locator="/tmp/x",
    )
    revision = store.append_revision(
        operation=OperationRef(kind="assemble"), artefact_ids=(artefact.artefact_id,)
    )
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())

    updated = store.update_head(run_id=run.run_id, revision_id=revision.revision_id)

    assert updated.head_revision_id == revision.revision_id
    assert store.load_manifest().head_revision_id == revision.revision_id


def test_update_head_refuses_a_revision_with_an_unresolvable_artefact_ref(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    revision = store.append_revision(
        operation=OperationRef(kind="assemble"),
        artefact_ids=(_bogus_id("artefact"),),
    )
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())

    with pytest.raises(UnresolvedClosureError):
        store.update_head(run_id=run.run_id, revision_id=revision.revision_id)

    assert store.load_manifest().head_revision_id is None


def test_update_head_refuses_a_run_that_does_not_hold_the_lease(tmp_path: Path) -> None:
    store = _store(tmp_path)
    revision = store.append_revision(operation=OperationRef(kind="assemble"))
    imposter_run = store.create_run(next_action=OperationRef(kind="assemble"))

    with pytest.raises(NotLeaseHolderError):
        store.update_head(run_id=imposter_run.run_id, revision_id=revision.revision_id)


def test_update_head_leaves_the_manifest_unchanged_and_readable_if_replace_fails(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    setup_store = BundleStore(root)
    setup_store.create_bundle()
    source_id = _registered_source(setup_store)
    artefact = setup_store.ingest_artefact(
        source_id=source_id,
        content=b"x",
        kind="audio",
        producer="p",
        acquisition_locator="/tmp/x",
    )
    revision = setup_store.append_revision(
        operation=OperationRef(kind="assemble"), artefact_ids=(artefact.artefact_id,)
    )
    run = setup_store.create_run(next_action=OperationRef(kind="assemble"))
    setup_store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    before = setup_store.load_manifest()

    def failing_replace(src: str, dst: str) -> None:
        raise OSError("simulated crash between temp write and rename")

    crashing_store = BundleStore(root, replace=failing_replace)

    with pytest.raises(OSError, match="simulated crash"):
        crashing_store.update_head(run_id=run.run_id, revision_id=revision.revision_id)

    after = BundleStore(root).load_manifest()
    assert after == before
    assert after.head_revision_id is None
    assert list(root.glob(".manifest.json.*")) == []


# -- leases -----------------------------------------------------------------


def test_lease_exclusion_refuses_a_second_concurrent_run_naming_the_holder(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    first = store.create_run(next_action=OperationRef(kind="assemble"))
    second = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=first.run_id, pid=os.getpid())

    with pytest.raises(LeaseHeldError, match=first.run_id):
        store.acquire_lease(run_id=second.run_id, pid=os.getpid())


def test_durable_state_transition_releases_the_lease(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    assert store.load_lease() is not None

    updated = store.release_lease(
        run_id=run.run_id,
        new_state=RunState.REVIEW_REQUIRED,
        next_action=OperationRef(kind="review-apply"),
    )

    assert updated.state == RunState.REVIEW_REQUIRED
    assert store.load_lease() is None


def test_release_lease_refuses_a_run_that_does_not_hold_the_lease(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    holder = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=holder.run_id, pid=os.getpid())
    imposter = store.create_run(next_action=OperationRef(kind="assemble"))

    with pytest.raises(NotLeaseHolderError):
        store.release_lease(
            run_id=imposter.run_id,
            new_state=RunState.FAILED,
            next_action=OperationRef(kind="assemble"),
        )


def test_a_released_run_can_be_resumed_without_take_over(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=first.run_id, pid=os.getpid())
    store.release_lease(
        run_id=first.run_id,
        new_state=RunState.FAILED,
        next_action=OperationRef(kind="assemble"),
    )

    resumer = store.create_run(
        next_action=OperationRef(kind="assemble"), resumes_run_id=first.run_id
    )
    lease = store.acquire_lease(run_id=resumer.run_id, pid=os.getpid())

    assert lease.run_id == resumer.run_id


def test_take_over_is_refused_while_the_holding_pid_is_alive(tmp_path: Path) -> None:
    store = _store(tmp_path)
    holder = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=holder.run_id, pid=os.getpid())  # our own pid: alive

    challenger = store.create_run(
        next_action=OperationRef(kind="assemble"), takeover_of_run_id=holder.run_id
    )

    with pytest.raises(TakeOverRefusedError, match="alive"):
        store.acquire_lease(run_id=challenger.run_id, pid=os.getpid(), take_over=True)


def test_take_over_is_allowed_once_the_holding_pid_is_dead(tmp_path: Path) -> None:
    store = _store(tmp_path)
    holder = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=holder.run_id, pid=_dead_pid())

    challenger = store.create_run(
        next_action=OperationRef(kind="assemble"), takeover_of_run_id=holder.run_id
    )
    lease = store.acquire_lease(
        run_id=challenger.run_id, pid=os.getpid(), take_over=True
    )

    assert lease.run_id == challenger.run_id
    # The dead run's own record is never rewritten by the run that took over.
    assert store.load_run(holder.run_id).state == RunState.RUNNING


def test_take_over_is_refused_when_there_is_no_lease_to_take_over(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    run = store.create_run(next_action=OperationRef(kind="assemble"))

    # No lease is held at all, so take_over=True has nothing to steal from —
    # acquire_lease should behave like an ordinary acquire.
    lease = store.acquire_lease(run_id=run.run_id, pid=os.getpid(), take_over=True)

    assert lease.run_id == run.run_id


# -- document projection -----------------------------------------------------


def test_document_head_is_no_document_yet_on_a_fresh_bundle(tmp_path: Path) -> None:
    store = _store(tmp_path)
    source_id = _registered_source(store)
    artefact = store.ingest_artefact(
        source_id=source_id,
        content=b"x",
        kind="audio",
        producer="p",
        acquisition_locator="/tmp/x",
    )

    state = store.document_head()

    assert isinstance(state, NoDocumentYet)
    assert state.candidate_artefact_ids == (artefact.artefact_id,)


def test_document_head_returns_the_head_revision_once_one_is_set(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    revision = store.append_revision(operation=OperationRef(kind="assemble"))
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    store.update_head(run_id=run.run_id, revision_id=revision.revision_id)

    state = store.document_head()

    assert state == revision
