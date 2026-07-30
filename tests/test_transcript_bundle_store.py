from __future__ import annotations

import os
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from jake_tools.transcripts.bundle.records import (
    ArtefactRecord,
    Lease,
    NoDocumentYet,
    OperationRef,
    RevisionRecord,
    RunState,
    SourceAssociation,
)
from jake_tools.transcripts.bundle.store import (
    ArtefactMetadataConflictError,
    BundleAlreadyExistsError,
    BundleStore,
    HeadNotAncestorError,
    InvalidIdError,
    InvalidRunTransitionError,
    LeaseHeldError,
    NextActionPreconditionError,
    NotABundleError,
    NotLeaseHolderError,
    RecordIdCollisionError,
    RunNotAcquirableError,
    TakeOverRefusedError,
    UnknownArtefactError,
    UnknownRevisionError,
    UnknownRunError,
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


def test_load_manifest_raises_a_typed_error_for_a_non_bundle_directory(
    tmp_path: Path,
) -> None:
    """Finding 11b: a bare FileNotFoundError from load_manifest on a
    directory that was never created via create_bundle lies about the
    cause -- it should say "not a bundle", typed."""
    store = BundleStore(tmp_path / "not-a-bundle")

    with pytest.raises(NotABundleError):
        store.load_manifest()


def test_locked_does_not_create_the_bundle_root_as_a_side_effect(
    tmp_path: Path,
) -> None:
    """Finding NEW-4: _locked() used to mkdir the bundle root before
    checking anything, so a locking call on a non-bundle path (e.g.
    register_source) left a directory containing only .store.lock behind
    even though the call correctly failed. Failing must not leave debris."""
    root = tmp_path / "not-a-bundle"
    store = BundleStore(root)

    with pytest.raises(NotABundleError):
        store.register_source(
            association=SourceAssociation.OPERATOR_ASSERTION, evidence="cli"
        )

    assert not root.exists()


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


def test_ingest_artefact_raises_on_metadata_conflict_for_the_same_acquisition(
    tmp_path: Path,
) -> None:
    """Finding 6: idempotency is keyed on (hash, source, locator), but a
    caller supplying different kind/producer/derived_from for that same
    triple must not silently get back a record whose metadata lies."""
    store = _store(tmp_path)
    source_id = _registered_source(store)
    store.ingest_artefact(
        source_id=source_id,
        content=b"same bytes",
        kind="audio",
        producer="ingest:a",
        acquisition_locator="/tmp/a.wav",
    )

    with pytest.raises(ArtefactMetadataConflictError):
        store.ingest_artefact(
            source_id=source_id,
            content=b"same bytes",
            kind="caption",
            producer="ingest:a",
            acquisition_locator="/tmp/a.wav",
        )


def test_concurrent_ingest_of_the_same_acquisition_identity_produces_one_record(
    tmp_path: Path,
) -> None:
    """Finding NEW-2: ingest_artefact's find-existing-then-write dedup
    decision is a read-modify-write, not a pure append -- it must be one
    atomic step across processes, or racing ingests of the exact same
    (hash, source, locator) triple mint one artefact record each instead
    of the single record M16 requires. Real threads, not a mocked race."""
    store = _store(tmp_path)
    source_id = _registered_source(store)
    racer_count = 6
    barrier = threading.Barrier(racer_count)
    results: list[ArtefactRecord | None] = [None] * racer_count

    def ingest(index: int) -> None:
        barrier.wait()
        results[index] = store.ingest_artefact(
            source_id=source_id,
            content=b"same bytes",
            kind="audio",
            producer="ingest:test",
            acquisition_locator="/tmp/a.wav",
        )

    threads = [threading.Thread(target=ingest, args=(i,)) for i in range(racer_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    artefact_ids = {record.artefact_id for record in results if record is not None}
    assert len(artefact_ids) == 1
    assert len(list((store.root / "artefacts").glob("*.json"))) == 1


def test_load_artefact_returns_the_stored_record(tmp_path: Path) -> None:
    store = _store(tmp_path)
    source_id = _registered_source(store)
    artefact = store.ingest_artefact(
        source_id=source_id,
        content=b"x",
        kind="audio",
        producer="p",
        acquisition_locator="/tmp/x",
    )

    assert store.load_artefact(artefact.artefact_id) == artefact


def test_load_artefact_raises_unknown_artefact_error_for_a_missing_id(
    tmp_path: Path,
) -> None:
    """Finding 10: artefact lookups raise an artefact-specific error, not
    UnknownRevisionError borrowed from a different record kind."""
    store = _store(tmp_path)

    with pytest.raises(UnknownArtefactError):
        store.load_artefact(_bogus_id("artefact"))


def test_ingest_artefact_raises_unknown_artefact_error_for_unresolved_derived_from(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    source_id = _registered_source(store)

    with pytest.raises(UnknownArtefactError):
        store.ingest_artefact(
            source_id=source_id,
            content=b"x",
            kind="transcript",
            producer="p",
            acquisition_locator="/tmp/x",
            derived_from=(_bogus_id("artefact"),),
        )


def test_load_artefact_cannot_escape_the_bundle_root_via_a_crafted_id(
    tmp_path: Path,
) -> None:
    """Finding 14: a store-boundary ID argument is untrusted input. Before
    the fix, `_artefact_path` string-concatenated the raw ID into a path,
    so an ID shaped like a traversal string resolved (and read) a file
    outside the bundle root -- this proves it now never gets that far."""
    secret = tmp_path / "secret.json"
    secret.write_text('{"leaked": true}', encoding="utf-8")
    store = _store(tmp_path)

    with pytest.raises(InvalidIdError):
        store.load_artefact("../secret")  # type: ignore[arg-type]


def test_load_run_cannot_escape_the_bundle_root_via_a_crafted_id(
    tmp_path: Path,
) -> None:
    secret = tmp_path / "secret.json"
    secret.write_text('{"leaked": true}', encoding="utf-8")
    store = _store(tmp_path)

    with pytest.raises(InvalidIdError):
        store.load_run("../secret")  # type: ignore[arg-type]


# -- revisions --------------------------------------------------------------


def test_append_revision_refuses_an_unresolvable_parent(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(UnknownRevisionError):
        store.append_revision(
            operation=OperationRef(kind="assemble"),
            parent_revision_ids=(_bogus_id("rev"),),
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


def test_write_json_exclusive_never_leaves_a_torn_target_if_the_link_step_crashes(
    tmp_path: Path,
) -> None:
    """Finding 5: writing straight to the target with O_EXCL could leave a
    torn, permanently-unwritable file behind if the process died mid-write
    (append-only means the target can never be rewritten to fix it).
    Writing to a temp file and linking it in means a crash before the link
    leaves the target simply absent -- retry-safe, never torn or poisoned."""
    root = tmp_path / "bundle"
    setup_store = BundleStore(root)
    setup_store.create_bundle()
    revision = RevisionRecord(
        revision_id=_bogus_id("rev"),
        bundle_id=setup_store.load_manifest().bundle_id,
        operation=OperationRef(kind="assemble"),
        created_at=datetime.now(UTC),
    )

    def failing_link(src: str, dst: str) -> None:
        raise OSError("simulated crash before the record becomes visible")

    crashing_store = BundleStore(root, link=failing_link)
    path = crashing_store._revision_path(revision.revision_id)

    with pytest.raises(OSError, match="simulated crash"):
        crashing_store._write_json_exclusive(
            path, revision, conflict_error=RecordIdCollisionError
        )

    assert not path.exists()

    # The ID is not permanently poisoned: a working store can still write it.
    recovery_store = BundleStore(root)
    recovery_store._write_json_exclusive(
        path, revision, conflict_error=RecordIdCollisionError
    )
    assert path.exists()


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


def test_update_head_refuses_to_move_to_an_unrelated_revision(tmp_path: Path) -> None:
    """Finding 9: a structurally-valid revision that isn't built on top of
    the current head must be refused -- otherwise the head could move
    backwards or sideways and orphan lineage (M18)."""
    store = _store(tmp_path)
    first_root = store.append_revision(operation=OperationRef(kind="assemble"))
    other_root = store.append_revision(operation=OperationRef(kind="assemble"))
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    store.update_head(run_id=run.run_id, revision_id=first_root.revision_id)

    with pytest.raises(HeadNotAncestorError):
        store.update_head(run_id=run.run_id, revision_id=other_root.revision_id)

    assert store.load_manifest().head_revision_id == first_root.revision_id


def test_update_head_allows_moving_forward_to_a_descendant_of_the_current_head(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    root_revision = store.append_revision(operation=OperationRef(kind="assemble"))
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    store.update_head(run_id=run.run_id, revision_id=root_revision.revision_id)

    child = store.append_revision(
        operation=OperationRef(kind="review-apply"),
        parent_revision_ids=(root_revision.revision_id,),
    )
    updated = store.update_head(run_id=run.run_id, revision_id=child.revision_id)

    assert updated.head_revision_id == child.revision_id


def test_capability_validator_can_call_back_into_a_locking_store_method(
    tmp_path: Path,
) -> None:
    """Finding NEW-3: _locked() opened a fresh file handle and flock'd it
    on every call, so a validate_capabilities callback invoked from inside
    update_head's own lock -- exactly the seam the future M4 capability
    registry will use to resolve components through other store methods
    -- would self-deadlock the moment it called back into any locking
    method (a second LOCK_EX from the same process, via a different fd,
    blocks on the lock this same thread already holds). _locked() is now
    reentrant per thread. Run on a background thread with a bounded join
    so a regression fails the test instead of hanging the whole suite."""
    completed = threading.Event()

    def validator(revision: RevisionRecord) -> None:
        store.register_source(
            association=SourceAssociation.OPERATOR_ASSERTION, evidence="from validator"
        )

    store = BundleStore(tmp_path / "bundle", validate_capabilities=validator)
    store.create_bundle()
    revision = store.append_revision(operation=OperationRef(kind="assemble"))
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())

    def run_update_head() -> None:
        store.update_head(run_id=run.run_id, revision_id=revision.revision_id)
        completed.set()

    thread = threading.Thread(target=run_update_head, daemon=True)
    thread.start()
    thread.join(timeout=5.0)

    assert completed.is_set(), (
        "update_head did not complete -- _locked likely deadlocked"
    )
    assert len(store.load_manifest().source_memberships) == 1


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


def test_concurrent_acquire_lease_calls_leave_exactly_one_winner(
    tmp_path: Path,
) -> None:
    """Finding 4: acquisition is a single O_CREAT|O_EXCL create under a
    process-wide flock, not a read-then-write race -- two real threads
    racing to acquire the same bundle's lease for two different runs must
    produce exactly one winner and one clean, typed loser, never both
    "succeeding" or a corrupted lease file."""
    store = _store(tmp_path)
    run_a = store.create_run(next_action=OperationRef(kind="assemble"))
    run_b = store.create_run(next_action=OperationRef(kind="assemble"))
    results: dict[str, object] = {}
    barrier = threading.Barrier(2)

    def try_acquire(run_id: str, key: str) -> None:
        barrier.wait()
        try:
            results[key] = store.acquire_lease(run_id=run_id, pid=os.getpid())
        except LeaseHeldError as exc:
            results[key] = exc

    threads = [
        threading.Thread(target=try_acquire, args=(run_a.run_id, "a")),
        threading.Thread(target=try_acquire, args=(run_b.run_id, "b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    winners = [outcome for outcome in results.values() if isinstance(outcome, Lease)]
    losers = [
        outcome for outcome in results.values() if isinstance(outcome, LeaseHeldError)
    ]
    assert len(winners) == 1
    assert len(losers) == 1
    held = store.load_lease()
    assert held is not None
    assert held.run_id == winners[0].run_id


def test_concurrent_register_source_calls_do_not_lose_an_update(tmp_path: Path) -> None:
    """Finding 7: register_source's manifest read-modify-write is flock-
    guarded, so two concurrent registrations both end up recorded -- one
    never silently clobbers the other's read of the membership list."""
    store = _store(tmp_path)
    barrier = threading.Barrier(2)

    def register(evidence: str) -> None:
        barrier.wait()
        store.register_source(
            association=SourceAssociation.OPERATOR_ASSERTION, evidence=evidence
        )

    threads = [
        threading.Thread(target=register, args=(f"cli-invocation-{i}",))
        for i in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    manifest = store.load_manifest()
    assert len(manifest.source_memberships) == 2
    assert {membership.evidence for membership in manifest.source_memberships} == {
        "cli-invocation-0",
        "cli-invocation-1",
    }


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


def test_release_lease_refuses_a_non_durable_target_state(tmp_path: Path) -> None:
    """Finding NEW-1: _replace_run_fields's edge check only fires when the
    target state differs from the run's current one, so
    release_lease(new_state=RUNNING) -- releasing into the run's own
    current state -- sailed through un-checked: the lease was unlinked
    without the run ever entering a durable state, permanently stranding
    it (RUNNING is not acquirable, and re-releasing raises
    NotLeaseHolderError since the lease is already gone). The explicit
    whitelist catches exactly this case, which the edge table cannot."""
    store = _store(tmp_path)
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())

    with pytest.raises(InvalidRunTransitionError):
        store.release_lease(
            run_id=run.run_id,
            new_state=RunState.RUNNING,
            next_action=OperationRef(kind="assemble"),
        )

    # Rejected outright: the run is still running and still holds the lease.
    assert store.load_run(run.run_id).state == RunState.RUNNING
    lease = store.load_lease()
    assert lease is not None
    assert lease.run_id == run.run_id


def test_release_lease_refuses_a_different_live_process_even_with_the_right_run_id(
    tmp_path: Path,
) -> None:
    """Finding NEW-5: release_lease authenticated by run_id alone, so any
    process that merely knew the run_id could release a LIVE run's lease
    out from under the process actively using it. A real (not faked)
    external process is spawned and kept alive to prove this."""
    store = _store(tmp_path)
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    other_process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(5)"]
    )
    try:
        store.acquire_lease(run_id=run.run_id, pid=other_process.pid)

        with pytest.raises(NotLeaseHolderError, match=str(other_process.pid)):
            store.release_lease(
                run_id=run.run_id,
                new_state=RunState.FAILED,
                next_action=OperationRef(kind="assemble"),
            )
    finally:
        other_process.kill()
        other_process.wait()

    lease = store.load_lease()
    assert lease is not None
    assert lease.pid == other_process.pid


def test_release_lease_allows_a_different_process_to_release_a_dead_pids_lease(
    tmp_path: Path,
) -> None:
    """The PID check only refuses a *live* mismatch -- a dead holder's
    lease may still be released by a different (recovering) process."""
    store = _store(tmp_path)
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=_dead_pid())

    updated = store.release_lease(
        run_id=run.run_id,
        new_state=RunState.FAILED,
        next_action=OperationRef(kind="assemble"),
    )

    assert updated.state == RunState.FAILED
    assert store.load_lease() is None


def test_release_lease_raises_a_typed_error_for_a_missing_next_action(
    tmp_path: Path,
) -> None:
    """Finding 11a: M2 requires every durable state to carry a next_action;
    omitting one must raise a typed precondition error naming that rule,
    not a bare pydantic ValidationError from RunRecord's own validator."""
    store = _store(tmp_path)
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())

    with pytest.raises(NextActionPreconditionError):
        store.release_lease(
            run_id=run.run_id, new_state=RunState.FAILED, next_action=None
        )


def test_release_lease_raises_a_typed_error_for_a_next_action_on_completed(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())

    with pytest.raises(NextActionPreconditionError):
        store.release_lease(
            run_id=run.run_id,
            new_state=RunState.COMPLETED,
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


def test_take_over_succeeds_as_an_ordinary_acquire_with_nothing_to_take_over(
    tmp_path: Path,
) -> None:
    """Finding 15 (renamed): a released run leaves no lease behind, so
    take_over=True has nothing to steal -- acquisition proceeds exactly as
    an ordinary acquire and succeeds. (Previously named "...is_refused..."
    while asserting success -- the name lied about the behaviour it proved.)
    """
    store = _store(tmp_path)
    stale = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=stale.run_id, pid=os.getpid())
    store.release_lease(
        run_id=stale.run_id,
        new_state=RunState.FAILED,
        next_action=OperationRef(kind="assemble"),
    )

    challenger = store.create_run(
        next_action=OperationRef(kind="assemble"), takeover_of_run_id=stale.run_id
    )
    lease = store.acquire_lease(
        run_id=challenger.run_id, pid=os.getpid(), take_over=True
    )

    assert lease.run_id == challenger.run_id


def test_acquire_lease_self_heals_a_lease_left_behind_by_a_crashed_release(
    tmp_path: Path,
) -> None:
    """Finding 1: a crash between release_lease's run-record write (to a
    durable state) and its lease unlink must not brick the bundle. The
    next acquire sees a lease whose holder's own record already says a
    non-running state -- a provable crash remnant (M2: durable-state entry
    releases the lease) -- and clears it automatically, no take_over needed.
    """
    root = tmp_path / "bundle"
    setup_store = BundleStore(root)
    setup_store.create_bundle()
    holder = setup_store.create_run(next_action=OperationRef(kind="assemble"))
    setup_store.acquire_lease(run_id=holder.run_id, pid=os.getpid())

    def replace_then_crash(src: str, dst: str) -> None:
        os.replace(src, dst)
        raise OSError("simulated crash after the run-record write, before lease unlink")

    crashing_store = BundleStore(root, replace=replace_then_crash)
    with pytest.raises(OSError, match="simulated crash"):
        crashing_store.release_lease(
            run_id=holder.run_id,
            new_state=RunState.FAILED,
            next_action=OperationRef(kind="assemble"),
        )

    recovery_store = BundleStore(root)
    assert recovery_store.load_run(holder.run_id).state == RunState.FAILED
    assert recovery_store.load_lease() is not None  # the stale remnant

    challenger = recovery_store.create_run(next_action=OperationRef(kind="assemble"))
    lease = recovery_store.acquire_lease(run_id=challenger.run_id, pid=os.getpid())

    assert lease.run_id == challenger.run_id


def test_acquire_lease_crash_after_run_write_leaves_no_stuck_lease(
    tmp_path: Path,
) -> None:
    """Finding 2: acquire_lease writes the run record to 'running' before
    the lease file, so a crash in between leaves no lease at all -- a
    fresh run can immediately acquire -- instead of a lease pointing at a
    run that never actually finished acquiring."""
    root = tmp_path / "bundle"
    setup_store = BundleStore(root)
    setup_store.create_bundle()
    run = setup_store.create_run(next_action=OperationRef(kind="assemble"))

    def replace_then_crash(src: str, dst: str) -> None:
        os.replace(src, dst)
        raise OSError(
            "simulated crash after the run-record write, before the lease write"
        )

    crashing_store = BundleStore(root, replace=replace_then_crash)
    with pytest.raises(OSError, match="simulated crash"):
        crashing_store.acquire_lease(run_id=run.run_id, pid=os.getpid())

    recovery_store = BundleStore(root)
    assert recovery_store.load_lease() is None

    fresh_run = recovery_store.create_run(next_action=OperationRef(kind="assemble"))
    lease = recovery_store.acquire_lease(run_id=fresh_run.run_id, pid=os.getpid())

    assert lease.run_id == fresh_run.run_id


def test_acquire_lease_recovers_the_orphaned_run_itself_with_no_lease(
    tmp_path: Path,
) -> None:
    """Finding NEW-6: the crash window above leaves the *original* run's
    own record stuck showing 'running' with no lease -- 'running' is
    otherwise not an acquirable state, so without this fix that run is
    immortally stranded (never acquirable, never releasable). A 'running'
    run with no lease held is provably not active, so it must remain
    acquirable -- recovering the same run_id, not just a fresh one."""
    root = tmp_path / "bundle"
    setup_store = BundleStore(root)
    setup_store.create_bundle()
    run = setup_store.create_run(next_action=OperationRef(kind="assemble"))

    def replace_then_crash(src: str, dst: str) -> None:
        os.replace(src, dst)
        raise OSError(
            "simulated crash after the run-record write, before the lease write"
        )

    crashing_store = BundleStore(root, replace=replace_then_crash)
    with pytest.raises(OSError, match="simulated crash"):
        crashing_store.acquire_lease(run_id=run.run_id, pid=os.getpid())

    recovery_store = BundleStore(root)
    assert recovery_store.load_run(run.run_id).state == RunState.RUNNING
    assert recovery_store.load_lease() is None

    lease = recovery_store.acquire_lease(run_id=run.run_id, pid=os.getpid())

    assert lease.run_id == run.run_id


def test_acquire_lease_refuses_a_completed_run_without_touching_the_lease(
    tmp_path: Path,
) -> None:
    """Finding 3: run acquirability is validated before the lease file is
    ever touched, so a run in a non-acquirable state (here: completed)
    never leaves a stuck lease behind."""
    store = _store(tmp_path)
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    store.release_lease(run_id=run.run_id, new_state=RunState.COMPLETED)

    with pytest.raises(RunNotAcquirableError):
        store.acquire_lease(run_id=run.run_id, pid=os.getpid())

    assert store.load_lease() is None


def test_acquire_lease_refuses_an_unknown_run_id_without_touching_the_lease(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    with pytest.raises(UnknownRunError):
        store.acquire_lease(run_id=_bogus_id("run"), pid=os.getpid())

    assert store.load_lease() is None


def test_take_over_refused_if_the_run_does_not_declare_a_takeover_target(
    tmp_path: Path,
) -> None:
    """Finding 8: take_over=True must leave an audit trail -- refused if
    the acquiring run's own record doesn't declare takeover_of_run_id."""
    store = _store(tmp_path)
    holder = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=holder.run_id, pid=_dead_pid())

    challenger = store.create_run(next_action=OperationRef(kind="assemble"))

    with pytest.raises(TakeOverRefusedError, match="takeover_of_run_id"):
        store.acquire_lease(run_id=challenger.run_id, pid=os.getpid(), take_over=True)


def test_take_over_refused_if_takeover_of_run_id_names_the_wrong_run(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    holder = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=holder.run_id, pid=_dead_pid())
    unrelated = store.create_run(next_action=OperationRef(kind="assemble"))

    challenger = store.create_run(
        next_action=OperationRef(kind="assemble"), takeover_of_run_id=unrelated.run_id
    )

    with pytest.raises(TakeOverRefusedError, match="does not name"):
        store.acquire_lease(run_id=challenger.run_id, pid=os.getpid(), take_over=True)


def test_create_run_refuses_a_resumes_run_id_that_does_not_exist(
    tmp_path: Path,
) -> None:
    """Finding 8: an audit trail pointing at nothing is worse than none."""
    store = _store(tmp_path)

    with pytest.raises(UnknownRunError):
        store.create_run(
            next_action=OperationRef(kind="assemble"), resumes_run_id=_bogus_id("run")
        )


def test_create_run_refuses_a_takeover_of_run_id_that_does_not_exist(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    with pytest.raises(UnknownRunError):
        store.create_run(
            next_action=OperationRef(kind="assemble"),
            takeover_of_run_id=_bogus_id("run"),
        )


def test_run_state_transition_refuses_an_illegal_edge(tmp_path: Path) -> None:
    """Finding 12: created -> completed (skipping running) is not a legal
    M2 edge, even though RunRecord's own field-level validator accepts the
    record shape in isolation -- the edge table catches what the record
    type alone cannot."""
    store = _store(tmp_path)
    run = store.create_run(next_action=OperationRef(kind="assemble"))

    with pytest.raises(InvalidRunTransitionError):
        store._replace_run_fields(run, state=RunState.COMPLETED)


def test_run_state_transition_refuses_re_running_a_completed_run(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    completed = store.release_lease(run_id=run.run_id, new_state=RunState.COMPLETED)

    with pytest.raises(InvalidRunTransitionError):
        store._replace_run_fields(
            completed,
            state=RunState.RUNNING,
            pid=os.getpid(),
            started_at=completed.created_at,
        )


def test_write_json_atomic_fsyncs_the_parent_directory_after_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 13: durability of the rename itself requires fsyncing the
    directory entry too, not just the file's own contents -- otherwise the
    rename can be lost on crash even though the data was synced. `os.fsync`
    is a real OS durability primitive with no observable return value, so
    this spies on it (still calling through to the real syscall) rather
    than faking file IO outright."""
    store = _store(tmp_path)
    synced_fds: list[int] = []
    real_fsync = os.fsync

    def spy_fsync(fd: int) -> None:
        synced_fds.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy_fsync)

    store.register_source(
        association=SourceAssociation.OPERATOR_ASSERTION, evidence="cli"
    )

    # One fsync for the temp file's own contents, one for the directory entry.
    assert len(synced_fds) >= 2


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
