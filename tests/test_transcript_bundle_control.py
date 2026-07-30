"""Behaviour tests for jake_tools.transcripts.bundle.control -- real
tmp-dir bundles throughout, no mocking of the store itself. CLI-level
wiring (option parsing, --json parity, help text) lives in
test_transcript_bundle_cli.py; this file exercises the orchestration
functions directly.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
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
from jake_tools.transcripts.bundle.control import (
    DEFAULT_BUNDLES_ROOT,
    AmbiguousResumeTargetError,
    BundleStagingError,
    ExecutorFailedError,
    ExecutorOutcome,
    NoExecutorRegisteredError,
    NoResumableRunError,
    NoTakeOverTargetError,
    RunNotResumableError,
    capability_failures,
    create_bundle,
    ingest_source,
    inspect_bundle,
    project_document,
    resume_run,
    status_bundle,
)
from jake_tools.transcripts.bundle.document import TranscriptDocumentV1
from jake_tools.transcripts.bundle.ids import mint_id
from jake_tools.transcripts.bundle.records import (
    NoDocumentYet,
    OperationRef,
    RunRecord,
    RunState,
    SourceAssociation,
)
from jake_tools.transcripts.bundle.registry import CapabilityKey, CapabilityStatus
from jake_tools.transcripts.bundle.store import BundleStore, UnknownSourceError


def _store(tmp_path: Path) -> BundleStore:
    store = BundleStore(tmp_path / "bundle")
    store.create_bundle()
    return store


def _notes_body(artefact_id: str) -> NotesComponentBody:
    return NotesComponentBody(
        notes_kind=NotesKind.PROVIDER_SUMMARY,
        source_artefact_id=artefact_id,
        authored=False,
        sections=(NotesSectionBody(title="T", text="text"),),
    )


def _participant_body(name: str) -> ParticipantSetComponentBody:
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


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def _durable_run(
    store: BundleStore,
    *,
    kind: str = "transcribe",
    state: RunState = RunState.REVIEW_REQUIRED,
) -> RunRecord:
    run = store.create_run(next_action=OperationRef(kind=kind, input_ids=()))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    return store.release_lease(
        run_id=run.run_id, new_state=state, next_action=run.next_action
    )


def _bundle_with_valid_head(store: BundleStore) -> tuple[str, str]:
    """Registers a source, ingests an artefact, adds a valid notes
    component, appends a revision, and moves the head to it. Returns
    (artefact_id, revision_id)."""
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
    return artefact.artefact_id, revision.revision_id


# -- bundle create ------------------------------------------------------------


def test_create_bundle_names_the_directory_after_the_minted_bundle_id(
    tmp_path: Path,
) -> None:
    created = create_bundle(tmp_path / "bundles")

    assert created.store.root == tmp_path / "bundles" / created.manifest.bundle_id
    assert created.store.root.is_dir()
    assert created.store.load_manifest().bundle_id == created.manifest.bundle_id


def test_create_bundle_defaults_to_the_m16_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    created = create_bundle()

    assert created.store.root == DEFAULT_BUNDLES_ROOT / created.manifest.bundle_id
    assert (tmp_path / DEFAULT_BUNDLES_ROOT / created.manifest.bundle_id).is_dir()


def test_create_bundle_leaves_no_staging_directory_behind(tmp_path: Path) -> None:
    root = tmp_path / "bundles"

    create_bundle(root)

    entries = list(root.iterdir())
    assert len(entries) == 1
    assert not entries[0].name.startswith(".")


def test_create_bundle_twice_produces_two_distinct_bundles(tmp_path: Path) -> None:
    root = tmp_path / "bundles"

    first = create_bundle(root)
    second = create_bundle(root)

    assert first.manifest.bundle_id != second.manifest.bundle_id
    assert len(list(root.iterdir())) == 2


def test_create_bundle_removes_the_staging_directory_on_rename_failure(
    tmp_path: Path,
) -> None:
    """MINOR 1 (adversarial review): a rename failure must not leave the
    staged `.bundle-creating-*` directory behind with nothing pointing
    at it -- it is removed, and the failure surfaces as a typed error."""
    root = tmp_path / "bundles"

    def failing_rename(_src: Path, _dst: Path) -> None:
        raise OSError("simulated rename failure")

    with pytest.raises(BundleStagingError, match="simulated rename failure"):
        create_bundle(root, rename=failing_rename)

    assert list(root.iterdir()) == []


# -- source ingest --------------------------------------------------------


def test_ingest_source_registers_a_new_operator_assertion_source_by_default(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    evidence = (
        "transcript source ingest --bundle X --kind notes --producer op /tmp/x.md"
    )

    outcome = ingest_source(
        store,
        source_id=None,
        association=SourceAssociation.OPERATOR_ASSERTION,
        evidence=evidence,
        content=b"hello",
        kind="notes",
        producer="op",
        acquisition_locator="/tmp/x.md",
    )

    assert outcome.already_ingested is False
    assert outcome.membership.association == SourceAssociation.OPERATOR_ASSERTION
    assert outcome.membership.evidence == evidence
    assert outcome.artefact.sha256 == hashlib.sha256(b"hello").hexdigest()
    assert store.load_manifest().source_memberships == (outcome.membership,)


def test_ingest_source_reusing_source_id_is_idempotent_on_repeat(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    first = ingest_source(
        store,
        source_id=None,
        association=SourceAssociation.OPERATOR_ASSERTION,
        evidence="cli",
        content=b"same bytes",
        kind="notes",
        producer="op",
        acquisition_locator="/tmp/x.md",
    )

    second = ingest_source(
        store,
        source_id=first.membership.source_id,
        association=SourceAssociation.OPERATOR_ASSERTION,
        evidence="unused -- reusing an existing source",
        content=b"same bytes",
        kind="notes",
        producer="op",
        acquisition_locator="/tmp/x.md",
    )

    assert second.already_ingested is True
    assert second.artefact.artefact_id == first.artefact.artefact_id
    assert second.membership.source_id == first.membership.source_id
    # Only one membership was ever registered -- reuse never mints another.
    assert len(store.load_manifest().source_memberships) == 1


def test_ingest_source_a_new_source_with_the_same_bytes_is_not_idempotent(
    tmp_path: Path,
) -> None:
    """M1 §3.2: the same bytes from a *different* source is a distinct
    acquisition, not a dedup hit -- ingest_source's own idempotency
    signal must agree."""
    store = _store(tmp_path)
    first = ingest_source(
        store,
        source_id=None,
        association=SourceAssociation.OPERATOR_ASSERTION,
        evidence="first source",
        content=b"same bytes",
        kind="notes",
        producer="op",
        acquisition_locator="/tmp/x.md",
    )

    second = ingest_source(
        store,
        source_id=None,
        association=SourceAssociation.OPERATOR_ASSERTION,
        evidence="second source",
        content=b"same bytes",
        kind="notes",
        producer="op",
        acquisition_locator="/tmp/x.md",
    )

    assert second.already_ingested is False
    assert second.artefact.artefact_id != first.artefact.artefact_id
    assert second.membership.source_id != first.membership.source_id


def test_ingest_source_rejects_an_unknown_source_id(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(UnknownSourceError):
        ingest_source(
            store,
            source_id="source_00000000-0000-7000-8000-000000000000",
            association=SourceAssociation.OPERATOR_ASSERTION,
            evidence="cli",
            content=b"x",
            kind="notes",
            producer="op",
            acquisition_locator="/tmp/x",
        )


# -- inspect ------------------------------------------------------------


def test_inspect_bundle_reports_no_document_yet_with_candidates(tmp_path: Path) -> None:
    store = _store(tmp_path)
    outcome = ingest_source(
        store,
        source_id=None,
        association=SourceAssociation.OPERATOR_ASSERTION,
        evidence="cli",
        content=b"x",
        kind="notes",
        producer="op",
        acquisition_locator="/tmp/x",
    )

    overview = inspect_bundle(store)

    assert isinstance(overview.document, NoDocumentYet)
    assert overview.artefact_ids == (outcome.artefact.artefact_id,)
    assert overview.revision_count == 0
    assert overview.components_by_kind == {}
    assert overview.sources == (outcome.membership,)
    assert overview.runs == ()


def test_inspect_bundle_reports_the_head_and_components_after_a_revision(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    artefact_id, revision_id = _bundle_with_valid_head(store)

    overview = inspect_bundle(store)

    assert isinstance(overview.document, TranscriptDocumentV1)
    assert overview.document.revision_id == revision_id
    assert overview.revision_count == 1
    assert overview.artefact_ids == (artefact_id,)
    assert overview.components_by_kind == {"notes": 1}
    assert len(overview.runs) == 1


# -- validate ------------------------------------------------------------


def test_project_document_no_document_yet_is_a_clean_state_not_an_error(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    document = project_document(store, revision_id=None)

    assert isinstance(document, NoDocumentYet)
    assert document.candidate_artefact_ids == ()


def test_capability_failures_is_empty_for_a_valid_head(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _bundle_with_valid_head(store)

    document = project_document(store, revision_id=None)

    assert isinstance(document, TranscriptDocumentV1)
    assert capability_failures(document) == ()


def test_capability_failures_reports_a_genuinely_broken_member(tmp_path: Path) -> None:
    """Mirrors test_transcript_bundle_document.py's own duplicate-section-id
    scenario, but *without* ever calling update_head -- validate's
    --revision path is exactly for inspecting a revision that never
    became (and, being broken, never could become) the head."""
    store = _store(tmp_path)
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

    shared_id = mint_id("seg")
    broken = NotesComponent(
        notes_kind=NotesKind.PROVIDER_SUMMARY,
        source_artefact_id=artefact.artefact_id,
        authored=False,
        sections=(
            NotesSection(section_id=shared_id, title="A", text="a"),
            NotesSection(section_id=shared_id, title="B", text="b"),
        ),
        component_id=mint_id("component"),
        content_hash=store_module._component_content_hash(
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
        created_at=artefact.created_at,
    )
    (store.root / "components" / f"{broken.component_id}.json").write_text(
        broken.model_dump_json(), encoding="utf-8"
    )
    revision = store.append_revision(
        operation=OperationRef(kind="assemble"),
        artefact_ids=(artefact.artefact_id,),
        component_ids=(broken.component_id,),
    )

    document = project_document(store, revision_id=revision.revision_id)

    assert isinstance(document, TranscriptDocumentV1)
    failures = capability_failures(document)
    assert len(failures) == 1
    assert failures[0].capability == CapabilityKey.NOTES_PROVIDER
    assert failures[0].member_id == broken.component_id
    # This revision never went through update_head -- it is not the head.
    assert store.load_manifest().head_revision_id is None


def test_capability_failures_reports_a_one_cardinality_top_level_failure(
    tmp_path: Path,
) -> None:
    """BLOCKER 2 (adversarial review): `participants.declared` is a
    one-cardinality key -- its registry validator returns a top-level
    `failed` status with an EMPTY `members` tuple when more than one
    candidate participant-set component is present. A scan limited to
    `record.members` never sees this; `capability_failures` must catch
    it via the top-level status too."""
    store = _store(tmp_path)
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
    first = store.add_component(_participant_body("Jake"))
    second = store.add_component(_participant_body("Not Jake"))
    revision = store.append_revision(
        operation=OperationRef(kind="assemble"),
        artefact_ids=(artefact.artefact_id,),
        component_ids=(first.component_id, second.component_id),
    )

    document = project_document(store, revision_id=revision.revision_id)

    assert isinstance(document, TranscriptDocumentV1)
    record = document.capabilities[CapabilityKey.PARTICIPANTS_DECLARED]
    assert record.status == CapabilityStatus.FAILED
    assert record.members == ()  # nothing for a members-only scan to find

    failures = capability_failures(document)
    assert len(failures) == 1
    assert failures[0].capability == CapabilityKey.PARTICIPANTS_DECLARED
    assert failures[0].member_id is None
    assert failures[0].detail != ""
    # Never went through update_head -- a one-cardinality failure like
    # this is blocked from ever becoming head in the first place.
    assert store.load_manifest().head_revision_id is None


# -- status ------------------------------------------------------------


def test_status_bundle_reports_no_lease_and_no_runs_for_a_fresh_bundle(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    status = status_bundle(store)

    assert status.lease is None
    assert status.lease_holder_alive is None
    assert status.runs == ()


def test_status_bundle_reports_a_durable_run(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _durable_run(store)

    status = status_bundle(store)

    assert status.lease is None
    assert len(status.runs) == 1
    assert status.runs[0].state == RunState.REVIEW_REQUIRED
    assert status.runs[0].next_action is not None
    assert status.runs[0].next_action.kind == "transcribe"


def test_status_bundle_reports_a_live_lease_holder(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run = store.create_run(next_action=OperationRef(kind="transcribe"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())

    status = status_bundle(store)

    assert status.lease is not None
    assert status.lease.run_id == run.run_id
    assert status.lease_holder_alive is True


def test_status_bundle_reports_a_dead_lease_holder(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run = store.create_run(next_action=OperationRef(kind="transcribe"))
    dead_pid = _dead_pid()
    store.acquire_lease(run_id=run.run_id, pid=dead_pid)

    status = status_bundle(store)

    assert status.lease is not None
    assert status.lease.pid == dead_pid
    assert status.lease_holder_alive is False


# -- resume ------------------------------------------------------------


def test_resume_run_raises_when_no_run_is_resumable(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(NoResumableRunError):
        resume_run(store, run_id=None, executors={})


def test_resume_run_raises_when_multiple_runs_are_resumable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _durable_run(store, kind="a")
    _durable_run(store, kind="b")

    with pytest.raises(AmbiguousResumeTargetError):
        resume_run(store, run_id=None, executors={})


def test_resume_run_auto_selects_the_single_resumable_run(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run = _durable_run(store, kind="only-one")

    with pytest.raises(NoExecutorRegisteredError, match="only-one"):
        resume_run(store, run_id=None, executors={})

    # Auto-selection targeted (and safely aborted out of) exactly this run.
    after = store.load_run(run.run_id)
    assert after.state == RunState.REVIEW_REQUIRED


def test_resume_run_rejects_a_run_that_is_not_in_a_durable_state(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    run = store.create_run(next_action=OperationRef(kind="transcribe"))

    with pytest.raises(RunNotResumableError):
        resume_run(store, run_id=run.run_id, executors={})


def test_resume_run_dispatches_to_the_registered_executor_and_moves_the_head(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    membership = store.register_source(
        association=SourceAssociation.OPERATOR_ASSERTION, evidence="cli"
    )
    artefact = store.ingest_artefact(
        source_id=membership.source_id,
        content=b"notes",
        kind="notes",
        producer="test",
        acquisition_locator="/tmp/n.md",
    )
    run = _durable_run(store, kind="transcribe")

    def fake_executor(
        inner_store: BundleStore, inner_run: RunRecord
    ) -> ExecutorOutcome:
        component = inner_store.add_component(_notes_body(artefact.artefact_id))
        revision = inner_store.append_revision(
            operation=inner_run.next_action or OperationRef(kind="transcribe"),
            artefact_ids=(artefact.artefact_id,),
            component_ids=(component.component_id,),
        )
        return ExecutorOutcome(
            state=RunState.COMPLETED, revision_id=revision.revision_id
        )

    result = resume_run(
        store, run_id=run.run_id, executors={"transcribe": fake_executor}
    )

    assert result.state == RunState.COMPLETED
    assert result.next_action is None
    head_revision_id = store.load_manifest().head_revision_id
    assert head_revision_id is not None
    assert store.load_revision(head_revision_id).operation.kind == "transcribe"
    assert store.load_lease() is None


def test_resume_run_durable_outcome_releases_the_lease_with_its_own_next_action(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    run = _durable_run(store, kind="review")
    new_next_action = OperationRef(kind="review", rationale="needs a human")

    def fake_executor(
        inner_store: BundleStore, inner_run: RunRecord
    ) -> ExecutorOutcome:
        return ExecutorOutcome(
            state=RunState.REVIEW_REQUIRED, next_action=new_next_action
        )

    result = resume_run(store, run_id=run.run_id, executors={"review": fake_executor})

    assert result.state == RunState.REVIEW_REQUIRED
    assert result.next_action == new_next_action
    assert store.load_lease() is None


def test_resume_run_aborts_safely_for_an_unregistered_operation_kind(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    run = _durable_run(store, kind="some-unimplemented-op")

    with pytest.raises(NoExecutorRegisteredError, match="some-unimplemented-op"):
        resume_run(store, run_id=run.run_id, executors={})

    after = store.load_run(run.run_id)
    assert after.state == RunState.REVIEW_REQUIRED
    assert after.next_action is not None
    assert after.next_action.kind == "some-unimplemented-op"
    assert store.load_lease() is None
    assert store.load_manifest().head_revision_id is None


def test_resume_run_executor_exception_moves_to_failed_and_retry_succeeds(
    tmp_path: Path,
) -> None:
    """BLOCKER 1 (adversarial review): an executor that raises must not
    leave the run stuck `running` with the lease held (permanently
    unresumable -- RunNotResumableError forever). It moves to `failed`
    (durable, resumable) with the exception recorded and the *same*
    next_action preserved, so a second resume retries the identical
    recorded action and can succeed."""
    store = _store(tmp_path)
    membership = store.register_source(
        association=SourceAssociation.OPERATOR_ASSERTION, evidence="cli"
    )
    artefact = store.ingest_artefact(
        source_id=membership.source_id,
        content=b"notes",
        kind="notes",
        producer="test",
        acquisition_locator="/tmp/n.md",
    )
    run = _durable_run(store, kind="transcribe")

    def raising_executor(
        _inner_store: BundleStore, _inner_run: RunRecord
    ) -> ExecutorOutcome:
        raise RuntimeError("boom")

    with pytest.raises(ExecutorFailedError, match="boom"):
        resume_run(store, run_id=run.run_id, executors={"transcribe": raising_executor})

    crashed = store.load_run(run.run_id)
    assert crashed.state == RunState.FAILED
    assert crashed.next_action is not None
    assert crashed.next_action.kind == "transcribe"
    assert crashed.next_action.input_ids == ()
    assert "boom" in crashed.next_action.rationale
    assert "RuntimeError" in crashed.next_action.rationale
    assert store.load_lease() is None
    assert store.load_manifest().head_revision_id is None

    def succeeding_executor(
        inner_store: BundleStore, inner_run: RunRecord
    ) -> ExecutorOutcome:
        component = inner_store.add_component(_notes_body(artefact.artefact_id))
        revision = inner_store.append_revision(
            operation=inner_run.next_action or OperationRef(kind="transcribe"),
            artefact_ids=(artefact.artefact_id,),
            component_ids=(component.component_id,),
        )
        return ExecutorOutcome(
            state=RunState.COMPLETED, revision_id=revision.revision_id
        )

    result = resume_run(
        store, run_id=run.run_id, executors={"transcribe": succeeding_executor}
    )

    assert result.state == RunState.COMPLETED
    assert store.load_manifest().head_revision_id is not None
    assert store.load_lease() is None


def test_resume_run_take_over_recovers_a_crashed_run(tmp_path: Path) -> None:
    """BLOCKER 1's --take-over half: a lease held by a dead PID on a run
    stuck `running` (its process was killed mid-dispatch) is the true
    crash case M2 reserves --take-over for. It creates a new run
    (`takeover_of_run_id` naming the stale one), never rewrites the
    stale run's own record, and dispatches the stale next_action."""
    store = _store(tmp_path)
    membership = store.register_source(
        association=SourceAssociation.OPERATOR_ASSERTION, evidence="cli"
    )
    artefact = store.ingest_artefact(
        source_id=membership.source_id,
        content=b"notes",
        kind="notes",
        producer="test",
        acquisition_locator="/tmp/n.md",
    )
    crashed = store.create_run(
        next_action=OperationRef(kind="transcribe", input_ids=(artefact.artefact_id,))
    )
    store.acquire_lease(run_id=crashed.run_id, pid=_dead_pid())

    def fake_executor(
        inner_store: BundleStore, inner_run: RunRecord
    ) -> ExecutorOutcome:
        component = inner_store.add_component(_notes_body(artefact.artefact_id))
        revision = inner_store.append_revision(
            operation=inner_run.next_action or OperationRef(kind="transcribe"),
            artefact_ids=(artefact.artefact_id,),
            component_ids=(component.component_id,),
        )
        return ExecutorOutcome(
            state=RunState.COMPLETED, revision_id=revision.revision_id
        )

    result = resume_run(
        store, run_id=None, executors={"transcribe": fake_executor}, take_over=True
    )

    assert result.state == RunState.COMPLETED
    assert result.takeover_of_run_id == crashed.run_id
    assert result.run_id != crashed.run_id
    assert store.load_manifest().head_revision_id is not None
    assert store.load_lease() is None
    # M2: "a run never edits another run's record" -- the stale run's own
    # record stays exactly as the crash left it.
    stale_after = store.load_run(crashed.run_id)
    assert stale_after.state == RunState.RUNNING


def test_resume_run_without_take_over_names_the_flag_for_a_crashed_run(
    tmp_path: Path,
) -> None:
    """Without --take-over, a run stuck `running` under a dead-PID lease
    gets the existing diagnosable refusal -- and it names --take-over,
    so an operator staring at `status` output knows the recovery path."""
    store = _store(tmp_path)
    run = store.create_run(next_action=OperationRef(kind="transcribe"))
    store.acquire_lease(run_id=run.run_id, pid=_dead_pid())

    with pytest.raises(RunNotResumableError, match="--take-over"):
        resume_run(store, run_id=run.run_id, executors={})


def test_resume_run_take_over_without_a_stale_lease_is_a_clean_error(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)

    with pytest.raises(NoTakeOverTargetError):
        resume_run(store, run_id=None, executors={}, take_over=True)
