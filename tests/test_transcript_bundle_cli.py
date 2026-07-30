"""CLI-level behaviour tests for the bundle control-plane commands
(`transcript bundle create`, `source ingest`, `inspect`, `validate`,
`status`, `resume`). Uses CliRunner with obj=AppContext(...) per
AGENTS.md; every bundle is a real tmp-dir bundle -- no mocking of
BundleStore. Orchestration-level edge cases live in
test_transcript_bundle_control.py; this file exercises Click wiring,
--json/human parity, and exit codes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

import jake_tools.transcripts.bundle.store as store_module
from jake_tools.cli import main
from jake_tools.cli.context import AppContext
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
from jake_tools.transcripts.bundle.control import ExecutorOutcome
from jake_tools.transcripts.bundle.ids import mint_id
from jake_tools.transcripts.bundle.records import (
    OperationRef,
    RunState,
    SourceAssociation,
)
from jake_tools.transcripts.bundle.store import BundleStore


def _invoke(*args: str, obj: AppContext | None = None):
    return CliRunner().invoke(main, ["transcript", *args], obj=obj)


def _create_bundle(tmp_path: Path) -> Path:
    result = _invoke("bundle", "create", "--root", str(tmp_path), "--json")
    assert result.exit_code == 0, result.output
    return Path(json.loads(result.output)["root"])


def _write_file(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


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


def _valid_head(store: BundleStore) -> tuple[str, str]:
    """Registers a source, ingests an artefact, adds a valid notes
    component, appends a revision, and moves the head to it. Returns
    (artefact_id, revision_id)."""
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


def _broken_head_revision(store: BundleStore) -> tuple[str, str]:
    """An assembly revision carrying a notes component with two sections
    sharing one section_id -- a genuine `notes.provider` member failure
    (M4) -- appended but never made head (a broken revision could never
    pass update_head's own capability gate). Returns
    (broken_component_id, revision_id). Mirrors the duplicate-section-id
    scenario in test_transcript_bundle_document.py."""
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
    return broken.component_id, revision.revision_id


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


# -- help / operator surface -----------------------------------------------


def test_transcript_help_lists_the_new_bundle_control_plane_commands() -> None:
    result = _invoke("--help")

    assert result.exit_code == 0
    for command in ("bundle", "source", "inspect", "validate", "status", "resume"):
        assert f"\n  {command}" in result.output


def test_bundle_create_help_states_the_layout_invariant() -> None:
    result = _invoke("bundle", "create", "--help")

    assert result.exit_code == 0
    assert "--root" in result.output
    assert "--json" in result.output


def test_source_ingest_help_states_the_m3_invariant() -> None:
    result = _invoke("source", "ingest", "--help")

    assert result.exit_code == 0
    for option in ("--bundle", "--kind", "--producer", "--source", "--provider-id"):
        assert option in result.output
    assert "operator-assertion" in result.output


def test_resume_help_documents_take_over() -> None:
    result = _invoke("resume", "--help")

    assert result.exit_code == 0
    assert "--take-over" in result.output
    assert "--run" in result.output


# -- create -> ingest -> inspect round trip --------------------------------


def test_create_ingest_inspect_round_trip_including_idempotent_reingest(
    tmp_path: Path,
) -> None:
    bundle_dir = _create_bundle(tmp_path)
    input_file = _write_file(tmp_path, "note.md", "hello world")

    first = _invoke(
        "source",
        "ingest",
        "--bundle",
        str(bundle_dir),
        "--kind",
        "obsidian-note",
        "--producer",
        "operator",
        str(input_file),
    )
    assert first.exit_code == 0, first.output
    assert "source_id:" in first.output
    assert "artefact_id:" in first.output
    source_id = first.output.splitlines()[0].split(": ", 1)[1]

    # Re-running against the *same* source with identical bytes is a clean,
    # honestly-reported no-op -- not a silent duplicate, not an error.
    second = _invoke(
        "source",
        "ingest",
        "--bundle",
        str(bundle_dir),
        "--kind",
        "obsidian-note",
        "--producer",
        "operator",
        "--source",
        source_id,
        str(input_file),
    )
    assert second.exit_code == 0, second.output
    assert "already ingested, artefact_id " in second.output

    inspected = _invoke("inspect", "--bundle", str(bundle_dir))
    assert inspected.exit_code == 0, inspected.output
    assert "sources (1):" in inspected.output
    assert "artefacts (1):" in inspected.output
    assert "head: no document yet (1 candidate artefact(s))" in inspected.output


def test_source_ingest_json_carries_the_already_ingested_flag(tmp_path: Path) -> None:
    bundle_dir = _create_bundle(tmp_path)
    input_file = _write_file(tmp_path, "note.md", "same bytes")
    common = [
        "source",
        "ingest",
        "--bundle",
        str(bundle_dir),
        "--kind",
        "obsidian-note",
        "--producer",
        "operator",
        "--json",
    ]

    first = json.loads(_invoke(*common, str(input_file)).output)
    assert first["already_ingested"] is False

    second = json.loads(
        _invoke(*common, "--source", first["source_id"], str(input_file)).output
    )
    assert second["already_ingested"] is True
    assert second["artefact_id"] == first["artefact_id"]


def test_source_ingest_rejects_more_than_one_association_flag(tmp_path: Path) -> None:
    bundle_dir = _create_bundle(tmp_path)
    input_file = _write_file(tmp_path, "note.md", "x")

    result = _invoke(
        "source",
        "ingest",
        "--bundle",
        str(bundle_dir),
        "--kind",
        "k",
        "--producer",
        "p",
        "--provider-id",
        "evt-1",
        "--note-embed",
        "obsidian://embed",
        str(input_file),
    )

    assert result.exit_code == 2
    assert "Use at most one of" in result.output


def test_source_ingest_provider_id_records_that_association(tmp_path: Path) -> None:
    bundle_dir = _create_bundle(tmp_path)
    input_file = _write_file(tmp_path, "note.md", "x")

    result = _invoke(
        "source",
        "ingest",
        "--bundle",
        str(bundle_dir),
        "--kind",
        "teams-transcript",
        "--producer",
        "teams",
        "--provider-id",
        "evt-123",
        "--json",
        str(input_file),
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["association"] == "provider-id"
    assert payload["evidence"] == "evt-123"


def test_source_ingest_default_evidence_is_the_genuine_invocation(
    tmp_path: Path,
) -> None:
    """MINOR 2 (adversarial review): the default operator-assertion
    evidence must be the genuine invocation -- every flag actually
    supplied -- not a hand-picked reconstruction that silently drops
    ones like --bundle/--json."""
    bundle_dir = _create_bundle(tmp_path)
    input_file = _write_file(tmp_path, "note.md", "hello")

    result = _invoke(
        "source",
        "ingest",
        "--bundle",
        str(bundle_dir),
        "--kind",
        "obsidian-note",
        "--producer",
        "operator",
        "--json",
        str(input_file),
    )

    assert result.exit_code == 0, result.output
    evidence = json.loads(result.output)["evidence"]
    assert "--bundle" in evidence
    assert str(bundle_dir) in evidence
    assert "--json" in evidence
    assert "--kind obsidian-note" in evidence
    assert "--producer operator" in evidence
    assert str(input_file) in evidence


def test_inspect_reports_typed_store_errors_as_a_clean_one_liner(
    tmp_path: Path,
) -> None:
    result = _invoke("inspect", "--bundle", str(tmp_path / "does-not-exist"))

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "not a bundle" in result.output


def test_inspect_reports_revision_count_and_heads_once_a_head_exists(
    tmp_path: Path,
) -> None:
    bundle_dir = _create_bundle(tmp_path)
    store = BundleStore(bundle_dir)
    _artefact_id, revision_id = _valid_head(store)

    human = _invoke("inspect", "--bundle", str(bundle_dir))
    machine = _invoke("inspect", "--bundle", str(bundle_dir), "--json")

    assert human.exit_code == 0
    assert f"revisions: 1 (heads: {revision_id})" in human.output
    payload = json.loads(machine.output)
    assert payload["revision_count"] == 1
    assert payload["heads"] == [revision_id]


# -- validate ------------------------------------------------------------


def test_validate_on_no_document_yet_is_a_clean_report_not_an_error(
    tmp_path: Path,
) -> None:
    bundle_dir = _create_bundle(tmp_path)

    result = _invoke("validate", "--bundle", str(bundle_dir))

    assert result.exit_code == 0
    assert "head: no document yet" in result.output


def test_validate_on_a_valid_head_exits_zero_with_no_failed_capabilities(
    tmp_path: Path,
) -> None:
    bundle_dir = _create_bundle(tmp_path)
    store = BundleStore(bundle_dir)
    _valid_head(store)

    result = _invoke("validate", "--bundle", str(bundle_dir), "--json")

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["capability_failures"] == []
    assert payload["capabilities"]["notes.provider"]["status"] == "present-validated"


def test_validate_exits_1_on_a_failed_member(tmp_path: Path) -> None:
    bundle_dir = _create_bundle(tmp_path)
    store = BundleStore(bundle_dir)
    broken_component_id, revision_id = _broken_head_revision(store)

    result = _invoke("validate", "--bundle", str(bundle_dir), "--revision", revision_id)

    assert result.exit_code == 1
    assert "failed capabilities (1):" in result.output
    assert broken_component_id in result.output


def test_validate_exits_1_on_a_one_cardinality_top_level_failure(
    tmp_path: Path,
) -> None:
    """BLOCKER 2 (adversarial review), end to end: two participant-set
    components on one revision make `participants.declared` (a
    one-cardinality key) fail at the top level, with no members at all.
    `validate` must exit 1 in both human and --json modes -- not exit 0
    while printing "failed" in the capability table."""
    bundle_dir = _create_bundle(tmp_path)
    store = BundleStore(bundle_dir)
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

    human = _invoke(
        "validate", "--bundle", str(bundle_dir), "--revision", revision.revision_id
    )
    machine = _invoke(
        "validate",
        "--bundle",
        str(bundle_dir),
        "--revision",
        revision.revision_id,
        "--json",
    )

    assert human.exit_code == 1
    assert "participants.declared" in human.output
    assert machine.exit_code == 1
    payload = json.loads(machine.output)
    failures = payload["capability_failures"]
    assert len(failures) == 1
    assert failures[0]["capability"] == "participants.declared"
    assert failures[0]["member_id"] is None


# -- status ------------------------------------------------------------


def test_status_shows_a_durable_run_and_a_live_lease_holder(tmp_path: Path) -> None:
    bundle_dir = _create_bundle(tmp_path)
    store = BundleStore(bundle_dir)
    run = store.create_run(next_action=OperationRef(kind="transcribe"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())

    result = _invoke("status", "--bundle", str(bundle_dir), "--json")

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["lease"]["run_id"] == run.run_id
    assert payload["lease"]["alive"] is True
    assert len(payload["runs"]) == 1
    assert payload["runs"][0]["state"] == "running"


def test_status_shows_a_dead_lease_holder(tmp_path: Path) -> None:
    bundle_dir = _create_bundle(tmp_path)
    store = BundleStore(bundle_dir)
    run = store.create_run(next_action=OperationRef(kind="transcribe"))
    store.acquire_lease(run_id=run.run_id, pid=_dead_pid())

    result = _invoke("status", "--bundle", str(bundle_dir))

    assert result.exit_code == 0
    assert "(dead)" in result.output


# -- resume ------------------------------------------------------------


def test_resume_dispatches_to_an_injected_executor_and_the_head_moves(
    tmp_path: Path,
) -> None:
    bundle_dir = _create_bundle(tmp_path)
    store = BundleStore(bundle_dir)
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
    run = store.create_run(next_action=OperationRef(kind="transcribe"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    store.release_lease(
        run_id=run.run_id,
        new_state=RunState.REVIEW_REQUIRED,
        next_action=run.next_action,
    )

    def fake_executor(inner_store: BundleStore, inner_run) -> ExecutorOutcome:
        component = inner_store.add_component(_notes_body(artefact.artefact_id))
        revision = inner_store.append_revision(
            operation=inner_run.next_action,
            artefact_ids=(artefact.artefact_id,),
            component_ids=(component.component_id,),
        )
        return ExecutorOutcome(
            state=RunState.COMPLETED, revision_id=revision.revision_id
        )

    app = AppContext(bundle_executors={"transcribe": fake_executor})
    result = _invoke("resume", "--bundle", str(bundle_dir), "--json", obj=app)

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["state"] == "completed"
    assert payload["head_revision_id"] is not None
    assert store.load_manifest().head_revision_id == payload["head_revision_id"]
    assert store.load_lease() is None


def test_resume_unregistered_kind_is_a_clean_error_and_the_run_is_abortable(
    tmp_path: Path,
) -> None:
    bundle_dir = _create_bundle(tmp_path)
    store = BundleStore(bundle_dir)
    run = store.create_run(next_action=OperationRef(kind="some-unimplemented-op"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    store.release_lease(
        run_id=run.run_id, new_state=RunState.FAILED, next_action=run.next_action
    )

    result = _invoke("resume", "--bundle", str(bundle_dir))

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert (
        "no executor registered for operation kind 'some-unimplemented-op'"
        in result.output
    )
    assert "transforms arrive in later slices" in result.output

    after = store.load_run(run.run_id)
    assert after.state == RunState.FAILED
    assert store.load_lease() is None

    # Safely abortable: resuming again (still no executor) behaves
    # identically -- nothing about the run was left half-mutated.
    again = _invoke("resume", "--bundle", str(bundle_dir))
    assert again.exit_code == 1
    assert again.output == result.output


def test_resume_with_no_resumable_run_is_a_clean_error(tmp_path: Path) -> None:
    bundle_dir = _create_bundle(tmp_path)

    result = _invoke("resume", "--bundle", str(bundle_dir))

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "no run in this bundle is in a durable state" in result.output


def test_resume_executor_exception_moves_to_failed_and_retry_succeeds(
    tmp_path: Path,
) -> None:
    """BLOCKER 1 (adversarial review), end to end: a crashing executor
    must not leave the run stuck `running` with the lease held. The
    --json error is a JSON envelope (MAJOR), the run is `failed`
    (durable, resumable), and a second resume with a working executor
    for the same kind succeeds -- proving the same recorded next_action
    survives the crash."""
    bundle_dir = _create_bundle(tmp_path)
    store = BundleStore(bundle_dir)
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
    run = store.create_run(next_action=OperationRef(kind="transcribe"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    store.release_lease(
        run_id=run.run_id,
        new_state=RunState.REVIEW_REQUIRED,
        next_action=run.next_action,
    )

    def raising_executor(_inner_store: BundleStore, _inner_run) -> ExecutorOutcome:
        raise RuntimeError("boom")

    crashing_app = AppContext(bundle_executors={"transcribe": raising_executor})
    crash_result = _invoke(
        "resume", "--bundle", str(bundle_dir), "--json", obj=crashing_app
    )

    assert crash_result.exit_code == 1
    payload = json.loads(crash_result.output)
    assert payload["error"]["type"] == "ExecutorFailedError"
    assert "boom" in payload["error"]["message"]

    after_crash = store.load_run(run.run_id)
    assert after_crash.state == RunState.FAILED
    assert store.load_lease() is None

    def succeeding_executor(inner_store: BundleStore, inner_run) -> ExecutorOutcome:
        component = inner_store.add_component(_notes_body(artefact.artefact_id))
        revision = inner_store.append_revision(
            operation=inner_run.next_action,
            artefact_ids=(artefact.artefact_id,),
            component_ids=(component.component_id,),
        )
        return ExecutorOutcome(
            state=RunState.COMPLETED, revision_id=revision.revision_id
        )

    retry_app = AppContext(bundle_executors={"transcribe": succeeding_executor})
    retry_result = _invoke(
        "resume", "--bundle", str(bundle_dir), "--json", obj=retry_app
    )

    assert retry_result.exit_code == 0, retry_result.output
    assert json.loads(retry_result.output)["state"] == "completed"


def test_resume_take_over_recovers_a_crashed_run(tmp_path: Path) -> None:
    """BLOCKER 1's --take-over half, end to end: a subprocess that
    acquires the lease and exits hard simulates the true crash case (a
    dead PID still holding the lease on a `running` run). Without
    --take-over, resume gives the existing diagnosable refusal naming
    the flag; with it, the crashed run is recovered via a fresh run
    record."""
    bundle_dir = _create_bundle(tmp_path)
    store = BundleStore(bundle_dir)
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

    def fake_executor(inner_store: BundleStore, inner_run) -> ExecutorOutcome:
        component = inner_store.add_component(_notes_body(artefact.artefact_id))
        revision = inner_store.append_revision(
            operation=inner_run.next_action,
            artefact_ids=(artefact.artefact_id,),
            component_ids=(component.component_id,),
        )
        return ExecutorOutcome(
            state=RunState.COMPLETED, revision_id=revision.revision_id
        )

    app = AppContext(bundle_executors={"transcribe": fake_executor})

    without_flag = _invoke(
        "resume", "--bundle", str(bundle_dir), "--run", crashed.run_id, obj=app
    )
    assert without_flag.exit_code == 1
    assert "--take-over" in without_flag.output

    with_flag = _invoke(
        "resume", "--bundle", str(bundle_dir), "--take-over", "--json", obj=app
    )

    assert with_flag.exit_code == 0, with_flag.output
    payload = json.loads(with_flag.output)
    assert payload["state"] == "completed"
    assert payload["head_revision_id"] is not None
    assert store.load_lease() is None


def test_inspect_json_and_human_output_carry_the_same_facts(tmp_path: Path) -> None:
    bundle_dir = _create_bundle(tmp_path)
    input_file = _write_file(tmp_path, "note.md", "hello")
    _invoke(
        "source",
        "ingest",
        "--bundle",
        str(bundle_dir),
        "--kind",
        "obsidian-note",
        "--producer",
        "operator",
        str(input_file),
    )

    human = _invoke("inspect", "--bundle", str(bundle_dir))
    machine = _invoke("inspect", "--bundle", str(bundle_dir), "--json")

    assert human.exit_code == 0
    assert machine.exit_code == 0
    payload = json.loads(machine.output)
    assert payload["bundle_id"] in human.output
    assert len(payload["sources"]) == 1
    assert payload["sources"][0]["source_id"] in human.output
    assert len(payload["artefact_ids"]) == 1
    assert payload["artefact_ids"][0] in human.output


# -- MAJOR: --json error envelopes on every command's typed-error path ------


def test_bundle_create_json_error_is_a_json_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_rename(_src: Path, _dst: Path) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "rename", failing_rename)

    result = _invoke("bundle", "create", "--root", str(tmp_path), "--json")

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    payload = json.loads(result.output)
    assert payload["error"]["type"] == "BundleStagingError"
    assert "simulated rename failure" in payload["error"]["message"]


def test_source_ingest_json_error_is_a_json_envelope(tmp_path: Path) -> None:
    input_file = _write_file(tmp_path, "note.md", "x")

    result = _invoke(
        "source",
        "ingest",
        "--bundle",
        str(tmp_path / "does-not-exist"),
        "--kind",
        "k",
        "--producer",
        "p",
        "--json",
        str(input_file),
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["type"] == "NotABundleError"
    assert "not a bundle" in payload["error"]["message"]


def test_inspect_json_error_is_a_json_envelope(tmp_path: Path) -> None:
    result = _invoke("inspect", "--bundle", str(tmp_path / "does-not-exist"), "--json")

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["type"] == "NotABundleError"


def test_validate_json_error_is_a_json_envelope(tmp_path: Path) -> None:
    result = _invoke("validate", "--bundle", str(tmp_path / "does-not-exist"), "--json")

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["type"] == "NotABundleError"


def test_status_json_error_is_a_json_envelope(tmp_path: Path) -> None:
    result = _invoke("status", "--bundle", str(tmp_path / "does-not-exist"), "--json")

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["type"] == "NotABundleError"


def test_resume_json_error_is_a_json_envelope(tmp_path: Path) -> None:
    result = _invoke("resume", "--bundle", str(tmp_path / "does-not-exist"), "--json")

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["type"] == "NotABundleError"
