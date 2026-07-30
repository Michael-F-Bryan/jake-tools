from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from jake_tools.transcripts.bundle.ids import mint_id
from jake_tools.transcripts.bundle.records import OperationRef, RunRecord, RunState

NOW = datetime.now(UTC)


def _operation() -> OperationRef:
    return OperationRef(kind="assemble", input_ids=("artefact_x",), config_hash="abc")


def test_run_record_rejects_a_non_terminal_state_with_no_next_action() -> None:
    with pytest.raises(ValidationError, match="next_action"):
        RunRecord(
            run_id=mint_id("run"),
            bundle_id=mint_id("bundle"),
            state=RunState.RUNNING,
            next_action=None,
            created_at=NOW,
        )


def test_run_record_rejects_completed_state_carrying_a_next_action() -> None:
    with pytest.raises(ValidationError, match="next_action"):
        RunRecord(
            run_id=mint_id("run"),
            bundle_id=mint_id("bundle"),
            state=RunState.COMPLETED,
            next_action=_operation(),
            created_at=NOW,
        )


def test_run_record_accepts_completed_state_with_no_next_action() -> None:
    record = RunRecord(
        run_id=mint_id("run"),
        bundle_id=mint_id("bundle"),
        state=RunState.COMPLETED,
        next_action=None,
        created_at=NOW,
    )

    assert record.next_action is None


def test_run_record_rejects_resuming_and_taking_over_at_the_same_time() -> None:
    with pytest.raises(ValidationError, match="resume"):
        RunRecord(
            run_id=mint_id("run"),
            bundle_id=mint_id("bundle"),
            state=RunState.RUNNING,
            next_action=_operation(),
            resumes_run_id=mint_id("run"),
            takeover_of_run_id=mint_id("run"),
            created_at=NOW,
        )
