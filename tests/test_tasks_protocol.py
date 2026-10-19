"""The MCP Tasks (2026-07-28) projection of the task service.

Not reachable over the transport (no host negotiates the extension yet);
these check the mapping itself, from real run directories created by the
real service, worker and SDK against the fake ``claude`` executable.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest
from conftest import install_fake_claude

from jake_tools.claude_runs import tasks_protocol as tp
from jake_tools.claude_runs.control import RunDefaults, StartRequest
from jake_tools.claude_runs.service import TaskService

DECLARED = {tp.CLIENT_CAPABILITIES_META: {"extensions": {tp.EXTENSION: {}}}}


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TaskService]:
    bin_dir, state = install_fake_claude(tmp_path / "fake")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_CLAUDE_STATE", str(state))
    (tmp_path / "work").mkdir()
    yield TaskService(
        tmp_path / "runs",
        RunDefaults(model="fake", max_turns=5, timeout_seconds=30, max_budget_usd=1),
    )


def start(service: TaskService, tmp_path: Path, brief: str, **extra: Any):
    return service.start(
        StartRequest(brief=brief, working_directory=tmp_path / "work", **extra)
    )


async def wait_for(service: TaskService, task_id: str, status: set[str]) -> None:
    deadline = time.monotonic() + 30
    while service.view(task_id).status not in status:
        assert time.monotonic() < deadline, service.view(task_id).status
        await anyio.sleep(0.1)


def test_negotiation_decides_the_result_shape(
    service: TaskService, tmp_path: Path
) -> None:
    started = start(service, tmp_path, "[[scenario:slow_tool:1]] go")
    view = service.view(started.task_id)

    plain = tp.start_response(started, view, None)
    native = tp.start_response(started, view, DECLARED)

    assert "resultType" not in plain  # an ordinary CallToolResult
    assert plain["structuredContent"]["task_id"] == started.task_id
    assert plain["isError"] is False
    assert native["resultType"] == "task"
    assert native["taskId"] == started.task_id
    assert native["status"] == "working"
    assert native["ttlMs"] is None and native["pollIntervalMs"] > 0
    assert native["createdAt"].endswith("Z")
    with pytest.raises(tp.TasksProtocolError) as info:
        tp.get_task(service, started.task_id, None)
    assert info.value.code == tp.MISSING_REQUIRED_CLIENT_CAPABILITY
    assert info.value.to_jsonrpc()["data"] == {
        "requiredCapabilities": {"extensions": {tp.EXTENSION: {}}}
    }


async def test_input_required_maps_to_keyed_elicitations_and_update(
    service: TaskService, tmp_path: Path
) -> None:
    started = start(
        service, tmp_path, "[[scenario:ask:Which colour?]] go", allow_questions=True
    )
    await wait_for(service, started.task_id, {"input_required"})

    waiting = tp.get_task(service, started.task_id, DECLARED)
    ((key, request),) = waiting["inputRequests"].items()
    ignored = tp.update_task(
        service,
        started.task_id,
        {"q99-unknown": {"anything": "goes"}, "also-unknown": 7},
        DECLARED,
    )
    with pytest.raises(tp.TasksProtocolError) as bad_action:
        tp.update_task(
            service,
            started.task_id,
            {key: {"action": "maybe", "content": {"answer": "x"}}},
            DECLARED,
        )
    with pytest.raises(tp.TasksProtocolError) as no_action:
        tp.update_task(
            service, started.task_id, {key: {"content": {"answer": "x"}}}, DECLARED
        )
    with pytest.raises(tp.TasksProtocolError) as no_answer:
        tp.update_task(service, started.task_id, {key: {"action": "accept"}}, DECLARED)
    ack = tp.update_task(
        service,
        started.task_id,
        {key: {"action": "accept", "content": {"answer": "green"}}},
        DECLARED,
    )
    await wait_for(service, started.task_id, {"completed", "failed", "cancelled"})
    done = tp.get_task(service, started.task_id, DECLARED)

    assert waiting["resultType"] == "complete"
    assert waiting["status"] == "input_required"
    assert request["method"] == "elicitation/create"
    assert request["params"]["message"] == "Which colour?"
    schema = request["params"]["requestedSchema"]
    assert schema["required"] == ["answer"]
    assert schema["properties"]["answer"]["examples"] == ["red", "green"]
    assert ignored == {"resultType": "complete"}  # never-issued keys ignored, unparsed
    assert {e.value.code for e in (bad_action, no_action, no_answer)} == {
        tp.INVALID_PARAMS
    }
    assert ack == {"resultType": "complete"}
    assert done["status"] == "completed"
    assert done["result"]["isError"] is False
    assert (
        done["result"]["content"][0]["text"]
        == "Answer was: Your parent answered: green"
    )
    assert "inputRequests" not in done


async def test_decline_and_cancel_actions_both_decline(
    service: TaskService, tmp_path: Path
) -> None:
    started = start(
        service, tmp_path, "[[scenario:ask_twice]] go", allow_questions=True
    )
    deadline = time.monotonic() + 30
    while len(service.view(started.task_id).pending_inputs) < 2:
        assert time.monotonic() < deadline
        await anyio.sleep(0.1)
    keys = list(tp.get_task(service, started.task_id, DECLARED)["inputRequests"])

    tp.update_task(
        service,
        started.task_id,
        {keys[0]: {"action": "decline"}, keys[1]: {"action": "cancel"}},
        DECLARED,
    )
    await wait_for(service, started.task_id, {"completed"})

    text = tp.get_task(service, started.task_id, DECLARED)["result"]["content"][0][
        "text"
    ]
    assert text.count("declined to answer") == 2


async def test_worker_failures_are_completed_tool_errors_not_protocol_failures(
    service: TaskService, tmp_path: Path
) -> None:
    started = start(service, tmp_path, "[[scenario:max_turns]] go")
    await wait_for(service, started.task_id, {"completed", "failed", "cancelled"})

    task = tp.get_task(service, started.task_id, DECLARED)

    assert service.view(started.task_id).status == "failed"  # the domain says failed
    assert task["status"] == "completed"  # the protocol: a tool result...
    assert task["result"]["isError"] is True  # ...that is an error
    assert task["result"]["structuredContent"]["termination_reason"] == "max_turns"
    assert "error" not in task


async def test_cancel_acks_and_unknown_tasks_are_invalid_params(
    service: TaskService, tmp_path: Path
) -> None:
    started = start(service, tmp_path, "[[scenario:hang:31101]] go")
    await anyio.sleep(1)

    ack = await tp.cancel_task(service, started.task_id, DECLARED)
    task = tp.get_task(service, started.task_id, DECLARED)

    assert ack == {"resultType": "complete"}
    assert task["status"] == "cancelled"
    assert task["statusMessage"] == "Cancelled"
    for call in (
        lambda: tp.get_task(service, "20261008T031500Z-1a2b3c4d", DECLARED),
        lambda: tp.update_task(service, "nope", {}, DECLARED),
    ):
        with pytest.raises(tp.TasksProtocolError) as info:
            call()
        assert info.value.code == tp.INVALID_PARAMS
    with pytest.raises(tp.TasksProtocolError) as info:
        await tp.cancel_task(service, "nope", DECLARED)
    assert info.value.code == tp.INVALID_PARAMS


def test_the_pinned_spec_revision_is_recorded() -> None:
    assert tp.SPEC_VERSION == "2026-07-28"
    assert len(tp.SPEC_REVISION) == 40
