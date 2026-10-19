"""Native MCP Tasks (2026-07-28) over the real stdio server.

Two kinds of client drive the real ``python -m jake_tools.mcp``: a
2026-07-28 ``mcp.Client`` that declares ``io.modelcontextprotocol/tasks``
per request (``tasks_client.py``), and the legacy ``initialize`` session the
rest of the suite uses. The server spawns the real worker, which runs the
real Claude Agent SDK against ``tests/fake_claude.py``: mechanics only.
``test_claude_tasks_live.py`` drives the same native path with real Claude.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import ClientSession
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult
from tasks_client import (
    EXTENSION,
    TERMINAL,
    CreateTaskResult,
    TaskParams,
    cancel_task,
    create_task,
    get_task,
    poll_task,
    start_modern_client,
    update_task,
)

from jake_tools.claude_runs.procs import process_table

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]

FAST: dict[str, Any] = {"model": "fake-model", "max_turns": 5, "max_budget_usd": 1.0}
DECLARED_META = {
    "io.modelcontextprotocol/clientCapabilities": {"extensions": {EXTENSION: {}}}
}
MISSING_CAPABILITY = -32021
INVALID_PARAMS = -32602


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


def start_args(workdir: Path, brief: str, **extra: Any) -> dict[str, Any]:
    return {
        "brief": brief,
        "working_directory": str(workdir),
        **FAST,
        "timeout_seconds": 60,
        **extra,
    }


async def create(session: ClientSession, arguments: dict[str, Any]) -> CreateTaskResult:
    created = await create_task(session, "claude_start", arguments)
    assert isinstance(created, CreateTaskResult), created
    return created


def is_empty_ack(result: dict[str, Any]) -> bool:
    """``resultType: complete`` and nothing else but the ``_meta`` stamp."""
    return result["resultType"] == "complete" and set(result) <= {"resultType", "_meta"}


def text_of(result: dict[str, Any]) -> str:
    return result["content"][0]["text"]


def sleepers(marker: str) -> list[int]:
    completed = subprocess.run(
        ["pgrep", "-x", "-f", f"sleep {marker}"], capture_output=True, text=True
    )
    return [int(pid) for pid in completed.stdout.split()]


# --- discovery -----------------------------------------------------------------


async def test_discovery_advertises_tasks_to_modern_clients_and_keeps_every_tool(
    mcp_server: ServerFactory, fake_env: dict[str, str]
) -> None:
    async with start_modern_client(fake_env) as client:
        modern = client.server_capabilities
        modern_tools = {tool.name for tool in (await client.list_tools()).tools}
    async with mcp_server(fake_env) as session:
        legacy = session.server_capabilities
        legacy_tools = {tool.name for tool in (await session.list_tools()).tools}

    assert modern.extensions == {EXTENSION: {}}
    assert legacy is not None and not legacy.extensions  # nowhere to put it
    # The ordinary tools stay for everyone, steering included.
    assert modern_tools == legacy_tools
    assert {
        "claude_start",
        "claude_status",
        "claude_poll",
        "claude_steer",
        "claude_update",
        "claude_cancel",
    } <= modern_tools


# --- creation and polling ----------------------------------------------------------


async def test_a_declaring_client_gets_a_durable_task_and_polls_it_to_a_result(
    fake_env: dict[str, str], workdir: Path
) -> None:
    async with start_modern_client(fake_env) as client:
        created = await create(
            client.session, start_args(workdir, "[[scenario:slow_tool:1]] go")
        )
        first = await get_task(client.session, created.task_id)
        final, seen = await poll_task(client.session, created.task_id, TERMINAL)
        # One task system: the ordinary tool sees the same task.
        status = await client.call_tool("claude_status", {"task_id": created.task_id})

    assert created.status == "working"
    assert created.ttl_ms is None and created.poll_interval_ms
    assert created.created_at.endswith("Z")
    assert first["resultType"] == "complete"
    assert first["taskId"] == created.task_id  # durable: resolves at once
    assert {task["status"] for task in seen[:-1]} <= {"working"}
    assert final["status"] == "completed"
    assert final["resultType"] == "complete"
    assert final["result"]["isError"] is False
    assert text_of(final["result"]) == "Done. Heard: nothing"
    assert final["result"]["structuredContent"]["termination_reason"] == "finished"
    assert "error" not in final and "inputRequests" not in final
    assert status.structured_content is not None
    assert status.structured_content["status"] == "completed"


async def test_call_tool_resolves_a_task_transparently(
    fake_env: dict[str, str], workdir: Path
) -> None:
    async with start_modern_client(fake_env) as client:
        result = await client.call_tool(
            "claude_start", start_args(workdir, "[[scenario:echo]] hello")
        )

    assert isinstance(result, CallToolResult)
    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["termination_reason"] == "finished"


async def test_a_failed_start_is_an_ordinary_tool_error_not_a_task(
    fake_env: dict[str, str], tmp_path: Path
) -> None:
    async with start_modern_client(fake_env) as client:
        result = await create_task(
            client.session,
            "claude_start",
            {"brief": "go", "working_directory": str(tmp_path / "missing")},
        )

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "invalid_argument"


# --- clarification ---------------------------------------------------------------------


async def test_keyed_input_requests_are_answered_with_tasks_update(
    fake_env: dict[str, str], workdir: Path
) -> None:
    async with start_modern_client(fake_env) as client:
        session = client.session
        created = await create(
            session,
            start_args(
                workdir, "[[scenario:ask:Which colour?]] go", allow_questions=True
            ),
        )
        waiting, _ = await poll_task(session, created.task_id, {"input_required"})
        again = await get_task(session, created.task_id)
        ((key, request),) = waiting["inputRequests"].items()
        ignored = await update_task(
            session, created.task_id, {"q99-never-issued": {"anything": 1}}
        )
        with pytest.raises(MCPError) as bad:
            await update_task(session, created.task_id, {key: {"action": "maybe"}})
        ack = await update_task(
            session,
            created.task_id,
            {key: {"action": "accept", "content": {"answer": "green"}}},
        )
        final, _ = await poll_task(session, created.task_id, TERMINAL)
        repeat = await update_task(
            session,
            created.task_id,
            {key: {"action": "accept", "content": {"answer": "red"}}},
        )

    assert waiting["statusMessage"] == "Waiting for an answer: Which colour?"
    assert again["inputRequests"] == waiting["inputRequests"]  # stable key
    assert request["method"] == "elicitation/create"
    assert request["params"]["mode"] == "form"
    assert request["params"]["message"] == "Which colour?"
    assert request["params"]["requestedSchema"]["required"] == ["answer"]
    assert is_empty_ack(ignored)
    assert bad.value.error.code == INVALID_PARAMS
    assert is_empty_ack(ack)
    assert final["status"] == "completed"
    assert text_of(final["result"]) == "Answer was: Your parent answered: green"
    assert is_empty_ack(repeat)  # no longer outstanding: ignored


async def test_a_partial_response_leaves_the_rest_outstanding(
    fake_env: dict[str, str], workdir: Path
) -> None:
    async with start_modern_client(fake_env) as client:
        session = client.session
        created = await create(
            session,
            start_args(workdir, "[[scenario:ask_twice]] go", allow_questions=True),
        )
        with anyio.fail_after(30):
            while True:
                waiting = await get_task(session, created.task_id)
                if len(waiting.get("inputRequests", {})) == 2:
                    break
                await anyio.sleep(0.1)
        first, second = sorted(waiting["inputRequests"])
        await update_task(
            session,
            created.task_id,
            {first: {"action": "accept", "content": {"answer": "one"}}},
        )
        with anyio.fail_after(30):
            while True:
                partial = await get_task(session, created.task_id)
                if set(partial.get("inputRequests", {})) == {second}:
                    break
                await anyio.sleep(0.1)
        await update_task(session, created.task_id, {second: {"action": "decline"}})
        final, _ = await poll_task(session, created.task_id, TERMINAL)

    assert partial["status"] == "input_required"
    assert final["status"] == "completed"
    assert "one" in text_of(final["result"])
    assert "declined to answer" in text_of(final["result"])


# --- terminal errors and cancellation ------------------------------------------


async def test_a_worker_failure_is_a_completed_task_with_an_error_result(
    fake_env: dict[str, str], workdir: Path
) -> None:
    async with start_modern_client(fake_env) as client:
        created = await create(
            client.session, start_args(workdir, "[[scenario:max_turns]] go")
        )
        final, seen = await poll_task(client.session, created.task_id, TERMINAL)

    assert final["status"] == "completed"  # never protocol "failed"
    assert final["result"]["isError"] is True
    assert final["result"]["structuredContent"]["termination_reason"] == "max_turns"
    assert "error" not in final
    assert "failed" not in {task["status"] for task in seen}


async def test_tasks_cancel_acks_and_stops_the_process_tree(
    fake_env: dict[str, str], workdir: Path
) -> None:
    marker = "31207"
    async with start_modern_client(fake_env) as client:
        created = await create(
            client.session, start_args(workdir, f"[[scenario:hang:{marker}]] go")
        )
        with anyio.fail_after(30):
            while not sleepers(marker):
                await anyio.sleep(0.1)
        await anyio.sleep(1)  # let the worker record the shell's group
        ack = await cancel_task(client.session, created.task_id)
        final, _ = await poll_task(client.session, created.task_id, TERMINAL)

    assert is_empty_ack(ack)
    assert final["status"] == "cancelled"
    assert "result" not in final and "error" not in final
    deadline = time.monotonic() + 10
    while sleepers(marker):
        assert time.monotonic() < deadline, "the tool shell outlived the cancel"
        await anyio.sleep(0.1)


# --- reconnect -----------------------------------------------------------------------


async def test_a_new_connection_and_server_process_resume_the_same_task(
    fake_env: dict[str, str], workdir: Path
) -> None:
    async with start_modern_client(fake_env) as client:
        ping = await client.call_tool("ping", {})
        assert ping.structured_content is not None
        first_pid = ping.structured_content["pid"]
        created = await create(
            client.session, start_args(workdir, "[[scenario:slow_tool:4]] go")
        )
        # Steering stays a domain tool, and works on a native task.
        steer = await client.call_tool(
            "claude_steer",
            {
                "task_id": created.task_id,
                "message": "after reconnect",
                "message_id": "m1",
            },
        )
        assert steer.is_error is False, steer.structured_content
        status = await client.call_tool("claude_status", {"task_id": created.task_id})
        assert status.structured_content is not None
        run_dir = Path(status.structured_content["run_dir"])
    # That server process has exited; the worker has not.
    record_pid = json.loads((run_dir / "process.json").read_text())["pid"]
    worker = process_table().get(record_pid)
    assert worker is not None and not worker.zombie

    async with start_modern_client(fake_env) as client:
        ping = await client.call_tool("ping", {})
        assert ping.structured_content is not None
        second_pid = ping.structured_content["pid"]
        final, _ = await poll_task(client.session, created.task_id, TERMINAL)

    assert second_pid != first_pid
    assert final["status"] == "completed"
    assert text_of(final["result"]) == "Done. Heard: after reconnect"


# --- capability rejection ------------------------------------------------------------


async def test_a_modern_client_that_does_not_declare_tasks_gets_ordinary_results(
    fake_env: dict[str, str], workdir: Path
) -> None:
    async with start_modern_client(fake_env, declare_tasks=False) as client:
        result = await create_task(
            client.session, "claude_start", start_args(workdir, "[[scenario:echo]] hi")
        )
        assert isinstance(result, CallToolResult), result
        assert result.structured_content is not None
        task_id = result.structured_content["task_id"]
        errors = []
        for call in (
            lambda: get_task(client.session, task_id),
            lambda: update_task(client.session, task_id, {}),
            lambda: cancel_task(client.session, task_id),
        ):
            with pytest.raises(MCPError) as info:
                await call()
            errors.append(info.value.error)

    assert result.is_error is False
    assert {error.code for error in errors} == {MISSING_CAPABILITY}
    assert errors[0].data == {"requiredCapabilities": {"extensions": {EXTENSION: {}}}}


async def test_a_legacy_client_never_gets_a_task_even_if_its_meta_claims_one(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        assert session.protocol_version == "2025-11-25"
        # The per-request capability key, smuggled into a legacy request:
        # a legacy connection cannot declare per-request capabilities.
        result = await session.call_tool(
            "claude_start",
            start_args(workdir, "[[scenario:echo]] hi"),
            meta=DECLARED_META,  # type: ignore[arg-type]
        )
        assert result.structured_content is not None
        task_id = result.structured_content["task_id"]
        with pytest.raises(MCPError) as info:
            await session.send_request(
                _get_request(task_id, DECLARED_META), CallToolResult
            )

    assert result.is_error is False
    assert result.structured_content["status"] == "working"
    assert info.value.error.code == MISSING_CAPABILITY


async def test_unknown_task_ids_are_invalid_params(
    fake_env: dict[str, str],
) -> None:
    async with start_modern_client(fake_env) as client:
        codes = []
        for call in (
            lambda: get_task(client.session, "20261008T031500Z-1a2b3c4d"),
            lambda: update_task(client.session, "../../etc", {}),
            lambda: cancel_task(client.session, "nope"),
        ):
            with pytest.raises(MCPError) as info:
                await call()
            codes.append(info.value.error.code)

    assert codes == [INVALID_PARAMS] * 3


def _get_request(task_id: str, meta: dict[str, Any]) -> Any:
    from tasks_client import GetTaskRequest

    return GetTaskRequest(params=TaskParams(task_id=task_id, _meta=meta))  # type: ignore[call-arg]
