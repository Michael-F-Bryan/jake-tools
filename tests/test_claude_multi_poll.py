"""Select-style waits through the real MCP server and worker/SDK transport."""

from __future__ import annotations

import time
from pathlib import Path

import anyio
import pytest
from test_claude_tasks import (
    ServerFactory,
    call,
    call_error,
    kinds,
    poll_until,
    start,
    terminal,
)


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


async def test_multi_poll_validates_all_entries_before_waiting(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(session, workdir, "[[scenario:echo]] go")
        task_id = started["task_id"]
        await poll_until(session, task_id, terminal)
        task = {"task_id": task_id}
        for arguments, code in (
            ({}, "invalid_argument"),
            ({"tasks": []}, "invalid_argument"),
            ({"tasks": [task] * 17}, "invalid_argument"),
            ({"tasks": [task, task]}, "invalid_argument"),
            ({"tasks": [task], "task_id": task_id}, "invalid_argument"),
            ({"tasks": [task], "cursor": "garbage"}, "invalid_argument"),
            ({"tasks": [task], "limit": 0}, "invalid_argument"),
            ({"tasks": [task], "wait_seconds": 26}, "invalid_argument"),
            ({"tasks": [task, {"task_id": "missing"}]}, "unknown_task"),
            (
                {
                    "tasks": [
                        task,
                        {
                            "task_id": "20261008T031500Z-1a2b3c4d",
                            "cursor": f"{task_id}:1:0",
                        },
                    ]
                },
                "unknown_task",
            ),
            (
                {"tasks": [{"task_id": task_id, "cursor": "garbage"}]},
                "invalid_argument",
            ),
            (
                {"tasks": [{"task_id": task_id, "cursor": f"{task_id}:99:999999"}]},
                "invalid_argument",
            ),
        ):
            began = time.monotonic()
            error = await call_error(
                session, "claude_poll", **{"wait_seconds": 20, **arguments}
            )
            assert error["code"] == code, arguments
            assert time.monotonic() - began < 2


async def test_multi_poll_has_one_deadline_and_cancelled_wait_leaves_workers_alone(
    mcp_server: ServerFactory, fake_env: dict[str, str], mcp_home: Path, workdir: Path
) -> None:
    from jake_tools.claude_runs.control import RunDefaults
    from jake_tools.claude_runs.service import PollTask, TaskService

    async with mcp_server(fake_env) as session:
        watched = []
        for _ in range(2):
            started = await start(session, workdir, "[[scenario:slow_tool:8]] go")
            _, _, cursor = await poll_until(
                session,
                started["task_id"],
                lambda _, events: "tool_use" in kinds(events),
            )
            watched.append({"task_id": started["task_id"], "cursor": cursor})
        began = time.monotonic()
        quiet = await call(session, "claude_poll", tasks=watched, wait_seconds=0.6)
        assert quiet["ready"] == [] and quiet["wake_reason"] == "deadline"
        assert 0.5 <= time.monotonic() - began < 1.1
        service = TaskService(
            mcp_home / ".local/state/jake-tools/runs",
            RunDefaults(
                model="fake", max_turns=5, timeout_seconds=60, max_budget_usd=1
            ),
        )
        with anyio.move_on_after(0.3) as scope:
            await service.poll_many(
                [PollTask(**item) for item in watched], wait_seconds=20
            )
        assert scope.cancelled_caught
        for item in watched:
            assert (await call(session, "claude_status", task_id=item["task_id"]))[
                "status"
            ] == "working"
            await call(session, "claude_cancel", task_id=item["task_id"])
        # Cancellation is a ready terminal outcome too, not just normal success.
        ready = await call(
            session, "claude_poll", tasks=watched, until="terminal", wait_seconds=0
        )
        assert len(ready["ready"]) == 2
        assert all(page["status"] == "cancelled" for page in ready["ready"])


async def test_multi_poll_preserves_cursor_error_mapping_if_log_changes_during_wait(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(session, workdir, "[[scenario:slow_tool:8]] go")
        page, _, cursor = await poll_until(
            session, started["task_id"], lambda _, events: "tool_use" in kinds(events)
        )
        log = Path(page["run_dir"]) / "events.jsonl"
        original = log.read_bytes()

        async def truncate_log() -> None:
            await anyio.sleep(0.2)
            log.write_text("")

        error = None
        try:
            async with anyio.create_task_group() as group:
                group.start_soon(truncate_log)
                error = await call_error(
                    session,
                    "claude_poll",
                    tasks=[{"task_id": started["task_id"], "cursor": cursor}],
                    wait_seconds=5,
                )
            assert error is not None and error["code"] == "invalid_argument"
        finally:
            log.write_bytes(original)
            await call(session, "claude_cancel", task_id=started["task_id"])


async def test_multi_poll_attention_includes_questions_and_does_not_consume_hidden_events(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(
            session, workdir, "[[scenario:ask:Which colour?]] go", allow_questions=True
        )
        task_id = started["task_id"]
        waiting, _, cursor = await poll_until(
            session, task_id, lambda page, _: page["status"] == "input_required"
        )
        quiet = await call(
            session,
            "claude_poll",
            tasks=[{"task_id": task_id}],
            until="terminal",
            wait_seconds=0,
        )
        assert quiet["ready"] == []  # progress is not terminal; no cursor is advanced
        attention = await call(
            session, "claude_poll", tasks=[{"task_id": task_id}], wait_seconds=0
        )
        assert attention["ready"][0]["events"][0]["seq"] == 1
        input_only = await call(
            session,
            "claude_poll",
            tasks=[{"task_id": task_id, "cursor": cursor}],
            wait_seconds=20,
        )
        assert input_only["ready"][0]["wake_reason"] == "input_required"
        assert input_only["ready"][0]["events"] == []
        request = waiting["pending_inputs"][0]["request_id"]
        await call(
            session,
            "claude_update",
            task_id=task_id,
            responses={request: {"answer": "green"}},
        )
        final = await call(
            session,
            "claude_poll",
            tasks=[{"task_id": task_id, "cursor": cursor}],
            until="terminal",
            wait_seconds=20,
        )
        assert final["ready"][0]["status"] == "completed"
        assert "green" in final["ready"][0]["result"]["final_text"]
