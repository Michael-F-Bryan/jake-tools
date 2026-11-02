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
            ({"tasks": [task], "token": "obs1.x"}, "invalid_argument"),
            ({"tasks": [task], "wait_seconds": 121}, "invalid_argument"),
            ({"tasks": [{"task_id": 7}]}, "invalid_argument"),
            ({"tasks": [task, {"task_id": "missing"}]}, "unknown_task"),
            (
                {"tasks": [task, {"task_id": "20261008T031500Z-1a2b3c4d"}]},
                "unknown_task",
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
            await poll_until(
                session,
                started["task_id"],
                lambda _, events: "tool_use" in kinds(events),
            )
            watched.append({"task_id": started["task_id"]})
        began = time.monotonic()
        quiet = await call(session, "claude_poll", tasks=watched, wait_seconds=0.6)
        assert quiet["ready"] == [] and quiet["wake_reason"] == "deadline"
        assert 0.5 <= time.monotonic() - began < 1.1
        # Every watched task is still reported, so a deadline is assessable.
        assert [t["status"] for t in quiet["tasks"]] == ["working", "working"]
        assert all(t["running_tool"]["tool"] == "Bash" for t in quiet["tasks"])
        tokens = [
            {"task_id": t["task_id"], "token": t["observation_token"]}
            for t in quiet["tasks"]
        ]
        service = TaskService(
            mcp_home / ".local/state/jake-tools/runs",
            RunDefaults(
                model="fake", max_turns=5, timeout_seconds=60, max_budget_usd=1
            ),
        )
        with anyio.move_on_after(0.3) as scope:
            await service.poll_many(
                [PollTask(**item) for item in tokens], wait_seconds=20
            )
        assert scope.cancelled_caught
        for item in watched:
            assert (await call(session, "claude_status", task_id=item["task_id"]))[
                "status"
            ] == "working"
            await call(session, "claude_cancel", task_id=item["task_id"])
        # Cancellation is a ready terminal outcome too, not just normal success.
        ready = await call(
            session, "claude_poll", tasks=tokens, until="terminal", wait_seconds=0
        )
        assert len(ready["ready"]) == 2
        assert all(task["status"] == "cancelled" for task in ready["tasks"])
        assert all(
            [w["category"] for w in task["changes"]["warnings"]] == ["cancel_requested"]
            for task in ready["tasks"]
        )


async def test_multi_poll_resynchronises_a_token_whose_log_changed_during_the_wait(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(session, workdir, "[[scenario:slow_tool:8]] go")
        page, _, _ = await poll_until(
            session, started["task_id"], lambda _, events: "tool_use" in kinds(events)
        )
        info = await call(session, "claude_status", task_id=started["task_id"])
        log = Path(info["run_dir"]) / "events.jsonl"
        original = log.read_bytes()
        log.write_text("")
        try:
            result = await call(
                session,
                "claude_poll",
                tasks=[
                    {"task_id": started["task_id"], "token": page["observation_token"]}
                ],
                wait_seconds=5,
            )
            (task,) = result["tasks"]
            assert task["token_status"] == "resynchronised"
            assert result["ready"] == [started["task_id"]]
        finally:
            log.write_bytes(original)
            await call(session, "claude_cancel", task_id=started["task_id"])


async def test_multi_poll_questions_interrupt_every_wait_mode(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(
            session, workdir, "[[scenario:ask:Which colour?]] go", allow_questions=True
        )
        task_id = started["task_id"]
        await poll_until(
            session, task_id, lambda page, _: page["status"] == "input_required"
        )
        for until in ("terminal", "attention", "progress"):
            began = time.monotonic()
            woke = await call(
                session,
                "claude_poll",
                tasks=[{"task_id": task_id, "token": started["observation_token"]}],
                until=until,
                wait_seconds=20,
            )
            assert time.monotonic() - began < 2, until
            (task,) = woke["tasks"]
            assert task["wake_reason"] == "question", until
            assert task["pending_questions"][0]["question"] == "Which colour?"
        request = task["pending_questions"][0]["request_id"]
        await call(
            session,
            "claude_update",
            task_id=task_id,
            responses={request: {"answer": "green"}},
        )
        final = await call(
            session,
            "claude_poll",
            tasks=[{"task_id": task_id, "token": task["observation_token"]}],
            until="terminal",
            wait_seconds=20,
        )
        (done,) = final["tasks"]
        assert done["status"] == "completed"
        assert "green" in done["result"]["final_text"]
        assert [
            (q["request_id"], q["state"]) for q in done["changes"]["questions"]
        ] == [(request, "answered")]
