"""Operator-configurable local task capacity (``claude.max_live_tasks``).

The admission check is ``live >= limit`` under ``<runs_dir>/.runs.lock``, where
``live`` is the number of live workers in the shared runs directory and
``limit`` is the *admitting process's own* configured value. These tests pin
that, over real processes: the real server and worker over stdio with a fake
``claude`` CLI (the only substitute, at the true external boundary), and real
sleeping process groups for the unit-level cases.
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from pathlib import Path
from typing import Any

import pytest
import test_claude_runs
from mcp import ClientSession
from test_claude_runs import DEFAULTS, _working
from test_claude_tasks import (
    FAST,
    call,
    call_error,
    poll_until,
    sleepers,
    start,
    terminal,
)

from jake_tools.claude_runs import (
    CapacityExceededError,
    RunDefaults,
    StartRequest,
    start_task,
)
from jake_tools.claude_runs.control import live_tasks
from jake_tools.claude_runs.rundir import read_state

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]
live_group = test_claude_runs.live_group  # the real-process-group fixture
LIMIT_VARIABLE = "JAKE_TOOLS_CLAUDE_MAX_LIVE_TASKS"


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


def runs_of(home: Path) -> Path:
    return home / ".local" / "state" / "jake-tools" / "runs"


def with_limit(env: dict[str, str], limit: int) -> dict[str, str]:
    return {**env, LIMIT_VARIABLE: str(limit)}


def task_dirs(runs_dir: Path) -> list[str]:
    return sorted(p.name for p in runs_dir.iterdir() if not p.name.startswith("."))


async def cancel_all(session: ClientSession, tasks: list[dict[str, Any]]) -> None:
    for task in tasks:
        await call(session, "claude_cancel", task_id=task["task_id"])


# --- unit level: the limit in RunDefaults, real process groups -----------------


def test_run_defaults_limit_is_two_unless_configured() -> None:
    assert DEFAULTS.max_live_tasks == 2
    assert DEFAULTS.model_copy(update={"max_live_tasks": 7}).max_live_tasks == 7
    with pytest.raises(ValueError, match="max_live_tasks"):
        RunDefaults(
            model="m",
            max_turns=1,
            timeout_seconds=1,
            max_budget_usd=1,
            max_live_tasks=0,
        )


def test_the_configured_limit_not_two_decides_admission(
    tmp_path: Path, live_group: Callable[[], int]
) -> None:
    """Three live, limit three: refused, with the configured limit in the error."""
    runs_dir = tmp_path / "runs"
    for _ in range(3):
        _working(runs_dir, tmp_path, live_group())
    before = task_dirs(runs_dir)
    defaults = DEFAULTS.model_copy(update={"max_live_tasks": 3})

    with pytest.raises(CapacityExceededError) as info:
        start_task(
            runs_dir, StartRequest(brief="x", working_directory=tmp_path), defaults
        )

    assert (info.value.live, info.value.limit) == (3, 3)
    assert "limit is 3" in str(info.value)
    assert task_dirs(runs_dir) == before


def test_a_limit_reduced_below_the_live_count_cancels_nothing_and_admits_nothing(
    tmp_path: Path, live_group: Callable[[], int]
) -> None:
    runs_dir = tmp_path / "runs"
    groups = [live_group() for _ in range(3)]
    dirs = [_working(runs_dir, tmp_path, pgid) for pgid in groups]
    reduced = DEFAULTS.model_copy(update={"max_live_tasks": 1})
    request = StartRequest(brief="x", working_directory=tmp_path)

    with pytest.raises(CapacityExceededError) as info:
        start_task(runs_dir, request, reduced)
    assert (info.value.live, info.value.limit) == (3, 1)
    # Nothing was cancelled or settled: every worker is still alive.
    assert len(live_tasks(runs_dir)) == 3
    for pgid in groups:
        os.killpg(pgid, 0)

    # Still refused while live >= limit as tasks finish on their own...
    os.killpg(groups[0], signal.SIGKILL)
    os.killpg(groups[1], signal.SIGKILL)
    with pytest.raises(CapacityExceededError) as info:
        start_task(runs_dir, request, reduced)
    assert (info.value.live, info.value.limit) == (1, 1)
    assert [read_state(d).status for d in dirs[:2]] == ["failed", "failed"]
    assert read_state(dirs[2]).status == "working"


# --- over stdio: real server, worker, fake CLI ---------------------------------


async def test_three_admitted_and_a_fourth_refused_with_a_limit_of_three(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path, mcp_home: Path
) -> None:
    env = with_limit(fake_env, 3)
    async with mcp_server(env) as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
        admitted = [
            await start(
                session, workdir, f"[[scenario:hang:3210{i}]] t{i}", request_id=f"r-{i}"
            )
            for i in range(3)
        ]
        fourth = await call_error(
            session, "claude_start", brief="d", working_directory=str(workdir), **FAST
        )
        retry = await start(
            session, workdir, "[[scenario:hang:32100]] t0", request_id="r-0"
        )
        await cancel_all(session, admitted)

    assert len({a["task_id"] for a in admitted}) == 3
    assert fourth["code"] == "capacity_exceeded"
    assert fourth["detail"] == {"live": 3, "limit": 3}
    assert "limit is 3" in fourth["message"]
    # An idempotent retry still finds its task when the slots are full.
    assert retry["replayed"] is True and retry["task_id"] == admitted[0]["task_id"]
    assert len(task_dirs(runs_of(mcp_home))) == 3  # the refused fourth made none
    assert [sleepers(f"3210{i}") for i in range(3)] == [[], [], []]
    # The description no longer promises a universal two.
    description = tools["claude_start"].description or ""
    assert "At most 2" not in description and "At most two" not in description
    assert "claude.max_live_tasks" in description


async def test_the_default_limit_is_still_two(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        admitted = [
            await start(session, workdir, f"[[scenario:hang:3220{i}]] t{i}")
            for i in range(2)
        ]
        third = await call_error(
            session, "claude_start", brief="c", working_directory=str(workdir), **FAST
        )
        await cancel_all(session, admitted)

    assert third["code"] == "capacity_exceeded"
    assert third["detail"] == {"live": 2, "limit": 2}


async def test_concurrent_servers_sharing_a_runs_directory_never_exceed_the_limit(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path, mcp_home: Path
) -> None:
    """Six server processes race six starts at limit three: exactly three win."""
    env = with_limit(fake_env, 3)
    async with AsyncExitStack() as stack:
        sessions = [await stack.enter_async_context(mcp_server(env)) for _ in range(6)]
        results = await asyncio.gather(
            *(
                s.call_tool(
                    "claude_start",
                    {
                        "brief": f"[[scenario:hang:3230{i}]] t{i}",
                        "working_directory": str(workdir),
                        **FAST,
                        "timeout_seconds": 60,
                    },
                )
                for i, s in enumerate(sessions)
            )
        )
        admitted = [r.structured_content for r in results if not r.is_error]
        refused = [r.structured_content for r in results if r.is_error]
        assert len(admitted) == 3, results
        assert all(r is not None for r in admitted + refused)
        await cancel_all(sessions[0], [r for r in admitted if r is not None])

    assert len(refused) == 3
    for error in refused:
        assert error is not None
        assert error["code"] == "capacity_exceeded"
        assert error["detail"] == {"live": 3, "limit": 3}
    assert len(task_dirs(runs_of(mcp_home))) == 3
    assert [sleepers(f"3230{i}") for i in range(6)] == [[]] * 6


async def test_servers_with_different_limits_share_the_live_count_not_a_ceiling(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    """Each server applies its own limit to the one shared live count.

    The limit-3 server fills the directory with three tasks. The limit-2
    server (a reduced limit relative to what is live) then refuses, reporting
    its own limit, and cancels nothing. When one task ends it still refuses
    (live 2 >= 2) while the limit-3 server admits again.
    """
    roomy_env, strict_env = with_limit(fake_env, 3), with_limit(fake_env, 2)
    async with mcp_server(roomy_env) as roomy, mcp_server(strict_env) as strict:
        admitted = [
            await start(roomy, workdir, f"[[scenario:hang:3240{i}]] t{i}")
            for i in range(3)
        ]
        refused_strict = await call_error(
            strict, "claude_start", brief="x", working_directory=str(workdir), **FAST
        )
        statuses = [
            (await call(strict, "claude_status", task_id=a["task_id"]))["status"]
            for a in admitted
        ]
        await call(roomy, "claude_cancel", task_id=admitted[2]["task_id"])
        still_refused = await call_error(
            strict, "claude_start", brief="x", working_directory=str(workdir), **FAST
        )
        fourth = await start(roomy, workdir, "[[scenario:hang:32403]] t3")
        full = await call_error(
            roomy, "claude_start", brief="x", working_directory=str(workdir), **FAST
        )
        await cancel_all(roomy, [*admitted[:2], fourth])

    assert refused_strict["detail"] == {"live": 3, "limit": 2}
    assert statuses == ["working"] * 3  # reducing the limit cancels nothing
    assert still_refused["detail"] == {"live": 2, "limit": 2}
    assert full["detail"] == {"live": 3, "limit": 3}
    assert [sleepers(f"3240{i}") for i in range(4)] == [[]] * 4


async def test_a_session_cannot_be_written_twice_even_with_free_capacity(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(with_limit(fake_env, 3)) as session:
        done = await start(session, workdir, "[[scenario:echo]] a")
        await poll_until(session, done["task_id"], terminal)
        held = await call(
            session,
            "claude_start",
            brief="[[hold]] keep going",
            resume_task_id=done["task_id"],
            **FAST,
            timeout_seconds=60,
        )
        busy = await call_error(
            session, "claude_start", brief="x", resume_task_id=done["task_id"], **FAST
        )
        other = await start(session, workdir, "[[scenario:hang:32501]] unrelated")
        await cancel_all(session, [held, other])

    assert busy["code"] == "session_busy"  # capacity was free: 1 of 3 live
    assert busy["detail"] == {
        "session_id": done["session_id"],
        "task_id": held["task_id"],
    }
