"""Delegated-task mechanics, hermetically: run directories, state machine,
liveness, capacity, argument validation and the three tools' error shapes.

Nothing here fakes a process. A "dead" process group is a real child that
has exited and been reaped; a "live" one is a real ``sleep`` started in its
own session and killed in teardown. The worker itself (the SDK call) is
covered by the ``slow`` tests in ``test_claude_runs_live.py``.
"""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.types import TextContent
from pydantic import ValidationError

from jake_tools.claude_runs import (
    MAX_LIVE_TASKS,
    CapacityExceededError,
    RunDefaults,
    RunSpec,
    RunState,
    StartRequest,
    UnknownTaskError,
    cancel_task,
    task_status,
)
from jake_tools.claude_runs.control import live_tasks, reconcile, start_task
from jake_tools.claude_runs.models import TASK_ID_PATTERN
from jake_tools.claude_runs.rundir import (
    LOCK_FILE,
    STATUS_FILE,
    create_run_dir,
    new_task_id,
    read_state,
    task_dir,
    update_state,
    write_text,
)
from jake_tools.claude_runs.spawn import (
    group_alive,
    reap_in_background,
    signal_group,
    terminate_group,
)
from jake_tools.config import load_config

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]

DEFAULTS = RunDefaults(
    model="claude-haiku-4-5", max_turns=3, timeout_seconds=30, max_budget_usd=0.1
)


# --- real process groups ------------------------------------------------------


@pytest.fixture
def live_group() -> Iterator[Callable[[], int]]:
    """Start a real ``sleep`` in its own session; returns its pgid. Killed after.

    Reaped from a background thread exactly as the server reaps its workers:
    a child that has exited but not been waited on is a zombie, and a
    zombie-only group still *exists* to ``killpg(pgid, 0)``.
    """
    started: list[subprocess.Popen[bytes]] = []

    def start() -> int:
        process = subprocess.Popen(
            ["sleep", "300"], start_new_session=True, stdin=subprocess.DEVNULL
        )
        started.append(process)
        reap_in_background(process)
        return process.pid

    yield start

    for process in started:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def dead_group() -> int:
    """A pgid that existed and is gone: a real child that exited and was reaped."""
    process = subprocess.Popen(["true"], start_new_session=True)
    process.wait()
    return process.pid


def _spec(task_id: str, working_directory: Path) -> RunSpec:
    return RunSpec(
        task_id=task_id,
        created_at=datetime.now(UTC),
        working_directory=working_directory,
        model="claude-haiku-4-5",
        max_turns=3,
        timeout_seconds=30,
        max_budget_usd=0.1,
    )


def _working(runs_dir: Path, working_directory: Path, pgid: int | None) -> Path:
    """A run directory whose status says ``working`` with ``pgid``."""
    spec = _spec(new_task_id(), working_directory)
    run_dir = create_run_dir(runs_dir, spec, "do the thing")
    update_state(run_dir, lambda s: s.model_copy(update={"pid": pgid, "pgid": pgid}))
    return run_dir


# --- task IDs and the run directory ------------------------------------------


def test_task_ids_are_utc_basic_timestamp_plus_eight_hex() -> None:
    now = datetime(2026, 10, 8, 3, 15, 0, tzinfo=UTC)

    task_id = new_task_id(now)

    assert task_id.startswith("20261008T031500Z-")
    assert TASK_ID_PATTERN.match(task_id)
    assert new_task_id(now) != task_id


def test_create_run_dir_writes_private_brief_spec_and_initial_state(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "state" / "runs"
    spec = _spec(new_task_id(), tmp_path)

    run_dir = create_run_dir(runs_dir, spec, "brief text\n")

    assert stat.S_IMODE(runs_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(run_dir.stat().st_mode) == 0o700
    assert (run_dir / "brief.md").read_text() == "brief text\n"
    assert stat.S_IMODE((run_dir / "brief.md").stat().st_mode) == 0o600
    assert RunSpec.model_validate_json((run_dir / "spec.json").read_bytes()) == spec
    # No status.json yet reads as the initial working state, not an error.
    state = read_state(run_dir)
    assert state.status == "working"
    assert state.task_id == spec.task_id
    assert state.pgid is None


def test_write_text_replaces_atomically_and_leaves_no_temp_files(
    tmp_path: Path,
) -> None:
    path = tmp_path / "status.json"
    write_text(path, "one")
    write_text(path, "two")

    assert path.read_text() == "two"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [p.name for p in tmp_path.iterdir()] == ["status.json"]


def test_update_state_serialises_concurrent_writers(tmp_path: Path) -> None:
    """Two processes doing read-modify-write never lose an update."""
    run_dir = create_run_dir(tmp_path / "runs", _spec(new_task_id(), tmp_path), "b")
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from jake_tools.claude_runs.rundir import update_state\n"
        "run_dir = Path(sys.argv[1]); field = sys.argv[2]\n"
        "for _ in range(25):\n"
        "    update_state(run_dir, lambda s: s.model_copy(update={field: (getattr(s, field) or 0) + 1}))\n"
    )
    workers = [
        subprocess.Popen([sys.executable, "-c", script, str(run_dir), field])
        for field in ("pid", "pgid")
    ]
    for worker in workers:
        assert worker.wait() == 0

    state = read_state(run_dir)
    assert (state.pid, state.pgid) == (25, 25)
    assert (run_dir / LOCK_FILE).exists()


def test_timestamps_serialise_as_rfc3339_utc(tmp_path: Path) -> None:
    run_dir = create_run_dir(tmp_path / "runs", _spec(new_task_id(), tmp_path), "b")
    state = update_state(run_dir, lambda s: s)

    payload = json.loads((run_dir / STATUS_FILE).read_text())
    assert payload["created_at"].endswith("Z")
    assert state.model_dump(mode="json")["created_at"] == payload["created_at"]


# --- resolving task IDs --------------------------------------------------------


@pytest.mark.parametrize(
    "task_id",
    ["", "nope", "../../etc", "20261008T031500Z-zzzzzzzz", "20261008T031500Z-1a2b3c4d"],
)
def test_task_dir_rejects_malformed_and_unknown_ids(
    tmp_path: Path, task_id: str
) -> None:
    with pytest.raises(UnknownTaskError):
        task_dir(tmp_path, task_id)


def test_task_status_for_unknown_id_raises_without_touching_the_filesystem(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "never-created"

    with pytest.raises(UnknownTaskError):
        task_status(runs_dir, "20261008T031500Z-1a2b3c4d")

    assert not runs_dir.exists()


# --- liveness and the worker_died reconciliation ------------------------------


def test_group_alive_reflects_real_process_groups(
    live_group: Callable[[], int],
) -> None:
    assert group_alive(live_group()) is True
    assert group_alive(dead_group()) is False
    assert group_alive(None) is False
    assert group_alive(0) is False  # never our own group


def test_reconcile_marks_a_working_task_with_a_dead_group_as_worker_died(
    tmp_path: Path,
) -> None:
    run_dir = _working(tmp_path / "runs", tmp_path, dead_group())

    state = reconcile(run_dir)

    assert state.status == "failed"
    assert state.termination_reason == "worker_died"
    assert state.finished_at is not None
    assert state.error
    assert read_state(run_dir) == state  # recorded, not just returned


def test_reconcile_leaves_a_live_task_alone(
    tmp_path: Path, live_group: Callable[[], int]
) -> None:
    run_dir = _working(tmp_path / "runs", tmp_path, live_group())

    state = reconcile(run_dir)

    assert state.status == "working"
    assert state.termination_reason is None


def test_reconcile_never_overwrites_a_terminal_state(tmp_path: Path) -> None:
    run_dir = _working(tmp_path / "runs", tmp_path, dead_group())
    done = update_state(
        run_dir,
        lambda s: s.model_copy(
            update={
                "status": "completed",
                "termination_reason": "finished",
                "finished_at": datetime.now(UTC),
                "final_text": "done",
            }
        ),
    )

    assert reconcile(run_dir) == done


def test_task_status_without_a_status_file_settles_as_worker_died(
    tmp_path: Path,
) -> None:
    """The creator died between spec.json and the spawn: no status, no process."""
    runs_dir = tmp_path / "runs"
    spec = _spec(new_task_id(), tmp_path)
    create_run_dir(runs_dir, spec, "brief")

    state = task_status(runs_dir, spec.task_id)

    assert state.status == "failed"
    assert state.termination_reason == "worker_died"


# --- capacity ------------------------------------------------------------------


def test_capacity_counts_only_live_groups_and_settles_dead_ones(
    tmp_path: Path, live_group: Callable[[], int]
) -> None:
    runs_dir = tmp_path / "runs"
    dead = _working(runs_dir, tmp_path, dead_group())
    _working(runs_dir, tmp_path, live_group())

    live = live_tasks(runs_dir)

    assert len(live) == 1
    assert read_state(dead).termination_reason == "worker_died"


def test_start_task_refuses_a_third_live_task_without_creating_anything(
    tmp_path: Path, live_group: Callable[[], int]
) -> None:
    runs_dir = tmp_path / "runs"
    for _ in range(MAX_LIVE_TASKS):
        _working(runs_dir, tmp_path, live_group())
    before = sorted(p.name for p in runs_dir.iterdir())
    request = StartRequest(brief="x", working_directory=tmp_path)

    with pytest.raises(CapacityExceededError) as info:
        start_task(runs_dir, request, DEFAULTS)

    assert info.value.live == MAX_LIVE_TASKS
    assert sorted(p.name for p in runs_dir.iterdir()) == before


# --- cancel ---------------------------------------------------------------------


async def test_cancel_returns_terminal_state_unchanged(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = _working(runs_dir, tmp_path, dead_group())
    done = update_state(
        run_dir,
        lambda s: s.model_copy(
            update={
                "status": "completed",
                "termination_reason": "finished",
                "finished_at": datetime.now(UTC),
            }
        ),
    )

    assert await cancel_task(runs_dir, run_dir.name) == done


async def test_cancel_kills_a_real_group_and_records_cancelled(
    tmp_path: Path,
) -> None:
    """A group that ignores SIGTERM still dies: the SIGKILL backstop runs."""
    runs_dir = tmp_path / "runs"
    process = subprocess.Popen(
        ["sh", "-c", "trap '' TERM; sleep 300 & wait"],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
    )
    reap_in_background(process)
    run_dir = _working(runs_dir, tmp_path, process.pid)
    try:
        started = time.monotonic()
        state = await cancel_task(runs_dir, run_dir.name)
        elapsed = time.monotonic() - started
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()

    assert state.status == "cancelled"
    assert state.termination_reason == "cancelled"
    assert state.finished_at is not None
    assert group_alive(process.pid) is False
    assert elapsed < 20


async def test_terminate_group_returns_quickly_for_a_cooperative_group() -> None:
    process = subprocess.Popen(
        ["sleep", "300"], start_new_session=True, stdin=subprocess.DEVNULL
    )
    reap_in_background(process)
    started = time.monotonic()
    try:
        dead = await terminate_group(process.pid)
    finally:
        process.wait()

    assert dead is True
    assert time.monotonic() - started < 5


def test_signalling_a_zombie_only_group_does_not_raise() -> None:
    """macOS answers EPERM for a group whose only member has exited unreaped."""
    process = subprocess.Popen(["true"], start_new_session=True)
    deadline = time.monotonic() + 5
    while process.poll() is None and time.monotonic() < deadline:
        # poll() reaps; we want the zombie, so look without waiting.
        break
    time.sleep(0.2)

    signal_group(process.pid, signal.SIGKILL)  # must not raise

    process.wait()
    assert group_alive(process.pid) is False


# --- argument validation --------------------------------------------------------


def test_start_request_requires_an_existing_absolute_directory(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="absolute"):
        StartRequest(brief="x", working_directory=Path("relative/dir"))
    with pytest.raises(ValidationError, match="existing directory"):
        StartRequest(brief="x", working_directory=tmp_path / "missing")
    file = tmp_path / "file"
    file.write_text("")
    with pytest.raises(ValidationError, match="existing directory"):
        StartRequest(brief="x", working_directory=file)


def test_start_request_rejects_blank_brief_bad_tools_and_non_positive_limits(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError):
        StartRequest(brief="   ", working_directory=tmp_path)
    with pytest.raises(ValidationError):
        StartRequest.model_validate(
            {"brief": "x", "working_directory": tmp_path, "tools": ("Task",)}
        )
    for field in ("max_turns", "timeout_seconds", "max_budget_usd"):
        with pytest.raises(ValidationError):
            StartRequest.model_validate(
                {"brief": "x", "working_directory": tmp_path, field: 0}
            )


def test_start_request_fills_defaults_and_keeps_explicit_values(tmp_path: Path) -> None:
    request = StartRequest(
        brief="x",
        working_directory=tmp_path,
        tools=("Bash", "Read", "Bash"),
        max_turns=7,
    )

    spec = request.to_spec("20261008T031500Z-1a2b3c4d", DEFAULTS)

    assert spec.tools == ("Bash", "Read")
    assert spec.max_turns == 7
    assert spec.model == DEFAULTS.model
    assert spec.timeout_seconds == DEFAULTS.timeout_seconds
    assert spec.max_budget_usd == DEFAULTS.max_budget_usd


def test_run_defaults_come_from_config(tmp_path: Path) -> None:
    config_dir = tmp_path / ".config" / "jake-tools"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        '[claude]\nmodel = "m"\neffort = "low"\nmax_turns = 3\n'
        "timeout_seconds = 12.5\nmax_budget_usd = 0.25\n"
    )

    defaults = RunDefaults.from_config(
        load_config(
            {"HOME": str(tmp_path), "XDG_CONFIG_HOME": str(tmp_path / ".config")}
        )
    )

    assert defaults == RunDefaults(
        model="m", effort="low", max_turns=3, timeout_seconds=12.5, max_budget_usd=0.25
    )


# --- the tools over stdio ------------------------------------------------------


async def test_stdio_lists_the_three_claude_tools(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
) -> None:
    async with mcp_server(mcp_env) as session:
        tools = {tool.name: tool for tool in (await session.list_tools()).tools}

    assert {"claude_start", "claude_status", "claude_cancel"} <= set(tools)
    start = tools["claude_start"].inputSchema
    assert start["required"] == ["brief", "working_directory"]
    assert set(start["properties"]) == {
        "brief",
        "working_directory",
        "tools",
        "model",
        "effort",
        "max_turns",
        "timeout_seconds",
        "max_budget_usd",
    }
    assert tools["claude_status"].inputSchema["required"] == ["task_id"]
    assert tools["claude_cancel"].inputSchema["required"] == ["task_id"]
    for name in ("claude_start", "claude_status", "claude_cancel"):
        assert tools[name].outputSchema is None


@pytest.mark.parametrize(
    ("arguments", "argument"),
    [
        ({"brief": "", "working_directory": "/"}, "brief"),
        ({"brief": "x", "working_directory": "relative"}, "working_directory"),
        (
            {"brief": "x", "working_directory": "/definitely/not/here"},
            "working_directory",
        ),
        ({"brief": "x", "working_directory": "/", "tools": ["Task"]}, "tools"),
        ({"brief": "x", "working_directory": "/", "max_turns": 0}, "max_turns"),
        (
            {"brief": "x", "working_directory": "/", "timeout_seconds": -1},
            "timeout_seconds",
        ),
        (
            {"brief": "x", "working_directory": "/", "max_budget_usd": 0},
            "max_budget_usd",
        ),
        ({"brief": "x", "working_directory": "/", "effort": "turbo"}, "effort"),
    ],
)
async def test_stdio_claude_start_rejects_bad_arguments_without_creating_a_run(
    mcp_server: ServerFactory,
    mcp_env: dict[str, str],
    mcp_home: Path,
    arguments: dict[str, object],
    argument: str,
) -> None:
    async with mcp_server(mcp_env) as session:
        result = await session.call_tool("claude_start", arguments)

    assert result.isError is True
    assert result.structuredContent is not None
    assert result.structuredContent["code"] == "invalid_argument"
    assert result.structuredContent["detail"] == {"argument": argument}
    runs_dir = mcp_home / ".local" / "state" / "jake-tools" / "runs"
    assert not runs_dir.exists() or not any(runs_dir.iterdir())


@pytest.mark.parametrize("tool", ["claude_status", "claude_cancel"])
@pytest.mark.parametrize("task_id", ["bogus", "20261008T031500Z-1a2b3c4d", "../x"])
async def test_stdio_status_and_cancel_report_unknown_task(
    mcp_server: ServerFactory, mcp_env: dict[str, str], tool: str, task_id: str
) -> None:
    async with mcp_server(mcp_env) as session:
        result = await session.call_tool(tool, {"task_id": task_id})

    assert result.isError is True
    assert result.structuredContent is not None
    assert result.structuredContent["code"] == "unknown_task"
    assert result.structuredContent["detail"] == {"task_id": task_id}
    text = result.content[0]
    assert isinstance(text, TextContent)
    assert json.loads(text.text) == result.structuredContent


async def test_stdio_status_reconciles_a_dead_group_from_a_fresh_server(
    mcp_server: ServerFactory, mcp_env: dict[str, str], mcp_home: Path
) -> None:
    """A run left ``working`` by a server that is gone reads as worker_died."""
    runs_dir = mcp_home / ".local" / "state" / "jake-tools" / "runs"
    run_dir = _working(runs_dir, mcp_home, dead_group())

    async with mcp_server(mcp_env) as session:
        result = await session.call_tool("claude_status", {"task_id": run_dir.name})

    assert result.isError is False
    state = RunState.model_validate(result.structuredContent)
    assert state.status == "failed"
    assert state.termination_reason == "worker_died"
    assert state.run_dir == run_dir


async def test_stdio_capacity_exceeded_with_two_live_groups(
    mcp_server: ServerFactory,
    mcp_env: dict[str, str],
    mcp_home: Path,
    live_group: Callable[[], int],
) -> None:
    runs_dir = mcp_home / ".local" / "state" / "jake-tools" / "runs"
    for _ in range(MAX_LIVE_TASKS):
        _working(runs_dir, mcp_home, live_group())

    async with mcp_server(mcp_env) as session:
        result = await session.call_tool(
            "claude_start", {"brief": "x", "working_directory": str(mcp_home)}
        )

    assert result.isError is True
    assert result.structuredContent is not None
    assert result.structuredContent["code"] == "capacity_exceeded"
    assert result.structuredContent["detail"] == {
        "live": MAX_LIVE_TASKS,
        "limit": MAX_LIVE_TASKS,
    }
    assert len(list(runs_dir.iterdir())) == MAX_LIVE_TASKS


@pytest.fixture
async def _unused() -> AsyncIterator[None]:  # keeps pytest-asyncio's mode exercised
    yield None
