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
from datetime import UTC, datetime, timedelta
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
from jake_tools.claude_runs.models import TASK_ID_PATTERN, GroupRecord, ProcessRecord
from jake_tools.claude_runs.procs import process_table
from jake_tools.claude_runs.rundir import (
    LOCK_FILE,
    PROCESS_FILE,
    RUNS_LOCK_FILE,
    STATUS_FILE,
    WORKER_LOCK_FILE,
    create_run_dir,
    new_task_id,
    read_process_record,
    read_state,
    task_dir,
    update_state,
    worker_lock_held,
    write_process_record,
    write_text,
)
from jake_tools.claude_runs.spawn import (
    group_alive,
    reap_in_background,
    record_process,
    signal_group,
    terminate_task,
    worker_environment,
)
from jake_tools.config import SECRET_VARIABLES, load_config

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


def _spec(
    task_id: str, working_directory: Path, created_at: datetime | None = None
) -> RunSpec:
    return RunSpec(
        task_id=task_id,
        created_at=created_at or datetime.now(UTC),
        working_directory=working_directory,
        model="claude-haiku-4-5",
        max_turns=3,
        timeout_seconds=30,
        max_budget_usd=0.1,
    )


def _working(runs_dir: Path, working_directory: Path, pid: int | None) -> Path:
    """A run directory whose status says ``working`` with worker ``pid``.

    ``process.json`` is recorded exactly as the server records it after a
    spawn: with the start-time token the process table shows now, or none
    if the process has already gone.
    """
    spec = _spec(new_task_id(), working_directory)
    run_dir = create_run_dir(runs_dir, spec, "do the thing")
    update_state(run_dir, lambda s: s.model_copy(update={"pid": pid, "pgid": pid}))
    if pid is not None:
        record_process(run_dir, pid)
    return run_dir


def _zombie(command: list[str]) -> subprocess.Popen[bytes]:
    """A child in its own session that has exited but not been reaped."""
    process = subprocess.Popen(command, start_new_session=True)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        info = process_table().get(process.pid)
        if info is not None and info.zombie:
            return process
        time.sleep(0.05)
    pytest.fail(f"pid {process.pid} did not become a zombie")


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
    [
        "",
        "nope",
        "../../etc",
        "20261008T031500Z-zzzzzzzz",
        "20261008T031500Z-1a2b3c4d",
        "20261008T031500Z-1a2b3c4d\n",  # ``$`` would accept this; fullmatch does not
    ],
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


def test_a_zombie_only_group_reads_dead_and_is_safe_to_signal() -> None:
    """An exited-but-unreaped process is not alive, however it looks to kill(2)."""
    process = _zombie(["true"])
    try:
        table = process_table()
        assert table[process.pid].zombie
        assert group_alive(process.pid) is False
        record = GroupRecord(
            pgid=process.pid, leader_started=table[process.pid].started
        )
        signal_group(record, signal.SIGKILL, table)  # must not raise
    finally:
        process.wait()


def test_signalling_our_own_group_or_init_is_refused_quietly() -> None:
    table = process_table()
    me = table[os.getpid()]
    leader = table.get(me.pgid)
    own = GroupRecord(
        pgid=me.pgid, leader_started=leader.started if leader else me.started
    )

    assert signal_group(own, signal.SIGKILL, table) is False  # we are still here
    assert (
        signal_group(GroupRecord(pgid=1, leader_started="x"), signal.SIGKILL, table)
        is False
    )
    assert (
        signal_group(GroupRecord(pgid=0, leader_started="x"), signal.SIGKILL, table)
        is False
    )


def test_a_reused_pid_is_neither_alive_nor_signalled(tmp_path: Path) -> None:
    """A recorded PID whose start time no longer matches is someone else's."""
    runs_dir = tmp_path / "runs"
    bystander = subprocess.Popen(
        ["sleep", "300"], start_new_session=True, stdin=subprocess.DEVNULL
    )
    reap_in_background(bystander)
    try:
        run_dir = _working(runs_dir, tmp_path, bystander.pid)
        write_process_record(
            run_dir,
            ProcessRecord(pid=bystander.pid, pgid=bystander.pid, started="older-run"),
        )

        state = reconcile(run_dir)
        time.sleep(0.5)

        assert state.termination_reason == "worker_died"
        assert bystander.poll() is None  # untouched
    finally:
        if bystander.poll() is None:
            os.killpg(bystander.pid, signal.SIGKILL)
        bystander.wait()


async def test_cancel_never_signals_a_reused_pid(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    bystander = subprocess.Popen(
        ["sleep", "300"], start_new_session=True, stdin=subprocess.DEVNULL
    )
    reap_in_background(bystander)
    try:
        run_dir = _working(runs_dir, tmp_path, bystander.pid)
        write_process_record(
            run_dir,
            ProcessRecord(pid=bystander.pid, pgid=bystander.pid, started="older-run"),
        )

        state = await cancel_task(runs_dir, run_dir.name)

        assert state.status == "failed"
        assert state.termination_reason == "worker_died"
        assert bystander.poll() is None
    finally:
        if bystander.poll() is None:
            os.killpg(bystander.pid, signal.SIGKILL)
        bystander.wait()


def test_a_zombie_worker_reads_dead(tmp_path: Path) -> None:
    """Linux containers without a reaping PID 1 leave exited workers as zombies."""
    runs_dir = tmp_path / "runs"
    process = subprocess.Popen(
        ["sleep", "0.3"], start_new_session=True, stdin=subprocess.DEVNULL
    )
    try:
        run_dir = _working(runs_dir, tmp_path, process.pid)  # recorded while alive
        assert read_process_record(run_dir) is not None
        assert read_process_record(run_dir).started is not None  # type: ignore[union-attr]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            info = process_table().get(process.pid)
            if info is not None and info.zombie:
                break
            time.sleep(0.05)
        else:
            pytest.fail("worker stand-in never became a zombie")

        state = reconcile(run_dir)

        assert state.status == "failed"
        assert state.termination_reason == "worker_died"
    finally:
        process.wait()


def test_worker_lock_is_liveness(tmp_path: Path) -> None:
    """A held ``.worker.lock`` means alive; a released one means gone."""
    runs_dir = tmp_path / "runs"
    run_dir = _working(runs_dir, tmp_path, dead_group())
    assert worker_lock_held(run_dir) is False
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, os, sys, time\n"
            "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)\n"
            "fcntl.flock(fd, fcntl.LOCK_EX)\n"
            "print('held', flush=True)\n"
            "time.sleep(300)\n",
            str(run_dir / WORKER_LOCK_FILE),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        assert worker_lock_held(run_dir) is True
        assert reconcile(run_dir).status == "working"  # the lock alone keeps it alive
    finally:
        holder.kill()
        holder.wait()

    assert worker_lock_held(run_dir) is False
    assert reconcile(run_dir).termination_reason == "worker_died"


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


def test_task_without_a_process_record_is_working_until_the_grace_ends(
    tmp_path: Path,
) -> None:
    """No process.json yet: still being spawned (working), unless it is stale."""
    runs_dir = tmp_path / "runs"
    fresh = _spec(new_task_id(), tmp_path)
    create_run_dir(runs_dir, fresh, "brief")
    assert task_status(runs_dir, fresh.task_id).status == "working"

    stale = _spec(new_task_id(), tmp_path, datetime.now(UTC) - timedelta(minutes=5))
    create_run_dir(runs_dir, stale, "brief")
    state = task_status(runs_dir, stale.task_id)

    assert state.status == "failed"
    assert state.termination_reason == "worker_died"
    assert not (runs_dir / stale.task_id / PROCESS_FILE).exists()


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
    before = sorted(p.name for p in runs_dir.iterdir() if not p.name.startswith("."))
    request = StartRequest(brief="x", working_directory=tmp_path)

    with pytest.raises(CapacityExceededError) as info:
        start_task(runs_dir, request, DEFAULTS)

    assert info.value.live == MAX_LIVE_TASKS
    after = sorted(p.name for p in runs_dir.iterdir() if not p.name.startswith("."))
    assert after == before  # only the runs lock file may have appeared


def test_start_time_token_is_the_same_from_a_process_in_another_locale() -> None:
    """The token is compared across processes; a locale must not change it."""
    pid = os.getpid()
    ours = process_table()[pid].started
    script = (
        "import sys\n"
        "from jake_tools.claude_runs.procs import process_table\n"
        "print(process_table()[int(sys.argv[1])].started)\n"
    )
    for locale in ("C", "de_DE.UTF-8", "en_AU.UTF-8"):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("LC_", "LANG"))}
        env["LC_ALL"] = locale
        completed = subprocess.run(
            [sys.executable, "-c", script, str(pid)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert completed.stdout.strip() == ours, locale


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


async def test_terminate_task_returns_quickly_for_a_cooperative_worker(
    tmp_path: Path,
) -> None:
    process = subprocess.Popen(
        ["sleep", "300"], start_new_session=True, stdin=subprocess.DEVNULL
    )
    reap_in_background(process)
    run_dir = _working(tmp_path / "runs", tmp_path, process.pid)
    started = time.monotonic()
    try:
        dead = await terminate_task(run_dir)
    finally:
        process.wait()

    assert dead is True
    assert time.monotonic() - started < 5


async def test_cancel_also_kills_recorded_groups_outside_the_worker_group(
    tmp_path: Path,
) -> None:
    """What the worker recorded in groups.json dies with it, start time matching."""
    runs_dir = tmp_path / "runs"
    worker = subprocess.Popen(
        ["sleep", "300"], start_new_session=True, stdin=subprocess.DEVNULL
    )
    shell = subprocess.Popen(
        ["sleep", "300"], start_new_session=True, stdin=subprocess.DEVNULL
    )
    for process in (worker, shell):
        reap_in_background(process)
    try:
        run_dir = _working(runs_dir, tmp_path, worker.pid)
        table = process_table()
        from jake_tools.claude_runs.models import GroupRecords
        from jake_tools.claude_runs.rundir import write_groups

        write_groups(
            run_dir,
            GroupRecords(
                groups=[
                    GroupRecord(pgid=shell.pid, leader_started=table[shell.pid].started)
                ]
            ),
        )

        state = await cancel_task(runs_dir, run_dir.name)

        assert state.status == "cancelled"
        deadline = time.monotonic() + 5
        while shell.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert shell.poll() is not None
        assert worker.poll() is not None
    finally:
        for process in (worker, shell):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_settling_a_dead_worker_kills_its_recorded_groups(tmp_path: Path) -> None:
    """reconcile() after a SIGKILLed worker still clears what it left running."""
    runs_dir = tmp_path / "runs"
    shell = subprocess.Popen(
        ["sleep", "300"], start_new_session=True, stdin=subprocess.DEVNULL
    )
    reap_in_background(shell)
    try:
        run_dir = _working(runs_dir, tmp_path, dead_group())
        from jake_tools.claude_runs.models import GroupRecords
        from jake_tools.claude_runs.rundir import write_groups

        table = process_table()
        write_groups(
            run_dir,
            GroupRecords(
                groups=[
                    GroupRecord(pgid=shell.pid, leader_started=table[shell.pid].started)
                ]
            ),
        )

        state = reconcile(run_dir)

        assert state.termination_reason == "worker_died"
        deadline = time.monotonic() + 5
        while shell.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert shell.poll() is not None
    finally:
        if shell.poll() is None:
            os.killpg(shell.pid, signal.SIGKILL)
        shell.wait()


# --- the worker's environment ---------------------------------------------------


def test_secret_variables_is_the_complete_list_config_reads() -> None:
    config = load_config({"HOME": "/nonexistent"})

    assert {s.name for s in config.settings() if s.secret} == set(SECRET_VARIABLES)


def test_worker_environment_drops_every_secret_and_its_file_variant() -> None:
    environ = {
        "HOME": "/h",
        "PATH": "/bin",
        "ANTHROPIC_API_KEY": "keep-me",
        "CLAUDE_CODE_OAUTH_TOKEN": "keep-me-too",
        **dict.fromkeys(SECRET_VARIABLES, "s3cret"),
        **{f"{name}_FILE": "/run/secrets/x" for name in SECRET_VARIABLES},
    }

    env = worker_environment(environ)

    assert env == {
        "HOME": "/h",
        "PATH": "/bin",
        "ANTHROPIC_API_KEY": "keep-me",
        "CLAUDE_CODE_OAUTH_TOKEN": "keep-me-too",
    }


def test_a_child_spawned_with_the_worker_environment_cannot_see_secrets() -> None:
    environ = {
        **os.environ,
        "CLOCKIFY_API_KEY": "s3cret",
        "JIRA_API_TOKEN_FILE": "/run/secrets/jira",
    }

    completed = subprocess.run(
        [sys.executable, "-c", "import json, os; print(json.dumps(dict(os.environ)))"],
        env=worker_environment(environ),
        capture_output=True,
        text=True,
        check=True,
    )

    seen = json.loads(completed.stdout)
    assert not any(k.startswith(tuple(SECRET_VARIABLES)) for k in seen), sorted(seen)
    assert seen["PATH"] == os.environ["PATH"]


# --- a real worker, briefly -----------------------------------------------------


def test_real_worker_holds_the_lock_records_itself_and_fails_cleanly(
    tmp_path: Path,
) -> None:
    """Spawn the real worker against a working directory that vanishes.

    The SDK refuses to start the CLI (no API call is made), so the whole
    lifecycle runs in a second or two: process.json recorded by the server,
    the lock held while the worker runs, ``failed/sdk_error`` written by the
    worker, the lock released on exit.
    """
    runs_dir = tmp_path / "runs"
    workdir = tmp_path / "gone-soon"
    workdir.mkdir()
    request = StartRequest(brief="hello", working_directory=workdir)
    workdir.rmdir()

    started = start_task(runs_dir, request, DEFAULTS)
    run_dir = started.run_dir
    record = read_process_record(run_dir)
    assert record is not None and record.started is not None
    assert (runs_dir / RUNS_LOCK_FILE).exists()

    deadline = time.monotonic() + 30
    saw_lock = False
    while time.monotonic() < deadline:
        if worker_lock_held(run_dir):
            saw_lock = True
        state = read_state(run_dir)
        if state.is_terminal:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"worker did not finish: {read_state(run_dir)}")

    assert saw_lock, "the worker never held its lock"
    assert state.status == "failed"
    assert state.termination_reason == "sdk_error"
    assert state.error and "gone-soon" in state.error
    assert state.started_at is not None
    assert (run_dir / "groups.json").exists() or True  # may be absent: nothing spawned
    deadline = time.monotonic() + 5
    while worker_lock_held(run_dir) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert worker_lock_held(run_dir) is False
    assert task_status(runs_dir, started.task_id) == state  # terminal stays put


def test_runs_lock_serialises_servers_sharing_a_runs_dir(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, os, sys, time\n"
            "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)\n"
            "fcntl.flock(fd, fcntl.LOCK_EX)\n"
            "print('held', flush=True)\n"
            "time.sleep(1.5)\n",
            str(runs_dir / RUNS_LOCK_FILE),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        began = time.monotonic()
        assert live_tasks(runs_dir) == []
        waited = time.monotonic() - began
    finally:
        holder.wait()

    assert waited >= 1.0, waited


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
    task_dirs = [p for p in runs_dir.iterdir() if not p.name.startswith(".")]
    assert len(task_dirs) == MAX_LIVE_TASKS  # nothing created; only the lock file


@pytest.fixture
async def _unused() -> AsyncIterator[None]:  # keeps pytest-asyncio's mode exercised
    yield None
