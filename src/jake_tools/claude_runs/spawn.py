"""Process primitives: spawn the worker, find what a task runs, kill it.

The worker is spawned with ``start_new_session=True``, so its PID is also
its session and process-group ID, and it outlives the server that spawned
it. That group is *not* the whole task, though: the CLI starts each tool
shell with ``setsid``, in a session and group of its own, so a ``killpg``
on the worker's group never reaches a running Bash tool. Two things close
that gap:

- The worker polls its descendant tree (:func:`track_descendant_groups`)
  and records every other group it finds, with the group leader's start
  time, in ``groups.json``. On Linux it also makes itself a child subreaper
  (:func:`become_subreaper`) so orphans of a killed CLI stay in its tree
  instead of vanishing to PID 1.
- Every kill (:func:`terminate_task` from the server, :func:`sweep_own_group`
  from the worker, :func:`settle_leftovers` when a dead worker is settled)
  signals the worker's group *and* every recorded group, each only while
  its leader's start time still matches what was recorded, so a PID the OS
  has reused is never signalled.

Liveness is "the worker holds ``.worker.lock``" (:func:`worker_alive`), with
"the recorded PID is running, not a zombie, and started when we recorded
it" covering the instant between the spawn and the lock. A zombie worker
reads as dead.

The worker's environment is the server's minus every secret variable
``config.py`` knows about (:func:`worker_environment`); the CLI's own auth
variables are left alone.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path

import anyio
from anyio import to_thread

from ..config import SECRET_VARIABLES
from .models import GroupRecord, GroupRecords, ProcessRecord
from .procs import (
    ProcessTable,
    descendants,
    group_live,
    group_members,
    pid_live,
    process_table,
)
from .rundir import (
    STDERR_FILE,
    STDOUT_FILE,
    open_private,
    read_groups,
    read_process_record,
    worker_lock_held,
    write_groups,
    write_process_record,
)

log = logging.getLogger(__name__)

WORKER_ARGS: tuple[str, ...] = ("-m", "jake_tools.mcp", "worker")

TERM_GRACE_SECONDS = 10.0
KILL_GRACE_SECONDS = 5.0
POLL_INTERVAL_SECONDS = 0.25
TRACK_INTERVAL_SECONDS = 0.5

PR_SET_CHILD_SUBREAPER = 36


# --- spawning -------------------------------------------------------------------


def worker_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The server's environment minus every secret and its ``_FILE`` variant.

    Derived from :data:`~jake_tools.config.SECRET_VARIABLES`, so a secret
    added there is dropped here without a second list to maintain.
    """
    source = os.environ if environ is None else environ
    dropped = {name for base in SECRET_VARIABLES for name in (base, f"{base}_FILE")}
    return {key: value for key, value in source.items() if key not in dropped}


def spawn_worker(
    run_dir: Path, environ: Mapping[str, str] | None = None
) -> subprocess.Popen[bytes]:
    """Start ``python -m jake_tools.mcp worker RUN_DIR`` in its own session.

    stdin is ``/dev/null``; stdout and stderr go to ``worker.stdout`` and
    ``worker.stderr`` in the run directory (mode ``0600``). The interpreter is
    this process's own, so the worker resolves to the same installed
    package. A daemon thread waits on the child so a worker that exits while
    this server is alive never lingers as a zombie; one that outlives the
    server is adopted by init.
    """
    stdout_fd = open_private(run_dir / STDOUT_FILE)
    try:
        stderr_fd = open_private(run_dir / STDERR_FILE)
    except BaseException:
        os.close(stdout_fd)
        raise
    try:
        process = subprocess.Popen(
            [sys.executable, *WORKER_ARGS, str(run_dir)],
            stdin=subprocess.DEVNULL,
            stdout=stdout_fd,
            stderr=stderr_fd,
            cwd=run_dir,
            env=worker_environment(environ),
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(stdout_fd)
        os.close(stderr_fd)
    reap_in_background(process)
    return process


def reap_in_background(process: subprocess.Popen[bytes]) -> threading.Thread:
    """Wait on ``process`` from a daemon thread so it is reaped when it exits."""
    thread = threading.Thread(
        target=process.wait, name=f"reap-worker-{process.pid}", daemon=True
    )
    thread.start()
    return thread


def record_process(
    run_dir: Path, pid: int, table: ProcessTable | None = None
) -> ProcessRecord:
    """Pin ``pid`` as the task's worker in ``process.json``.

    If the process is already gone, the record carries no start time and
    reads as dead from then on.
    """
    info = (table if table is not None else process_table()).get(pid)
    record = ProcessRecord(
        pid=pid,
        pgid=info.pgid if info is not None else pid,
        started=info.started if info is not None else None,
    )
    write_process_record(run_dir, record)
    return record


def become_subreaper() -> bool:
    """On Linux, adopt orphaned descendants instead of letting PID 1 have them.

    With this set, a tool shell whose CLI parent was SIGKILLed is reparented
    to the worker, stays in its descendant tree, and is reaped by it. Other
    platforms have no equivalent; ``groups.json`` carries them.
    """
    if sys.platform != "linux":
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0
    except OSError, AttributeError:  # pragma: no cover - exotic libc
        return False


# --- liveness --------------------------------------------------------------------


def worker_alive(
    run_dir: Path, record: ProcessRecord | None, table: ProcessTable
) -> bool:
    """Whether the task's worker is still running.

    The lock is the truth once the worker holds it; before that (the few
    hundred milliseconds between the spawn and the worker's first lines)
    the recorded PID has to be running, not a zombie, and the same process
    we recorded.
    """
    if worker_lock_held(run_dir):
        return True
    return record is not None and pid_live(table, record.pid, record.started)


def group_alive(pgid: int | None) -> bool:
    """Whether ``pgid`` has a live (non-zombie) member right now.

    ``None`` and IDs ``<= 1`` are never alive; nothing we record can mean
    init's group or our own. A convenience over :func:`~.procs.group_live`
    for callers without a table in hand.
    """
    if pgid is None or pgid <= 1:
        return False
    return group_live(process_table(), pgid)


def group_pids(pgid: int) -> set[int]:
    """The PIDs currently in ``pgid``, zombies included."""
    return group_members(process_table(), pgid)


def group_is_ours(table: ProcessTable, group: GroupRecord) -> bool:
    """Whether the group's leader is still the process we recorded.

    A group whose leader is gone (or was never seen) cannot be told apart
    from a reused ID, so it is treated as not ours and left alone.
    """
    leader = table.get(group.pgid)
    return (
        leader is not None
        and group.leader_started is not None
        and leader.started == group.leader_started
    )


def task_groups(run_dir: Path) -> list[GroupRecord]:
    """The worker's own group followed by every group it recorded."""
    groups: list[GroupRecord] = []
    record = read_process_record(run_dir)
    if record is not None:
        groups.append(GroupRecord(pgid=record.pgid, leader_started=record.started))
    groups.extend(read_groups(run_dir).groups)
    return groups


def live_groups(table: ProcessTable, groups: list[GroupRecord]) -> list[GroupRecord]:
    """The recorded groups that are still ours and still have a live member."""
    return [
        group
        for group in groups
        if group_is_ours(table, group) and group_live(table, group.pgid)
    ]


def task_dead(run_dir: Path, table: ProcessTable) -> bool:
    return not worker_alive(run_dir, read_process_record(run_dir), table) and not (
        live_groups(table, task_groups(run_dir))
    )


# --- signalling -------------------------------------------------------------------


def signal_group(group: GroupRecord, sig: signal.Signals, table: ProcessTable) -> bool:
    """Send ``sig`` to the group if its leader still matches; True if sent.

    Never signals IDs ``<= 1`` or this process's own group, whatever was
    recorded. ``ESRCH`` (gone) and ``EPERM`` (macOS: only zombies left;
    anywhere: another user's processes) are not errors: there is nothing
    we can usefully signal and the liveness poll decides the outcome.
    """
    if group.pgid <= 1 or group.pgid == os.getpgrp():
        return False
    if not group_is_ours(table, group):
        return False
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(group.pgid, sig)
    return True


def signal_pid(
    pid: int, started: str, sig: signal.Signals, table: ProcessTable
) -> bool:
    """Send ``sig`` to one process if it is still the one we saw; True if sent."""
    if pid <= 1 or pid == os.getpid() or not pid_live(table, pid, started):
        return False
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(pid, sig)
    return True


async def terminate_task(
    run_dir: Path,
    *,
    term_grace: float = TERM_GRACE_SECONDS,
    kill_grace: float = KILL_GRACE_SECONDS,
) -> bool:
    """SIGTERM everything the task runs, wait, SIGKILL it, wait.

    "Everything" is the worker's group plus every group in ``groups.json``,
    re-read on each pass because the worker may record more while it is
    shutting down. SIGTERM lets the worker cancel its anyio scope and have
    the SDK close the CLI cleanly; SIGKILL is the backstop. True when the
    worker and every recorded group are confirmed dead.
    """
    for sig, grace in ((signal.SIGTERM, term_grace), (signal.SIGKILL, kill_grace)):
        table = await to_thread.run_sync(process_table)
        for group in task_groups(run_dir):
            signal_group(group, sig, table)
        if await _wait_task_dead(run_dir, grace):
            return True
    return task_dead(run_dir, await to_thread.run_sync(process_table))


async def _wait_task_dead(run_dir: Path, timeout: float) -> bool:
    with anyio.move_on_after(timeout):
        while True:
            table = await to_thread.run_sync(process_table)
            if task_dead(run_dir, table):
                return True
            await anyio.sleep(POLL_INTERVAL_SECONDS)
    return False


def settle_leftovers(run_dir: Path, *, grace: float = 2.0) -> int:
    """Kill what a dead worker left running; returns how many groups were hit.

    Called by the server when it settles a task as ``worker_died``: the
    worker never got to sweep, so its recorded groups (a tool shell in its
    own session, typically) would otherwise run on. Synchronous and short:
    SIGTERM, up to ``grace`` seconds, SIGKILL.
    """
    table = process_table()
    targets = live_groups(table, task_groups(run_dir))
    if not targets:
        return 0
    for group in targets:
        signal_group(group, signal.SIGTERM, table)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        time.sleep(0.1)
        table = process_table()
        if not live_groups(table, targets):
            return len(targets)
    for group in live_groups(table, targets):
        signal_group(group, signal.SIGKILL, table)
    return len(targets)


def sweep_own_group(run_dir: Path, *, grace: float = 3.0) -> int:
    """Kill every process the task still runs, from inside the worker.

    Called before the worker records a terminal state, so that "the task
    ended" also means "its processes are gone". Targets are this process's
    live descendants (by PID) and every recorded group (by ``killpg``, while
    its leader still matches). Normally there is nothing to do: the SDK has
    closed the CLI and the CLI has cleaned up its tools. When the CLI was
    SIGKILLed mid-tool its shell lives on in its own session; this is what
    reaches it. Returns how many processes and groups were signalled.
    """
    me = os.getpid()
    table = process_table()
    pids = {pid for pid in descendants(table, me) if not table[pid].zombie}
    groups = live_groups(table, read_groups(run_dir).groups)
    if not pids and not groups:
        _reap_adopted_children()
        return 0
    for pid in pids:
        signal_pid(pid, table[pid].started, signal.SIGTERM, table)
    for group in groups:
        signal_group(group, signal.SIGTERM, table)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        time.sleep(0.1)
        _reap_adopted_children()
        table = process_table()
        if not _still_live(table, me, pids) and not live_groups(table, groups):
            return len(pids) + len(groups)
    for pid in _still_live(table, me, pids):
        signal_pid(pid, table[pid].started, signal.SIGKILL, table)
    for group in live_groups(table, groups):
        signal_group(group, signal.SIGKILL, table)
    _reap_adopted_children()
    return len(pids) + len(groups)


def _still_live(table: ProcessTable, me: int, pids: set[int]) -> set[int]:
    live = {pid for pid in descendants(table, me) if not table[pid].zombie}
    return live & pids


def _reap_adopted_children() -> None:
    """Reap children the subreaper setting handed us (Linux only).

    Elsewhere the worker's only child is the CLI, which the SDK reaps
    itself; a stray ``waitpid(-1)`` there could race the SDK's own waiter.
    """
    if sys.platform != "linux":
        return
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


# --- tracking ----------------------------------------------------------------------


async def track_descendant_groups(
    run_dir: Path, *, interval: float = TRACK_INTERVAL_SECONDS
) -> None:
    """Record every other process group in this process's tree, as it appears.

    Runs for the life of the worker. ``groups.json`` is an append-only set:
    once a group has been seen it stays recorded, so a shell that has since
    been orphaned to PID 1 (macOS has no subreaper) can still be reached.
    """
    me = os.getpid()
    my_pgid = os.getpgid(0)
    known = {group.pgid: group for group in read_groups(run_dir).groups}
    while True:
        table = await to_thread.run_sync(process_table)
        changed = False
        for pid in descendants(table, me):
            pgid = table[pid].pgid
            if pgid == my_pgid:
                continue
            leader = table.get(pgid)
            started = leader.started if leader is not None else None
            current = known.get(pgid)
            if current is None or (current.leader_started is None and started):
                known[pgid] = GroupRecord(pgid=pgid, leader_started=started)
                changed = True
        if changed:
            write_groups(run_dir, GroupRecords(groups=list(known.values())))
            log.info("recorded process groups %s", sorted(known))
        await anyio.sleep(interval)
