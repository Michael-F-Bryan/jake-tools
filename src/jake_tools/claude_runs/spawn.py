"""Process primitives for the worker: spawn in its own group, reap, kill.

The worker is spawned with ``start_new_session=True``, so its PID is also
its session and process-group ID. Everything the SDK and the CLI start
underneath it stays in that group, which is what makes a task cancellable
as a unit (``killpg``) and lets it outlive the server that spawned it.

Liveness is "does the process group still exist", checked with signal 0.
That is the only thing a later server, which never held the ``Popen``,
can know.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import anyio

from .rundir import STDERR_FILE, STDOUT_FILE, open_private

WORKER_ARGS: tuple[str, ...] = ("-m", "jake_tools.mcp", "worker")

TERM_GRACE_SECONDS = 10.0
KILL_GRACE_SECONDS = 5.0
POLL_INTERVAL_SECONDS = 0.1


def spawn_worker(run_dir: Path) -> subprocess.Popen[bytes]:
    """Start ``python -m jake_tools.mcp worker RUN_DIR`` in its own session.

    stdin is ``/dev/null``; stdout and stderr go to ``worker.stdout`` and
    ``worker.stderr`` in the run directory (mode ``0600``). The interpreter is
    this process's own, so the worker resolves to the same installed package,
    and the environment is passed through unchanged: the process environment
    is the only environment. A daemon thread waits on the child so a worker
    that exits while this server is alive never lingers as a zombie; one that
    outlives the server is adopted and reaped by init.
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


def group_alive(pgid: int | None) -> bool:
    """Whether any process in group ``pgid`` still exists.

    ``None`` and non-positive IDs are never alive: ``killpg(0, ...)`` would
    address *this* process's group, and nothing we record can mean that.

    A process that has exited but not been reaped (a zombie) still counts as
    existing here, which is why the server reaps every worker it spawns (see
    :func:`reap_in_background`) and why anything else that parents a group
    it later checks must do the same.
    """
    if pgid is None or pgid <= 1:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, but belongs to another user
    return True


def signal_group(pgid: int, sig: signal.Signals) -> None:
    """Send ``sig`` to every process in ``pgid``.

    A vanished group (``ESRCH``) is fine. So is ``EPERM``: macOS returns it
    when the group holds only zombies (exited, not yet reaped), and any
    platform returns it for another user's processes; in both cases there
    is nothing we can usefully signal and the liveness poll decides the
    outcome.
    """
    if pgid <= 1 or pgid == os.getpgrp():
        raise ValueError(f"refusing to signal process group {pgid}")
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


def group_members(pgid: int) -> set[int]:
    """The PIDs currently in process group ``pgid``.

    Linux is read from ``/proc`` (always present, unlike ``pgrep`` in slim
    images); elsewhere ``pgrep -g`` is used. Zombies are included on both.
    """
    proc = Path("/proc")
    if proc.is_dir():
        members: set[int] = set()
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text()
            except OSError:
                continue  # exited while we were looking
            # ``pid (comm) state ppid pgrp ...``; comm may contain spaces.
            fields = stat.rsplit(")", 1)[-1].split()
            if len(fields) > 2 and fields[2] == str(pgid):
                members.add(int(entry.name))
        return members
    completed = subprocess.run(
        ["pgrep", "-g", str(pgid)], capture_output=True, text=True, check=False
    )
    return {int(token) for token in completed.stdout.split()}


def sweep_own_group(*, grace: float = 3.0) -> int:
    """Kill every other process left in this process's group; returns how many.

    The worker calls this before recording a terminal state. Normally there
    is nothing to do: the SDK has closed the CLI and the CLI has cleaned up
    its tools. When the CLI had to be SIGKILLed mid-tool, the tool's shell
    and whatever it ran are orphaned but still in our group; this is what
    makes "the task ended" also mean "its processes are gone" without
    waiting for the server's group SIGKILL.
    """
    me = os.getpid()
    pgid = os.getpgid(0)
    others = group_members(pgid) - {me}
    if not others:
        return 0
    for pid in others:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL_SECONDS)
        if not (group_members(pgid) - {me}):
            break
    for pid in group_members(pgid) - {me}:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)
    return len(others)


async def wait_group_dead(
    pgid: int, timeout: float, *, interval: float = POLL_INTERVAL_SECONDS
) -> bool:
    """Poll until the group is gone or ``timeout`` elapses; never blocks the loop."""
    with anyio.move_on_after(timeout):
        while group_alive(pgid):
            await anyio.sleep(interval)
        return True
    return not group_alive(pgid)


async def terminate_group(
    pgid: int,
    *,
    term_grace: float = TERM_GRACE_SECONDS,
    kill_grace: float = KILL_GRACE_SECONDS,
) -> bool:
    """SIGTERM the group, wait, SIGKILL it, wait. True if it is confirmed gone.

    SIGTERM lets the worker cancel its anyio scope and have the SDK close the
    CLI cleanly; SIGKILL is the backstop for a worker that does not get there
    in time.
    """
    signal_group(pgid, signal.SIGTERM)
    if await wait_group_dead(pgid, term_grace):
        return True
    signal_group(pgid, signal.SIGKILL)
    return await wait_group_dead(pgid, kill_grace)
