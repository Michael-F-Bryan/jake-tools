"""A snapshot of the process table: parent, group, zombie state, start time.

Everything that decides whether a task's processes are alive, or whether a
recorded PID still names the process it was recorded for, reads one of
these snapshots. Linux is read from ``/proc`` (always present, unlike
``ps`` in slim images); elsewhere ``ps`` is parsed.

``started`` is an opaque per-process token that a reused PID will not
repeat: the start time in clock ticks since boot on Linux, ``ps``'s
``lstart`` elsewhere. Compare it for equality, nothing more.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

PROC = Path("/proc")


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    pgid: int
    zombie: bool
    started: str


ProcessTable = dict[int, ProcessInfo]


def process_table() -> ProcessTable:
    """Every visible process, keyed by PID."""
    if PROC.is_dir():
        return _from_proc()
    return _from_ps()


def descendants(table: ProcessTable, root: int) -> set[int]:
    """Every PID below ``root`` in the parent tree (not ``root`` itself)."""
    children: dict[int, list[int]] = {}
    for info in table.values():
        children.setdefault(info.ppid, []).append(info.pid)
    found: set[int] = set()
    frontier = [root]
    while frontier:
        pid = frontier.pop()
        for child in children.get(pid, ()):
            if child not in found:
                found.add(child)
                frontier.append(child)
    return found


def group_members(table: ProcessTable, pgid: int) -> set[int]:
    """Every PID in ``pgid``, zombies included."""
    return {info.pid for info in table.values() if info.pgid == pgid}


def group_live(table: ProcessTable, pgid: int) -> bool:
    """Whether ``pgid`` has any member that is not a zombie.

    A group of nothing but zombies has nothing left to signal and will
    vanish once something reaps it; it counts as dead.
    """
    return any(info.pgid == pgid and not info.zombie for info in table.values())


def pid_live(table: ProcessTable, pid: int, started: str | None) -> bool:
    """Whether ``pid`` is running, is not a zombie, and is the same process.

    A ``started`` of ``None`` never matches: it was recorded for a process
    that had already gone.
    """
    info = table.get(pid)
    return (
        info is not None
        and not info.zombie
        and started is not None
        and info.started == started
    )


def _from_proc() -> ProcessTable:
    table: ProcessTable = {}
    for entry in PROC.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue  # exited while we were looking
        # ``pid (comm) state ppid pgrp session ... starttime``; comm may hold
        # spaces and parentheses, so split after the last ``)``.
        fields = stat.rsplit(")", 1)[-1].split()
        if len(fields) < 20:
            continue
        pid = int(entry.name)
        table[pid] = ProcessInfo(
            pid=pid,
            ppid=int(fields[1]),
            pgid=int(fields[2]),
            zombie=fields[0] == "Z",
            started=fields[19],
        )
    return table


def _from_ps() -> ProcessTable:
    # ``lstart`` is formatted per locale; the token is compared between
    # processes that may not share one (a hermetic server, a container), so
    # pin it. Everything else about the environment is left alone.
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,pgid=,stat=,lstart="],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "LC_ALL": "C", "LANG": "C"},
    )
    table: ProcessTable = {}
    for line in completed.stdout.splitlines():
        parts = line.split(maxsplit=4)
        if len(parts) < 5:
            continue
        pid, ppid, pgid, stat, started = parts
        try:
            info = ProcessInfo(
                pid=int(pid),
                ppid=int(ppid),
                pgid=int(pgid),
                zombie="Z" in stat,
                started=started.strip(),
            )
        except ValueError:
            continue
        table[info.pid] = info
    return table
