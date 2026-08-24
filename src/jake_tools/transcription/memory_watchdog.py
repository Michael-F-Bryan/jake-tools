"""Process-level memory watchdog: kill this process before the OS does.

The real-meeting acceptance run this guards against didn't just fail — it
kernel-panicked the machine. A runaway allocation drove this process's
anonymous memory into swap; swap grew to sixteen swapfiles until "LOW swap
space" starved watchdogd, and the *kernel* panicked, not just the process.
Metal's own allocator can reject a too-large allocation cleanly (a fast,
catchable exception) — but that only covers wired GPU memory. Some of what
blew up was ordinary swappable CPU-side memory, which Metal's allocator
never sees and can't protect against. Watching this process's own resident
memory and self-terminating before the OS runs out of options is the only
thing that reliably keeps a bad run from taking the whole machine with it.
"""

from __future__ import annotations

import os
import resource
import sys
import threading
from collections.abc import Callable

MemorySampler = Callable[[], int]
"""Returns this process's resident memory in bytes."""

MEMORY_WATCHDOG_EXIT_CODE = 87
"""Distinct from a normal exception exit (1) or a signal-kill exit (128+n,
e.g. 137 for SIGKILL) so a dead run's exit code unambiguously points at
*this* watchdog rather than an ordinary crash or the OS's own OOM killer."""

DEFAULT_MEMORY_BUDGET_FRACTION = 0.6
"""Fraction of total physical RAM the watchdog allows before terminating.

parakeet-mlx, pyannote, and torch all keep caches on top of live tensors,
and the goal is to terminate *this* process while the kernel still has
slack to react — not to use every last byte before swap runs out. 60%
leaves headroom for the OS, other processes, and the gap between samples.
"""

DEFAULT_POLL_INTERVAL_SECONDS = 2.0


def current_rss_bytes() -> int:
    """This process's resident memory high-water mark, in bytes.

    `ru_maxrss` is a high-water mark, not "memory right now" — it never
    decreases within a process's lifetime, even after memory is freed. For
    a watchdog whose only job is "has this process ever gotten dangerously
    large", that's the right semantics, not a bug: memory that peaked and
    was then reclaimed was never actually a threat to the machine. macOS
    reports `ru_maxrss` in bytes (Linux reports kibibytes — irrelevant
    here, since the transcription pipeline this guards is
    macOS/Apple-Silicon-only).
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def default_memory_budget_bytes() -> int:
    """~60% of total physical RAM, read from the OS rather than hard-coded."""
    total_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return int(total_bytes * DEFAULT_MEMORY_BUDGET_FRACTION)


def terminate_over_budget(message: str) -> None:
    """Log why, then hard-kill the process.

    `os._exit` (not `sys.exit`/`raise`) is deliberate: this normally runs
    on a daemon thread, where a raised `SystemExit` would only end that
    thread, not the process — and this is precisely the moment normal
    interpreter cleanup (atexit handlers, GC of the huge tensors that got
    us here) is itself a memory-pressure risk. A hard exit is the safe
    choice here, not a shortcut.
    """
    print(message, file=sys.stderr, flush=True)
    os._exit(MEMORY_WATCHDOG_EXIT_CODE)


class MemoryWatchdog:
    """Daemon-thread poll loop that kills this process if its resident
    memory crosses `budget_bytes`.

    Start it before any model loading begins — model loading is exactly
    where the un-chunked-ASR failure this guards against blew memory up —
    and stop it once the risky work is done. `sampler` and `terminate` are
    constructor seams: tests inject fakes for both so they can exercise the
    budget comparison and termination path without touching real process
    memory or actually killing the test process.
    """

    def __init__(
        self,
        *,
        budget_bytes: int | None = None,
        sampler: MemorySampler = current_rss_bytes,
        terminate: Callable[[str], None] = terminate_over_budget,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None:
        self.budget_bytes = (
            budget_bytes if budget_bytes is not None else default_memory_budget_bytes()
        )
        self._sampler = sampler
        self._terminate = terminate
        self._poll_interval_seconds = poll_interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def check_once(self) -> bool:
        """Sample memory once; terminate (and return True) if over budget.

        This is the same check the poll loop runs on each tick — exposed
        as its own method so tests can drive it directly, without
        threading or waiting on real elapsed time.
        """
        rss = self._sampler()
        if rss <= self.budget_bytes:
            return False
        self._terminate(
            f"memory watchdog: resident memory {rss:,} bytes exceeded the "
            f"{self.budget_bytes:,} byte budget — terminating to avoid a "
            "kernel panic (see jake_tools.transcription.memory_watchdog "
            "module docstring)"
        )
        return True

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._poll_loop, name="memory-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the poll loop to stop and wait for it to actually exit.

        Safe to call even if `start()` was never called or `check_once()`
        already terminated the process on its own — this only reaches
        that unreachable second case if `terminate` was swapped out (as
        tests do), since the real `terminate_over_budget` never returns.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval_seconds + 1.0)

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self._poll_interval_seconds):
            if self.check_once():
                return
