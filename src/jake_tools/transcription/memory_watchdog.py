"""Process-level memory watchdog: a best-effort in-process backstop.

The real-meeting acceptance run this guards against didn't just fail — it
kernel-panicked the machine. A runaway allocation drove this process's
anonymous memory into swap; swap grew to sixteen swapfiles until "LOW swap
space" starved watchdogd, and the *kernel* panicked, not just the process.

**Honest scope**: this is a Python daemon thread sampling `resource.
getrusage` every couple of seconds and calling `os._exit` if the result is
over budget. That only helps if the thread actually gets scheduled with
the GIL free. Under the exact conditions that caused the original panic —
severe swap thrash, or a single blocking libc/Metal allocation call on the
main thread that never releases the GIL — the OS scheduler can starve this
thread indefinitely, so it may never sample or terminate before the kernel
gives up first. This module does **not** reliably prevent a repeat of the
original incident on its own; it is one layer among several, not the
whole answer:

- `mx.set_memory_limit`/`torch.mps.set_per_process_memory_fraction`
  (`asr.py`) are synchronous, in the allocating call itself — they don't
  depend on any thread being scheduled, so they're the more dependable
  defense against the primary single-huge-allocation vector (the failure
  mode this watchdog is weakest against, since Metal's own allocator
  already rejects that case cleanly and gives a catchable exception).
- Until a genuinely reliable bound exists — an out-of-process supervisor
  (a sibling process polling RSS via `psutil`/`ps` and sending `SIGKILL`,
  or an OS-enforced resource limit) — a first real run of any newly
  widened workload should be watched from outside this process too (a
  human or an external script watching swap/RSS, ready to force-kill).
  Building that supervisor was raised and deliberately deferred as a
  follow-up (not an oversight); it is the known gap this module doesn't
  close.

What this watchdog *does* add: coverage for the failure class the
accelerator caps can't see at all — ordinary swappable CPU-side memory
growth (part of what caused the original panic), which never goes through
Metal's allocator. It's real protection for that gap, just not a
guarantee, because it can only run when the scheduler lets it.
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

    **Tuning risk this creates**: because the mark only accumulates, a run
    whose ASR stage legitimately peaked near the budget will start
    diarisation already close to the trip point even though `_run_asr`'s
    own cleanup (`del` + `gc.collect()` + `mx.clear_cache()`) genuinely
    freed that memory — real current usage dropped, but the sampler can't
    see that. The budget therefore behaves more like "60% of the worst
    single stage plus whatever the next stage needs" than "60% of
    steady-state usage," which makes a false-positive kill on a run with
    real headroom left more likely than the raw 60% figure suggests. The
    failure direction is the safe one (over-conservative, not
    under-conservative) — but it's a real tuning consideration if the
    budget ever needs lowering.
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

    A best-effort in-process backstop, not a hard resource limit — see the
    module docstring for what it can and can't guarantee (it can be
    starved of scheduling under the exact swap-thrash conditions it exists
    to catch). Start it before any model loading begins — model loading is
    exactly where the un-chunked-ASR failure this guards against blew
    memory up — and stop it once the risky work is done. `sampler` and
    `terminate` are constructor seams: tests inject fakes for both so they
    can exercise the budget comparison and termination path without
    touching real process memory or actually killing the test process.
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
