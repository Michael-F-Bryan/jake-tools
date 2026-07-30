"""Wall-clock timeout for blocking, in-process stage calls (ASR, diarisation).

The worker runs stages serially in the main thread of a short-lived
process, so a SIGALRM-based timer is sufficient here — no thread pool or
subprocess cancellation machinery is needed. This is POSIX-only (fine:
the worker targets macOS/Linux). The ffmpeg prepare stage doesn't use
this — it gets a timeout for free from ``subprocess.run(timeout=...)``.
"""

from __future__ import annotations

import signal
from collections.abc import Iterator
from contextlib import contextmanager


class StageTimeoutError(Exception):
    """Raised when a stage exceeds its configured wall-clock budget."""


@contextmanager
def enforce_timeout(seconds: float) -> Iterator[None]:
    def _on_alarm(signum: int, frame: object) -> None:
        raise StageTimeoutError(f"stage exceeded its {seconds}s timeout")

    previous_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
