"""Real SIGALRM-based timeout behaviour (no mocking of signal/time)."""

from __future__ import annotations

import time

import pytest

from inference_worker.timeouts import StageTimeoutError, enforce_timeout


def test_enforce_timeout_raises_when_budget_is_exceeded():
    with pytest.raises(StageTimeoutError), enforce_timeout(0.1):
        time.sleep(2)


def test_enforce_timeout_allows_work_within_budget():
    with enforce_timeout(2.0):
        time.sleep(0.05)
    # No exception: the alarm must have been cancelled by the context
    # manager's exit, not merely outlived.


def test_enforce_timeout_does_not_leak_a_pending_alarm():
    with pytest.raises(StageTimeoutError), enforce_timeout(0.1):
        time.sleep(2)

    # If the previous alarm wasn't cancelled/cleared, this sleep would be
    # interrupted by a stale SIGALRM.
    time.sleep(0.3)
