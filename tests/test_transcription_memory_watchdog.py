"""Behaviour of `MemoryWatchdog` and the budget it enforces.

`MemoryWatchdog.check_once` — the same check its background poll loop
runs — is exercised directly with an injected `sampler` and `terminate`,
never the real `resource.getrusage`-backed sampler or the real
`os._exit`-based `terminate_over_budget` (which would kill the test
process). This is the seam the class was built around, not a workaround.
"""

from __future__ import annotations

import os

import pytest

from jake_tools.transcription.memory_watchdog import (
    DEFAULT_MEMORY_BUDGET_FRACTION,
    MemoryWatchdog,
    default_memory_budget_bytes,
)

# --- check_once: fires over budget, not under -------------------------------


def test_check_once_terminates_when_sampler_reports_over_budget() -> None:
    terminated: list[str] = []
    watchdog = MemoryWatchdog(
        budget_bytes=1_000,
        sampler=lambda: 1_001,
        terminate=terminated.append,
    )

    fired = watchdog.check_once()

    assert fired is True
    assert len(terminated) == 1
    assert "1,001" in terminated[0]
    assert "1,000" in terminated[0]


def test_check_once_does_not_terminate_when_sampler_reports_under_budget() -> None:
    terminated: list[str] = []
    watchdog = MemoryWatchdog(
        budget_bytes=1_000,
        sampler=lambda: 500,
        terminate=terminated.append,
    )

    fired = watchdog.check_once()

    assert fired is False
    assert terminated == []


def test_check_once_does_not_terminate_exactly_at_budget() -> None:
    """The boundary is inclusive of the budget itself — only strictly over
    it counts as a breach, matching a sampler that reads a genuinely flat
    steady-state process at exactly its configured ceiling."""
    terminated: list[str] = []
    watchdog = MemoryWatchdog(
        budget_bytes=1_000,
        sampler=lambda: 1_000,
        terminate=terminated.append,
    )

    fired = watchdog.check_once()

    assert fired is False
    assert terminated == []


# --- budget derivation -------------------------------------------------------


def test_default_memory_budget_bytes_is_a_fraction_of_physical_ram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_phys_pages = 4_000
    fake_page_size = 4_096
    total_bytes = fake_phys_pages * fake_page_size

    def _fake_sysconf(name: str) -> int:
        return {"SC_PHYS_PAGES": fake_phys_pages, "SC_PAGE_SIZE": fake_page_size}[name]

    monkeypatch.setattr(os, "sysconf", _fake_sysconf)

    budget = default_memory_budget_bytes()

    assert budget == int(total_bytes * DEFAULT_MEMORY_BUDGET_FRACTION)


def test_memory_watchdog_defaults_budget_to_default_memory_budget_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`budget_bytes=None` (the constructor default) must resolve to the
    same ~60%-of-RAM policy `default_memory_budget_bytes()` computes, not a
    separately hard-coded number that could drift from it."""
    monkeypatch.setattr(
        "jake_tools.transcription.memory_watchdog.default_memory_budget_bytes",
        lambda: 12_345,
    )

    watchdog = MemoryWatchdog()

    assert watchdog.budget_bytes == 12_345


def test_memory_watchdog_uses_an_explicit_budget_over_the_default() -> None:
    watchdog = MemoryWatchdog(budget_bytes=999)

    assert watchdog.budget_bytes == 999
