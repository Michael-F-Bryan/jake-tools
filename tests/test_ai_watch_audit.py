from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from jake_tools.ai_usage import AITotals
from jake_tools.ai_watch.audit import (
    InvalidSinceError,
    audit_cutoff,
    find_runs,
    parse_since_days,
)
from jake_tools.ai_watch.manifest import write_manifest
from jake_tools.ai_watch.models import AiWatchCommandOptions, RunStatus
from jake_tools.ai_watch.paths import AiWatchPaths


def test_parse_since_days_supports_day_and_week_units() -> None:
    assert parse_since_days("7d") == 7
    assert parse_since_days("2w") == 14


def test_parse_since_days_rejects_unsupported_unit() -> None:
    with pytest.raises(InvalidSinceError):
        parse_since_days("2m")


def test_parse_since_days_rejects_non_numeric_count() -> None:
    with pytest.raises(InvalidSinceError):
        parse_since_days("xd")


def test_audit_cutoff_uses_injected_today() -> None:
    assert audit_cutoff(since="7d", today=date(2026, 7, 30)) == date(2026, 7, 23)
    assert audit_cutoff(since="2w", today=date(2026, 7, 30)) == date(2026, 7, 16)


def test_find_runs_returns_empty_list_when_base_dir_missing(tmp_path: Path) -> None:
    assert find_runs(tmp_path, cutoff=date(2026, 1, 1)) == []


def test_find_runs_excludes_runs_before_cutoff(tmp_path: Path) -> None:
    old_paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 1)).create()
    new_paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 20)).create()
    for paths, target_date in (
        (old_paths, date(2026, 7, 1)),
        (new_paths, date(2026, 7, 20)),
    ):
        write_manifest(
            paths=paths,
            options=AiWatchCommandOptions(target_date=target_date, base_dir=tmp_path),
            status=RunStatus.OK,
            surfaced_count=0,
            speculative_count=0,
            failed_stages=[],
            summary=AITotals(),
        )

    runs = find_runs(tmp_path, cutoff=date(2026, 7, 10))

    assert [run.run_id for run in runs] == ["2026-07-20"]


def test_find_runs_skips_run_dirs_without_a_manifest(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 20)).create()
    assert not paths.manifest.exists()

    assert find_runs(tmp_path, cutoff=date(2026, 1, 1)) == []
