from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from jake_tools.daily_report.paths import DailyReportPaths
from jake_tools.daily_report.preflight import (
    build_session_manifest,
    day_epoch_bounds,
    write_session_manifest,
)


def test_daily_report_paths_create_dated_tree(tmp_path: Path) -> None:
    paths = DailyReportPaths.for_date(tmp_path, date(2026, 6, 21)).create()

    assert paths.root == tmp_path / "daily-report-2026-06-21"
    assert paths.report == paths.root / "report.md"
    assert paths.summary == paths.root / "summary.json"
    assert paths.manifest == paths.root / "manifest.json"
    assert paths.lane_events == paths.root / "lane-events.jsonl"
    for directory in [
        paths.subtasks,
        paths.evidence,
        paths.prompts,
        paths.logs,
        paths.drafts,
    ]:
        assert directory.is_dir()


def create_state_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            model TEXT,
            parent_session_id TEXT,
            started_at REAL NOT NULL,
            ended_at REAL,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            estimated_cost_usd REAL,
            cost_status TEXT,
            cost_source TEXT,
            title TEXT
        )
        """
    )
    conn.commit()


def epoch(value: str) -> float:
    return datetime.fromisoformat(value).replace(tzinfo=ZoneInfo("Australia/Perth")).timestamp()


def test_build_session_manifest_uses_epoch_bounds_in_local_timezone(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    create_state_db(db)
    conn = sqlite3.connect(db)
    conn.executemany(
        """
        INSERT INTO sessions (
            id, source, model, parent_session_id, started_at, ended_at,
            input_tokens, output_tokens, total_tokens, estimated_cost_usd,
            cost_status, cost_source, title
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("before", "cli", "m", None, epoch("2026-06-20T23:59:59"), None, 1, 1, 2, 0.1, "ok", "test", "Before"),
            ("inside", "discord", "gpt", "parent", epoch("2026-06-21T08:30:00"), None, 10, 5, 15, 0.2, "ok", "test", "Inside"),
            ("after", "cli", "m", None, epoch("2026-06-22T00:00:00"), None, 1, 1, 2, 0.1, "ok", "test", "After"),
        ],
    )
    conn.commit()

    manifest = build_session_manifest(db, date(2026, 6, 21))

    assert (manifest.start_epoch, manifest.end_epoch) == day_epoch_bounds(
        date(2026, 6, 21), "Australia/Perth"
    )
    assert [session.id for session in manifest.sessions] == ["inside"]
    assert manifest.sessions[0].parent_session_id == "parent"
    assert manifest.sessions[0].total_tokens == 15
    assert manifest.missing_columns == []


def test_build_session_manifest_degrades_when_optional_columns_are_missing(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            started_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
        ("session-1", "cli", epoch("2026-06-21T12:00:00")),
    )
    conn.commit()

    manifest = build_session_manifest(db, date(2026, 6, 21))

    assert [session.id for session in manifest.sessions] == ["session-1"]
    assert manifest.sessions[0].model is None
    assert "model" in manifest.missing_columns
    assert "parent_session_id" in manifest.missing_columns


def test_write_session_manifest_persists_json(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    create_state_db(db)
    manifest = build_session_manifest(db, date(2026, 6, 21))
    output = tmp_path / "evidence" / "session-manifest.json"

    write_session_manifest(output, manifest)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["target_date"] == "2026-06-21"
    assert payload["sessions"] == []
