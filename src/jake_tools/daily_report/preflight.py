from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field


class SessionManifestEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    source: str
    started_at: float
    ended_at: float | None
    model: str | None = None
    parent_session_id: str | None = None
    title: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    cost_status: str | None = None
    cost_source: str | None = None


class SessionManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    target_date: date
    timezone_name: str = Field(serialization_alias="timezone")
    start_epoch: float
    end_epoch: float
    sessions: list[SessionManifestEntry]
    missing_columns: list[str] = Field(default_factory=list)


def day_epoch_bounds(target_date: date, timezone_name: str) -> tuple[float, float]:
    tz = ZoneInfo(timezone_name)
    start = datetime.combine(target_date, time.min, tzinfo=tz)
    end = start + timedelta(days=1)
    return start.timestamp(), end.timestamp()


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _select_expr(columns: set[str], name: str, fallback: str = "NULL") -> str:
    if name in columns:
        return name
    return f"{fallback} AS {name}"


def build_session_manifest(
    state_db: Path,
    target_date: date,
    *,
    timezone_name: str = "Australia/Perth",
) -> SessionManifest:
    start_epoch, end_epoch = day_epoch_bounds(target_date, timezone_name)
    conn = sqlite3.connect(state_db)
    conn.row_factory = sqlite3.Row
    columns = _table_columns(conn, "sessions")
    required = {
        "id",
        "source",
        "started_at",
        "ended_at",
        "model",
        "parent_session_id",
        "title",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "cost_status",
        "cost_source",
    }
    missing = sorted(required - columns)
    select_columns = [
        _select_expr(columns, "id"),
        _select_expr(columns, "source", "'unknown'"),
        _select_expr(columns, "started_at"),
        _select_expr(columns, "ended_at"),
        _select_expr(columns, "model"),
        _select_expr(columns, "parent_session_id"),
        _select_expr(columns, "title"),
        _select_expr(columns, "input_tokens"),
        _select_expr(columns, "output_tokens"),
        _select_expr(columns, "total_tokens"),
        _select_expr(columns, "estimated_cost_usd"),
        _select_expr(columns, "cost_status"),
        _select_expr(columns, "cost_source"),
    ]
    rows = conn.execute(
        f"""
        SELECT {", ".join(select_columns)}
        FROM sessions
        WHERE started_at >= ? AND started_at < ?
        ORDER BY started_at ASC
        """,
        (start_epoch, end_epoch),
    ).fetchall()
    return SessionManifest(
        target_date=target_date,
        timezone_name=timezone_name,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        missing_columns=missing,
        sessions=[SessionManifestEntry.model_validate(dict(row)) for row in rows],
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_session_manifest(path: Path, manifest: SessionManifest) -> None:
    write_json(path, manifest.model_dump(mode="json", by_alias=True))
