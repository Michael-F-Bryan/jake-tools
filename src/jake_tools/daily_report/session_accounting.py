from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..ai_usage import AIStageStats, AITotals, Usage, build_ai_totals


class SessionUsageRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    model: str | None = None
    provider: str | None = None
    api_calls: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "SessionUsageRow":
        return cls(
            id=str(row.get("id") or "session"),
            model=_optional_str(row.get("model")),
            provider=_optional_str(row.get("provider")),
            api_calls=_non_negative_int(row.get("api_calls"), default=1),
            input_tokens=_non_negative_int(row.get("input_tokens")),
            output_tokens=_non_negative_int(row.get("output_tokens")),
            cache_read_tokens=_non_negative_int(row.get("cache_read_tokens")),
            cache_write_tokens=_non_negative_int(row.get("cache_write_tokens")),
            reasoning_tokens=_non_negative_int(row.get("reasoning_tokens")),
            prompt_tokens=_non_negative_int(row.get("prompt_tokens")),
            completion_tokens=_non_negative_int(row.get("completion_tokens")),
            total_tokens=_non_negative_int(row.get("total_tokens")),
            estimated_cost_usd=_non_negative_float(row.get("estimated_cost_usd")),
        )


def aggregate_parent_child_from_db(
    session_db: Any, parent_session_id: str
) -> AITotals | None:
    if session_db is None:
        return None
    conn: sqlite3.Connection | None = None
    owns_connection = False
    try:
        if isinstance(session_db, sqlite3.Connection):
            conn = session_db
        else:
            conn = sqlite3.connect(Path(session_db))
            owns_connection = True
        conn.row_factory = sqlite3.Row
        columns = table_columns(conn, "sessions")
        if not {"id", "parent_session_id"}.issubset(columns):
            return None
        select_columns = [
            select_expr(columns, "id"),
            select_expr(columns, "model"),
            select_expr(columns, "provider"),
            select_expr(columns, "api_calls", "1"),
            select_expr(columns, "input_tokens", "0"),
            select_expr(columns, "output_tokens", "0"),
            select_expr(columns, "cache_read_tokens", "0"),
            select_expr(columns, "cache_write_tokens", "0"),
            select_expr(columns, "reasoning_tokens", "0"),
            select_expr(columns, "prompt_tokens", "0"),
            select_expr(columns, "completion_tokens", "0"),
            select_expr(columns, "total_tokens", "0"),
            select_expr(columns, "estimated_cost_usd", "0"),
        ]
        rows = conn.execute(
            f"""
            SELECT {", ".join(select_columns)}
            FROM sessions
            WHERE id = ? OR parent_session_id = ?
            ORDER BY id ASC
            """,
            (parent_session_id, parent_session_id),
        ).fetchall()
        if not rows:
            return None
        return totals_from_session_rows(
            [SessionUsageRow.from_mapping(dict(row)) for row in rows]
        )
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return None
    finally:
        if owns_connection and conn is not None:
            conn.close()


def totals_from_session_rows(rows: Sequence[SessionUsageRow]) -> AITotals:
    stage_stats = [
        AIStageStats(
            stage=row.id,
            usage=Usage(
                model=row.model,
                provider=row.provider,
                api_calls=row.api_calls,
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                cache_read_tokens=row.cache_read_tokens,
                cache_write_tokens=row.cache_write_tokens,
                reasoning_tokens=row.reasoning_tokens,
                prompt_tokens=row.prompt_tokens,
                completion_tokens=row.completion_tokens,
                total_tokens=row.total_tokens,
                estimated_cost_usd=row.estimated_cost_usd,
            ),
        )
        for row in rows
    ]
    return build_ai_totals(stage_stats)


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def select_expr(columns: set[str], name: str, fallback: str = "NULL") -> str:
    if name in columns:
        return name
    return f"{fallback} AS {name}"


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _non_negative_int(value: Any, *, default: int = 0) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else 0


def _non_negative_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed > 0 else 0.0
