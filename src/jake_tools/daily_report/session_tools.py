"""In-process tools granted to the worker-agent lanes.

Hermes stays the daily driver, so the sessions a daily report reviews still
live in ``~/.hermes/state.db``. This exposes a read-only full-text search over
that store as an MCP tool the lane agent can call, instead of handing it a
shell and hoping.

The connection is opened ``mode=ro`` so the lane's read-only contract is
enforced by SQLite rather than by prompt wording.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from claude_agent_sdk import McpSdkServerConfig, create_sdk_mcp_server, tool
from pydantic import BaseModel

SESSION_SERVER_NAME = "sessions"
SESSION_SEARCH_TOOL = f"mcp__{SESSION_SERVER_NAME}__search"

MAX_LIMIT = 50
SNIPPET_TOKENS = 24


class SessionSearchHit(BaseModel):
    session_id: str
    title: str | None = None
    role: str
    timestamp: float
    snippet: str


def search_sessions(
    state_db: Path,
    query: str,
    *,
    start_epoch: float,
    end_epoch: float,
    limit: int = 10,
) -> list[SessionSearchHit]:
    """Full-text search message bodies from sessions started within the day.

    Returns an empty list when the store is missing or the FTS query is
    malformed — a lane should degrade to "no evidence found", not crash.
    """

    if not state_db.exists():
        return []

    bounded = max(1, min(limit, MAX_LIMIT))
    try:
        conn = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
    except sqlite3.Error:
        return []

    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT
                m.session_id AS session_id,
                s.title AS title,
                m.role AS role,
                m.timestamp AS timestamp,
                snippet(messages_fts, 0, '', '', '…', {SNIPPET_TOKENS}) AS snippet
            FROM messages_fts
            JOIN messages m ON m.id = messages_fts.rowid
            JOIN sessions s ON s.id = m.session_id
            WHERE messages_fts MATCH ?
              AND s.started_at >= ? AND s.started_at < ?
            ORDER BY m.timestamp ASC
            LIMIT ?
            """,
            (query, start_epoch, end_epoch, bounded),
        ).fetchall()
    except sqlite3.Error:
        # A bad MATCH expression is user input, not a bug worth failing the lane.
        return []
    finally:
        conn.close()

    return [SessionSearchHit.model_validate(dict(row)) for row in rows]


def session_search_server(
    state_db: Path, *, start_epoch: float, end_epoch: float
) -> McpSdkServerConfig:
    """An MCP server exposing :func:`search_sessions` scoped to one day."""

    @tool(
        "search",
        "Full-text search the bodies of Hermes sessions started on the report's "
        "target date. Returns matching message snippets with their session IDs, "
        "which are the only valid values for cited_session_ids.",
        {"query": str, "limit": int},
    )
    async def search(args: dict[str, Any]) -> dict[str, Any]:
        hits = search_sessions(
            state_db,
            str(args.get("query", "")),
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            limit=int(args.get("limit", 10) or 10),
        )
        payload = [hit.model_dump(mode="json") for hit in hits]
        return {"content": [{"type": "text", "text": json.dumps(payload, indent=2)}]}

    return create_sdk_mcp_server(
        name=SESSION_SERVER_NAME, version="1.0.0", tools=[search]
    )
