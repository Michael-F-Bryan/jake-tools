"""clockify_time_entries MCP adapter."""

from __future__ import annotations

import functools

from mcp.server.mcpserver import MCPServer

from ...clockify_time import TimeEntriesResult, time_entries
from ...config import Config
from ..server import structured_tool
from ._clockify import call_clockify


def register(app: MCPServer, config: Config) -> None:
    @structured_tool(
        app,
        name="clockify_time_entries",
        description=(
            "Read authenticated-user entries overlapping timezone-aware [start,end). Scans ALL user history on every call (no lookback/date filtering), preserving original boundaries and running status. limit 1..500; cursor is bound to workspace/user/range/timezone/limit and detects selected-history changes. Running entries evaluated at observed_at, preserved on continuation. Only allowlisted workspaces; defaults to first. Names must resolve or the call fails. Coverage lists upstream pages; concurrent writes are not an atomic snapshot."
            "\n\nBefore using this tool, read `skill://clockify-time` via this server's "
            "`read_resource` capability, unless its contents are already loaded in the "
            "current context. The guide defines workspace scope, pagination, write "
            "verification and uncertain-write recovery rules. Reuse the loaded guide for "
            "subsequent calls."
        ),
    )
    async def clockify_time_entries(
        start: str,
        end: str,
        workspace_id: str | None = None,
        timezone: str = "UTC",
        limit: int = 100,
        cursor: str | None = None,
    ) -> TimeEntriesResult:
        return await call_clockify(
            config,
            workspace_id,
            functools.partial(
                time_entries,
                start=start,
                end=end,
                timezone=timezone,
                limit=limit,
                cursor=cursor,
            ),
        )
