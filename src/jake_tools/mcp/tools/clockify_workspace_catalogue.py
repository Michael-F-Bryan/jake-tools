"""clockify_workspace_catalogue MCP adapter."""

from __future__ import annotations

import functools

from mcp.server.mcpserver import MCPServer

from ...clockify_time import WorkspaceCatalogue, workspace_catalogue
from ...config import Config
from ..server import structured_tool
from ._clockify import call_clockify


def register(app: MCPServer, config: Config) -> None:
    @structured_tool(
        app,
        name="clockify_workspace_catalogue",
        description="Read all clients, archived/active projects, and ACTIVE/DONE tasks in an allowlisted workspace (default first). Explicit archived/status flags and coverage; bounded upstream pagination errors rather than silent truncation. No Jira credentials required.",
    )
    async def clockify_workspace_catalogue(
        workspace_id: str | None = None,
    ) -> WorkspaceCatalogue:
        return await call_clockify(
            config, workspace_id, functools.partial(workspace_catalogue)
        )
