"""clockify_create_time_entry MCP adapter."""

from __future__ import annotations

import functools

from mcp.server.mcpserver import MCPServer

from ...clockify_time import CreateTimeEntryResult, create_time_entry
from ...config import Config
from ..server import structured_tool
from ._clockify import call_clockify


def register(app: MCPServer, config: Config) -> None:
    @structured_tool(
        app,
        name="clockify_create_time_entry",
        description=(
            "Create one COMPLETED entry for the authenticated user. start/end must be aware, end>start. description and billable required; task requires and must belong to project. Allowlisted workspace only (default first). POST once, then GET and verify all fields. Error outcome distinguishes post_uncertain, verification_unavailable, verification_failed and includes known created_entry_id. NEVER automatically retry an uncertain write; inspect history first."
            "\n\nBefore using this tool, read `skill://clockify-time` via this server's "
            "`read_resource` capability, unless its contents are already loaded in the "
            "current context. The guide defines workspace scope, pagination, write "
            "verification and uncertain-write recovery rules. Reuse the loaded guide for "
            "subsequent calls."
        ),
    )
    async def clockify_create_time_entry(
        start: str,
        end: str,
        description: str,
        billable: bool,
        workspace_id: str | None = None,
        project_id: str | None = None,
        task_id: str | None = None,
    ) -> CreateTimeEntryResult:
        return await call_clockify(
            config,
            workspace_id,
            functools.partial(
                create_time_entry,
                start=start,
                end=end,
                description=description,
                billable=billable,
                project_id=project_id,
                task_id=task_id,
            ),
        )
