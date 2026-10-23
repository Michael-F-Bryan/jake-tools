"""``claude_list``: find delegated tasks, newest first, from disk."""

from anyio import to_thread
from mcp.server.mcpserver import MCPServer

from ...claude_runs.models import RunStatus
from ...claude_runs.service import LIST_DEFAULT_TASKS, TaskList
from ...config import Config
from ..server import structured_tool
from ._claude import task_errors, task_service


def register(app: MCPServer, config: Config) -> None:
    service = task_service(config)

    @structured_tool(
        app,
        name="claude_list",
        description=(
            "List delegated tasks, newest first (to the second): task_id, a one-line label "
            "from the brief, status, termination_reason, status_message, "
            "times, session_id, resumable and handoff_status. Use it to "
            "recover task IDs after a restart, then claude_poll for detail. "
            "Pages by `before` (pass the previous page's next_before; null "
            "means nothing older); `limit` 1-100 (default 20); optional "
            "`status` filter. Unreadable run directories are listed in "
            "`unreadable`, not fatal."
            "\n\nBefore using this tool, read `skill://claude-delegation` via this server's "
            "`read_resource` capability, unless its contents are already loaded in the "
            "current context. The guide defines task scope, permissions, supervision, "
            "cancellation and recovery rules. Reuse the loaded guide for subsequent "
            "calls."
        ),
    )
    async def claude_list(
        before: str | None = None,
        limit: int = LIST_DEFAULT_TASKS,
        status: RunStatus | None = None,
    ) -> TaskList:
        with task_errors():
            return await to_thread.run_sync(
                lambda: service.list(before=before, limit=limit, status=status)
            )
