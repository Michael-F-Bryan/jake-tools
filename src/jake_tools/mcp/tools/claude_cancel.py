"""``claude_cancel``: stop a delegated task's whole process tree."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from ...claude_runs import RunState
from ...config import Config
from ..server import structured_tool
from ._claude import task_errors, task_service


def register(app: MCPServer, config: Config) -> None:
    service = task_service(config)

    @structured_tool(
        app,
        name="claude_cancel",
        description=(
            "Cancel a delegated task. A task that already ended is returned "
            "unchanged. Otherwise cancel_requested_at is recorded, its process "
            "groups get SIGTERM, up to 10 s to exit cleanly, then SIGKILL and "
            "up to 5 s more, and the state is recorded as cancelled once every "
            "process is confirmed gone (failed, with an error, if not). "
            "Queued steering messages become not_delivered and pending "
            "questions expired. Returns the same shape as claude_status. "
            "Cancellation never undoes filesystem or external changes the "
            "task already made; inspect before retrying."
            "\n\nBefore using this tool, read `skill://claude-delegation` via this server's "
            "`read_resource` capability, unless its contents are already loaded in the "
            "current context. The guide defines task scope, permissions, supervision, "
            "cancellation and recovery rules. Reuse the loaded guide for subsequent "
            "calls."
        ),
    )
    async def claude_cancel(task_id: str) -> RunState:
        with task_errors():
            return await service.cancel(task_id)
