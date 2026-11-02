"""``claude_cancel``: stop a delegated task's whole process tree."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from ...claude_runs.service import CancelResult
from ...config import Config
from ..server import structured_tool
from ._claude import GUIDE, task_errors, task_service


def register(app: MCPServer, config: Config) -> None:
    service = task_service(config)

    @structured_tool(
        app,
        name="claude_cancel",
        omit_none=True,
        description=(
            "Cancel a delegated task: SIGTERM to every process group it used, up "
            "to 10 s, then SIGKILL and up to 5 s more. termination says what was "
            "established: confirmed (every process seen gone), unconfirmed "
            "(status failed; error says what survived), already_ended, or "
            "ended_before_cancel (its own outcome stands). Queued steering becomes "
            "not_delivered and pending questions expire. rollback is always none: "
            "inspect what the task changed before retrying." + GUIDE
        ),
    )
    async def claude_cancel(task_id: str) -> CancelResult:
        with task_errors():
            return await service.cancel(task_id)
