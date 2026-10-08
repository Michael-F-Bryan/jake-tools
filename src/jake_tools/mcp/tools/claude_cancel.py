"""``claude_cancel``: stop a delegated task's whole process group."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ...claude_runs import RunState
from ...claude_runs.control import cancel_task
from ...claude_runs.rundir import UnknownTaskError
from ...config import Config
from ..server import structured_tool
from .claude_status import unknown_task_error


def register(app: FastMCP, config: Config) -> None:
    runs_dir = config.runs_dir.value

    @structured_tool(
        app,
        name="claude_cancel",
        description=(
            "Cancel a delegated task. A task that already ended is returned "
            "unchanged. Otherwise its process group gets SIGTERM, up to 10 s to "
            "exit cleanly, then SIGKILL and up to 5 s more, and the state is "
            "recorded as cancelled (or failed if the group could not be "
            "confirmed dead). Returns the same shape as claude_status. "
            "Cancellation never undoes filesystem or external changes the "
            "task already made; inspect before retrying."
        ),
    )
    async def claude_cancel(task_id: str) -> RunState:
        try:
            return await cancel_task(runs_dir, task_id)
        except UnknownTaskError as exc:
            raise unknown_task_error(exc) from exc
