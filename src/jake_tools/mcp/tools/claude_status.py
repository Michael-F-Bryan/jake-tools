"""``claude_status``: the current state of a delegated task, from disk."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ...claude_runs import RunState
from ...claude_runs.control import task_status
from ...claude_runs.rundir import UnknownTaskError
from ...config import Config
from ..errors import ToolError
from ..server import structured_tool


def register(app: FastMCP, config: Config) -> None:
    runs_dir = config.runs_dir.value

    @structured_tool(
        app,
        name="claude_status",
        description=(
            "The state of a delegated task: status (working, completed, failed, "
            "cancelled), timestamps, termination_reason (finished, max_turns, "
            "timeout, budget, cancelled, worker_died, sdk_error), final_text "
            "(the agent's final answer), usage and cost when known, error "
            "detail, and run_dir (read transcript.jsonl there for the full "
            "message stream). `completed` means the run ended normally; it is "
            "not verification of the worker's claims. Works across server "
            "restarts; unknown or malformed IDs return unknown_task."
        ),
    )
    async def claude_status(task_id: str) -> RunState:
        try:
            return task_status(runs_dir, task_id)
        except UnknownTaskError as exc:
            raise unknown_task_error(exc) from exc


def unknown_task_error(exc: UnknownTaskError) -> ToolError:
    return ToolError(
        "unknown_task", "no such delegated task", detail={"task_id": exc.task_id}
    )
