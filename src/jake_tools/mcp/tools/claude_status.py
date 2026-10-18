"""``claude_status``: the current state of a delegated task, from disk."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ...claude_runs import RunState
from ...config import Config
from ..server import structured_tool
from ._claude import task_errors, task_service


def register(app: FastMCP, config: Config) -> None:
    service = task_service(config)

    @structured_tool(
        app,
        name="claude_status",
        description=(
            "A snapshot of a delegated task, returned at once: status "
            "(working, input_required, completed, failed, cancelled), "
            "timestamps, termination_reason (finished, max_turns, timeout, "
            "budget, cancelled, worker_died, sdk_error, resume_failed), "
            "status_message, final_text (the agent's final answer, in full in "
            "final_text_path), session_id, usage and cost when known, error "
            "detail and run_dir. Prefer claude_poll, which waits for change "
            "and returns progress events. `completed` means the run ended "
            "normally; it is not verification of the worker's claims. Works "
            "across server restarts; unknown or malformed IDs return "
            "unknown_task."
        ),
    )
    async def claude_status(task_id: str) -> RunState:
        with task_errors():
            return service.status(task_id)
