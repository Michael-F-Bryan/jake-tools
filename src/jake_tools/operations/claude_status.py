"""``claude_status``: the current state of a delegated task, from disk."""

from __future__ import annotations

from anyio import to_thread

from ..claude_runs import RunState
from ..config import Config
from . import Handler, Operation
from ._claude import task_errors, task_service

DESCRIPTION = (
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
)


def build(config: Config) -> Handler:
    service = task_service(config)

    async def claude_status(task_id: str) -> RunState:
        with task_errors():
            return await to_thread.run_sync(service.status, task_id)

    return claude_status


OPERATION = Operation(
    name="claude_status",
    capability="claude",
    skill="claude-delegation",
    description=DESCRIPTION,
    build=build,
)
