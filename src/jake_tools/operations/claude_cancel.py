"""``claude_cancel``: stop a delegated task's whole process tree."""

from __future__ import annotations

from ..claude_runs import RunState
from ..config import Config
from . import Handler, Operation
from ._claude import task_errors, task_service

DESCRIPTION = (
    "Cancel a delegated task. A task that already ended is returned "
    "unchanged. Otherwise cancel_requested_at is recorded, its process "
    "groups get SIGTERM, up to 10 s to exit cleanly, then SIGKILL and "
    "up to 5 s more, and the state is recorded as cancelled once every "
    "process is confirmed gone (failed, with an error, if not). "
    "Queued steering messages become not_delivered and pending "
    "questions expired. Returns the same shape as claude_status. "
    "Cancellation never undoes filesystem or external changes the "
    "task already made; inspect before retrying."
)


def build(config: Config) -> Handler:
    service = task_service(config)

    async def claude_cancel(task_id: str) -> RunState:
        with task_errors():
            return await service.cancel(task_id)

    return claude_cancel


OPERATION = Operation(
    name="claude_cancel",
    capability="claude",
    skill="claude-delegation",
    description=DESCRIPTION,
    build=build,
)
