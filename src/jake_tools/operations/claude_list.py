"""``claude_list``: find delegated tasks, newest first, from disk."""

from typing import Annotated

from anyio import to_thread
from pydantic import Field

from ..claude_runs.models import RunStatus
from ..claude_runs.service import LIST_DEFAULT_TASKS, LIST_MAX_TASKS, TaskList
from ..config import Config
from . import Handler, Operation
from ._claude import task_errors, task_service

DESCRIPTION = "List delegated tasks, newest first: task_id, a one-line label from the brief, status, termination_reason, status_message, times, session_id, resumable and handoff_status. Use it to recover task IDs after a restart. Page with `before` (the previous next_before); `limit` 1-100; optional `status` filter. Unreadable run directories are listed in `unreadable`."


def build(config: Config) -> Handler:
    service = task_service(config)

    async def claude_list(
        before: str | None = None,
        limit: Annotated[int, Field(ge=1, le=LIST_MAX_TASKS)] = LIST_DEFAULT_TASKS,
        status: RunStatus | None = None,
    ) -> TaskList:
        with task_errors():
            return await to_thread.run_sync(
                lambda: service.list(before=before, limit=limit, status=status)
            )

    return claude_list


OPERATION = Operation(
    name="claude_list",
    capability="claude",
    skill="claude-delegation",
    description=DESCRIPTION,
    build=build,
    omit_none=True,
)
