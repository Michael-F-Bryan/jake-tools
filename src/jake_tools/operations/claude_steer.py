"""``claude_steer``: send a running delegated task a message."""

from __future__ import annotations

from anyio import to_thread

from ..claude_runs.inbox import MAX_MESSAGE_CHARS, SteerSubmission
from ..claude_runs.service import SteerResult
from ..config import Config
from . import Handler, Operation
from ._claude import task_errors, task_service

DESCRIPTION = (
    "Send a running delegated task an extra user message in the same "
    "conversation. `message_id` (1-128 of [A-Za-z0-9._:-]) makes it idempotent "
    "(replayed=true; another message under the same id is request_conflict). "
    "`message` up to "
    f"{MAX_MESSAGE_CHARS}"
    " chars. state: queued, delivered (written to Claude), in_conversation "
    "(added after the running tool call), or not_delivered (reason says why). "
    "None proves the model acted on it. Limits and tools never change; an ended"
    " task is task_not_running."
)


def build(config: Config) -> Handler:
    service = task_service(config)

    async def claude_steer(task_id: str, message: str, message_id: str) -> SteerResult:
        with task_errors():
            submission = SteerSubmission.model_validate(
                {"message": message, "message_id": message_id}
            )
            return await to_thread.run_sync(service.steer, task_id, submission)

    return claude_steer


OPERATION = Operation(
    name="claude_steer",
    capability="claude",
    skill="claude-delegation",
    description=DESCRIPTION,
    build=build,
    omit_none=True,
)
