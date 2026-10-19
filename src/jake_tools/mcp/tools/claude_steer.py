"""``claude_steer``: send a running delegated task a message."""

from __future__ import annotations

from anyio import to_thread
from mcp.server.mcpserver import MCPServer

from ...claude_runs.inbox import MAX_MESSAGE_CHARS, SteerSubmission
from ...claude_runs.service import SteerResult
from ...config import Config
from ..server import structured_tool
from ._claude import task_errors, task_service


def register(app: MCPServer, config: Config) -> None:
    service = task_service(config)

    @structured_tool(
        app,
        name="claude_steer",
        description=(
            "Send a running delegated task an extra user message (a "
            "correction or more context) in the same Claude conversation. "
            "`message_id` (1-128 of [A-Za-z0-9._:-], your choice) makes it "
            "idempotent: retrying the same id and message returns its current "
            "state (replayed=true); the same id with another message is "
            f"request_conflict. `message` up to {MAX_MESSAGE_CHARS} chars. "
            "state is queued (stored, not yet sent), delivered (written to "
            "Claude's input), in_conversation (Claude Code added it to the "
            "conversation; mid-turn messages land after the running tool call "
            "finishes, which is never interrupted) or not_delivered (the task "
            "ended first; see reason). None of these proves the model acted "
            "on it: check its later events and output. Limits and tools never "
            "change. Track progress with claude_poll. An ended task returns "
            "task_not_running."
        ),
    )
    async def claude_steer(task_id: str, message: str, message_id: str) -> SteerResult:
        with task_errors():
            submission = SteerSubmission.model_validate(
                {"message": message, "message_id": message_id}
            )
            return await to_thread.run_sync(service.steer, task_id, submission)
