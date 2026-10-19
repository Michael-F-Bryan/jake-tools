"""``claude_update``: answer a delegated task's clarification requests."""

from __future__ import annotations

from typing import Any

from anyio import to_thread
from mcp.server.fastmcp import FastMCP
from pydantic import TypeAdapter

from ...claude_runs.inbox import MAX_ANSWER_CHARS, InputResponse
from ...claude_runs.service import UpdateResult
from ...config import Config
from ..errors import ToolError
from ..server import structured_tool
from ._claude import task_errors, task_service

_RESPONSES = TypeAdapter(dict[str, InputResponse])


def register(app: FastMCP, config: Config) -> None:
    service = task_service(config)

    @structured_tool(
        app,
        name="claude_update",
        description=(
            "Respond to a delegated task's clarification requests (its "
            "pending_inputs, status input_required). `responses` maps each "
            'request_id to {"answer": "..."} or {"action": "decline"}; '
            f"answers up to {MAX_ANSWER_CHARS} chars. You may answer some and "
            "leave others pending. Each key reports accepted, duplicate (the "
            "same response was already applied), conflict (it was settled "
            "differently), not_pending (it expired when the task ended) or "
            "unknown. The worker resumes the same turn with the answer as its "
            "tool result; poll to see it continue. This answers questions "
            "only: it never grants tools or permissions."
        ),
    )
    async def claude_update(
        task_id: str, responses: dict[str, dict[str, Any]]
    ) -> UpdateResult:
        if not responses:
            raise ToolError(
                "invalid_argument",
                "responses: at least one response is required",
                detail={"argument": "responses"},
            )
        with task_errors():
            parsed = _RESPONSES.validate_python(responses)
            return await to_thread.run_sync(service.respond, task_id, parsed)
