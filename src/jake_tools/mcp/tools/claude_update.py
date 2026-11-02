"""``claude_update``: answer a delegated task's clarification requests."""

from typing import Annotated, Literal

from anyio import to_thread
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, ConfigDict, Field

from ...claude_runs.inbox import MAX_ANSWER_CHARS, InputResponse
from ...claude_runs.service import UpdateResult
from ...config import Config
from ..server import structured_tool
from ._claude import GUIDE, task_errors, task_service


class Answer(BaseModel):
    """Answer the question."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=MAX_ANSWER_CHARS, pattern=r"\S")
    action: Literal["accept"] = "accept"


class Decline(BaseModel):
    """Tell the worker no answer is coming."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["decline"]


def register(app: MCPServer, config: Config) -> None:
    service = task_service(config)

    @structured_tool(
        app,
        name="claude_update",
        omit_none=True,
        description=(
            "Respond to a delegated task's pending questions. `responses` maps "
            'each request_id to {"answer": "..."} or {"action": "decline"}; you '
            "may leave others pending. Each key reports accepted, duplicate "
            "(already applied identically), conflict (settled differently), "
            "not_pending (expired) or unknown. The worker continues the same turn "
            "with the answer. It answers questions only: it never grants tools "
            "or permissions." + GUIDE
        ),
    )
    async def claude_update(
        task_id: str,
        responses: Annotated[dict[str, Answer | Decline], Field(min_length=1)],
    ) -> UpdateResult:
        with task_errors():
            parsed = {
                key: InputResponse(action="decline")
                if isinstance(value, Decline)
                else InputResponse(action="accept", answer=value.answer)
                for key, value in responses.items()
            }
            return await to_thread.run_sync(service.respond, task_id, parsed)
