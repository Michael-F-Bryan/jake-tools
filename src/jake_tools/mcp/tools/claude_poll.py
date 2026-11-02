"""``claude_poll``: wait (boundedly) for a delegated task to need attention."""

from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ...claude_runs.control import ArgumentError
from ...claude_runs.observe import Observation, PollUntil
from ...claude_runs.service import (
    DEFAULT_WAIT_SECONDS,
    MAX_POLL_TASKS,
    MAX_WAIT_SECONDS,
    MultiObservation,
    PollTask,
)
from ...config import Config
from ..server import structured_tool
from ._claude import GUIDE, task_errors, task_service


def register(app: MCPServer, config: Config) -> None:
    service = task_service(config)

    @structured_tool(
        app,
        name="claude_poll",
        omit_none=True,
        description=(
            "Wait up to wait_seconds for a delegated task to need you, then return "
            "a compact snapshot: status, progress, last_activity_at, running_tool, "
            "pending_questions, changes since your observation_token (grouped "
            "denials, warnings, steering and question changes), and result once it "
            "ended. Returns at once for a pending question or an ended task; "
            "until=attention (default) also for a new denial, a warning or "
            "undeliverable steering; until=progress also for routine activity; "
            "until=terminal only for questions and ends. Pass back "
            "observation_token; reads never consume anything and a bad token "
            "resynchronises. Never pages history: use claude_transcript. "
            "`tasks` (1-16 {task_id, token?}) watches several under one deadline."
            + GUIDE
        ),
    )
    async def claude_poll(
        task_id: str | None = None,
        token: str | None = None,
        wait_seconds: Annotated[
            float, Field(ge=0, le=MAX_WAIT_SECONDS)
        ] = DEFAULT_WAIT_SECONDS,
        until: PollUntil = "attention",
        tasks: Annotated[
            list[PollTask] | None, Field(min_length=1, max_length=MAX_POLL_TASKS)
        ] = None,
    ) -> Observation | MultiObservation:
        with task_errors():
            if tasks is not None:
                if task_id is not None or token is not None:
                    raise ArgumentError("tasks", "use tasks or task_id/token, not both")
                return await service.poll_many(
                    tasks, wait_seconds=wait_seconds, until=until
                )
            if task_id is None:
                raise ArgumentError("task_id", "provide task_id or tasks")
            return await service.poll(
                task_id, token=token, wait_seconds=wait_seconds, until=until
            )
