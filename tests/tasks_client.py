"""A minimal MCP Tasks (2026-07-28) client, on ``mcp``'s client extension API.

``mcp`` 2.3 ships no Tasks client, so this is the client half the tests
drive the real server with: a :class:`TasksClient` extension that declares
``io.modelcontextprotocol/tasks`` in every request's envelope and claims
the ``resultType: "task"`` shape, plus the ``tasks/*`` requests. It is a
test client, not production code; it proves the server's wire behaviour,
not any host's.

Results of ``tasks/*`` come back as the raw wire mapping (camelCase keys),
so assertions read exactly what the server sent.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any, Literal

import anyio
import mcp.types as types
from mcp import Client, StdioServerParameters
from mcp.client import ClaimContext, ClientExtension, ResultClaim
from mcp.client.session import ClientSession
from pydantic import BaseModel, ConfigDict

EXTENSION = "io.modelcontextprotocol/tasks"
TERMINAL = frozenset({"completed", "failed", "cancelled"})


class Raw(BaseModel):
    """Any result, kept as the wire sent it."""

    model_config = ConfigDict(extra="allow")


class CreateTaskResult(types.Result):
    """``CreateTaskResult``: ``Result & Task`` with ``resultType: "task"``."""

    result_type: Literal["task"] = "task"
    task_id: str
    status: str
    status_message: str | None = None
    created_at: str
    last_updated_at: str
    ttl_ms: int | None
    poll_interval_ms: int | None = None


class TaskParams(types.RequestParams):
    task_id: str


class UpdateTaskParams(TaskParams):
    input_responses: dict[str, Any]


class GetTaskRequest(types.Request[TaskParams, Literal["tasks/get"]]):
    method: Literal["tasks/get"] = "tasks/get"
    params: TaskParams
    name_param = "taskId"


class UpdateTaskRequest(types.Request[UpdateTaskParams, Literal["tasks/update"]]):
    method: Literal["tasks/update"] = "tasks/update"
    params: UpdateTaskParams
    name_param = "taskId"


class CancelTaskRequest(types.Request[TaskParams, Literal["tasks/cancel"]]):
    method: Literal["tasks/cancel"] = "tasks/cancel"
    params: TaskParams
    name_param = "taskId"


class TasksClient(ClientExtension):
    """Declares the extension; resolves a task by polling it to the end.

    The resolver is what a fixed-shape host does with a task (the spec's
    backwards-compatibility note): ``Client.call_tool`` returns the final
    ``CallToolResult``. It cannot answer questions; a test that needs to
    calls ``session.call_tool(..., allow_claimed=True)`` and drives
    ``tasks/*`` itself.
    """

    identifier = EXTENSION

    def __init__(self, *, max_poll_seconds: float = 0.5) -> None:
        self.max_poll_seconds = max_poll_seconds

    def claims(self) -> Sequence[ResultClaim[Any]]:
        return [
            ResultClaim(
                result_type="task", model=CreateTaskResult, resolve=self._resolve
            )
        ]

    async def _resolve(
        self, created: CreateTaskResult, ctx: ClaimContext
    ) -> types.CallToolResult:
        while True:
            task = await get_task(ctx.session, created.task_id)
            if task["status"] == "completed":
                return types.CallToolResult.model_validate(task["result"])
            if task["status"] in TERMINAL:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=task["status"])],
                    is_error=True,
                )
            if task["status"] == "input_required":
                raise RuntimeError(f"task {created.task_id} needs input")
            await anyio.sleep(
                min(task.get("pollIntervalMs", 1000) / 1000, self.max_poll_seconds)
            )


async def create_task(
    session: ClientSession, tool: str, arguments: Mapping[str, Any]
) -> types.CallToolResult | types.Result:
    """``tools/call`` that hands back a claimed task instead of resolving it."""
    return await session.call_tool(tool, dict(arguments), allow_claimed=True)


async def get_task(session: ClientSession, task_id: str) -> dict[str, Any]:
    result = await session.send_request(
        GetTaskRequest(params=TaskParams(task_id=task_id)), Raw
    )
    return result.model_dump()


async def update_task(
    session: ClientSession, task_id: str, responses: Mapping[str, Any]
) -> dict[str, Any]:
    result = await session.send_request(
        UpdateTaskRequest(
            params=UpdateTaskParams(task_id=task_id, input_responses=dict(responses))
        ),
        Raw,
    )
    return result.model_dump()


async def cancel_task(session: ClientSession, task_id: str) -> dict[str, Any]:
    result = await session.send_request(
        CancelTaskRequest(params=TaskParams(task_id=task_id)), Raw
    )
    return result.model_dump()


async def poll_task(
    session: ClientSession,
    task_id: str,
    until: frozenset[str] | set[str],
    *,
    timeout: float = 60,
    interval: float = 0.2,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Poll ``tasks/get`` until the status is in ``until``; every response seen."""
    seen: list[dict[str, Any]] = []
    with anyio.fail_after(timeout):
        while True:
            task = await get_task(session, task_id)
            seen.append(task)
            if task["status"] in until:
                return task, seen
            await anyio.sleep(interval)


@asynccontextmanager
async def start_modern_client(
    env: Mapping[str, str],
    *,
    declare_tasks: bool = True,
    args: tuple[str, ...] = ("-m", "jake_tools.mcp", "serve"),
) -> AsyncIterator[Client]:
    """The real stdio server, through a 2026-07-28 ``Client``.

    ``declare_tasks=False`` is a modern client that does not declare the
    extension: it must be treated exactly like a legacy one for tasks.
    """
    params = StdioServerParameters(
        command=sys.executable, args=list(args), env=dict(env)
    )
    extensions = [TasksClient()] if declare_tasks else []
    async with Client(params, extensions=extensions) as client:
        assert client.protocol_version == "2026-07-28", client.protocol_version
        yield client
