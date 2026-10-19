"""The native MCP Tasks extension: ``claude_start`` as a polled task.

``io.modelcontextprotocol/tasks`` 2026-07-28, served through ``mcp``'s
:class:`~mcp.server.extension.Extension` surface: the server advertises the
extension in ``server/discover``, binds ``tasks/get``, ``tasks/update`` and
``tasks/cancel``, and intercepts ``tools/call``. The mapping itself is
:mod:`jake_tools.claude_runs.tasks_protocol`, over the same
:class:`~jake_tools.claude_runs.service.TaskService` the ``claude_*`` tools
call: one task system, two views of it.

A request gets native behaviour only when it is on a 2026-07-28 connection
**and** its own ``_meta`` envelope declares the extension. A legacy
(``initialize``) connection never does, whatever it declared at handshake:
the spec makes the capability per-request. So:

- ``claude_start`` runs exactly as the ordinary tool (same arguments,
  validation and errors). If that request declared the extension and the
  start succeeded, the result is a ``CreateTaskResult`` for the
  already-durable task; otherwise it is the tool's own ``CallToolResult``.
- ``tasks/*`` from a request that did not declare the extension is
  ``-32021`` (missing required client capability).

Steering, bounded-wait polling and transcripts stay on the ordinary tools.
Subscriptions (``notifications/tasks``) are not implemented.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from anyio import to_thread
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.extension import Extension, MethodBinding
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolRequestParams, CallToolResult, RequestParams
from mcp.types.version import MODERN_PROTOCOL_VERSIONS
from pydantic import ValidationError

from ..claude_runs import ClaudeStartResult
from ..claude_runs import tasks_protocol as tp
from ..claude_runs.rundir import UnknownTaskError
from ..claude_runs.service import TaskService

TASK_TOOL = "claude_start"
"""The one tool that may answer with a task."""

log = logging.getLogger(__name__)


class TaskParams(RequestParams):
    """``tasks/get`` and ``tasks/cancel``: ``{taskId}``."""

    task_id: str


class UpdateTaskParams(TaskParams):
    """``tasks/update``: ``{taskId, inputResponses}``."""

    input_responses: dict[str, Any]


class TasksExtension(Extension):
    identifier = tp.EXTENSION

    def __init__(self, service: TaskService) -> None:
        self._service = service

    def methods(self) -> Sequence[MethodBinding]:
        return (
            MethodBinding("tasks/get", TaskParams, self._get),
            MethodBinding("tasks/update", UpdateTaskParams, self._update),
            MethodBinding("tasks/cancel", TaskParams, self._cancel),
        )

    async def intercept_tool_call(
        self,
        params: CallToolRequestParams,
        ctx: ServerRequestContext[Any, Any],
        call_next: CallNext,
    ) -> HandlerResult:
        result = await call_next(ctx)
        meta = request_meta(ctx)
        if (
            params.name != TASK_TOOL
            or not tp.client_declares_tasks(meta)
            or not isinstance(result, CallToolResult)
            or result.is_error
        ):
            return result
        try:
            started = ClaudeStartResult.model_validate(result.structured_content)
            view = await to_thread.run_sync(self._service.view, started.task_id)
        except OSError, ValidationError, UnknownTaskError:
            # The task exists; an ordinary result is still a valid answer to a
            # declaring client, and it names the task.
            log.exception("could not project a started task; returning the tool result")
            return result
        return tp.create_task_result(view, meta)

    async def _get(
        self, ctx: ServerRequestContext[Any, Any], params: TaskParams
    ) -> HandlerResult:
        meta = request_meta(ctx)
        with _jsonrpc_errors():
            return await to_thread.run_sync(
                tp.get_task, self._service, params.task_id, meta
            )

    async def _update(
        self, ctx: ServerRequestContext[Any, Any], params: UpdateTaskParams
    ) -> HandlerResult:
        meta = request_meta(ctx)
        with _jsonrpc_errors():
            return await to_thread.run_sync(
                tp.update_task,
                self._service,
                params.task_id,
                params.input_responses,
                meta,
            )

    async def _cancel(
        self, ctx: ServerRequestContext[Any, Any], params: TaskParams
    ) -> HandlerResult:
        meta = request_meta(ctx)
        with _jsonrpc_errors():
            return await tp.cancel_task(self._service, params.task_id, meta)


def request_meta(ctx: ServerRequestContext[Any, Any]) -> Mapping[str, Any] | None:
    """This request's ``_meta``, if it can carry per-request capabilities.

    Only a 2026-07-28 envelope does; a legacy connection's ``_meta`` is
    treated as declaring nothing.
    """
    if ctx.protocol_version not in MODERN_PROTOCOL_VERSIONS:
        return None
    return ctx.meta


@contextmanager
def _jsonrpc_errors() -> Iterator[None]:
    try:
        yield
    except tp.TasksProtocolError as exc:
        raise MCPError(exc.code, exc.message, exc.data) from exc
