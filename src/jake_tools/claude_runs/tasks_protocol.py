"""The task service projected onto the MCP Tasks extension.

Spec: ``io.modelcontextprotocol/tasks``, version ``2026-07-28``, fetched
from ``modelcontextprotocol/ext-tasks`` at revision :data:`SPEC_REVISION`
(``specification/2026-07-28/tasks.md``).

Wired to the transport by :mod:`jake_tools.mcp.tasks`, which passes each
request's ``_meta`` (``None`` on a legacy connection, which cannot carry
per-request capabilities). Every function here that returns a native
shape first checks that ``_meta`` declares the extension
(:func:`require_tasks`), so a request without it can never get one. These
functions are written against the same :class:`~.service.TaskService` and
:class:`~.service.TaskView` the ``claude_*`` tools use, so the mapping is
decided once. Subscriptions (``notifications/tasks``) are not implemented.

Mapping (domain -> protocol):

- ``working`` -> ``working``; ``input_required`` -> ``input_required`` with
  ``inputRequests`` keyed by our request IDs (unique for a task's life, as
  the spec requires), each an ``elicitation/create`` form asking for one
  string ``answer`` (``choices`` become ``examples``: suggestions, not an
  enum).
- A finished task -> ``completed`` with the ``CallToolResult`` that
  ``claude_start`` would have returned had it run synchronously:
  ``isError`` false when the worker finished, **true** for every other
  worker outcome (turn, budget or time limit, SDK or resume failure, worker
  death). Those are tool-level results; the spec reserves ``failed`` for
  JSON-RPC errors, and the worker's ``failed`` is not one. ``cancelled`` ->
  ``cancelled``.
- Protocol ``failed`` is never produced: an unreadable task is a JSON-RPC
  ``-32603`` on ``tasks/get`` itself.
- ``tasks/update``: only keyed ``inputResponses``. Keys not currently
  outstanding are ignored without looking at their payload, as the spec
  recommends. For an outstanding key, ``action`` must be ``accept`` (with
  a string ``content.answer``), ``decline`` or ``cancel``; the last two both
  decline (the worker is told no answer is coming). Anything else is
  ``-32602``. The ack is empty.
- ``tasks/cancel``: an acknowledgement of intent. The underlying
  :meth:`~.service.TaskService.cancel` stops the process tree, but the ack
  carries no claim that it did; status reports the outcome.
- ``ttlMs`` is ``null``: this server never purges run directories (Claude
  Code's own session retention is separate and not promised).
  ``pollIntervalMs`` is a suggestion.

Not part of the standard, and not projected: ``claude_poll``'s bounded
wait and cursor-paged events, and ``claude_steer``'s arbitrary messages.
They remain domain extensions on the ordinary tools.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from .inbox import InputResponse
from .rundir import UnknownTaskError
from .service import SUGGESTED_POLL_INTERVAL_MS, TaskService, TaskView

EXTENSION = "io.modelcontextprotocol/tasks"
SPEC_VERSION = "2026-07-28"
SPEC_REVISION = "899c1bbc869567670093face3591b39c28ce3219"
CLIENT_CAPABILITIES_META = "io.modelcontextprotocol/clientCapabilities"

INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
MISSING_REQUIRED_CLIENT_CAPABILITY = -32021


class TasksProtocolError(Exception):
    """A JSON-RPC error a ``tasks/*`` method must return."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_jsonrpc(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return error


def client_declares_tasks(meta: Mapping[str, Any] | None) -> bool:
    """Whether a request's ``params._meta`` declares the extension."""
    if not meta:
        return False
    capabilities = meta.get(CLIENT_CAPABILITIES_META)
    if not isinstance(capabilities, Mapping):
        return False
    extensions = capabilities.get("extensions")
    return isinstance(extensions, Mapping) and EXTENSION in extensions


def require_tasks(meta: Mapping[str, Any] | None) -> None:
    if not client_declares_tasks(meta):
        raise TasksProtocolError(
            MISSING_REQUIRED_CLIENT_CAPABILITY,
            "Missing required client capability",
            {"requiredCapabilities": {"extensions": {EXTENSION: {}}}},
        )


def create_task_result(
    view: TaskView, meta: Mapping[str, Any] | None
) -> dict[str, Any]:
    """The ``CreateTaskResult`` for a started task.

    Call it only once the task is durable (``claude_start`` has returned),
    so ``tasks/get`` resolves. Raises ``-32021`` unless the request declared
    the extension: a non-declaring request gets the ordinary tool result.
    """
    require_tasks(meta)
    return {"resultType": "task", **task_object(view)}


def task_object(view: TaskView) -> dict[str, Any]:
    """The base ``Task`` fields."""
    status = "completed" if view.status == "failed" else view.status
    task: dict[str, Any] = {
        "taskId": view.task_id,
        "status": status,
        "createdAt": _iso(view.created_at),
        "lastUpdatedAt": _iso(view.last_updated_at),
        "ttlMs": None,
        "pollIntervalMs": SUGGESTED_POLL_INTERVAL_MS,
    }
    if view.status_message:
        task["statusMessage"] = view.status_message
    return task


def detailed_task(view: TaskView) -> dict[str, Any]:
    """A ``DetailedTask``: the task plus its status-specific payload."""
    task = task_object(view)
    if view.status == "input_required":
        task["inputRequests"] = {
            request.request_id: {
                "method": "elicitation/create",
                "params": {
                    "mode": "form",
                    "message": request.question,
                    "requestedSchema": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                **(
                                    {"examples": request.choices}
                                    if request.choices
                                    else {}
                                ),
                            }
                        },
                        "required": ["answer"],
                    },
                },
            }
            for request in view.pending_inputs
        }
    elif view.status in ("completed", "failed"):
        task["result"] = call_tool_result(view)
    return task


def call_tool_result(view: TaskView) -> dict[str, Any]:
    """The ``CallToolResult`` a finished task stands for."""
    result = view.result
    assert result is not None, "only a finished task has a result"
    payload = result.model_dump(mode="json")
    text = result.final_text or result.error or view.status_message or ""
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": payload,
        "isError": result.termination_reason != "finished",
    }


def get_task(
    service: TaskService, task_id: str, meta: Mapping[str, Any] | None
) -> dict[str, Any]:
    """``tasks/get``."""
    require_tasks(meta)
    view = _view(service, task_id)
    return {"resultType": "complete", **detailed_task(view)}


def update_task(
    service: TaskService,
    task_id: str,
    input_responses: Mapping[str, Any],
    meta: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """``tasks/update``: apply keyed ``inputResponses``; an empty ack."""
    require_tasks(meta)
    outstanding = {r.request_id for r in _view(service, task_id).pending_inputs}
    responses = {
        key: _input_response(key, response)
        for key, response in input_responses.items()
        if key in outstanding
    }
    if responses:
        # A key settled since the view above reports duplicate/conflict in
        # the domain result; the protocol ignores it, as it would any key
        # that is no longer outstanding.
        service.respond(task_id, responses)
    return {"resultType": "complete"}


async def cancel_task(
    service: TaskService, task_id: str, meta: Mapping[str, Any] | None
) -> dict[str, Any]:
    """``tasks/cancel``: request cancellation; an empty ack (intent only)."""
    require_tasks(meta)
    _view(service, task_id)
    await service.cancel(task_id)
    return {"resultType": "complete"}


def _input_response(key: str, response: Any) -> InputResponse:
    if not isinstance(response, Mapping):
        raise TasksProtocolError(
            INVALID_PARAMS, f"inputResponses[{key!r}] is not an object"
        )
    action = response.get("action")
    if action in ("decline", "cancel"):
        return InputResponse(action="decline")
    if action != "accept":
        raise TasksProtocolError(
            INVALID_PARAMS,
            f"inputResponses[{key!r}].action must be accept, decline or cancel",
        )
    content = response.get("content")
    answer = content.get("answer") if isinstance(content, Mapping) else None
    try:
        return InputResponse(action="accept", answer=answer)
    except ValidationError as exc:
        raise TasksProtocolError(
            INVALID_PARAMS, f"inputResponses[{key!r}] needs content.answer as a string"
        ) from exc


def _view(service: TaskService, task_id: str) -> TaskView:
    try:
        return service.view(task_id)
    except UnknownTaskError as exc:
        raise TasksProtocolError(
            INVALID_PARAMS, "Failed to retrieve task: Task not found"
        ) from exc
    except (OSError, ValidationError) as exc:
        raise TasksProtocolError(
            INTERNAL_ERROR, "Failed to retrieve task: its state could not be read"
        ) from exc


def _iso(value: Any) -> str:
    return value.isoformat().replace("+00:00", "Z")
