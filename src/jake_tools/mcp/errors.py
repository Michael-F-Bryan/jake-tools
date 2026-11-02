"""The MCP server's error model.

Every failure a tool reports is a structured result with ``isError=true`` and
a stable ``code`` field, never a JSON-RPC error (those carry only numbers).
Handlers raise :class:`ToolError`; the registration wrapper in
:mod:`jake_tools.mcp.server` turns it into the result. Messages name the
integration and the variable, never a value, and ``detail`` carries only
what a caller can act on (an offending argument name, a task ID, a digest).
"""

from __future__ import annotations

import json
from typing import Any, Literal

from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, ConfigDict, ValidationError

ToolErrorCode = Literal[
    "missing_credentials",
    "invalid_argument",
    "access_denied",
    "plan_stale",
    "unknown_task",
    "capacity_exceeded",
    "worker_failed",
    "request_conflict",
    "session_busy",
    "resume_unavailable",
    "task_not_running",
    "unsupported_store_schema",
    "acquisition_failed",
    "upstream_error",
    "internal_error",
]
"""The stable error codes.

The first nine come from the design. ``upstream_error`` is for a remote API
(Clockify, Jira) failing after credentials were present, and
``internal_error`` is the wrapper's catch-all for an exception a handler did
not translate; both are additions the design's list left implicit.

The delegated-task tools add four: ``request_conflict`` (a caller key such
as ``request_id`` or ``message_id`` reused with a different payload),
``session_busy`` (the Claude session to resume is still being written),
``resume_unavailable`` (the task cannot be resumed at all) and
``task_not_running`` (the task can no longer take a message).
"""


class ToolErrorPayload(BaseModel):
    """The structured content of an error result."""

    model_config = ConfigDict(frozen=True)

    code: ToolErrorCode
    message: str
    detail: dict[str, Any] | None = None


class ToolError(Exception):
    """Raised by a tool handler; becomes an ``isError`` result with a code."""

    def __init__(
        self,
        code: ToolErrorCode,
        message: str,
        *,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code: ToolErrorCode = code
        self.message = message
        self.detail = detail

    @property
    def payload(self) -> ToolErrorPayload:
        return ToolErrorPayload(
            code=self.code, message=self.message, detail=self.detail
        )


def invalid_argument(exc: ValidationError) -> ToolError:
    """The ``invalid_argument`` error for a Pydantic validation failure."""
    first = exc.errors()[0]
    location = first.get("loc") or ()
    argument = str(location[0]) if location else None
    message = first.get("msg", "invalid argument")
    message = message.removeprefix("Value error, ")
    prefix = f"{argument}: " if argument else ""
    detail = {"argument": argument} if argument else None
    return ToolError("invalid_argument", f"{prefix}{message}", detail=detail)


def compact_json(payload: Any) -> str:
    """JSON with no insignificant whitespace.

    A host that shows the model the text copy (Hermes drops the identical
    structured copy and embeds this text as a JSON string, escaping it) pays
    for every space and newline twice; compact text keeps that minimal.
    """
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def error_result(error: ToolError) -> CallToolResult:
    """The wire form of a :class:`ToolError`."""
    payload = error.payload.model_dump(mode="json", exclude_none=True)
    return CallToolResult(
        content=[TextContent(type="text", text=compact_json(payload))],
        structured_content=payload,
        is_error=True,
    )


def success_result(model: BaseModel, *, omit_none: bool = False) -> CallToolResult:
    """The wire form of a tool's typed result: the same compact JSON in both
    content slots (the MCP backwards-compatible dual emit). ``omit_none``
    drops ``None`` fields, for tools whose contract says an absent field is
    null (every such field is optional in the result model)."""
    payload = model.model_dump(mode="json", exclude_none=omit_none)
    return CallToolResult(
        content=[TextContent(type="text", text=compact_json(payload))],
        structured_content=payload,
        is_error=False,
    )
