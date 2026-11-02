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
from typing import Any

from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel

from ..operations._claude import invalid_argument
from ..operations.errors import ToolError, ToolErrorCode, ToolErrorPayload

__all__ = [
    "ToolError",
    "ToolErrorCode",
    "ToolErrorPayload",
    "error_result",
    "invalid_argument",
    "success_result",
]


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
