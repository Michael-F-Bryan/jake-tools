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

from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel

from ..operations.errors import ToolError, ToolErrorCode, ToolErrorPayload

__all__ = [
    "ToolError",
    "ToolErrorCode",
    "ToolErrorPayload",
    "error_result",
    "success_result",
]


def error_result(error: ToolError) -> CallToolResult:
    """The wire form of a :class:`ToolError`."""
    payload = error.payload.model_dump(mode="json", exclude_none=True)
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, indent=2))],
        structured_content=payload,
        is_error=True,
    )


def success_result(model: BaseModel) -> CallToolResult:
    """The wire form of a tool's typed result: JSON in both content slots."""
    payload = model.model_dump(mode="json")
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, indent=2))],
        structured_content=payload,
        is_error=False,
    )
