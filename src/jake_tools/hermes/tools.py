"""Shared operations as Hermes tools: schema, handler, result encoding.

The schema is the one the MCP server advertises: both come from the
handler's signature through ``mcp``'s ``func_metadata``, which also
validates the arguments Hermes passes. Handlers are synchronous on purpose:
Hermes bridges async plugin handlers with a fixed wait, which a long
transcription would exceed, and the profile's credentials must be read in
the caller's context before any thread hop.

Results are the operation's model as JSON. Failures are
``{"error": message, "code": ..., "detail": ...}`` with the same stable codes
and messages as the MCP server's ``isError`` results; an unexpected exception
is logged and reported as ``internal_error`` without its text.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import anyio
from mcp.server.mcpserver.utilities.func_metadata import FuncMetadata, func_metadata
from pydantic import BaseModel, ValidationError

from ..config import Config
from ..operations import Operation
from ..operations._claude import invalid_argument
from ..operations.errors import ToolError

log = logging.getLogger(__name__)

ConfigFactory = Callable[[], Config]
ToolHandler = Callable[..., str]


def hermes_description(operation: Operation) -> str:
    return (
        f"{operation.description.rstrip()}\n\nBefore using this tool, load the "
        f"`jake-tools:{operation.skill}` skill with skill_view, unless its contents "
        f"are already loaded in the current context. {operation.skill_guide} Reuse "
        "the loaded guide for subsequent calls."
    )


def tool_schema(operation: Operation, metadata: FuncMetadata) -> dict[str, Any]:
    """The OpenAI-style function schema Hermes registers."""
    return {
        "name": operation.name,
        "description": hermes_description(operation),
        "parameters": metadata.arg_model.model_json_schema(by_alias=True),
    }


def signature_metadata(operation: Operation, config: Config) -> FuncMetadata:
    """The argument model of ``operation``'s handler (its signature is static)."""
    return func_metadata(operation.build(config), structured_output=False)


def make_handler(
    operation: Operation, metadata: FuncMetadata, config_factory: ConfigFactory
) -> ToolHandler:
    """A Hermes handler that builds the operation for the calling profile."""

    def handler(args: dict[str, Any] | None = None, **_context: Any) -> str:
        try:
            config = config_factory()
        except Exception as exc:
            # Hermes' profile scoping and Jake's configuration both raise
            # messages written for the user that name settings, not values.
            log.warning(
                "%s: no configuration for this profile: %s", operation.name, exc
            )
            return error_json(
                ToolError(
                    "internal_error",
                    f"{operation.name} could not resolve this profile's configuration: {exc}",
                )
            )
        try:
            arguments = metadata.validate_arguments(dict(args or {}))
            function = operation.build(config)
            result = run_coroutine(lambda: function(**arguments))
        except ToolError as exc:
            log.info("%s failed: %s: %s", operation.name, exc.code, exc.message)
            return error_json(exc)
        except ValidationError as exc:
            return error_json(invalid_argument(exc))
        except Exception:
            log.exception("%s raised an untranslated exception", operation.name)
            return error_json(
                ToolError(
                    "internal_error",
                    f"{operation.name} failed unexpectedly; see the Hermes log",
                )
            )
        return success_json(result)

    handler.__name__ = operation.name
    return handler


def run_coroutine(make: Callable[[], Awaitable[BaseModel]]) -> BaseModel:
    """Run an operation to completion from synchronous code.

    Off any event loop (Hermes' tool threads) it runs here. A caller already
    inside a running loop gets a fresh thread carrying its context variables
    (the profile's secret scope) instead; either way there is no deadline
    beyond the operation's own.
    """

    async def main() -> BaseModel:
        return await make()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return anyio.run(main)
    context = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="jake-tools") as pool:
        return pool.submit(context.run, anyio.run, main).result()


def success_json(result: BaseModel) -> str:
    return json.dumps(result.model_dump(mode="json"), ensure_ascii=False)


def error_json(error: ToolError) -> str:
    payload = error.payload.model_dump(mode="json", exclude_none=True)
    return json.dumps({"error": payload["message"], **payload}, ensure_ascii=False)
