"""Clockify-only configuration and safe domain-error translation."""

from __future__ import annotations

import functools
from collections.abc import Callable

import anyio.to_thread
from pydantic import BaseModel

from ...clockify import ClockifyClient, ClockifyError, validate_clockify_id
from ...clockify_time import CreateTimeEntryError
from ...config import Config
from ..errors import ToolError


def clockify_client(config: Config) -> ClockifyClient:
    key = config.clockify_api_key.value
    if not key:
        raise ToolError(
            "missing_credentials",
            "Set CLOCKIFY_API_KEY in the server environment",
            detail={"variables": ["CLOCKIFY_API_KEY"]},
        )
    try:
        return ClockifyClient(api_key=key, base_url=config.clockify_api_base_url.value)
    except ClockifyError as exc:
        raise ToolError(
            "invalid_argument", "Invalid Clockify API configuration"
        ) from exc


async def call_clockify[ResultT: BaseModel](
    config: Config,
    workspace_id: str | None,
    operation: Callable[[ClockifyClient, str], ResultT],
) -> ResultT:
    allowed = config.clockify_workspaces.value
    if not allowed:
        raise ToolError(
            "access_denied",
            "Configure clockify.workspaces (JAKE_TOOLS_CLOCKIFY_WORKSPACES); no workspace is authorised",
        )
    workspace = workspace_id if workspace_id is not None else allowed[0]
    try:
        validate_clockify_id(workspace)
    except ValueError as exc:
        raise ToolError("invalid_argument", str(exc)) from exc
    if workspace not in allowed:
        raise ToolError("access_denied", "Workspace is not in clockify.workspaces")
    client = clockify_client(config)
    try:
        return await anyio.to_thread.run_sync(
            functools.partial(operation, client, workspace)
        )
    except CreateTimeEntryError as exc:
        raise ToolError(
            "upstream_error",
            str(exc),
            detail={
                "integration": "clockify",
                "outcome": exc.outcome,
                "created_entry_id": exc.entry_id,
                "retry_safe": False,
            },
        ) from exc
    except ClockifyError as exc:
        message = exc.safe_summary or str(exc)
        if config.clockify_api_key.value and len(config.clockify_api_key.value) >= 6:
            message = message.replace(
                config.clockify_api_key.value, "[CLOCKIFY_API_KEY]"
            )
        raise ToolError(
            "upstream_error",
            "Clockify error: " + message,
            detail={"integration": "clockify"},
        ) from exc
    except ValueError as exc:
        raise ToolError("invalid_argument", str(exc)) from exc
