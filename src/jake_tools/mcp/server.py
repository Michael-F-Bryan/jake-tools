"""The FastMCP app: tool and resource registration only.

Tool logic lives in :mod:`jake_tools.mcp.tools`; each module there exposes a
``register(app, config)`` that calls :func:`structured_tool`. Packaged skills
are served as ``skill://<name>`` resources so an agent can fetch the text
matching the running server even if its local copy is stale.
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.resources import FileResource
from mcp.types import CallToolResult
from pydantic import AnyUrl, BaseModel

from ..config import Config
from .errors import ToolError, error_result, success_result

SERVER_NAME = "jake-tools"
SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

log = logging.getLogger(__name__)


def build_server(config: Config) -> FastMCP:
    """Assemble the server for one process. Tools read ``config`` by closure."""
    from .tools import register_all

    app = FastMCP(SERVER_NAME, instructions=_INSTRUCTIONS)
    register_all(app, config)
    for skill_dir in packaged_skills():
        app.add_resource(
            FileResource(
                uri=AnyUrl(f"skill://{skill_dir.name}"),
                name=skill_dir.name,
                description=f"The packaged {skill_dir.name} skill, as installed by install-skills.",
                mime_type="text/markdown",
                path=skill_dir / "SKILL.md",
            )
        )
    return app


def packaged_skills() -> tuple[Path, ...]:
    """Every ``skills/<name>/SKILL.md`` directory shipped with the package."""
    if not SKILLS_DIR.is_dir():
        return ()
    return tuple(
        sorted(path for path in SKILLS_DIR.iterdir() if (path / "SKILL.md").is_file())
    )


def structured_tool(
    app: FastMCP, *, name: str, description: str
) -> Callable[
    [Callable[..., Awaitable[BaseModel]]], Callable[..., Awaitable[BaseModel]]
]:
    """Register an async handler that returns a Pydantic model.

    The handler's parameters become the tool's input schema. Its result is
    sent as structured content (and as JSON text); a :class:`ToolError` it
    raises becomes an ``isError`` result carrying the error payload; any
    other exception is logged with its traceback and reported as
    ``internal_error`` without the exception's text, which could carry a
    value we must not leak.

    No ``outputSchema`` is advertised: an error result's structured content
    is the error payload, which would not validate against a success schema.
    """

    def decorator(
        fn: Callable[..., Awaitable[BaseModel]],
    ) -> Callable[..., Awaitable[BaseModel]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> CallToolResult:
            try:
                result = await fn(*args, **kwargs)
            except ToolError as exc:
                log.info("%s failed: %s: %s", name, exc.code, exc.message)
                return error_result(exc)
            except Exception:
                log.exception("%s raised an untranslated exception", name)
                return error_result(
                    ToolError(
                        "internal_error",
                        f"{name} failed unexpectedly; see the server's stderr log",
                    )
                )
            return success_result(result)

        signature = inspect.signature(fn).replace(return_annotation=CallToolResult)
        wrapper.__signature__ = signature  # type: ignore[attr-defined]
        wrapper.__annotations__ = {**fn.__annotations__, "return": CallToolResult}
        app.tool(name=name, description=description)(wrapper)
        return fn

    return decorator


_INSTRUCTIONS = (
    "jake-tools: Jira to Clockify reconciliation and bounded Claude delegation. "
    "Every tool returns JSON as structured content. Failures are results with "
    "isError=true and a stable `code` field (missing_credentials, "
    "invalid_argument, access_denied, plan_stale, unknown_task, "
    "capacity_exceeded, worker_failed, upstream_error, internal_error). "
    "The packaged skills are served as skill://<name> resources."
)
