"""The MCPServer app: tool, resource and extension registration only.

Tool logic lives in :mod:`jake_tools.operations`, shared with the native
Hermes adapter; :mod:`jake_tools.mcp.tools` registers each operation through
:func:`structured_tool`. Packaged skills
are served as ``skill://<name>`` resources so an agent can fetch the text
matching the running server even if its local copy is stale. The native
MCP Tasks extension (:mod:`jake_tools.mcp.tasks`) is the one extension.
"""

from __future__ import annotations

import functools
import inspect
import logging
import typing
from collections.abc import Awaitable, Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError as FrameworkToolError
from mcp.server.mcpserver.exceptions import UnexpectedToolError
from mcp.server.mcpserver.resources import FileResource
from mcp.types import CallToolResult, InputRequiredResult
from pydantic import BaseModel, ValidationError

from ..config import Config
from .errors import ToolError, error_result, invalid_argument, success_result

SERVER_NAME = "jake-tools"
SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

log = logging.getLogger(__name__)


def build_server(config: Config) -> MCPServer:
    """Assemble the server for one process. Tools read ``config`` by closure."""
    from ..operations._claude import task_service
    from .tasks import TasksExtension
    from .tools import register_all

    app = JakeMCPServer(
        SERVER_NAME,
        instructions=_INSTRUCTIONS,
        version=package_version(),
        extensions=[TasksExtension(task_service(config))],
    )
    register_all(app, config)
    for skill_dir in packaged_skills():
        app.add_resource(
            FileResource(
                uri=f"skill://{skill_dir.name}",
                name=skill_dir.name,
                description=skill_description(skill_dir / "SKILL.md"),
                mime_type="text/markdown",
                path=skill_dir / "SKILL.md",
            )
        )
        for reference in sorted((skill_dir / "references").rglob("*.md")):
            relative = reference.relative_to(skill_dir).as_posix()
            app.add_resource(
                FileResource(
                    uri=f"skill://{skill_dir.name}/{relative}",
                    name=f"{skill_dir.name}/{relative}",
                    description=f"Supporting reference for {skill_dir.name}; read only when needed.",
                    mime_type="text/markdown",
                    path=reference,
                )
            )
    return app


class JakeMCPServer(MCPServer):
    """``MCPServer`` whose argument-validation failures keep the error contract.

    The framework validates a call's arguments against the handler's
    signature before the handler runs, and reports a failure as plain text
    (``Error executing tool <name>: 1 validation error for ...``) with no
    structured payload. Here that failure becomes the same structured
    ``invalid_argument`` result a handler would return, naming the argument.
    """

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: Context | None = None
    ) -> CallToolResult | InputRequiredResult:
        try:
            return await super().call_tool(name, arguments, context)
        except FrameworkToolError as exc:
            cause = exc.__cause__
            if isinstance(exc, UnexpectedToolError) or not isinstance(
                cause, ValidationError
            ):
                raise
            log.info("%s rejected its arguments", name)
            return error_result(invalid_argument(cause))


def skill_description(path: Path) -> str:
    """Read the resource's description from its packaged YAML frontmatter."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        raise ValueError(f"{path}: missing YAML frontmatter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError(f"{path}: unterminated YAML frontmatter") from exc
    metadata = yaml.safe_load("\n".join(lines[1:end]) + "\n")
    description = metadata.get("description") if isinstance(metadata, dict) else None
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{path}: frontmatter needs a non-empty description string")
    return description


def package_version() -> str:
    try:
        return version("jake-tools")
    except PackageNotFoundError:  # pragma: no cover - only outside an install
        return "unknown"


def packaged_skills() -> tuple[Path, ...]:
    """Every ``skills/<name>/SKILL.md`` directory shipped with the package."""
    if not SKILLS_DIR.is_dir():
        return ()
    return tuple(
        sorted(path for path in SKILLS_DIR.iterdir() if (path / "SKILL.md").is_file())
    )


def structured_tool(
    app: MCPServer, *, name: str, description: str, omit_none: bool = False
) -> Callable[
    [Callable[..., Awaitable[BaseModel]]], Callable[..., Awaitable[BaseModel]]
]:
    """Register an async handler that returns a Pydantic model.

    The handler's parameters become the tool's input schema. Its result is
    sent as structured content (and as the same compact JSON text; with
    ``omit_none``, ``None`` fields are left out); a :class:`ToolError` it
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
            return success_result(result, omit_none=omit_none)

        # Resolved against the handler's own module, so a handler may use
        # ``from __future__ import annotations`` and module-level constants
        # in ``Annotated`` bounds.
        signature = inspect.signature(fn, eval_str=True).replace(
            return_annotation=CallToolResult
        )
        wrapper.__signature__ = signature  # type: ignore[attr-defined]
        wrapper.__annotations__ = {
            **typing.get_type_hints(fn, include_extras=True),
            "return": CallToolResult,
        }
        app.tool(name=name, description=description)(wrapper)
        return fn

    return decorator


_INSTRUCTIONS = (
    "jake-tools: Clockify history, catalogue, verified completed-entry creation, Jira reconciliation, bounded Claude delegation, clean-verbatim polishing of a saved transcript through Claude (`polish_transcript`; sends transcript text, never audio) and, when an endpoint is configured, local audio transcription with speaker diarisation (`transcribe`). "
    "Clockify history/catalogue/create require clockify.workspaces allowlisting. Never automatically retry uncertain entry writes. "
    "Every tool returns JSON as structured content. Failures are results with "
    "isError=true and a stable `code` field (missing_credentials, "
    "invalid_argument, access_denied, plan_stale, unknown_task, "
    "capacity_exceeded, worker_failed, request_conflict, session_busy, "
    "resume_unavailable, task_not_running, upstream_error, internal_error). "
    "The packaged skills are served as skill://<name> resources. "
    "A 2026-07-28 client that declares io.modelcontextprotocol/tasks on a "
    "claude_start request gets a native task (tasks/get, tasks/update, "
    "tasks/cancel); every other client gets the ordinary result."
)
