"""``claude_start``: delegate a bounded, non-interactive Claude task."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from ...claude_runs import BUILTIN_TOOLS, ClaudeStartResult
from ...claude_runs.control import (
    MAX_LIVE_TASKS,
    CapacityExceededError,
    RunDefaults,
    StartRequest,
    WorkerSpawnError,
    start_task,
)
from ...config import Config
from ..errors import ToolError
from ..server import structured_tool


def register(app: FastMCP, config: Config) -> None:
    runs_dir = config.runs_dir.value
    defaults = RunDefaults.from_config(config)

    @structured_tool(
        app,
        name="claude_start",
        description=(
            "Start a delegated Claude task and return immediately with its "
            "task_id. `brief` is the full task text, context and acceptance "
            "criteria (the worker sees nothing else). `working_directory` must "
            "be an absolute path to an existing directory. `tools` is the "
            f"explicit built-in grant from {', '.join(BUILTIN_TOOLS)}; the "
            "default is none, and Bash is a broad capability, not a sandbox. "
            "`model`, `effort` (low|medium|high|xhigh|max), `max_turns`, "
            "`timeout_seconds` and `max_budget_usd` default to the server's "
            "configuration and are hard limits. At most "
            f"{MAX_LIVE_TASKS} tasks run at once (capacity_exceeded otherwise). "
            "Poll with claude_status; stop with claude_cancel."
        ),
    )
    async def claude_start(
        brief: str,
        working_directory: str,
        tools: list[str] | None = None,
        model: str | None = None,
        effort: str | None = None,
        max_turns: int | None = None,
        timeout_seconds: float | None = None,
        max_budget_usd: float | None = None,
    ) -> ClaudeStartResult:
        try:
            request = StartRequest.model_validate(
                {
                    "brief": brief,
                    "working_directory": working_directory,
                    "tools": tuple(tools or ()),
                    "model": model,
                    "effort": effort,
                    "max_turns": max_turns,
                    "timeout_seconds": timeout_seconds,
                    "max_budget_usd": max_budget_usd,
                }
            )
        except ValidationError as exc:
            raise _invalid_argument(exc) from exc
        try:
            return start_task(runs_dir, request, defaults)
        except CapacityExceededError as exc:
            raise ToolError(
                "capacity_exceeded",
                str(exc),
                detail={"live": exc.live, "limit": exc.limit},
            ) from exc
        except WorkerSpawnError as exc:
            raise ToolError("worker_failed", str(exc)) from exc


def _invalid_argument(exc: ValidationError) -> ToolError:
    first = exc.errors()[0]
    location = first.get("loc") or ()
    argument = str(location[0]) if location else None
    message = first.get("msg", "invalid argument")
    message = message.removeprefix("Value error, ")
    prefix = f"{argument}: " if argument else ""
    detail = {"argument": argument} if argument else None
    return ToolError("invalid_argument", f"{prefix}{message}", detail=detail)
