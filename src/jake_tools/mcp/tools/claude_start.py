"""``claude_start``: delegate a bounded Claude task, or resume one."""

from __future__ import annotations

from typing import get_args

from anyio import to_thread
from mcp.server.mcpserver import MCPServer

from ...claude import EffortLevel
from ...claude_runs import ClaudeStartResult
from ...claude_runs.control import MAX_LIVE_TASKS, StartRequest
from ...config import Config
from ..server import structured_tool
from ._claude import GUIDE, task_errors, task_service


def register(app: MCPServer, config: Config) -> None:
    service = task_service(config)

    @structured_tool(
        app,
        name="claude_start",
        omit_none=True,
        description=(
            "Start a delegated Claude task (or, with resume_task_id, continue a "
            "finished task's conversation as a new task) and return once it is "
            "durable: task_id, session_id, observation_token for claude_poll, and "
            "config as normalised. `brief` is everything the worker sees besides "
            "working_directory (absolute, existing). `tools` exposes Claude Code "
            "built-ins by name (default none; a name the session lacks fails the "
            "task as tools_unavailable); `allowed_tools` is permission rules that "
            "auto-approve calls, each for a tool in `tools` ('Read', "
            "'Bash(git status)', 'Edit(/src/**)'); omitted approves every exposed "
            "tool, [] approves none. Other calls needing approval are denied. "
            "allow_questions lets the worker ask you (answer with claude_update). "
            f"model, effort ({'|'.join(get_args(EffortLevel))}), max_turns, "
            "timeout_seconds (wall clock) and max_budget_usd default to the "
            "server's configuration and are hard limits; nothing is inherited on "
            "resume. request_id makes retries safe (replayed=true). At most "
            f"{MAX_LIVE_TASKS} tasks run at once (capacity_exceeded)." + GUIDE
        ),
    )
    async def claude_start(
        brief: str,
        working_directory: str | None = None,
        tools: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        model: str | None = None,
        effort: str | None = None,
        max_turns: int | None = None,
        timeout_seconds: float | None = None,
        max_budget_usd: float | None = None,
        resume_task_id: str | None = None,
        request_id: str | None = None,
        allow_questions: bool = False,
    ) -> ClaudeStartResult:
        with task_errors():
            request = StartRequest.model_validate(
                {
                    "brief": brief,
                    "working_directory": working_directory,
                    "tools": tuple(tools or ()),
                    "allowed_tools": (
                        None if allowed_tools is None else tuple(allowed_tools)
                    ),
                    "model": model,
                    "effort": effort,
                    "max_turns": max_turns,
                    "timeout_seconds": timeout_seconds,
                    "max_budget_usd": max_budget_usd,
                    "resume_task_id": resume_task_id,
                    "request_id": request_id,
                    "allow_questions": allow_questions,
                }
            )
            return await to_thread.run_sync(service.start, request)
