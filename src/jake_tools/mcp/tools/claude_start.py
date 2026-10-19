"""``claude_start``: delegate a bounded Claude task, or resume one."""

from __future__ import annotations

from typing import get_args

from anyio import to_thread
from mcp.server.mcpserver import MCPServer

from ...claude import EffortLevel
from ...claude_runs import BUILTIN_TOOLS, ClaudeStartResult
from ...claude_runs.control import MAX_LIVE_TASKS, StartRequest
from ...config import Config
from ..server import structured_tool
from ._claude import task_errors, task_service


def register(app: MCPServer, config: Config) -> None:
    service = task_service(config)

    @structured_tool(
        app,
        name="claude_start",
        description=(
            "Start a delegated Claude task and return once it is durably "
            "created, with its task_id and Claude session_id. `brief` is the "
            "full task text, context and acceptance criteria (the worker sees "
            "nothing else). `working_directory` must be an absolute path to an "
            "existing directory. `tools` is the explicit built-in grant from "
            f"{', '.join(BUILTIN_TOOLS)}; the default is none, and Bash is a "
            "broad capability, not a sandbox. `allow_questions=true` lets the "
            "worker ask you questions (status input_required; answer with "
            "claude_update); leave it off unless you will watch with "
            "claude_poll, or an unanswered question stalls the task until its "
            "timeout. "
            f"`model`, `effort` ({'|'.join(get_args(EffortLevel))}), `max_turns`, "
            "`timeout_seconds` (wall clock, including time waiting for your "
            "answers) and `max_budget_usd` default to the server's "
            "configuration and are hard limits. `resume_task_id` continues a "
            "finished task's Claude conversation as a NEW task: tools and "
            "limits (and allow_questions) are this call's own (nothing is "
            "inherited), and "
            "working_directory must be the parent's (omit it to use it). "
            "`request_id` (1-128 of [A-Za-z0-9._:-]) makes the call safe to "
            "retry: the same id and request return the original task "
            "(replayed=true); the same id with a different request fails "
            "with request_conflict. At most "
            f"{MAX_LIVE_TASKS} tasks run at once (capacity_exceeded otherwise). "
            "Observe with claude_poll, steer with claude_steer, stop with "
            "claude_cancel."
        ),
    )
    async def claude_start(
        brief: str,
        working_directory: str | None = None,
        tools: list[str] | None = None,
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
