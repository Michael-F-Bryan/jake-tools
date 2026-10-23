"""``claude_poll``: wait (boundedly) for a delegated task to change."""

from mcp.server.mcpserver import MCPServer

from ...claude_runs.control import ArgumentError
from ...claude_runs.service import (
    DEFAULT_WAIT_SECONDS,
    MAX_WAIT_SECONDS,
    POLL_DEFAULT_EVENTS,
    POLL_MAX_EVENTS,
    POLL_TEXT_CHARS,
    MultiPollResult,
    PollResult,
    PollTask,
    PollUntil,
)
from ...config import Config
from ..server import structured_tool
from ._claude import task_errors, task_service


def register(app: MCPServer, config: Config) -> None:
    service = task_service(config)

    @structured_tool(
        app,
        name="claude_poll",
        description=(
            "Wait up to `wait_seconds` (0-"
            f"{MAX_WAIT_SECONDS:g}, default {DEFAULT_WAIT_SECONDS:g}) for a "
            "delegated task to have something new, then return its state. It "
            "returns early on new events after `cursor`, on input_required "
            "(answer pending_inputs with claude_update) or when the task ends; "
            "wake_reason says which. The wait is not the task's timeout, and "
            "cancelling a poll never affects the task. Returns status, "
            "termination_reason, status_message (latest progress), "
            "last_activity_at (agent activity, not a heartbeat), usage, "
            "limits (with deadline_at), turns_used, permission_denials "
            "(requested operation, runtime_message), pending_inputs, steering message "
            f"states, up to `limit` (1-{POLL_MAX_EVENTS}, default "
            f"{POLL_DEFAULT_EVENTS}) normalised events (assistant_text, "
            "tool_use, tool_result, turn_result, permission_denied, notice, "
            "message_*, input_*, handoff_*, "
            f"finished; text cut to {POLL_TEXT_CHARS} chars, marked "
            "text_truncated), next_cursor, has_more, and, once ended, result "
            "(final_text, final_text_path, session_id, resumable, usage, "
            "error, last_narration, handoff). Pass next_cursor back to get only newer "
            "events; omit cursor to start from the first. A cursor from "
            "another task, malformed or past the end is invalid_argument. "
            "Alternatively supply `tasks` "
            "(1-16 distinct {task_id, cursor?} objects), not task_id/cursor: "
            "wait under one shared deadline and return {ready: [poll results], "
            "wake_reason: ready|deadline, waited_seconds}. All ready tasks are "
            "returned in request order; ready is empty on timeout. `limit` is "
            "per task; keep cursors for tasks not returned. `until` defaults "
            "to attention (events, input or terminal); terminal ignores "
            "progress/questions and waits for any completed, failed or "
            "cancelled task. Remove terminal tasks after collecting them."
            "\n\nBefore using this tool, read `skill://claude-delegation` via this server's "
            "`read_resource` capability, unless its contents are already loaded in the "
            "current context. The guide defines task scope, permissions, supervision, "
            "cancellation and recovery rules. Reuse the loaded guide for subsequent "
            "calls."
        ),
    )
    async def claude_poll(
        task_id: str | None = None,
        cursor: str | None = None,
        wait_seconds: float = DEFAULT_WAIT_SECONDS,
        limit: int = POLL_DEFAULT_EVENTS,
        tasks: list[PollTask] | None = None,
        until: PollUntil = "attention",
    ) -> PollResult | MultiPollResult:
        with task_errors():
            if tasks is not None:
                if task_id is not None or cursor is not None:
                    raise ArgumentError(
                        "tasks", "use tasks or task_id/cursor, not both"
                    )
                return await service.poll_many(
                    tasks, wait_seconds=wait_seconds, limit=limit, until=until
                )
            if task_id is None:
                raise ArgumentError("task_id", "provide task_id or tasks")
            return await service.poll(
                task_id,
                cursor=cursor,
                wait_seconds=wait_seconds,
                limit=limit,
                until=until,
            )
