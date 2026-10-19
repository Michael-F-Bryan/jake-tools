"""``claude_transcript``: page through a task's normalised events."""

from __future__ import annotations

import functools

from anyio import to_thread
from mcp.server.mcpserver import MCPServer

from ...claude_runs.service import (
    TRANSCRIPT_DEFAULT_EVENTS,
    TRANSCRIPT_DEFAULT_TEXT_CHARS,
    TRANSCRIPT_MAX_EVENTS,
    TRANSCRIPT_MAX_TEXT_CHARS,
    TranscriptPage,
)
from ...config import Config
from ..server import structured_tool
from ._claude import task_errors, task_service


def register(app: MCPServer, config: Config) -> None:
    service = task_service(config)

    @structured_tool(
        app,
        name="claude_transcript",
        description=(
            "Read a delegated task's normalised event log by cursor, without "
            "parsing raw SDK JSON: the agent's public narration, tool calls "
            "(input summary) and results, steering and clarification events, "
            "per-turn results and the end. Never includes private thinking. "
            f"`limit` 1-{TRANSCRIPT_MAX_EVENTS} (default "
            f"{TRANSCRIPT_DEFAULT_EVENTS}); `max_text_chars` 1-"
            f"{TRANSCRIPT_MAX_TEXT_CHARS} (default {TRANSCRIPT_DEFAULT_TEXT_CHARS}) "
            "per event, with text_truncated/text_chars when cut. Tool results "
            "are stored up to 8000 chars; raw_transcript_path is the full raw "
            "SDK stream, a diagnostic escape hatch. Runs created before event "
            "logs existed report legacy_run=true and no events."
        ),
    )
    async def claude_transcript(
        task_id: str,
        cursor: str | None = None,
        limit: int = TRANSCRIPT_DEFAULT_EVENTS,
        max_text_chars: int = TRANSCRIPT_DEFAULT_TEXT_CHARS,
    ) -> TranscriptPage:
        with task_errors():
            return await to_thread.run_sync(
                functools.partial(
                    service.transcript,
                    task_id,
                    cursor=cursor,
                    limit=limit,
                    max_text_chars=max_text_chars,
                )
            )
