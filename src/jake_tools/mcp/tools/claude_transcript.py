"""``claude_transcript``: page through a task's normalised events."""

from __future__ import annotations

import functools
from typing import Annotated

from anyio import to_thread
from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ...claude_runs.service import (
    TRANSCRIPT_DEFAULT_EVENTS,
    TRANSCRIPT_DEFAULT_TEXT_CHARS,
    TRANSCRIPT_MAX_EVENTS,
    TRANSCRIPT_MAX_TEXT_CHARS,
    TranscriptPage,
)
from ...config import Config
from ..server import structured_tool
from ._claude import GUIDE, task_errors, task_service


def register(app: MCPServer, config: Config) -> None:
    service = task_service(config)

    @structured_tool(
        app,
        name="claude_transcript",
        omit_none=True,
        description=(
            "Page a delegated task's history: its normalised events (narration, "
            "tool calls and results, denials, steering, questions, per-turn "
            "results, the end; never private thinking) from `cursor` (omit for "
            "the first). Pass next_cursor while has_more. Transcript cursors are "
            "not observation tokens. Text is cut to max_chars per event "
            "(text_truncated, text_chars). Read it when a poll's changes point "
            "here or you need the detail; supervision does not need it." + GUIDE
        ),
    )
    async def claude_transcript(
        task_id: str,
        cursor: str | None = None,
        limit: Annotated[
            int, Field(ge=1, le=TRANSCRIPT_MAX_EVENTS)
        ] = TRANSCRIPT_DEFAULT_EVENTS,
        max_text_chars: Annotated[
            int, Field(ge=1, le=TRANSCRIPT_MAX_TEXT_CHARS)
        ] = TRANSCRIPT_DEFAULT_TEXT_CHARS,
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
