"""``claude_status``: a delegated task's detail, or its result text in chunks."""

import functools
from typing import Annotated, Literal

from anyio import to_thread
from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ...claude_runs.service import (
    RESULT_CHUNK_DEFAULT,
    RESULT_CHUNK_MAX,
    ResultPart,
    ResultText,
    TaskView,
)
from ...config import Config
from ..server import structured_tool
from ._claude import GUIDE, task_errors, task_service


def register(app: MCPServer, config: Config) -> None:
    service = task_service(config)

    @structured_tool(
        app,
        name="claude_status",
        omit_none=True,
        description=(
            "A delegated task's detail now, without waiting: configuration and "
            "limits (requested, normalised, and exposed_tools as the runtime "
            "reported them), usage and cost, permission denials, steering "
            "messages, pending questions and the result (final text excerpt, "
            "diagnostics). view=result returns only result text, in chunks: "
            "part final_text|handoff|error from `offset`, up to max_chars; follow "
            "next_offset until it is absent. `completed` means the run ended "
            "normally, not that its claims are verified." + GUIDE
        ),
    )
    async def claude_status(
        task_id: str,
        view: Literal["detail", "result"] = "detail",
        part: ResultPart = "final_text",
        offset: Annotated[int, Field(ge=0)] = 0,
        max_chars: Annotated[
            int, Field(ge=1, le=RESULT_CHUNK_MAX)
        ] = RESULT_CHUNK_DEFAULT,
    ) -> TaskView | ResultText:
        with task_errors():
            if view == "result":
                return await to_thread.run_sync(
                    functools.partial(
                        service.result_text,
                        task_id,
                        part=part,
                        offset=offset,
                        max_chars=max_chars,
                    )
                )
            return await to_thread.run_sync(service.detail, task_id)
