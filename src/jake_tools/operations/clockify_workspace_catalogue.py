"""clockify_workspace_catalogue MCP adapter."""

from __future__ import annotations

import functools

from ..clockify_time import WorkspaceCatalogue, workspace_catalogue
from ..config import Config
from . import Handler, Operation
from ._clockify import call_clockify

DESCRIPTION = "Read all clients, archived/active projects, and ACTIVE/DONE tasks in an allowlisted workspace (default first). Explicit archived/status flags and coverage; bounded upstream pagination errors rather than silent truncation. No Jira credentials required."


def build(config: Config) -> Handler:
    async def clockify_workspace_catalogue(
        workspace_id: str | None = None,
    ) -> WorkspaceCatalogue:
        return await call_clockify(
            config, workspace_id, functools.partial(workspace_catalogue)
        )

    return clockify_workspace_catalogue


OPERATION = Operation(
    name="clockify_workspace_catalogue",
    capability="clockify",
    skill="clockify-time",
    description=DESCRIPTION,
    build=build,
)
