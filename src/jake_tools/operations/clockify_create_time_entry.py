"""clockify_create_time_entry MCP adapter."""

from __future__ import annotations

import functools

from ..clockify_time import CreateTimeEntryResult, create_time_entry
from ..config import Config
from . import Handler, Operation
from ._clockify import call_clockify

DESCRIPTION = "Create one COMPLETED entry for the authenticated user. start/end must be aware, end>start. description and billable required; task requires and must belong to project. Allowlisted workspace only (default first). POST once, then GET and verify all fields. Error outcome distinguishes post_uncertain, verification_unavailable, verification_failed and includes known created_entry_id. NEVER automatically retry an uncertain write; inspect history first."


def build(config: Config) -> Handler:
    async def clockify_create_time_entry(
        start: str,
        end: str,
        description: str,
        billable: bool,
        workspace_id: str | None = None,
        project_id: str | None = None,
        task_id: str | None = None,
    ) -> CreateTimeEntryResult:
        return await call_clockify(
            config,
            workspace_id,
            functools.partial(
                create_time_entry,
                start=start,
                end=end,
                description=description,
                billable=billable,
                project_id=project_id,
                task_id=task_id,
            ),
        )

    return clockify_create_time_entry


OPERATION = Operation(
    name="clockify_create_time_entry",
    capability="clockify",
    skill="clockify-time",
    description=DESCRIPTION,
    build=build,
)
