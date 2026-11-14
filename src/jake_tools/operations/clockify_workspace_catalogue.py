"""clockify_workspace_catalogue MCP adapter."""

from __future__ import annotations

import functools

from ..clockify_time import WorkItemLookup, WorkspaceCatalogue, lookup_work_items
from ..clockify_time import workspace_catalogue as read_workspace_catalogue
from ..config import Config
from . import Handler, Operation
from ._clockify import call_clockify
from .errors import ToolError

DESCRIPTION = """\
Read Clockify clients, projects and tasks in an allowlisted workspace (default first). No Jira credentials required.

keys omitted: the full catalogue - all clients, archived/active projects, and ACTIVE/DONE tasks, with explicit archived/status flags and coverage; bounded upstream pagination errors rather than silent truncation. Large.

keys given (1..50 Jira keys, e.g. ["SF-304"]): exact work-item lookup, a compact answer instead of the catalogue. Every distinct key gets one result in request order: resolved (exactly one task, whose name starts with the key at its very first character, or project, whose note's FIRST Jira key is it - later keys in a note own nothing; with its project, client and archived/completed flags), ambiguous (several candidates, none preferred; at most 10 listed, candidate_count/truncated say if more; this bounds the answer, not that it fits a host's inline result limit), not_found (absent from the declared scope), or incomplete (part of the scope could not be read, so absence or uniqueness is unproven). Never a similar-name or substring match. coverage states the scope, how many projects were read, unread projects with their failure (reading stops after 20; the rest are unattempted), and upstream_requests.

client_ids / project_ids (lookup only) declare where to look; both intersect, and not_found then means absent from that scope. Scoping is what reduces upstream requests (tasks are read per in-scope project); an unscoped lookup makes the same requests as the full catalogue and only returns less.
"""


def build(config: Config) -> Handler:
    async def clockify_workspace_catalogue(
        workspace_id: str | None = None,
        keys: list[str] | None = None,
        client_ids: list[str] | None = None,
        project_ids: list[str] | None = None,
    ) -> WorkspaceCatalogue | WorkItemLookup:
        if keys is None:
            if client_ids is not None or project_ids is not None:
                raise ToolError(
                    "invalid_argument",
                    "client_ids and project_ids scope an exact lookup: pass keys, or omit all three for the full catalogue",
                    detail={"argument": "keys"},
                )
            return await call_clockify(config, workspace_id, read_workspace_catalogue)
        return await call_clockify(
            config,
            workspace_id,
            functools.partial(
                lookup_work_items,
                keys=keys,
                client_ids=client_ids,
                project_ids=project_ids,
            ),
        )

    return clockify_workspace_catalogue


OPERATION = Operation(
    name="clockify_workspace_catalogue",
    capability="clockify",
    skill="clockify-time",
    description=DESCRIPTION,
    build=build,
)
