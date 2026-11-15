"""Authenticated-user history, inventory, and verified completed-entry writes."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError

from .clockify import (
    ClockifyClient,
    ClockifyClientRecord,
    ClockifyError,
    ClockifyPaginationError,
    ClockifyProject,
    ClockifyTask,
    ClockifyTimeEntry,
    PaginationFailure,
    TaskStatus,
    jira_key_of_project_note,
    jira_key_of_task_name,
    validate_clockify_id,
)
from .jira import JiraError, normalise_jira_key


class Coverage(BaseModel):
    complete: bool = True
    reads: list[dict[str, object]]
    strategy: str
    limitations: str = (
        "Upstream pagination is not an atomic snapshot; concurrent changes may affect coverage. "
        "A scan ends on an explicit Last-Page header; a response without one is assumed to end at "
        "the first page shorter than the requested page-size (reads[].completion says which)."
    )


class CatalogueTask(ClockifyTask):
    completed: bool


class WorkspaceCatalogue(BaseModel):
    workspace_id: str
    clients: list[ClockifyClientRecord]
    projects: list[ClockifyProject]
    tasks: list[CatalogueTask]
    coverage: Coverage


def workspace_catalogue(
    client: ClockifyClient, workspace_id: str
) -> WorkspaceCatalogue:
    validate_clockify_id(workspace_id)
    first_read = len(client.pagination_reads)
    clients = {
        c.id: c
        for archived in (False, True)
        for c in client.get_clients(workspace_id, archived=archived)
    }
    projects = {
        p.id: p
        for archived in (False, True)
        for p in client.get_projects(workspace_id, archived=archived)
    }
    tasks: dict[str, ClockifyTask] = {}
    for project in projects.values():
        validate_clockify_id(project.id)
        for active in (False, True):
            for task in client.get_tasks(workspace_id, project.id, active=active):
                if task.project_id != project.id:
                    raise ClockifyError(
                        "Clockify returned a task belonging to another project"
                    )
                tasks[task.id] = task
    return WorkspaceCatalogue(
        workspace_id=workspace_id,
        clients=sorted(clients.values(), key=lambda c: c.id),
        projects=sorted(projects.values(), key=lambda p: p.id),
        tasks=[
            CatalogueTask(**task.model_dump(), completed=task.status == "DONE")
            for task in sorted(tasks.values(), key=lambda t: (t.project_id, t.id))
        ],
        coverage=Coverage(
            reads=client.pagination_reads[first_read:],
            strategy="All archived/active clients and projects, and ACTIVE/DONE tasks; bounded upstream pagination, never truncated.",
        ),
    )


MAX_LOOKUP_KEYS = 50
MAX_LOOKUP_SCOPE_IDS = 50
# A key that many records claim is ambiguous long before this; the cap only
# bounds the answer, and the count and ``truncated`` say what was cut.
MAX_CANDIDATES_PER_KEY = 10
# Stop reading after this many projects fail: the rest are reported as
# unattempted, so a down upstream costs neither requests nor response size
# proportional to the workspace.
MAX_UNREAD_PROJECTS = 20

LookupOutcome = Literal["resolved", "ambiguous", "not_found", "incomplete"]


class ClientRef(BaseModel):
    id: str
    name: str
    archived: bool


class ProjectRef(BaseModel):
    id: str
    name: str
    archived: bool


class TaskRef(BaseModel):
    id: str
    name: str
    status: TaskStatus
    completed: bool


class WorkItemCandidate(BaseModel):
    """One Clockify record that claims a key, with everything that owns it.

    ``kind`` says how the key is claimed: a task whose name leads with the
    key (from its first character), or a project whose note's first Jira key it is (the Jira sync conventions).
    ``client`` is null only for a project with no client.
    """

    kind: Literal["task", "project"]
    project: ProjectRef
    client: ClientRef | None
    task: TaskRef | None = None


class KeyResult(BaseModel):
    """``resolved``: exactly one candidate and the scope was read in full.
    ``ambiguous``: more than one; none is preferred. ``not_found``: none, and
    the declared scope was read in full. ``incomplete``: part of the scope
    could not be read, so absence or uniqueness is unproven; any candidates
    seen are kept as evidence, never as an answer.
    """

    key: str
    outcome: LookupOutcome
    # At most MAX_CANDIDATES_PER_KEY, in project/task ID order.
    candidates: list[WorkItemCandidate]
    # Candidates found in total; ``truncated`` when ``candidates`` holds fewer.
    candidate_count: int
    truncated: bool = False


class LookupScope(BaseModel):
    client_ids: list[str] | None
    project_ids: list[str] | None
    # True only when no client/project scope narrowed the search.
    workspace_wide: bool


class UnreadProject(BaseModel):
    project_id: str
    # Set when the failure was a pagination fault; otherwise ``error`` carries
    # the host-free summary. Never upstream content.
    failure: PaginationFailure | None = None
    error: str | None = None


class LookupCoverage(BaseModel):
    complete: bool
    scope: LookupScope
    projects_total: int
    projects_in_scope: int
    projects_read: int
    # At most MAX_UNREAD_PROJECTS failed projects; reading stops there.
    unread: list[UnreadProject]
    # In-scope projects never read because reading stopped.
    unattempted: int = 0
    # HTTP requests this lookup made, counted at the client. Not response size.
    upstream_requests: int
    strategy: str
    limitations: str = (
        "Upstream pagination is not an atomic snapshot; concurrent changes may affect coverage. "
        "A scan ends on an explicit Last-Page header; a response without one is assumed to end at "
        "the first page shorter than the requested page-size (reads[].completion says which)."
    )


class WorkItemLookup(BaseModel):
    workspace_id: str
    results: list[KeyResult]
    coverage: LookupCoverage


class ContradictionError(ClockifyError):
    """Two reads of one record disagree: neither is an answer."""


def _observed_once[RecordT: BaseModel](
    label: str,
    records: Iterable[RecordT],
    *,
    contradiction: type[ClockifyError] = ClockifyError,
) -> dict[str, RecordT]:
    """Index records by ID across overlapping scans.

    The same record read twice and identical is one record. Read twice with
    different content (for example a task whose status changed between the
    ACTIVE and DONE scans) is a contradiction: raised, never resolved by
    taking whichever scan came last.
    """
    seen: dict[str, RecordT] = {}
    for record in records:
        record_id = str(record.model_dump()["id"])
        if record_id in seen and seen[record_id] != record:
            raise contradiction(
                f"Clockify returned contradictory observations of a {label}; "
                "it changed between reads",
                safe_summary=f"a {label} was observed with contradictory content "
                "in overlapping scans",
            )
        seen[record_id] = record
    return seen


def _scope_ids(label: str, ids: list[str] | None) -> list[str] | None:
    if ids is None:
        return None
    if not 1 <= len(ids) <= MAX_LOOKUP_SCOPE_IDS:
        raise ValueError(
            f"{label} must hold 1..{MAX_LOOKUP_SCOPE_IDS} IDs (omit it for no scope)"
        )
    return list(dict.fromkeys(validate_clockify_id(i) for i in ids))


def lookup_work_items(
    client: ClockifyClient,
    workspace_id: str,
    *,
    keys: list[str],
    client_ids: list[str] | None = None,
    project_ids: list[str] | None = None,
) -> WorkItemLookup:
    """Resolve exact Jira work-item keys to Clockify IDs, per key.

    A key is claimed by a task whose name leads with it or a project whose
    note's first Jira key it is (the Jira sync's ownership conventions, never a
    substring or similar-name match). Archived and completed records are
    included and flagged. ``client_ids``/``project_ids`` declare where to
    look; together they intersect. They are what removes upstream requests:
    clients and projects are always listed (at least two paginated scans each,
    archived and not), but tasks are read per in-scope project. Without a scope this
    reads exactly what the full catalogue reads and only the answer is smaller.
    """
    validate_clockify_id(workspace_id)
    try:
        wanted = list(dict.fromkeys(normalise_jira_key(key) for key in keys))
    except JiraError as exc:
        raise ValueError(str(exc)) from exc
    if not 1 <= len(wanted) <= MAX_LOOKUP_KEYS:
        raise ValueError(
            f"keys must hold 1..{MAX_LOOKUP_KEYS} distinct Jira keys "
            "(omit keys for the full catalogue)"
        )
    scope_clients = _scope_ids("client_ids", client_ids)
    scope_projects = _scope_ids("project_ids", project_ids)

    first_request = client.requests_made
    clients = _observed_once(
        "client",
        (
            c
            for archived in (False, True)
            for c in client.get_clients(workspace_id, archived=archived)
        ),
    )
    projects = _observed_once(
        "project",
        (
            p
            for archived in (False, True)
            for p in client.get_projects(workspace_id, archived=archived)
        ),
    )
    for label, scoped, known in (
        ("client", scope_clients, clients),
        ("project", scope_projects, projects),
    ):
        missing = [i for i in scoped or [] if i not in known]
        if missing:
            raise ValueError(f"Unknown {label} ID in this workspace: {missing[0]}")
    in_scope = [
        p
        for p in sorted(projects.values(), key=lambda p: p.id)
        if (scope_projects is None or p.id in scope_projects)
        and (scope_clients is None or p.client_id in scope_clients)
    ]

    claims: dict[str, list[tuple[ClockifyProject, ClockifyTask | None]]] = {}
    for project in in_scope:
        key = jira_key_of_project_note(project.note)
        if key:
            claims.setdefault(key, []).append((project, None))
    unread: list[UnreadProject] = []
    unattempted = 0
    for project in in_scope:
        if len(unread) >= MAX_UNREAD_PROJECTS:
            unattempted += 1
            continue
        try:
            scans: list[ClockifyTask] = []
            for active in (False, True):
                scans += client.get_tasks(workspace_id, project.id, active=active)
            tasks = _observed_once("task", scans, contradiction=ContradictionError)
        except ClockifyPaginationError as exc:
            unread.append(UnreadProject(project_id=project.id, failure=exc.failure))
            continue
        except ContradictionError as exc:
            # A task changed between the two reads: no answer about it is
            # trustworthy, so this project's tasks contribute nothing.
            unread.append(UnreadProject(project_id=project.id, error=str(exc)))
            continue
        except ClockifyError as exc:
            if exc.safe_summary is None:
                raise  # not an upstream read failure: never guess a summary
            unread.append(UnreadProject(project_id=project.id, error=exc.safe_summary))
            continue
        for task in tasks.values():
            if task.project_id != project.id:
                raise ClockifyError(
                    "Clockify returned a task belonging to another project"
                )
            key = jira_key_of_task_name(task.name)
            if key:
                claims.setdefault(key, []).append((project, task))

    def candidate(
        project: ClockifyProject, task: ClockifyTask | None
    ) -> WorkItemCandidate:
        owner: ClockifyClientRecord | None = None
        if project.client_id:
            owner = clients.get(project.client_id)
            if owner is None:
                raise ClockifyError(
                    "Clockify returned a project with an unknown client"
                )
        return WorkItemCandidate(
            kind="task" if task else "project",
            project=ProjectRef(
                id=project.id, name=project.name, archived=project.archived
            ),
            client=ClientRef(id=owner.id, name=owner.name, archived=owner.archived)
            if owner
            else None,
            task=TaskRef(
                id=task.id,
                name=task.name,
                status=task.status,
                completed=task.status == "DONE",
            )
            if task
            else None,
        )

    results: list[KeyResult] = []
    for key in wanted:
        found = [
            candidate(project, task)
            for project, task in sorted(
                claims.get(key, []), key=lambda c: (c[0].id, c[1].id if c[1] else "")
            )
        ]
        outcome: LookupOutcome
        total = len(found)
        if total > 1:
            outcome = "ambiguous"
        elif unread:
            outcome = "incomplete"
        else:
            outcome = "resolved" if found else "not_found"
        results.append(
            KeyResult(
                key=key,
                outcome=outcome,
                candidates=found[:MAX_CANDIDATES_PER_KEY],
                candidate_count=total,
                truncated=total > MAX_CANDIDATES_PER_KEY,
            )
        )
    workspace_wide = scope_clients is None and scope_projects is None
    return WorkItemLookup(
        workspace_id=workspace_id,
        results=results,
        coverage=LookupCoverage(
            complete=not unread and not unattempted,
            scope=LookupScope(
                client_ids=scope_clients,
                project_ids=scope_projects,
                workspace_wide=workspace_wide,
            ),
            projects_total=len(projects),
            projects_in_scope=len(in_scope),
            projects_read=len(in_scope) - len(unread) - unattempted,
            unread=unread,
            unattempted=unattempted,
            upstream_requests=client.requests_made - first_request,
            strategy="Exact key match only: a task name starting with the key at its first character, or a project note whose first Jira key it is. "
            "All archived/active clients and projects are listed, then ACTIVE/DONE tasks of each in-scope project; "
            "never a similar-name match. not_found means absent from the stated scope only.",
        ),
    )


def interval(start: str, end: str) -> tuple[datetime, datetime]:
    try:
        a, b = datetime.fromisoformat(start), datetime.fromisoformat(end)
        if a.utcoffset() is None or b.utcoffset() is None:
            raise ValueError("missing timezone")
    except ValueError as exc:
        raise ValueError("start and end must be timezone-aware ISO timestamps") from exc
    if b <= a:
        raise ValueError("end must be after start")
    return a, b


class HistoryEntry(BaseModel):
    entry: ClockifyTimeEntry
    running: bool
    project_name: str | None
    task_name: str | None
    display_start: str
    display_end: str | None


class TimeEntriesResult(BaseModel):
    workspace_id: str
    user_id: str
    start: str
    end: str
    timezone: str
    observed_at: datetime
    entries: list[HistoryEntry]
    count: int
    total: int
    next_cursor: str | None
    coverage: Coverage


class HistoryCursor(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    snapshot: str
    offset: int
    observed_at: AwareDatetime


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def time_entries(
    client: ClockifyClient,
    workspace_id: str,
    *,
    start: str,
    end: str,
    timezone: str = "UTC",
    limit: int = 100,
    cursor: str | None = None,
) -> TimeEntriesResult:
    a, b = interval(start, end)
    validate_clockify_id(workspace_id)
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError("timezone must be an IANA timezone name") from exc
    first_read = len(client.pagination_reads)
    user = client.get_user()
    validate_clockify_id(user.id)
    query = _digest(
        [
            workspace_id,
            user.id,
            a.astimezone(UTC).isoformat(),
            b.astimezone(UTC).isoformat(),
            timezone,
            limit,
        ]
    )
    continuation: HistoryCursor | None = None
    if cursor is not None:
        try:
            if len(cursor) > 4096:
                raise ValueError("cursor too large")
            continuation = HistoryCursor.model_validate_json(
                base64.b64decode(cursor, altchars=b"-_", validate=True)
            )
            if continuation.query != query or continuation.offset < 1:
                raise ValueError("cursor query mismatch")
        except (ValueError, ValidationError) as exc:
            raise ValueError(
                "Invalid cursor or cursor belongs to a different query"
            ) from exc
    observed_at = continuation.observed_at if continuation else datetime.now(UTC)
    records = client.get_time_entries(workspace_id, user.id)
    selected: list[ClockifyTimeEntry] = []
    for record in records:
        if record.user_id != user.id or record.workspace_id != workspace_id:
            raise ClockifyError(
                "Clockify returned a time entry outside the requested user/workspace"
            )
        begin, finish = record.time_interval.instants()
        effective_end = finish if finish is not None else observed_at
        if begin < b and effective_end > a and effective_end > begin:
            selected.append(record)
    selected.sort(key=lambda r: (r.time_interval.instants()[0], r.id))
    snapshot = _digest([r.model_dump(mode="json") for r in selected])
    offset = continuation.offset if continuation else 0
    if continuation and (continuation.snapshot != snapshot or offset >= len(selected)):
        raise ValueError(
            "History changed or cursor is invalid; restart pagination without cursor"
        )
    projects: dict[str, ClockifyProject] = {}
    tasks: dict[tuple[str, str], ClockifyTask] = {}
    entries: list[HistoryEntry] = []
    for record in selected[offset : offset + limit]:
        project_name = task_name = None
        if record.project_id:
            validate_clockify_id(record.project_id)
            if record.project_id not in projects:
                projects[record.project_id] = client.get_project(
                    workspace_id, record.project_id
                )
            if projects[record.project_id].id != record.project_id:
                raise ClockifyError("Clockify returned a different project")
            project_name = projects[record.project_id].name
        if record.task_id:
            if not record.project_id:
                raise ClockifyError("Clockify returned a task without a project")
            validate_clockify_id(record.task_id)
            key = (record.project_id, record.task_id)
            if key not in tasks:
                tasks[key] = client.get_task(workspace_id, *key)
            if (
                tasks[key].id != record.task_id
                or tasks[key].project_id != record.project_id
            ):
                raise ClockifyError(
                    "Clockify returned a task belonging to another project"
                )
            task_name = tasks[key].name
        begin, finish = record.time_interval.instants()
        entries.append(
            HistoryEntry(
                entry=record,
                running=finish is None,
                project_name=project_name,
                task_name=task_name,
                display_start=begin.astimezone(zone).isoformat(),
                display_end=finish.astimezone(zone).isoformat() if finish else None,
            )
        )
    next_cursor = None
    if offset + len(entries) < len(selected):
        token = HistoryCursor(
            query=query,
            snapshot=snapshot,
            offset=offset + len(entries),
            observed_at=observed_at,
        )
        next_cursor = base64.urlsafe_b64encode(
            token.model_dump_json().encode()
        ).decode()
    return TimeEntriesResult(
        workspace_id=workspace_id,
        user_id=user.id,
        start=start,
        end=end,
        timezone=timezone,
        observed_at=observed_at,
        entries=entries,
        count=len(entries),
        total=len(selected),
        next_cursor=next_cursor,
        coverage=Coverage(
            reads=client.pagination_reads[first_read:],
            strategy="Scans ALL authenticated-user entries without upstream date filters on every call; overlap filtered locally. Name resolution is required (failures are errors, not silent gaps). Cost is proportional to all user history, not just the requested range.",
        ),
    )


class CreateTimeEntryResult(BaseModel):
    workspace_id: str
    entry: ClockifyTimeEntry
    verified: Literal[True] = True


class CreateTimeEntryError(ClockifyError):
    def __init__(self, outcome: str, entry_id: str | None = None) -> None:
        super().__init__(
            "Clockify create outcome: "
            + outcome
            + ". Do not automatically retry; inspect history first."
        )
        self.outcome = outcome
        self.entry_id = entry_id


def create_time_entry(
    client: ClockifyClient,
    workspace_id: str,
    *,
    start: str,
    end: str,
    description: str,
    billable: bool,
    project_id: str | None = None,
    task_id: str | None = None,
) -> CreateTimeEntryResult:
    a, b = interval(start, end)
    validate_clockify_id(workspace_id)
    if project_id is not None:
        validate_clockify_id(project_id)
    if task_id is not None:
        validate_clockify_id(task_id)
        if project_id is None:
            raise ValueError("task_id requires project_id")
    user = client.get_user()
    if project_id is not None:
        project = client.get_project(workspace_id, project_id)
        if project.id != project_id:
            raise ClockifyError("Clockify returned a different project")
    if task_id is not None and project_id is not None:
        task = client.get_task(workspace_id, project_id, task_id)
        if task.id != task_id or task.project_id != project_id:
            raise ValueError("task_id does not belong to project_id")
    payload: dict[str, object] = {
        "start": a.astimezone(UTC).isoformat(),
        "end": b.astimezone(UTC).isoformat(),
        "description": description,
        "billable": billable,
        "projectId": project_id,
        "taskId": task_id,
    }
    try:
        entry_id = client.create_time_entry(workspace_id, payload)
    except ClockifyError as exc:
        # Even a server error can follow a committed write. No retries.
        raise CreateTimeEntryError("post_uncertain") from exc
    try:
        validate_clockify_id(entry_id)
        entry = client.get_time_entry(workspace_id, entry_id)
        actual_start, actual_end = entry.time_interval.instants()
    except (ClockifyError, ValueError) as exc:
        raise CreateTimeEntryError("verification_unavailable", entry_id) from exc
    if (
        entry.id != entry_id
        or entry.user_id != user.id
        or entry.workspace_id != workspace_id
        or actual_start != a
        or actual_end != b
        or entry.description != description
        or entry.billable != billable
        or entry.project_id != project_id
        or entry.task_id != task_id
    ):
        raise CreateTimeEntryError("verification_failed", entry_id)
    return CreateTimeEntryResult(workspace_id=workspace_id, entry=entry)
