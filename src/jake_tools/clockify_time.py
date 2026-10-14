"""Authenticated-user history, inventory, and verified completed-entry writes."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError

from .clockify import (
    ClockifyClient,
    ClockifyClientRecord,
    ClockifyError,
    ClockifyProject,
    ClockifyTask,
    ClockifyTimeEntry,
    validate_clockify_id,
)


class Coverage(BaseModel):
    complete: bool = True
    reads: list[dict[str, object]]
    strategy: str
    limitations: str = "Upstream pagination is not an atomic snapshot; concurrent changes may affect coverage."


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
