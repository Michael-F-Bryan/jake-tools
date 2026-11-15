from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal, TypeVar

import requests
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from .http import HttpSession, UpstreamError
from .jira import normalise_jira_key

CLOCKIFY_API_ROOT = "https://api.clockify.me/api/v1"

JsonObject = dict[str, object]
JsonValue = JsonObject | list[object] | str | int | float | bool | None
TaskStatus = Literal["ACTIVE", "DONE"]
ModelT = TypeVar("ModelT", bound=BaseModel)


class ClockifyError(UpstreamError):
    pass


PaginationReason = Literal[
    "duplicate_within_page",
    "repeated_across_pages",
    "repeated_page",
    "oversized_page",
    "page_bound",
    "invalid_last_page",
    "empty_page_not_last",
]

# What the response's ``Last-Page`` header said. ``absent`` is not an error:
# the legacy short-page rule applies. ``invalid`` is any other value.
LastPageSignal = Literal["true", "false", "absent", "invalid"]


class PaginationFailure(BaseModel):
    """Why a scan could not establish coverage.

    Counts and the endpoint's resource name only: never record IDs, entry
    content or credentials, since upstream IDs are not validated data.
    """

    model_config = ConfigDict(frozen=True)

    reason: PaginationReason
    # The last path segment of the scanned list, e.g. ``time-entries``.
    resource: str
    page: int
    page_size: int
    page_records: int
    # Distinct records accepted on earlier pages of this scan.
    records_seen: int
    # Occurrences on this page of an ID already seen within the page or on an
    # earlier page.
    duplicate_count: int
    # The ``Last-Page`` header on the failing page (the value itself is never
    # echoed).
    last_page: LastPageSignal = "absent"


class ClockifyPaginationError(ClockifyError):
    """A paginated read is incomplete. Nothing was deduplicated or retried."""

    def __init__(self, message: str, failure: PaginationFailure) -> None:
        super().__init__(message, safe_summary=message)
        self.failure = failure

    def detail(self) -> dict[str, object]:
        return {
            "complete": False,
            "strategy": "single page-number scan from page 1; no retry, deduplication or merge",
            "failure": self.failure.model_dump(mode="json"),
        }


def _last_page_signal(headers: Mapping[str, str] | None) -> LastPageSignal:
    """Read the documented ``Last-Page`` header (name case-insensitive)."""
    if headers is None:
        return "absent"
    values = [v for k, v in headers.items() if str(k).lower() == "last-page"]
    if not values:
        return "absent"
    if len(values) > 1:
        return "invalid"
    value = str(values[0]).strip().lower()
    return "true" if value == "true" else "false" if value == "false" else "invalid"


def validate_clockify_id(value: str) -> str:
    """Accept opaque IDs, never URL path/query syntax (including escapes)."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ValueError("Clockify IDs must contain only letters, digits, _ or -")
    return value


class ClockifyInterval(BaseModel):
    model_config = ConfigDict(frozen=True)
    start: str
    end: str | None = None

    def instants(self) -> tuple[datetime, datetime | None]:
        try:
            start = datetime.fromisoformat(self.start)
            end = datetime.fromisoformat(self.end) if self.end is not None else None
            if start.utcoffset() is None or (
                end is not None and end.utcoffset() is None
            ):
                raise ValueError("missing timezone")
        except ValueError as exc:
            raise ClockifyError(
                "Clockify returned an invalid or timezone-naive timestamp"
            ) from exc
        if end is not None and end < start:
            raise ClockifyError("Clockify returned an invalid time interval")
        return start, end


class ClockifyTimeEntry(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    id: str
    user_id: str = Field(alias="userId")
    workspace_id: str = Field(alias="workspaceId")
    description: str = ""
    project_id: str | None = Field(default=None, alias="projectId")
    task_id: str | None = Field(default=None, alias="taskId")
    billable: bool
    time_interval: ClockifyInterval = Field(alias="timeInterval")


def clockify_project_name_for_jira(summary: str) -> str:
    project_name = summary.strip()
    if not project_name:
        raise ClockifyError("Jira summary is required for a Clockify project name")
    return project_name


def clockify_task_name_for_jira(key: str, summary: str) -> str:
    task_summary = summary.strip()
    if not task_summary:
        raise ClockifyError("Jira summary is required for a Clockify task name")
    return f"{normalise_jira_key(key)} {task_summary}"


def project_note_for(key: str) -> str:
    """The Clockify project note convention used to track a Jira key."""
    return f"Jira: {normalise_jira_key(key)}"


JIRA_REFERENCE_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")


def jira_key_of_task_name(name: str) -> str | None:
    """The Jira key a task name leads with (``KEY summary``), if any.

    The one definition of task ownership, shared by Jira sync and exact
    lookup so they never disagree about which task a key belongs to. A key
    mentioned later in the name does not make the task that key's owner.
    """
    match = JIRA_REFERENCE_PATTERN.match(name)
    return match.group(1) if match else None


def jira_key_of_project_note(note: str) -> str | None:
    """The FIRST Jira key in a project note (``Jira: KEY``), if any.

    Later keys in the same note own nothing.
    """
    match = JIRA_REFERENCE_PATTERN.search(note)
    return match.group(1) if match else None


class ClockifyUser(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    id: str
    name: str = ""
    email: str = ""
    active_workspace: str = Field(default="", alias="activeWorkspace")
    default_workspace: str = Field(default="", alias="defaultWorkspace")


class ClockifyWorkspaceRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str


class ClockifyClientRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    archived: bool = False


class ClockifyProject(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    id: str
    name: str
    note: str = ""
    archived: bool = False
    billable: bool = False
    color: str = "#039BE5"
    is_public: bool = Field(default=False, alias="public")
    client_id: str = Field(default="", alias="clientId")


class ClockifyTask(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    id: str
    name: str
    project_id: str = Field(alias="projectId")
    status: TaskStatus


class JiraIssueRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    summary: str

    @property
    def project_name(self) -> str:
        return clockify_project_name_for_jira(self.summary)

    @property
    def task_name(self) -> str:
        return clockify_task_name_for_jira(self.key, self.summary)

    @property
    def project_note(self) -> str:
        return project_note_for(self.key)


class ClockifyClient:
    # The page size we request, not a documented provider maximum. A page
    # shorter than this is assumed to be the last only when the response has
    # no ``Last-Page`` header; an explicit ``Last-Page`` always wins. A private
    # class attribute (rather than a bare literal at each call site) so tests
    # can shrink it to exercise the continuation logic without building a
    # 5000-record fixture.
    _PAGE_SIZE: int = 5000
    _MAX_PAGES: int = 1000

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = CLOCKIFY_API_ROOT,
        session: HttpSession | None = None,
    ) -> None:
        if not api_key.strip():
            raise ClockifyError("Clockify API key is required")
        normalised_base_url = base_url.strip().rstrip("/")
        if not normalised_base_url:
            raise ClockifyError("Clockify API base URL is required")
        self._api_key = api_key
        self._base_url = normalised_base_url
        self._session = session or requests.Session()
        self.pagination_reads: list[dict[str, object]] = []
        # Every HTTP request attempted, whether or not it succeeded. Distinct
        # from ``pagination_reads`` (one entry per completed list scan, however
        # many pages it took) so a caller can report upstream cost honestly.
        self.requests_made: int = 0

    def get_time_entries(
        self, workspace_id: str, user_id: str
    ) -> list[ClockifyTimeEntry]:
        path = f"/workspaces/{validate_clockify_id(workspace_id)}/user/{validate_clockify_id(user_id)}/time-entries"
        # Deliberately no date filter: entries starting arbitrarily far back
        # can still overlap the requested interval.
        return self._get_paginated_list(ClockifyTimeEntry, path, params={})

    def create_time_entry(self, workspace_id: str, payload: JsonObject) -> str:
        path = f"/workspaces/{validate_clockify_id(workspace_id)}/time-entries"
        data = self._request_json("POST", path, payload=payload, follow_redirects=False)
        if not isinstance(data, dict) or not isinstance(data.get("id"), str):
            raise ClockifyError("Create response did not contain an entry ID")
        # Preserve the known ID even if it cannot safely be interpolated into
        # a GET path. The domain reports it as an unverified created entry.
        return str(data["id"])

    def get_time_entry(self, workspace_id: str, entry_id: str) -> ClockifyTimeEntry:
        path = f"/workspaces/{validate_clockify_id(workspace_id)}/time-entries/{validate_clockify_id(entry_id)}"
        return self._validate(ClockifyTimeEntry, self._request_json("GET", path), path)

    def get_user(self) -> ClockifyUser:
        return self._validate(ClockifyUser, self._request_json("GET", "/user"), "/user")

    def get_workspace_ids(self) -> list[str]:
        """Ids of every workspace the API key's user belongs to."""
        path = "/workspaces"
        records = self._validate_list(
            ClockifyWorkspaceRef, self._request_json("GET", path), path
        )
        return [record.id for record in records]

    def get_clients(
        self, workspace_id: str, *, archived: bool | None = None
    ) -> list[ClockifyClientRecord]:
        path = f"/workspaces/{validate_clockify_id(workspace_id)}/clients"
        params: dict[str, object] = (
            {} if archived is None else {"archived": str(archived).lower()}
        )
        return self._get_paginated_list(ClockifyClientRecord, path, params=params)

    def get_projects(
        self,
        workspace_id: str,
        *,
        archived: bool,
    ) -> list[ClockifyProject]:
        path = f"/workspaces/{validate_clockify_id(workspace_id)}/projects"
        projects = self._get_paginated_list(
            ClockifyProject,
            path,
            params={
                "archived": str(archived).lower(),
                "hydrated": "false",
            },
        )
        return [project for project in projects if project.archived is archived]

    def get_project(self, workspace_id: str, project_id: str) -> ClockifyProject:
        path = f"/workspaces/{validate_clockify_id(workspace_id)}/projects/{validate_clockify_id(project_id)}"
        return self._validate(
            ClockifyProject,
            self._request_json("GET", path),
            path,
        )

    def get_tasks(
        self,
        workspace_id: str,
        project_id: str,
        *,
        active: bool,
    ) -> list[ClockifyTask]:
        path = f"/workspaces/{validate_clockify_id(workspace_id)}/projects/{validate_clockify_id(project_id)}/tasks"
        return self._get_paginated_list(
            ClockifyTask,
            path,
            params={"is-active": str(active).lower()},
        )

    def get_task(
        self,
        workspace_id: str,
        project_id: str,
        task_id: str,
    ) -> ClockifyTask:
        path = f"/workspaces/{validate_clockify_id(workspace_id)}/projects/{validate_clockify_id(project_id)}/tasks/{validate_clockify_id(task_id)}"
        return self._validate(
            ClockifyTask,
            self._request_json("GET", path),
            path,
        )

    def create_project(
        self,
        workspace_id: str,
        *,
        name: str,
        note: str,
        client_id: str,
    ) -> ClockifyProject:
        path = f"/workspaces/{validate_clockify_id(workspace_id)}/projects"
        payload: JsonObject = {
            "billable": False,
            "clientId": client_id,
            "isPublic": True,
            "name": name,
            "note": note,
        }
        return self._validate(
            ClockifyProject,
            self._request_json("POST", path, payload=payload),
            path,
        )

    def update_project_name(
        self,
        workspace_id: str,
        project: ClockifyProject,
        name: str,
    ) -> ClockifyProject:
        path = f"/workspaces/{validate_clockify_id(workspace_id)}/projects/{validate_clockify_id(project.id)}"
        payload: JsonObject = {
            "archived": project.archived,
            "billable": project.billable,
            "clientId": project.client_id,
            "color": project.color,
            "isPublic": project.is_public,
            "name": name,
            "note": project.note,
        }
        return self._validate(
            ClockifyProject,
            self._request_json("PUT", path, payload=payload),
            path,
        )

    def create_task(
        self,
        workspace_id: str,
        project_id: str,
        *,
        name: str,
    ) -> ClockifyTask:
        path = f"/workspaces/{validate_clockify_id(workspace_id)}/projects/{validate_clockify_id(project_id)}/tasks"
        return self._validate(
            ClockifyTask,
            self._request_json("POST", path, payload={"name": name}),
            path,
        )

    def update_task(
        self,
        workspace_id: str,
        task: ClockifyTask,
        *,
        name: str | None = None,
        status: TaskStatus | None = None,
    ) -> ClockifyTask:
        """Rename and/or change the status of a task.

        Any field not being changed is carried forward from `task` (the
        caller's snapshot of the record, typically taken when a sync plan
        was built). Since that snapshot can be stale by the time this runs,
        re-fetch the task first and refuse to write if it has drifted from
        the snapshot: otherwise a rename-only call would silently PUT back
        the snapshot's stale status, clobbering a status change that
        happened remotely in between.
        """
        current = self.get_task(workspace_id, task.project_id, task.id)
        if current.name != task.name or current.status != task.status:
            raise ClockifyError(
                f"Clockify task {task.id} has drifted since the sync plan was built: "
                f"expected name={task.name!r} status={task.status!r}, "
                f"found name={current.name!r} status={current.status!r}"
            )

        path = f"/workspaces/{validate_clockify_id(workspace_id)}/projects/{validate_clockify_id(task.project_id)}/tasks/{validate_clockify_id(task.id)}"
        payload: JsonObject = {
            "name": name if name is not None else task.name,
            "status": status if status is not None else task.status,
        }
        return self._validate(
            ClockifyTask,
            self._request_json("PUT", path, payload=payload),
            path,
        )

    def _get_paginated_list(
        self,
        model: type[ModelT],
        path: str,
        *,
        params: dict[str, object],
    ) -> list[ModelT]:
        results: list[ModelT] = []
        page = 1
        seen: set[str] = set()
        previous_pages: list[list[str]] = []
        signal: LastPageSignal = "absent"

        def incomplete(
            reason: PaginationReason, what: str, ids: list[str], duplicates: int
        ) -> ClockifyPaginationError:
            return ClockifyPaginationError(
                f"Clockify pagination {what}; coverage is incomplete",
                PaginationFailure(
                    reason=reason,
                    resource=path.rsplit("/", 1)[-1],
                    page=page,
                    page_size=self._PAGE_SIZE,
                    page_records=len(ids),
                    records_seen=len(seen),
                    duplicate_count=duplicates,
                    last_page=signal,
                ),
            )

        while page <= self._MAX_PAGES:
            page_params = {**params, "page-size": self._PAGE_SIZE, "page": page}
            data, headers = self._send_json("GET", path, params=page_params)
            signal = _last_page_signal(headers)
            items = self._validate_list(model, data, path)
            ids = [str(item.model_dump()["id"]) for item in items]
            if signal == "invalid":
                raise incomplete(
                    "invalid_last_page", "returned an invalid Last-Page header", ids, 0
                )
            if len(items) > self._PAGE_SIZE:
                raise incomplete("oversized_page", "returned an oversized page", ids, 0)
            # Offset pagination is not a snapshot: a repeat means records
            # moved between pages while we read, and a record may equally have
            # moved the other way unseen. Never deduplicate and claim coverage.
            repeats = 0
            known = set(seen)
            for record_id in ids:
                repeats += record_id in known
                known.add(record_id)
            if ids in previous_pages:
                raise incomplete(
                    "repeated_page", "repeated a page of record IDs", ids, repeats
                )
            if repeats:
                reason: PaginationReason = (
                    "duplicate_within_page"
                    if len(ids) != len(set(ids))
                    else "repeated_across_pages"
                )
                raise incomplete(reason, "repeated record IDs", ids, repeats)
            if signal == "false" and not ids:
                # Promised more data but gave none: no progress is possible.
                raise incomplete(
                    "empty_page_not_last",
                    "returned an empty page that is not the last",
                    ids,
                    0,
                )
            seen.update(ids)
            previous_pages.append(ids)
            results.extend(items)
            # An explicit Last-Page decides; only a response without the header
            # falls back to "a short page is the last" (legacy assumption).
            if signal == "true" or (
                signal == "absent" and len(items) < self._PAGE_SIZE
            ):
                self.pagination_reads.append(
                    {
                        "path": path,
                        "filters": params,
                        "pages": page,
                        "records": len(results),
                        "page_size": self._PAGE_SIZE,
                        "completion": (
                            "last_page_header"
                            if signal == "true"
                            else "short_page_assumed_last"
                        ),
                    }
                )
                return results
            page += 1
        page = self._MAX_PAGES
        raise incomplete("page_bound", "exceeded the page bound", [], 0)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: JsonObject | None = None,
        params: dict[str, object] | None = None,
        follow_redirects: bool = True,
    ) -> JsonValue:
        return self._send_json(
            method,
            path,
            payload=payload,
            params=params,
            follow_redirects=follow_redirects,
        )[0]

    def _send_json(
        self,
        method: str,
        path: str,
        *,
        payload: JsonObject | None = None,
        params: dict[str, object] | None = None,
        follow_redirects: bool = True,
    ) -> tuple[JsonValue, Mapping[str, str] | None]:
        """One request: the decoded body and that response's own headers."""
        for segment in path.strip("/").split("/"):
            validate_clockify_id(segment)
        request_kwargs: dict[str, Any] = {
            "headers": {
                "Accept": "application/json",
                "X-Api-Key": self._api_key,
            },
            "json": payload,
            "timeout": 30,
        }
        if not follow_redirects:
            request_kwargs["allow_redirects"] = False
        if params is not None:
            request_kwargs["params"] = params

        self.requests_made += 1
        try:
            response = self._session.request(
                method,
                f"{self._base_url}{path}",
                **request_kwargs,
            )
        except requests.RequestException as exc:
            raise ClockifyError(
                f"Clockify request failed for {method} {path}: {exc}",
                safe_summary=f"{method} {path} failed: {type(exc).__name__}",
            ) from exc

        if response.status_code >= 400 or (
            not follow_redirects and response.status_code >= 300
        ):
            body = str(response.text)[:500]
            raise ClockifyError(
                f"Clockify request failed for {method} {path}: "
                f"{response.status_code} {response.reason}\n{body}",
                safe_summary=f"{method} {path} returned HTTP {response.status_code}",
            )

        headers = getattr(response, "headers", None)
        if not response.content:
            return None, headers

        try:
            return response.json(), headers
        except ValueError as exc:
            body = str(response.text)[:500]
            raise ClockifyError(
                f"Clockify returned invalid JSON for {method} {path}: {exc}; "
                f"body={body!r}",
                safe_summary=f"{method} {path} returned invalid JSON",
            ) from exc

    @staticmethod
    def _validate(
        model: type[ModelT],
        data: JsonValue,
        path: str,
    ) -> ModelT:
        try:
            return model.model_validate(data)
        except ValidationError as exc:
            raise ClockifyError(
                f"Clockify returned invalid data for {path}: {exc}",
                safe_summary=f"{path} returned invalid data",
            ) from exc

    @staticmethod
    def _validate_list(
        model: type[ModelT],
        data: JsonValue,
        path: str,
    ) -> list[ModelT]:
        try:
            return TypeAdapter(list[model]).validate_python(data)
        except ValidationError as exc:
            raise ClockifyError(
                f"Clockify returned invalid data for {path}: {exc}",
                safe_summary=f"{path} returned invalid data",
            ) from exc
