from __future__ import annotations

from typing import Any, Literal, TypeVar

import requests
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from .http import HttpSession
from .jira import normalise_jira_key

CLOCKIFY_API_ROOT = "https://api.clockify.me/api/v1"

JsonObject = dict[str, object]
JsonValue = JsonObject | list[object] | str | int | float | bool | None
TaskStatus = Literal["ACTIVE", "DONE"]
ModelT = TypeVar("ModelT", bound=BaseModel)


class ClockifyError(RuntimeError):
    pass


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
    # Clockify caps each list response at this many records; a page shorter
    # than this is the last one. A private class attribute (rather than a
    # bare literal at each call site) so tests can shrink it to exercise the
    # continuation logic without building a 5000-record fixture.
    _PAGE_SIZE: int = 5000

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

    def get_user(self) -> ClockifyUser:
        return self._validate(ClockifyUser, self._request_json("GET", "/user"), "/user")

    def get_workspace_ids(self) -> list[str]:
        """Ids of every workspace the API key's user belongs to."""
        path = "/workspaces"
        records = self._validate_list(
            ClockifyWorkspaceRef, self._request_json("GET", path), path
        )
        return [record.id for record in records]

    def get_clients(self, workspace_id: str) -> list[ClockifyClientRecord]:
        path = f"/workspaces/{workspace_id}/clients"
        return self._get_paginated_list(ClockifyClientRecord, path, params={})

    def get_projects(
        self,
        workspace_id: str,
        *,
        archived: bool,
    ) -> list[ClockifyProject]:
        path = f"/workspaces/{workspace_id}/projects"
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
        path = f"/workspaces/{workspace_id}/projects/{project_id}"
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
        path = f"/workspaces/{workspace_id}/projects/{project_id}/tasks"
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
        path = f"/workspaces/{workspace_id}/projects/{project_id}/tasks/{task_id}"
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
        path = f"/workspaces/{workspace_id}/projects"
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
        path = f"/workspaces/{workspace_id}/projects/{project.id}"
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
        path = f"/workspaces/{workspace_id}/projects/{project_id}/tasks"
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

        path = f"/workspaces/{workspace_id}/projects/{task.project_id}/tasks/{task.id}"
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
        while True:
            page_params = {**params, "page-size": self._PAGE_SIZE, "page": page}
            data = self._request_json("GET", path, params=page_params)
            items = self._validate_list(model, data, path)
            results.extend(items)
            if len(items) < self._PAGE_SIZE:
                return results
            page += 1

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: JsonObject | None = None,
        params: dict[str, object] | None = None,
    ) -> JsonValue:
        request_kwargs: dict[str, Any] = {
            "headers": {
                "Accept": "application/json",
                "X-Api-Key": self._api_key,
            },
            "json": payload,
            "timeout": 30,
        }
        if params is not None:
            request_kwargs["params"] = params

        try:
            response = self._session.request(
                method,
                f"{self._base_url}{path}",
                **request_kwargs,
            )
        except requests.RequestException as exc:
            raise ClockifyError(
                f"Clockify request failed for {method} {path}: {exc}"
            ) from exc

        if response.status_code >= 400:
            body = str(response.text)[:500]
            raise ClockifyError(
                f"Clockify request failed for {method} {path}: "
                f"{response.status_code} {response.reason}\n{body}"
            )

        if not response.content:
            return None

        try:
            return response.json()
        except ValueError as exc:
            body = str(response.text)[:500]
            raise ClockifyError(
                f"Clockify returned invalid JSON for {method} {path}: {exc}; "
                f"body={body!r}"
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
                f"Clockify returned invalid data for {path}: {exc}"
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
                f"Clockify returned invalid data for {path}: {exc}"
            ) from exc
