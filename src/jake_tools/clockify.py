from __future__ import annotations

import os
import re
from typing import Any, Protocol, cast

import requests
from pydantic import BaseModel, ConfigDict, Field

CLOCKIFY_API_ROOT = "https://api.clockify.me/api/v1"
JIRA_KEY_PATTERN = re.compile(r"^[A-Z][A-Z0-9]+-\d+$")

JsonObject = dict[str, object]


class HttpSession(Protocol):
    def request(self, method: str, url: str, **kwargs: Any) -> Any: ...


class ClockifyError(RuntimeError):
    pass


def normalise_jira_key(key: str) -> str:
    normalised = key.strip().upper()
    if not JIRA_KEY_PATTERN.fullmatch(normalised):
        raise ClockifyError(f"Invalid Jira issue key: {key!r}")
    return normalised


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


class ClockifyUser(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    id: str
    name: str = ""
    email: str = ""
    active_workspace: str = Field(default="", alias="activeWorkspace")
    default_workspace: str = Field(default="", alias="defaultWorkspace")


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
        return f"Jira: {normalise_jira_key(self.key)}"


class ClockifyClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = CLOCKIFY_API_ROOT,
        session: HttpSession | None = None,
    ) -> None:
        if not api_key.strip():
            raise ClockifyError("Clockify API key is required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()

    def get_user(self) -> ClockifyUser:
        data = self._request_json("GET", "/user")
        return ClockifyUser.model_validate(data)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: JsonObject | None = None,
    ) -> JsonObject:
        try:
            response = self._session.request(
                method,
                f"{self._base_url}{path}",
                headers={
                    "Accept": "application/json",
                    "X-Api-Key": self._api_key,
                },
                json=payload,
                timeout=30,
            )
        except requests.RequestException as exc:
            raise ClockifyError(f"Clockify request failed: {exc}") from exc

        if response.status_code >= 400:
            raise ClockifyError(
                f"Clockify request failed: {response.status_code} {response.reason}\n{response.text}"
            )

        if not response.content:
            return {}

        return cast(JsonObject, response.json())


def clockify_api_key_from_env() -> str:
    api_key = os.environ.get("CLOCKIFY_API_KEY", "").strip()
    if not api_key:
        raise ClockifyError(
            "Clockify API key is required. Set CLOCKIFY_API_KEY or pass --api-key."
        )
    return api_key


def clockify_base_url_from_env() -> str:
    return os.environ.get("CLOCKIFY_API_BASE_URL", CLOCKIFY_API_ROOT).strip()
