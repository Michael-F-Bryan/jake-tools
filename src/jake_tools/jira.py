from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Literal
from urllib.parse import urlsplit

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .http import HttpSession

JIRA_KEY_PATTERN = re.compile(r"^[A-Z][A-Z0-9]+-\d+$")
JIRA_PROJECT_KEY_PATTERN = re.compile(r"^[A-Z][A-Z0-9]+$")
JIRA_FIELDS: tuple[str, ...] = (
    "summary",
    "status",
    "assignee",
    "issuetype",
    "parent",
)

# Jira's three built-in status categories. Every workflow status maps onto
# exactly one of these regardless of how many custom statuses a project has.
JiraStatusCategory = Literal["To Do", "In Progress", "Done"]

# The Jira statuses treated as "currently being worked" for the default
# assigned-active sync scope.
ACTIVE_ISSUE_STATUSES: tuple[str, ...] = ("In Progress", "Blocked", "In Review")


class JiraError(RuntimeError):
    pass


def normalise_jira_key(key: str) -> str:
    normalised = key.strip().upper()
    if not JIRA_KEY_PATTERN.fullmatch(normalised):
        raise JiraError(f"Invalid Jira issue key: {key!r}")
    return normalised


class JiraIssue(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    key: str
    summary: str
    status: str
    status_category: JiraStatusCategory = Field(alias="statusCategory")
    assignee: str | None = None
    issue_type: str = Field(default="", alias="issueType")
    parent_key: str | None = Field(default=None, alias="parentKey")
    parent_summary: str | None = Field(default=None, alias="parentSummary")


class _RestStatusCategory(BaseModel):
    name: JiraStatusCategory


class _RestStatus(BaseModel):
    name: str
    status_category: _RestStatusCategory = Field(alias="statusCategory")


class _RestNamedValue(BaseModel):
    name: str


class _RestAssignee(BaseModel):
    display_name: str = Field(alias="displayName")


class _RestParentFields(BaseModel):
    summary: str


class _RestParent(BaseModel):
    key: str
    fields: _RestParentFields


class _RestIssueFields(BaseModel):
    summary: str
    status: _RestStatus
    assignee: _RestAssignee | None = None
    issue_type: _RestNamedValue = Field(alias="issuetype")
    parent: _RestParent | None = None


class _RestIssue(BaseModel):
    key: str
    fields: _RestIssueFields

    def to_domain(self) -> JiraIssue:
        parent = self.fields.parent
        return JiraIssue(
            key=self.key,
            summary=self.fields.summary,
            status=self.fields.status.name,
            statusCategory=self.fields.status.status_category.name,
            assignee=(
                self.fields.assignee.display_name if self.fields.assignee else None
            ),
            issueType=self.fields.issue_type.name,
            parentKey=parent.key if parent else None,
            parentSummary=parent.fields.summary if parent else None,
        )


class _RestSearchPage(BaseModel):
    issues: list[_RestIssue]
    next_page_token: str | None = Field(default=None, alias="nextPageToken")


class JiraClient:
    _SEARCH_PATH = "/rest/api/3/search/jql"
    _PAGE_SIZE = 100

    def __init__(
        self,
        *,
        base_url: str,
        email: str,
        api_token: str,
        session: HttpSession | None = None,
    ) -> None:
        self._base_url = self._normalise_base_url(base_url)
        self._email = email.strip()
        self._api_token = api_token.strip()
        if not self._email:
            raise JiraError("Jira email is required")
        if not self._api_token:
            raise JiraError("Jira API token is required")
        self._session = session or requests.Session()

    def get_active_assigned_issues(self, project_key: str) -> list[JiraIssue]:
        project = self._normalise_project_key(project_key)
        statuses = ", ".join(f'"{status}"' for status in ACTIVE_ISSUE_STATUSES)
        jql = (
            f"project = {project} AND assignee = currentUser() "
            f"AND status in ({statuses}) ORDER BY key"
        )
        return self._search(jql)

    def get_issue(self, key: str) -> JiraIssue:
        normalised = normalise_jira_key(key)
        path = f"/rest/api/3/issue/{normalised}"
        payload = self._request_json(
            "GET",
            path,
            params={"fields": ",".join(JIRA_FIELDS)},
        )
        try:
            return _RestIssue.model_validate(payload).to_domain()
        except ValidationError as exc:
            raise JiraError(
                f"Jira returned invalid issue data for {normalised}: {exc}"
            ) from exc

    def get_issues(self, keys: Iterable[str]) -> list[JiraIssue]:
        normalised = sorted({normalise_jira_key(key) for key in keys})
        if not normalised:
            return []
        return self._search(f"key in ({','.join(normalised)}) ORDER BY key")

    def _search(self, jql: str) -> list[JiraIssue]:
        issues: list[JiraIssue] = []
        next_page_token: str | None = None
        while True:
            payload: dict[str, object] = {
                "jql": jql,
                "fields": list(JIRA_FIELDS),
                "maxResults": self._PAGE_SIZE,
            }
            if next_page_token is not None:
                payload["nextPageToken"] = next_page_token

            data = self._request_json("POST", self._SEARCH_PATH, payload=payload)
            try:
                page = _RestSearchPage.model_validate(data)
            except ValidationError as exc:
                raise JiraError(f"Jira returned invalid search data: {exc}") from exc

            issues.extend(issue.to_domain() for issue in page.issues)
            if page.next_page_token is None:
                return issues
            next_page_token = page.next_page_token

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, object] | None = None,
        params: dict[str, object] | None = None,
    ) -> object:
        request_kwargs: dict[str, Any] = {
            "auth": (self._email, self._api_token),
            "headers": {
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            "timeout": 30,
        }
        if payload is not None:
            request_kwargs["json"] = payload
        if params is not None:
            request_kwargs["params"] = params

        try:
            response = self._session.request(
                method,
                f"{self._base_url}{path}",
                **request_kwargs,
            )
        except requests.RequestException as exc:
            raise JiraError(f"Jira request failed for {method} {path}: {exc}") from exc

        if response.status_code >= 400:
            body = str(response.text)[:500]
            raise JiraError(
                f"Jira request failed for {method} {path}: "
                f"{response.status_code} {response.reason}\n{body}"
            )

        try:
            return response.json()
        except ValueError as exc:
            body = str(response.text)[:500]
            raise JiraError(
                f"Jira returned invalid JSON for {method} {path}: {exc}; "
                f"response={body!r}"
            ) from exc

    @staticmethod
    def _normalise_base_url(base_url: str) -> str:
        normalised = base_url.strip().rstrip("/")
        if not normalised:
            raise JiraError("Jira base URL is required")
        if "://" not in normalised:
            normalised = f"https://{normalised}"
        parsed = urlsplit(normalised)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise JiraError(f"Invalid Jira base URL: {base_url!r}")
        return normalised

    @staticmethod
    def _normalise_project_key(key: str) -> str:
        normalised = key.strip().upper()
        if not JIRA_PROJECT_KEY_PATTERN.fullmatch(normalised):
            raise JiraError(f"Invalid Jira project key: {key!r}")
        return normalised
