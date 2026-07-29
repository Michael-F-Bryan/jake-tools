from __future__ import annotations

import json
import re
import shlex
import subprocess
from collections.abc import Callable, Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
JIRA_KEY_PATTERN = re.compile(r"^[A-Z][A-Z0-9]+-\d+$")
JIRA_PROJECT_KEY_PATTERN = re.compile(r"^[A-Z][A-Z0-9]+$")

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


class _AcliStatusCategory(BaseModel):
    name: JiraStatusCategory


class _AcliStatus(BaseModel):
    name: str
    status_category: _AcliStatusCategory = Field(alias="statusCategory")


class _AcliNamedValue(BaseModel):
    name: str


class _AcliAssignee(BaseModel):
    display_name: str = Field(alias="displayName")


class _AcliParentFields(BaseModel):
    summary: str


class _AcliParent(BaseModel):
    key: str
    fields: _AcliParentFields


class _AcliIssueFields(BaseModel):
    summary: str
    status: _AcliStatus
    assignee: _AcliAssignee | None = None
    issue_type: _AcliNamedValue | None = Field(default=None, alias="issuetype")
    parent: _AcliParent | None = None


class _AcliIssue(BaseModel):
    key: str
    fields: _AcliIssueFields

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
            issueType=self.fields.issue_type.name if self.fields.issue_type else "",
            parentKey=parent.key if parent else None,
            parentSummary=parent.fields.summary if parent else None,
        )


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


class AcliJiraClient:
    def __init__(self, runner: CommandRunner | None = None) -> None:
        self._runner = runner or _run_command

    def get_active_assigned_issues(self, project_key: str) -> list[JiraIssue]:
        project = self._normalise_project_key(project_key)
        statuses = ", ".join(f'"{status}"' for status in ACTIVE_ISSUE_STATUSES)
        jql = (
            f"project = {project} AND assignee = currentUser() "
            f"AND status in ({statuses}) ORDER BY key"
        )
        # One search call fetches everything the planner needs (issue type
        # and parent), instead of a search followed by a per-result
        # "workitem view" subprocess just to hydrate those two fields.
        # assignee is omitted: the JQL already scopes to currentUser().
        return self._search(jql, fields="key,summary,status,issuetype,parent")

    def get_issue(self, key: str) -> JiraIssue:
        return self._view(self._normalise_key(key))

    def get_issues(self, keys: Iterable[str]) -> list[JiraIssue]:
        normalised = sorted({self._normalise_key(key) for key in keys})
        if not normalised:
            return []

        jql = f"key in ({','.join(normalised)}) ORDER BY key"
        # Fetch every field JiraIssue declares so issue_type/parent_key
        # aren't silent "not fetched" sentinels for planner logic that
        # branches on them (e.g. Project / Phase detection).
        return self._search(jql, fields="key,summary,status,assignee,issuetype,parent")

    def _search(self, jql: str, *, fields: str) -> list[JiraIssue]:
        payload = self._run_json(
            [
                "acli",
                "jira",
                "workitem",
                "search",
                "--jql",
                jql,
                "--fields",
                fields,
                "--json",
                "--paginate",
            ]
        )
        try:
            issues = TypeAdapter(list[_AcliIssue]).validate_python(payload)
        except ValidationError as exc:
            raise JiraError(f"acli returned invalid Jira issue data: {exc}") from exc
        return [issue.to_domain() for issue in issues]

    def _view(self, key: str) -> JiraIssue:
        payload = self._run_json(
            [
                "acli",
                "jira",
                "workitem",
                "view",
                key,
                "--fields",
                "*all",
                "--json",
            ]
        )
        try:
            return _AcliIssue.model_validate(payload).to_domain()
        except ValidationError as exc:
            raise JiraError(
                f"acli returned invalid Jira issue data for {key}: {exc}"
            ) from exc

    def _run_json(self, command: list[str]) -> object:
        command_text = shlex.join(command)
        try:
            result = self._runner(command)
        except OSError as exc:
            raise JiraError(f"Unable to run {command_text}: {exc}") from exc

        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise JiraError(
                f"acli command failed ({result.returncode}): {command_text}: {detail}"
            )

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            output = result.stdout[:500]
            raise JiraError(
                f"acli returned invalid JSON for {command_text}: {exc}; "
                f"stdout={output!r}"
            ) from exc

    @staticmethod
    def _normalise_key(key: str) -> str:
        return normalise_jira_key(key)

    @staticmethod
    def _normalise_project_key(key: str) -> str:
        normalised = key.strip().upper()
        if not JIRA_PROJECT_KEY_PATTERN.fullmatch(normalised):
            raise JiraError(f"Invalid Jira project key: {key!r}")
        return normalised
