from __future__ import annotations

import json
import re
import shlex
import subprocess
from collections.abc import Callable, Iterable

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from .clockify import ClockifyError, normalise_jira_key

CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
JIRA_PROJECT_KEY_PATTERN = re.compile(r"^[A-Z][A-Z0-9]+$")


class JiraError(RuntimeError):
    pass


class JiraIssue(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    key: str
    summary: str
    status: str
    status_category: str = Field(alias="statusCategory")
    assignee: str | None = None
    issue_type: str = Field(default="", alias="issueType")
    parent_key: str | None = Field(default=None, alias="parentKey")
    parent_summary: str | None = Field(default=None, alias="parentSummary")


class _AcliStatusCategory(BaseModel):
    name: str


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
        jql = (
            f"project = {project} AND assignee = currentUser() "
            'AND status in ("In Progress", "Blocked", "In Review") ORDER BY key'
        )
        candidates = self._search(
            jql,
            fields="key,summary,status,priority",
        )
        return [self._view(issue.key) for issue in candidates]

    def get_issue(self, key: str) -> JiraIssue:
        return self._view(self._normalise_key(key))

    def get_issues(self, keys: Iterable[str]) -> list[JiraIssue]:
        normalised = sorted({self._normalise_key(key) for key in keys})
        if not normalised:
            return []

        jql = f"key in ({','.join(normalised)}) ORDER BY key"
        return self._search(jql, fields="key,summary,status,assignee")

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
        try:
            return normalise_jira_key(key)
        except ClockifyError as exc:
            raise JiraError(str(exc)) from exc

    @staticmethod
    def _normalise_project_key(key: str) -> str:
        normalised = key.strip().upper()
        if not JIRA_PROJECT_KEY_PATTERN.fullmatch(normalised):
            raise JiraError(f"Invalid Jira project key: {key!r}")
        return normalised
