from __future__ import annotations

import dataclasses
import json

from click.testing import CliRunner

from jake_tools.cli.clockify import clockify
from jake_tools.cli.context import AppContext, ClockifyConfig, JiraConfig
from jake_tools.clockify import (
    CLOCKIFY_API_ROOT,
    ClockifyClientRecord,
    ClockifyProject,
    ClockifyTask,
    ClockifyUser,
    TaskStatus,
)
from jake_tools.jira import JiraError, JiraIssue


class FakeClockifyClient:
    """A minimal ``ClockifyInventoryClient``: only ``get_user`` is exercised
    by the ``whoami`` tests below, so every other method just documents
    that it is unused."""

    def __init__(self, *, api_key: str = "test-key", base_url: str = "") -> None:
        self.api_key = api_key
        self.base_url = base_url

    def get_user(self) -> ClockifyUser:
        return ClockifyUser(
            id="user-123",
            name="Michael Bryan",
            email="michael@example.test",
            activeWorkspace="workspace-1",
            defaultWorkspace="workspace-2",
        )

    def get_clients(self, workspace_id: str) -> list[ClockifyClientRecord]:
        raise AssertionError("not expected")

    def get_projects(
        self, workspace_id: str, *, archived: bool
    ) -> list[ClockifyProject]:
        raise AssertionError("not expected")

    def get_tasks(
        self, workspace_id: str, project_id: str, *, active: bool
    ) -> list[ClockifyTask]:
        raise AssertionError("not expected")

    def get_project(self, workspace_id: str, project_id: str) -> ClockifyProject:
        raise AssertionError("not expected")

    def get_task(
        self, workspace_id: str, project_id: str, task_id: str
    ) -> ClockifyTask:
        raise AssertionError("not expected")

    def create_project(
        self, workspace_id: str, *, name: str, note: str, client_id: str
    ) -> ClockifyProject:
        raise AssertionError("not expected")

    def update_project_name(
        self, workspace_id: str, project: ClockifyProject, name: str
    ) -> ClockifyProject:
        raise AssertionError("not expected")

    def create_task(
        self, workspace_id: str, project_id: str, *, name: str
    ) -> ClockifyTask:
        raise AssertionError("not expected")

    def update_task(
        self,
        workspace_id: str,
        task: ClockifyTask,
        *,
        name: str | None = None,
        status: TaskStatus | None = None,
    ) -> ClockifyTask:
        raise AssertionError("not expected")


def test_whoami_prints_current_clockify_user() -> None:
    client = FakeClockifyClient()
    app = AppContext(clockify_client_factory=lambda _config: client)
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        ["--api-key", "test-key", "whoami"],
        obj=app,
    )

    assert result.exit_code == 0
    assert "ID: user-123" in result.output
    assert "Name: Michael Bryan" in result.output
    assert "Email: michael@example.test" in result.output
    assert "Active workspace: workspace-1" in result.output
    assert "Default workspace: workspace-2" in result.output


def test_whoami_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("CLOCKIFY_API_KEY", raising=False)
    runner = CliRunner()

    result = runner.invoke(clockify, ["whoami"])

    assert result.exit_code != 0
    assert "CLOCKIFY_API_KEY" in result.output


def test_api_key_resolves_from_env_when_the_flag_is_omitted(monkeypatch) -> None:
    monkeypatch.setenv("CLOCKIFY_API_KEY", "env-key")
    captured: list[ClockifyConfig] = []

    def factory(config: ClockifyConfig) -> FakeClockifyClient:
        captured.append(config)
        return FakeClockifyClient()

    app = AppContext(clockify_client_factory=factory)
    runner = CliRunner()

    result = runner.invoke(clockify, ["whoami"], obj=app)

    assert result.exit_code == 0
    assert captured == [
        ClockifyConfig(api_key="env-key", api_base_url=CLOCKIFY_API_ROOT)
    ]


def test_api_base_url_resolves_from_env_when_the_flag_is_omitted(monkeypatch) -> None:
    monkeypatch.setenv("CLOCKIFY_API_BASE_URL", "https://clockify.example.test/api/v1")
    captured: list[ClockifyConfig] = []

    def factory(config: ClockifyConfig) -> FakeClockifyClient:
        captured.append(config)
        return FakeClockifyClient()

    app = AppContext(clockify_client_factory=factory)
    runner = CliRunner()

    result = runner.invoke(clockify, ["--api-key", "test-key", "whoami"], obj=app)

    assert result.exit_code == 0
    assert captured == [
        ClockifyConfig(
            api_key="test-key", api_base_url="https://clockify.example.test/api/v1"
        )
    ]


def test_api_base_url_falls_back_to_the_default_root(monkeypatch) -> None:
    monkeypatch.delenv("CLOCKIFY_API_BASE_URL", raising=False)
    captured: list[ClockifyConfig] = []

    def factory(config: ClockifyConfig) -> FakeClockifyClient:
        captured.append(config)
        return FakeClockifyClient()

    app = AppContext(clockify_client_factory=factory)
    runner = CliRunner()

    result = runner.invoke(clockify, ["--api-key", "test-key", "whoami"], obj=app)

    assert result.exit_code == 0
    assert captured == [
        ClockifyConfig(api_key="test-key", api_base_url=CLOCKIFY_API_ROOT)
    ]


def test_jira_name_renders_project_without_key_and_task_with_key() -> None:
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        [
            "jira-name",
            "SF-353",
            "Vehicle Control Logic - Preliminary Architecture",
        ],
    )

    assert result.exit_code == 0
    assert "project: Vehicle Control Logic - Preliminary Architecture" in result.output
    assert (
        "task: SF-353 Vehicle Control Logic - Preliminary Architecture" in result.output
    )
    assert "note: Jira: SF-353" in result.output


def test_jira_name_can_emit_json() -> None:
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        ["jira-name", "sf-4", "Zoda - Internal Tooling & Upkeep", "--json"],
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "project": "Zoda - Internal Tooling & Upkeep",
        "task": "SF-4 Zoda - Internal Tooling & Upkeep",
        "note": "Jira: SF-4",
    }


class FakeJiraSyncClockifyClient:
    def __init__(self, *, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.operations: list[str] = []
        self.task_status: TaskStatus = "ACTIVE"

    @property
    def project(self) -> ClockifyProject:
        return ClockifyProject(
            id="project-1",
            name="Production Vehicle",
            note="Jira: SF-131",
            archived=False,
            billable=False,
            color="#689F38",
            public=True,
            clientId="client-1",
        )

    @property
    def task(self) -> ClockifyTask:
        return ClockifyTask(
            id="task-1",
            name="SF-304 Bench test PX4",
            projectId="project-1",
            status=self.task_status,
        )

    def get_user(self) -> ClockifyUser:
        return ClockifyUser(id="user-1", activeWorkspace="workspace-1")

    def get_clients(self, workspace_id: str) -> list[ClockifyClientRecord]:
        return [ClockifyClientRecord(id="client-1", name="Sunfish Robotics")]

    def get_projects(
        self,
        workspace_id: str,
        *,
        archived: bool,
    ) -> list[ClockifyProject]:
        return [self.project]

    def get_tasks(
        self,
        workspace_id: str,
        project_id: str,
        *,
        active: bool,
    ) -> list[ClockifyTask]:
        return [self.task] if active is (self.task_status == "ACTIVE") else []

    def create_project(
        self,
        workspace_id: str,
        *,
        name: str,
        note: str,
        client_id: str,
    ) -> ClockifyProject:
        raise AssertionError("not expected")

    def update_project_name(
        self,
        workspace_id: str,
        project: ClockifyProject,
        name: str,
    ) -> ClockifyProject:
        raise AssertionError("not expected")

    def create_task(
        self,
        workspace_id: str,
        project_id: str,
        *,
        name: str,
    ) -> ClockifyTask:
        raise AssertionError("not expected")

    def update_task(
        self,
        workspace_id: str,
        task: ClockifyTask,
        *,
        name: str | None = None,
        status: TaskStatus | None = None,
    ) -> ClockifyTask:
        self.operations.append(f"{task.id}:{status}")
        updated = task.model_copy(
            update={
                "name": name if name is not None else task.name,
                "status": status if status is not None else task.status,
            }
        )
        self.task_status = updated.status
        return updated

    def get_project(self, workspace_id: str, project_id: str) -> ClockifyProject:
        return self.project

    def get_task(
        self,
        workspace_id: str,
        project_id: str,
        task_id: str,
    ) -> ClockifyTask:
        return self.task


class FakeJiraClient:
    def get_active_assigned_issues(self, project_key: str) -> list[JiraIssue]:
        return []

    def get_issues(self, keys: object) -> list[JiraIssue]:
        return [
            JiraIssue(
                key="SF-131",
                summary="Production Vehicle",
                status="In Progress",
                statusCategory="In Progress",
                issueType="Project / Phase",
            ),
            JiraIssue(
                key="SF-304",
                summary="Bench test PX4",
                status="Done",
                statusCategory="Done",
                assignee="Michael Bryan",
                issueType="Task",
                parentKey="SF-131",
                parentSummary="Production Vehicle",
            ),
        ]

    def get_issue(self, key: str) -> JiraIssue:
        return JiraIssue(
            key="SF-304",
            summary="Bench test PX4",
            status="In Progress",
            statusCategory="In Progress",
            assignee="David Htet",
            issueType="Task",
            parentKey="SF-131",
            parentSummary="Production Vehicle",
        )


def install_sync_fakes() -> tuple[FakeJiraSyncClockifyClient, AppContext]:
    client = FakeJiraSyncClockifyClient(
        api_key="test-key",
        base_url="https://clockify.example.test/api/v1",
    )
    app = AppContext(
        clockify_client_factory=lambda _config: client,
        jira_client_factory=lambda _config: FakeJiraClient(),
    )
    return client, app


def test_jira_sync_defaults_to_json_dry_run() -> None:
    client, app = install_sync_fakes()
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        ["--api-key", "test-key", "jira-sync", "--json"],
        obj=app,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "dry-run"
    assert payload["workspaceId"] == "workspace-1"
    assert payload["scope"] == {
        "kind": "assigned-active",
        "jiraProject": "SF",
        "issueKeys": [],
    }
    assert payload["inventory"] == {
        "activeIssues": 0,
        "jiraIssues": 2,
        "projects": 1,
        "tasks": 1,
    }
    assert payload["actions"] == [
        {
            "kind": "MARK_TASK_DONE",
            "jiraKey": "SF-304",
            "currentName": "SF-304 Bench test PX4",
            "desiredName": "SF-304 Bench test PX4",
            "projectKey": None,
            "projectId": "project-1",
            "taskId": "task-1",
            "jiraStatus": "Done",
            "message": "",
        }
    ]
    assert client.operations == []


def test_jira_sync_resolves_typed_jira_config_from_environment() -> None:
    client, app = install_sync_fakes()
    captured: list[JiraConfig] = []

    def jira_factory(config: JiraConfig) -> FakeJiraClient:
        captured.append(config)
        return FakeJiraClient()

    app = dataclasses.replace(app, jira_client_factory=jira_factory)
    result = CliRunner().invoke(
        clockify,
        ["--api-key", "test-key", "jira-sync", "--json"],
        obj=app,
        env={
            "JIRA_BASE_URL": "sunfishrobotics.atlassian.net",
            "JIRA_EMAIL": "michael@example.test",
            "JIRA_API_TOKEN": "secret-token",
        },
    )

    assert result.exit_code == 0
    assert client.operations == []
    assert captured == [
        JiraConfig(
            base_url="sunfishrobotics.atlassian.net",
            email="michael@example.test",
            api_token="secret-token",
        )
    ]


def test_jira_sync_human_dry_run_is_concise() -> None:
    _client, app = install_sync_fakes()
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        ["--api-key", "test-key", "jira-sync", "--dry-run"],
        obj=app,
    )

    assert result.exit_code == 0
    assert (
        result.output
        == "Dry run: 1 change\nMARK_TASK_DONE SF-304 — Jira status: Done\n"
    )


def test_jira_sync_apply_executes_and_reports_changes() -> None:
    client, app = install_sync_fakes()
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        ["--api-key", "test-key", "jira-sync", "--apply", "--json"],
        obj=app,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "apply"
    assert payload["applied"] == 1
    assert payload["verified"] is True
    assert client.operations == ["task-1:DONE"]


def test_jira_sync_can_target_issue_assigned_to_someone_else() -> None:
    client, app = install_sync_fakes()
    client.task_status = "DONE"
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        [
            "--api-key",
            "test-key",
            "jira-sync",
            "--issue",
            "sf-304",
            "--json",
        ],
        obj=app,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["scope"] == {
        "kind": "issues",
        "jiraProject": "SF",
        "issueKeys": ["SF-304"],
    }
    assert [(action["kind"], action["jiraKey"]) for action in payload["actions"]] == [
        ("REACTIVATE_TASK", "SF-304")
    ]
    assert client.operations == []


class EmptyJiraClient:
    def get_active_assigned_issues(self, project_key: str) -> list[JiraIssue]:
        return []

    def get_issues(self, keys: object) -> list[JiraIssue]:
        return []

    def get_issue(self, key: str) -> JiraIssue:
        raise AssertionError("not expected")


class FailingJiraClient:
    def get_active_assigned_issues(self, project_key: str) -> list[JiraIssue]:
        raise JiraError("Jira unavailable")

    def get_issues(self, keys: object) -> list[JiraIssue]:
        raise AssertionError("not expected")

    def get_issue(self, key: str) -> JiraIssue:
        raise AssertionError("not expected")


class DuplicateTaskClockifyClient(FakeJiraSyncClockifyClient):
    def get_tasks(
        self,
        workspace_id: str,
        project_id: str,
        *,
        active: bool,
    ) -> list[ClockifyTask]:
        if not active:
            return []
        return [
            self.task,
            self.task.model_copy(update={"id": "task-2"}),
        ]


def test_jira_sync_reports_when_no_changes_are_required() -> None:
    _client, app = install_sync_fakes()
    app = dataclasses.replace(
        app, jira_client_factory=lambda _config: EmptyJiraClient()
    )
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        ["--api-key", "test-key", "jira-sync", "--dry-run"],
        obj=app,
    )

    assert result.exit_code == 0
    assert (
        result.output
        == "No Clockify changes required for active Jira issues assigned to currentUser().\n"
    )


def test_jira_sync_reports_conflicts_as_json_and_exits_nonzero() -> None:
    client = DuplicateTaskClockifyClient(
        api_key="test-key",
        base_url="https://clockify.example.test/api/v1",
    )
    app = AppContext(
        clockify_client_factory=lambda _config: client,
        jira_client_factory=lambda _config: FakeJiraClient(),
    )
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        ["--api-key", "test-key", "jira-sync", "--dry-run", "--json"],
        obj=app,
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert [(action["kind"], action["jiraKey"]) for action in payload["actions"]] == [
        ("CONFLICT", "SF-304")
    ]
    assert client.operations == []


def test_jira_sync_reports_backend_errors_without_polluting_json() -> None:
    _client, app = install_sync_fakes()
    app = dataclasses.replace(
        app, jira_client_factory=lambda _config: FailingJiraClient()
    )
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        ["--api-key", "test-key", "jira-sync", "--json"],
        obj=app,
    )

    assert result.exit_code == 1
    assert result.output == "Error: Jira unavailable\n"
