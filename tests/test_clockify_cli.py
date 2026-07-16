from __future__ import annotations

import importlib
import json

from click.testing import CliRunner

from jake_tools.cli.clockify import clockify
from jake_tools.clockify import (
    ClockifyClientRecord,
    ClockifyProject,
    ClockifyTask,
    ClockifyUser,
    TaskStatus,
)
from jake_tools.clockify_jira_sync import JiraIssue

clockify_cli = importlib.import_module("jake_tools.cli.clockify")


class FakeClockifyClient:
    def __init__(self, *, api_key: str, base_url: str) -> None:
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


def test_whoami_prints_current_clockify_user(monkeypatch) -> None:
    monkeypatch.setattr(clockify_cli, "ClockifyClient", FakeClockifyClient)
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        ["--api-key", "test-key", "whoami"],
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
            status="ACTIVE",
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
        return [self.task] if active else []

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
        return task.model_copy(
            update={
                "name": name if name is not None else task.name,
                "status": status if status is not None else task.status,
            }
        )


class FakeAcliJiraClient:
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


def install_sync_fakes(monkeypatch) -> FakeJiraSyncClockifyClient:
    client = FakeJiraSyncClockifyClient(
        api_key="test-key",
        base_url="https://clockify.example.test/api/v1",
    )
    monkeypatch.setattr(clockify_cli, "ClockifyClient", lambda **_: client)
    monkeypatch.setattr(clockify_cli, "AcliJiraClient", FakeAcliJiraClient)
    return client


def test_jira_sync_defaults_to_json_dry_run(monkeypatch) -> None:
    client = install_sync_fakes(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        ["--api-key", "test-key", "jira-sync", "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "dry-run"
    assert payload["workspaceId"] == "workspace-1"
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


def test_jira_sync_human_dry_run_is_concise(monkeypatch) -> None:
    install_sync_fakes(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        ["--api-key", "test-key", "jira-sync", "--dry-run"],
    )

    assert result.exit_code == 0
    assert (
        result.output
        == "Dry run: 1 change\nMARK_TASK_DONE SF-304 — Jira status: Done\n"
    )


def test_jira_sync_apply_executes_and_reports_changes(monkeypatch) -> None:
    client = install_sync_fakes(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        ["--api-key", "test-key", "jira-sync", "--apply", "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "apply"
    assert payload["applied"] == 1
    assert client.operations == ["task-1:DONE"]
