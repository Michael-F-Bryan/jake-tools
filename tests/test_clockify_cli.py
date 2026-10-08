from __future__ import annotations

import json

from click.testing import CliRunner

from jake_tools.cli.clockify import ClockifyOptions, JiraOptions, clockify
from jake_tools.clockify import (
    CLOCKIFY_API_ROOT,
    ClockifyClientRecord,
    ClockifyError,
    ClockifyProject,
    ClockifyTask,
    ClockifyUser,
    TaskStatus,
)
from jake_tools.clockify_jira_sync import SyncReport
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


def test_whoami_prints_current_clockify_user(monkeypatch) -> None:
    client = FakeClockifyClient()
    monkeypatch.setattr(ClockifyOptions, "inventory_client", lambda self: client)
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        ["whoami", "--api-key", "test-key"],
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
    captured: list[ClockifyOptions] = []

    def fake_inventory_client(self: ClockifyOptions) -> FakeClockifyClient:
        captured.append(self)
        return FakeClockifyClient()

    monkeypatch.setattr(ClockifyOptions, "inventory_client", fake_inventory_client)
    runner = CliRunner()

    result = runner.invoke(clockify, ["whoami"])

    assert result.exit_code == 0
    assert captured == [
        ClockifyOptions(api_key="env-key", api_base_url=CLOCKIFY_API_ROOT)
    ]


def test_api_base_url_resolves_from_env_when_the_flag_is_omitted(monkeypatch) -> None:
    monkeypatch.setenv("CLOCKIFY_API_BASE_URL", "https://clockify.example.test/api/v1")
    captured: list[ClockifyOptions] = []

    def fake_inventory_client(self: ClockifyOptions) -> FakeClockifyClient:
        captured.append(self)
        return FakeClockifyClient()

    monkeypatch.setattr(ClockifyOptions, "inventory_client", fake_inventory_client)
    runner = CliRunner()

    result = runner.invoke(clockify, ["whoami", "--api-key", "test-key"])

    assert result.exit_code == 0
    assert captured == [
        ClockifyOptions(
            api_key="test-key", api_base_url="https://clockify.example.test/api/v1"
        )
    ]


def test_api_base_url_falls_back_to_the_default_root(monkeypatch) -> None:
    monkeypatch.delenv("CLOCKIFY_API_BASE_URL", raising=False)
    captured: list[ClockifyOptions] = []

    def fake_inventory_client(self: ClockifyOptions) -> FakeClockifyClient:
        captured.append(self)
        return FakeClockifyClient()

    monkeypatch.setattr(ClockifyOptions, "inventory_client", fake_inventory_client)
    runner = CliRunner()

    result = runner.invoke(clockify, ["whoami", "--api-key", "test-key"])

    assert result.exit_code == 0
    assert captured == [
        ClockifyOptions(api_key="test-key", api_base_url=CLOCKIFY_API_ROOT)
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


def install_sync_fakes(monkeypatch) -> FakeJiraSyncClockifyClient:
    """Patch the clockify/jira constructor seams with in-memory fakes.

    CLI-level tests are thin: they monkeypatch the client-constructor
    methods on the options models rather than exercising real HTTP clients
    or reconciliation logic (that lives at the library seam, tested
    directly against ``clockify_jira_sync``).
    """
    client = FakeJiraSyncClockifyClient(
        api_key="test-key",
        base_url="https://clockify.example.test/api/v1",
    )
    monkeypatch.setattr(ClockifyOptions, "inventory_client", lambda self: client)
    monkeypatch.setattr(JiraOptions, "inventory_client", lambda self: FakeJiraClient())
    return client


def test_jira_sync_defaults_to_json_dry_run(monkeypatch) -> None:
    client = install_sync_fakes(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        ["jira-sync", "--api-key", "test-key", "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    # The CLI prints exactly the SyncReport the MCP tool returns.
    report = SyncReport.model_validate(payload)
    assert payload == report.model_dump(mode="json")
    assert payload["mode"] == "preview"
    assert payload["applied"] is False
    assert payload["failure"] is None
    assert payload["workspace_id"] == "workspace-1"
    assert payload["client_id"] == "client-1"
    assert payload["jira_project"] == "SF"
    assert payload["clockify_client"] == "Sunfish Robotics"
    assert payload["scope"] == {
        "kind": "assigned-active",
        "jira_project": "SF",
        "issue_keys": [],
    }
    assert payload["inventory"] == {
        "active_issues": 0,
        "jira_issues": 2,
        "projects": 1,
        "tasks": 1,
    }
    assert payload["actions"] == [
        {
            "kind": "MARK_TASK_DONE",
            "jira_key": "SF-304",
            "current_name": "SF-304 Bench test PX4",
            "desired_name": "SF-304 Bench test PX4",
            "project_key": None,
            "project_id": "project-1",
            "task_id": "task-1",
            "jira_status": "Done",
            "message": "",
            "applied": False,
            "verified": False,
        }
    ]
    assert payload["conflicts"] == []
    assert len(payload["plan_digest"]) == 64
    assert client.operations == []


def test_jira_sync_resolves_typed_jira_config_from_environment(monkeypatch) -> None:
    client = install_sync_fakes(monkeypatch)
    captured: list[JiraOptions] = []

    def fake_inventory_client(self: JiraOptions) -> FakeJiraClient:
        captured.append(self)
        return FakeJiraClient()

    monkeypatch.setattr(JiraOptions, "inventory_client", fake_inventory_client)
    result = CliRunner().invoke(
        clockify,
        ["jira-sync", "--api-key", "test-key", "--json"],
        env={
            "JIRA_BASE_URL": "sunfishrobotics.atlassian.net",
            "JIRA_EMAIL": "michael@example.test",
            "JIRA_API_TOKEN": "secret-token",
        },
    )

    assert result.exit_code == 0
    assert client.operations == []
    assert captured == [
        JiraOptions(
            base_url="sunfishrobotics.atlassian.net",
            email="michael@example.test",
            api_token="secret-token",
        )
    ]


def test_jira_sync_human_dry_run_is_concise(monkeypatch) -> None:
    install_sync_fakes(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        ["jira-sync", "--api-key", "test-key", "--dry-run"],
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
        ["jira-sync", "--api-key", "test-key", "--apply", "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "apply"
    assert payload["applied"] is True
    assert payload["failure"] is None
    assert [
        (action["kind"], action["applied"], action["verified"])
        for action in payload["actions"]
    ] == [("MARK_TASK_DONE", True, True)]
    assert client.operations == ["task-1:DONE"]


def test_jira_sync_can_target_issue_assigned_to_someone_else(monkeypatch) -> None:
    client = install_sync_fakes(monkeypatch)
    client.task_status = "DONE"
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        [
            "jira-sync",
            "--api-key",
            "test-key",
            "--issue",
            "sf-304",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["scope"] == {
        "kind": "issues",
        "jira_project": "SF",
        "issue_keys": ["SF-304"],
    }
    assert [(action["kind"], action["jira_key"]) for action in payload["actions"]] == [
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


def test_jira_sync_reports_when_no_changes_are_required(monkeypatch) -> None:
    install_sync_fakes(monkeypatch)
    monkeypatch.setattr(JiraOptions, "inventory_client", lambda self: EmptyJiraClient())
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        ["jira-sync", "--api-key", "test-key", "--dry-run"],
    )

    assert result.exit_code == 0
    assert (
        result.output
        == "No Clockify changes required for active Jira issues assigned to currentUser().\n"
    )


def test_jira_sync_reports_conflicts_as_json_and_exits_nonzero(monkeypatch) -> None:
    client = DuplicateTaskClockifyClient(
        api_key="test-key",
        base_url="https://clockify.example.test/api/v1",
    )
    monkeypatch.setattr(ClockifyOptions, "inventory_client", lambda self: client)
    monkeypatch.setattr(JiraOptions, "inventory_client", lambda self: FakeJiraClient())
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        ["jira-sync", "--api-key", "test-key", "--dry-run", "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["actions"] == []
    assert [conflict["jira_key"] for conflict in payload["conflicts"]] == ["SF-304"]
    assert client.operations == []


def install_conflict_fakes(monkeypatch) -> DuplicateTaskClockifyClient:
    client = DuplicateTaskClockifyClient(
        api_key="test-key",
        base_url="https://clockify.example.test/api/v1",
    )
    monkeypatch.setattr(ClockifyOptions, "inventory_client", lambda self: client)
    monkeypatch.setattr(JiraOptions, "inventory_client", lambda self: FakeJiraClient())
    return client


def test_jira_sync_apply_with_conflicts_prints_report_and_writes_nothing(
    monkeypatch,
) -> None:
    client = install_conflict_fakes(monkeypatch)

    result = CliRunner().invoke(
        clockify, ["jira-sync", "--api-key", "test-key", "--apply", "--json"]
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["applied"] is False
    assert [conflict["jira_key"] for conflict in payload["conflicts"]] == ["SF-304"]
    assert client.operations == []


def test_jira_sync_apply_with_conflicts_in_text_mode_is_an_error(monkeypatch) -> None:
    client = install_conflict_fakes(monkeypatch)

    result = CliRunner().invoke(
        clockify, ["jira-sync", "--api-key", "test-key", "--apply"]
    )

    assert result.exit_code == 1
    assert result.output.startswith(
        "Error: Refusing to apply a plan with conflicts: SF-304: "
    )
    assert client.operations == []


class FailingWriteClockifyClient(FakeJiraSyncClockifyClient):
    def update_task(
        self,
        workspace_id: str,
        task: ClockifyTask,
        *,
        name: str | None = None,
        status: TaskStatus | None = None,
    ) -> ClockifyTask:
        raise ClockifyError("Clockify request failed for PUT /tasks: 500 Boom")


def install_failing_write_fakes(monkeypatch) -> None:
    client = FailingWriteClockifyClient(
        api_key="test-key",
        base_url="https://clockify.example.test/api/v1",
    )
    monkeypatch.setattr(ClockifyOptions, "inventory_client", lambda self: client)
    monkeypatch.setattr(JiraOptions, "inventory_client", lambda self: FakeJiraClient())


def test_jira_sync_apply_failure_prints_report_with_failure_and_exits_nonzero(
    monkeypatch,
) -> None:
    install_failing_write_fakes(monkeypatch)

    result = CliRunner().invoke(
        clockify, ["jira-sync", "--api-key", "test-key", "--apply", "--json"]
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["mode"] == "apply"
    assert payload["failure"] == {
        "jira_key": "SF-304",
        "kind": "MARK_TASK_DONE",
        "message": "Clockify request failed for PUT /tasks: 500 Boom",
    }
    assert [action["applied"] for action in payload["actions"]] == [False]


def test_jira_sync_apply_failure_in_text_mode_prints_only_the_error(
    monkeypatch,
) -> None:
    install_failing_write_fakes(monkeypatch)

    result = CliRunner().invoke(
        clockify, ["jira-sync", "--api-key", "test-key", "--apply"]
    )

    assert result.exit_code == 1
    assert result.output == "Error: Clockify request failed for PUT /tasks: 500 Boom\n"


def test_jira_sync_reports_backend_errors_without_polluting_json(monkeypatch) -> None:
    install_sync_fakes(monkeypatch)
    monkeypatch.setattr(
        JiraOptions, "inventory_client", lambda self: FailingJiraClient()
    )
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        ["jira-sync", "--api-key", "test-key", "--json"],
    )

    assert result.exit_code == 1
    assert result.output == "Error: Jira unavailable\n"
