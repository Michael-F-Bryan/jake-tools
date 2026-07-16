from __future__ import annotations

import json
import subprocess

import pytest

from jake_tools.clockify import ClockifyProject, ClockifyTask, TaskStatus
from jake_tools.clockify_jira_sync import (
    AcliJiraClient,
    JiraError,
    JiraIssue,
    SyncActionKind,
    plan_jira_sync,
)


class FakeCommandRunner:
    def __init__(self, *results: subprocess.CompletedProcess[str]) -> None:
        self.results = list(results)
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        return self.results.pop(0)


def completed(payload: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps(payload),
        stderr="",
    )


def test_acli_jira_client_hydrates_active_assigned_issues() -> None:
    runner = FakeCommandRunner(
        completed(
            [
                {
                    "key": "SF-427",
                    "fields": {
                        "summary": "Evaluate PX4 external control methods",
                        "status": {
                            "name": "In Progress",
                            "statusCategory": {"name": "In Progress"},
                        },
                    },
                }
            ]
        ),
        completed(
            {
                "key": "SF-427",
                "fields": {
                    "summary": "Evaluate PX4 external control methods",
                    "status": {
                        "name": "In Progress",
                        "statusCategory": {"name": "In Progress"},
                    },
                    "issuetype": {"name": "Task"},
                    "assignee": {"displayName": "Michael Bryan"},
                    "parent": {
                        "key": "SF-131",
                        "fields": {
                            "summary": "Production Vehicle - Investigations and Overhead"
                        },
                    },
                },
            }
        ),
    )
    client = AcliJiraClient(runner=runner)

    issues = client.get_active_assigned_issues("SF")

    assert issues == [
        JiraIssue(
            key="SF-427",
            summary="Evaluate PX4 external control methods",
            status="In Progress",
            statusCategory="In Progress",
            assignee="Michael Bryan",
            issueType="Task",
            parentKey="SF-131",
            parentSummary="Production Vehicle - Investigations and Overhead",
        )
    ]
    assert runner.commands[0] == [
        "acli",
        "jira",
        "workitem",
        "search",
        "--jql",
        'project = SF AND assignee = currentUser() AND status in ("In Progress", "Blocked", "In Review") ORDER BY key',
        "--fields",
        "key,summary,status,priority",
        "--json",
    ]
    assert runner.commands[1] == [
        "acli",
        "jira",
        "workitem",
        "view",
        "SF-427",
        "--fields",
        "*all",
        "--json",
    ]


def test_acli_jira_client_batches_issue_lookup_by_key() -> None:
    runner = FakeCommandRunner(
        completed(
            [
                {
                    "key": "SF-304",
                    "fields": {
                        "summary": "Bench test PX4",
                        "status": {
                            "name": "Done",
                            "statusCategory": {"name": "Done"},
                        },
                        "assignee": {"displayName": "Michael Bryan"},
                    },
                },
                {
                    "key": "SF-438",
                    "fields": {
                        "summary": "Check newer flight controller boards",
                        "status": {
                            "name": "To Do",
                            "statusCategory": {"name": "To Do"},
                        },
                        "assignee": None,
                    },
                },
            ]
        )
    )
    client = AcliJiraClient(runner=runner)

    issues = client.get_issues(["SF-438", "SF-304", "SF-438"])

    assert [issue.key for issue in issues] == ["SF-304", "SF-438"]
    assert issues[0].status_category == "Done"
    assert issues[1].assignee is None
    assert runner.commands[0][5] == "key in (SF-304,SF-438) ORDER BY key"


def test_acli_jira_client_skips_command_for_empty_issue_lookup() -> None:
    runner = FakeCommandRunner()
    client = AcliJiraClient(runner=runner)

    assert client.get_issues([]) == []
    assert runner.commands == []


def test_acli_jira_client_reports_command_failure() -> None:
    runner = FakeCommandRunner(
        subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="not authenticated",
        )
    )
    client = AcliJiraClient(runner=runner)

    with pytest.raises(JiraError, match="not authenticated"):
        client.get_issues(["SF-304"])


def test_acli_jira_client_reports_malformed_json() -> None:
    runner = FakeCommandRunner(
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="not-json",
            stderr="",
        )
    )
    client = AcliJiraClient(runner=runner)

    with pytest.raises(JiraError, match="invalid JSON"):
        client.get_issues(["SF-304"])


def jira_issue(
    key: str,
    summary: str,
    *,
    status: str = "To Do",
    status_category: str = "To Do",
    assignee: str | None = "Michael Bryan",
    issue_type: str = "Task",
    parent_key: str | None = "SF-131",
    parent_summary: str | None = "Production Vehicle",
) -> JiraIssue:
    return JiraIssue(
        key=key,
        summary=summary,
        status=status,
        statusCategory=status_category,
        assignee=assignee,
        issueType=issue_type,
        parentKey=parent_key,
        parentSummary=parent_summary,
    )


def clockify_project(
    key: str,
    name: str,
    *,
    project_id: str | None = None,
    archived: bool = False,
) -> ClockifyProject:
    return ClockifyProject(
        id=project_id or f"project-{key}",
        name=name,
        note=f"Jira: {key}",
        archived=archived,
        billable=False,
        color="#689F38",
        public=True,
        clientId="client-1",
    )


def clockify_task(
    key: str,
    summary: str,
    *,
    project_id: str = "project-SF-131",
    task_id: str | None = None,
    status: TaskStatus = "ACTIVE",
) -> ClockifyTask:
    return ClockifyTask(
        id=task_id or f"task-{key}",
        name=f"{key} {summary}",
        projectId=project_id,
        status=status,
    )


def test_plan_creates_missing_projects_and_task_for_active_issues() -> None:
    phase = jira_issue(
        "SF-1",
        "Software in the Loop (SITL) / Simulator",
        status="In Progress",
        status_category="In Progress",
        issue_type="Project / Phase",
        parent_key=None,
        parent_summary=None,
    )
    task = jira_issue(
        "SF-427",
        "Evaluate PX4 external control methods",
        status="In Progress",
        status_category="In Progress",
    )
    parent = jira_issue(
        "SF-131",
        "Production Vehicle",
        status="In Progress",
        status_category="In Progress",
        issue_type="Project / Phase",
        parent_key=None,
        parent_summary=None,
    )

    plan = plan_jira_sync(
        active_issues=[phase, task],
        jira_issues=[phase, task, parent],
        projects=[],
        tasks=[],
    )

    assert [(action.kind, action.jira_key) for action in plan.actions] == [
        (SyncActionKind.CREATE_PROJECT, "SF-1"),
        (SyncActionKind.CREATE_PROJECT, "SF-131"),
        (SyncActionKind.CREATE_TASK, "SF-427"),
    ]
    assert plan.actions[-1].project_key == "SF-131"
    assert (
        plan.actions[-1].desired_name == "SF-427 Evaluate PX4 external control methods"
    )


def test_plan_renames_drift_and_completes_done_jira_tasks() -> None:
    project = clockify_project("SF-131", "Old project name")
    tasks = [
        clockify_task("SF-305", "Wet test PX4 using Tom's thesis bot"),
        clockify_task("SF-304", "Bench test PX4"),
    ]
    jira = [
        jira_issue("SF-131", "Production Vehicle", issue_type="Project / Phase"),
        jira_issue("SF-305", "Wet test PX4 using Zoda"),
        jira_issue(
            "SF-304",
            "Bench test PX4",
            status="Done",
            status_category="Done",
        ),
    ]

    plan = plan_jira_sync(
        active_issues=[],
        jira_issues=jira,
        projects=[project],
        tasks=tasks,
    )

    assert [(action.kind, action.jira_key) for action in plan.actions] == [
        (SyncActionKind.RENAME_PROJECT, "SF-131"),
        (SyncActionKind.RENAME_TASK, "SF-305"),
        (SyncActionKind.MARK_TASK_DONE, "SF-304"),
    ]


def test_plan_reactivates_done_task_when_it_is_active_and_assigned() -> None:
    project = clockify_project("SF-131", "Production Vehicle")
    task = clockify_task(
        "SF-427",
        "Old summary",
        status="DONE",
    )
    issue = jira_issue(
        "SF-427",
        "Evaluate PX4 external control methods",
        status="In Progress",
        status_category="In Progress",
    )

    plan = plan_jira_sync(
        active_issues=[issue],
        jira_issues=[issue],
        projects=[project],
        tasks=[task],
    )

    assert len(plan.actions) == 1
    action = plan.actions[0]
    assert action.kind == SyncActionKind.REACTIVATE_TASK
    assert action.desired_name == "SF-427 Evaluate PX4 external control methods"


def test_plan_ignores_archived_project_and_creates_usable_replacements() -> None:
    archived = clockify_project(
        "SF-131",
        "Production Vehicle (Jira SF-131)",
        archived=True,
    )
    archived_task = clockify_task(
        "SF-427",
        "Evaluate PX4 external control methods",
        project_id=archived.id,
    )
    issue = jira_issue(
        "SF-427",
        "Evaluate PX4 external control methods",
        status="In Progress",
        status_category="In Progress",
    )

    plan = plan_jira_sync(
        active_issues=[issue],
        jira_issues=[issue],
        projects=[archived],
        tasks=[archived_task],
    )

    assert [(action.kind, action.jira_key) for action in plan.actions] == [
        (SyncActionKind.CREATE_PROJECT, "SF-131"),
        (SyncActionKind.CREATE_TASK, "SF-427"),
    ]


def test_plan_does_not_deactivate_todo_or_reassigned_tasks() -> None:
    project = clockify_project("SF-131", "Production Vehicle")
    tasks = [
        clockify_task("SF-115", "Get Virtual Anchor running in SITL"),
        clockify_task("SF-353", "Vehicle Control Logic"),
    ]
    jira = [
        jira_issue("SF-115", "Get Virtual Anchor running in SITL"),
        jira_issue(
            "SF-353",
            "Vehicle Control Logic",
            status="In Progress",
            status_category="In Progress",
            assignee="Taylor Odishoo",
        ),
    ]

    plan = plan_jira_sync(
        active_issues=[],
        jira_issues=jira,
        projects=[project],
        tasks=tasks,
    )

    assert plan.actions == ()


def test_plan_reports_ambiguous_active_task_duplicates_as_conflict() -> None:
    project = clockify_project("SF-131", "Production Vehicle")
    tasks = [
        clockify_task("SF-427", "Evaluate PX4", task_id="task-1"),
        clockify_task("SF-427", "Evaluate PX4", task_id="task-2"),
    ]
    issue = jira_issue(
        "SF-427",
        "Evaluate PX4",
        status="In Progress",
        status_category="In Progress",
    )

    plan = plan_jira_sync(
        active_issues=[issue],
        jira_issues=[issue],
        projects=[project],
        tasks=tasks,
    )

    assert len(plan.actions) == 1
    assert plan.actions[0].kind == SyncActionKind.CONFLICT
    assert "multiple active Clockify tasks" in plan.actions[0].message
