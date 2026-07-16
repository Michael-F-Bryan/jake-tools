from __future__ import annotations

from collections.abc import Iterable

import pytest

from jake_tools.clockify import (
    ClockifyClientRecord,
    ClockifyProject,
    ClockifyTask,
    ClockifyUser,
    TaskStatus,
)
from jake_tools.clockify_jira_sync import (
    SyncAction,
    SyncActionKind,
    SyncApplyError,
    SyncPlan,
    SyncPreparationError,
    apply_sync_plan,
    plan_jira_sync,
    prepare_jira_sync,
)
from jake_tools.jira import JiraIssue


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


def test_plan_conflicts_when_existing_task_cannot_belong_to_missing_parent() -> None:
    wrong_project = clockify_project("SF-999", "Unrelated project")
    task = clockify_task(
        "SF-427",
        "Old summary",
        project_id=wrong_project.id,
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
        projects=[wrong_project],
        tasks=[task],
    )

    assert [(action.kind, action.jira_key) for action in plan.actions] == [
        (SyncActionKind.CONFLICT, "SF-427"),
        (SyncActionKind.CREATE_PROJECT, "SF-131"),
    ]
    assert all(action.kind != SyncActionKind.RENAME_TASK for action in plan.actions)
    assert "different active project" in plan.actions[0].message


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


def test_plan_keeps_todo_or_reassigned_tasks_active_while_syncing_names() -> None:
    project = clockify_project("SF-131", "Production Vehicle")
    tasks = [
        clockify_task("SF-115", "Old Virtual Anchor summary"),
        clockify_task("SF-353", "Old vehicle-control summary"),
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

    assert [(action.kind, action.jira_key) for action in plan.actions] == [
        (SyncActionKind.RENAME_TASK, "SF-115"),
        (SyncActionKind.RENAME_TASK, "SF-353"),
    ]


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


class FakeClockifySyncClient:
    def __init__(self) -> None:
        self.operations: list[str] = []
        self.project_note_override: str | None = None

    def create_project(
        self,
        workspace_id: str,
        *,
        name: str,
        note: str,
        client_id: str,
    ) -> ClockifyProject:
        self.operations.append(f"create_project:{note}")
        key = note.removeprefix("Jira: ")
        return ClockifyProject(
            id=f"created-{key}",
            name=name,
            note=note,
            archived=False,
            billable=False,
            color="#689F38",
            public=True,
            clientId=client_id,
        )

    def update_project_name(
        self,
        workspace_id: str,
        project: ClockifyProject,
        name: str,
    ) -> ClockifyProject:
        self.operations.append(f"rename_project:{project.id}")
        return project.model_copy(
            update={
                "name": name,
                "note": self.project_note_override or project.note,
            }
        )

    def create_task(
        self,
        workspace_id: str,
        project_id: str,
        *,
        name: str,
    ) -> ClockifyTask:
        self.operations.append(f"create_task:{project_id}")
        return ClockifyTask(
            id=f"created-{name.split()[0]}",
            name=name,
            projectId=project_id,
            status="ACTIVE",
        )

    def update_task(
        self,
        workspace_id: str,
        task: ClockifyTask,
        *,
        name: str | None = None,
        status: TaskStatus | None = None,
    ) -> ClockifyTask:
        self.operations.append(f"update_task:{task.id}:{status or task.status}")
        return task.model_copy(
            update={
                "name": name if name is not None else task.name,
                "status": status if status is not None else task.status,
            }
        )


def test_apply_sync_plan_creates_project_before_dependent_task() -> None:
    client = FakeClockifySyncClient()
    plan = SyncPlan(
        actions=(
            SyncAction(
                kind=SyncActionKind.CREATE_PROJECT,
                jira_key="SF-131",
                project_key="SF-131",
                desired_name="Production Vehicle",
            ),
            SyncAction(
                kind=SyncActionKind.CREATE_TASK,
                jira_key="SF-427",
                project_key="SF-131",
                desired_name="SF-427 Evaluate PX4 external control methods",
            ),
        )
    )

    result = apply_sync_plan(
        plan,
        clockify=client,
        workspace_id="workspace-1",
        client_id="client-1",
        projects=[],
        tasks=[],
    )

    assert client.operations == [
        "create_project:Jira: SF-131",
        "create_task:created-SF-131",
    ]
    assert [applied.action.kind for applied in result.applied] == [
        SyncActionKind.CREATE_PROJECT,
        SyncActionKind.CREATE_TASK,
    ]
    assert result.applied[-1].task is not None
    assert result.applied[-1].task.project_id == "created-SF-131"


def test_apply_sync_plan_updates_names_and_task_statuses() -> None:
    client = FakeClockifySyncClient()
    project = clockify_project("SF-131", "Old project")
    rename_task = clockify_task("SF-305", "Old summary")
    reactivate_task = clockify_task("SF-427", "Old summary", status="DONE")
    complete_task = clockify_task("SF-304", "Bench test PX4")
    plan = SyncPlan(
        actions=(
            SyncAction(
                kind=SyncActionKind.RENAME_PROJECT,
                jira_key="SF-131",
                project_id=project.id,
                desired_name="Production Vehicle",
            ),
            SyncAction(
                kind=SyncActionKind.REACTIVATE_TASK,
                jira_key="SF-427",
                task_id=reactivate_task.id,
                desired_name="SF-427 Evaluate PX4 external control methods",
            ),
            SyncAction(
                kind=SyncActionKind.RENAME_TASK,
                jira_key="SF-305",
                task_id=rename_task.id,
                desired_name="SF-305 Wet test PX4 using Zoda",
            ),
            SyncAction(
                kind=SyncActionKind.MARK_TASK_DONE,
                jira_key="SF-304",
                task_id=complete_task.id,
                desired_name=complete_task.name,
            ),
        )
    )

    result = apply_sync_plan(
        plan,
        clockify=client,
        workspace_id="workspace-1",
        client_id="client-1",
        projects=[project],
        tasks=[rename_task, reactivate_task, complete_task],
    )

    assert result.applied[0].project is not None
    assert result.applied[0].project.name == "Production Vehicle"
    assert result.applied[1].task is not None
    assert result.applied[1].task.status == "ACTIVE"
    assert result.applied[2].task is not None
    assert result.applied[2].task.name == "SF-305 Wet test PX4 using Zoda"
    assert result.applied[3].task is not None
    assert result.applied[3].task.status == "DONE"


def test_apply_sync_plan_rejects_renamed_project_with_changed_jira_note() -> None:
    client = FakeClockifySyncClient()
    client.project_note_override = "Jira: SF-999"
    project = clockify_project("SF-131", "Old project")
    plan = SyncPlan(
        actions=(
            SyncAction(
                kind=SyncActionKind.RENAME_PROJECT,
                jira_key="SF-131",
                project_id=project.id,
                desired_name="Production Vehicle",
            ),
        )
    )

    with pytest.raises(SyncApplyError, match="project note"):
        apply_sync_plan(
            plan,
            clockify=client,
            workspace_id="workspace-1",
            client_id="client-1",
            projects=[project],
            tasks=[],
        )


def test_apply_sync_plan_refuses_conflicts_before_mutating() -> None:
    client = FakeClockifySyncClient()
    plan = SyncPlan(
        actions=(
            SyncAction(
                kind=SyncActionKind.CONFLICT,
                jira_key="SF-427",
                message="multiple active Clockify tasks reference SF-427",
            ),
        )
    )

    with pytest.raises(SyncApplyError, match="SF-427"):
        apply_sync_plan(
            plan,
            clockify=client,
            workspace_id="workspace-1",
            client_id="client-1",
            projects=[],
            tasks=[],
        )

    assert client.operations == []


class FakeClockifyInventoryClient(FakeClockifySyncClient):
    def __init__(
        self,
        *,
        clients: list[ClockifyClientRecord],
        projects: list[ClockifyProject],
        active_tasks: list[ClockifyTask],
        done_tasks: list[ClockifyTask],
    ) -> None:
        super().__init__()
        self.clients = clients
        self.projects = projects
        self.active_tasks = active_tasks
        self.done_tasks = done_tasks

    def get_user(self) -> ClockifyUser:
        return ClockifyUser(id="user-1", activeWorkspace="workspace-1")

    def get_clients(self, workspace_id: str) -> list[ClockifyClientRecord]:
        return self.clients

    def get_projects(
        self,
        workspace_id: str,
        *,
        archived: bool,
    ) -> list[ClockifyProject]:
        assert archived is False
        return self.projects

    def get_tasks(
        self,
        workspace_id: str,
        project_id: str,
        *,
        active: bool,
    ) -> list[ClockifyTask]:
        source = self.active_tasks if active else self.done_tasks
        return [task for task in source if task.project_id == project_id]


class FakeJiraInventoryClient:
    def __init__(
        self,
        *,
        active_issues: list[JiraIssue],
        issues: list[JiraIssue],
    ) -> None:
        self.active_issues = active_issues
        self.issues = {issue.key: issue for issue in issues}
        self.requested_keys: list[str] = []

    def get_active_assigned_issues(self, project_key: str) -> list[JiraIssue]:
        assert project_key == "SF"
        return self.active_issues

    def get_issues(self, keys: Iterable[str]) -> list[JiraIssue]:
        self.requested_keys = sorted(keys)
        return [self.issues[key] for key in self.requested_keys if key in self.issues]


def test_prepare_jira_sync_collects_inventory_and_builds_plan() -> None:
    project = clockify_project("SF-131", "Production Vehicle")
    active_task = clockify_task("SF-304", "Bench test PX4")
    done_task = clockify_task("SF-427", "Evaluate PX4", status="DONE")
    active_issue = jira_issue(
        "SF-427",
        "Evaluate PX4 external control methods",
        status="In Progress",
        status_category="In Progress",
    )
    issues = [
        jira_issue("SF-131", "Production Vehicle", issue_type="Project / Phase"),
        jira_issue(
            "SF-304",
            "Bench test PX4",
            status="Done",
            status_category="Done",
        ),
        active_issue,
    ]
    clockify = FakeClockifyInventoryClient(
        clients=[ClockifyClientRecord(id="client-1", name="Sunfish Robotics")],
        projects=[project],
        active_tasks=[active_task],
        done_tasks=[done_task],
    )
    jira = FakeJiraInventoryClient(active_issues=[active_issue], issues=issues)

    snapshot = prepare_jira_sync(
        clockify=clockify,
        jira=jira,
        jira_project="SF",
        clockify_client="Sunfish Robotics",
    )

    assert snapshot.workspace_id == "workspace-1"
    assert snapshot.client_id == "client-1"
    assert {task.id for task in snapshot.tasks} == {active_task.id, done_task.id}
    assert jira.requested_keys == ["SF-131", "SF-304", "SF-427"]
    assert [(action.kind, action.jira_key) for action in snapshot.plan.actions] == [
        (SyncActionKind.REACTIVATE_TASK, "SF-427"),
        (SyncActionKind.MARK_TASK_DONE, "SF-304"),
    ]


def test_prepare_jira_sync_requires_exact_active_clockify_client() -> None:
    clockify = FakeClockifyInventoryClient(
        clients=[
            ClockifyClientRecord(
                id="client-1",
                name="Sunfish Robotics",
                archived=True,
            )
        ],
        projects=[],
        active_tasks=[],
        done_tasks=[],
    )
    jira = FakeJiraInventoryClient(active_issues=[], issues=[])

    with pytest.raises(SyncPreparationError, match="Sunfish Robotics"):
        prepare_jira_sync(
            clockify=clockify,
            jira=jira,
            jira_project="SF",
            clockify_client="Sunfish Robotics",
        )
