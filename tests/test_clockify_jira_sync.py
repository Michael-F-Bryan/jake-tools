from __future__ import annotations

from collections.abc import Iterable

import pytest
from pydantic import ValidationError

from jake_tools.clockify import (
    ClockifyClientRecord,
    ClockifyError,
    ClockifyProject,
    ClockifyTask,
    ClockifyUser,
    TaskStatus,
)
from jake_tools.clockify_jira_sync import (
    AppliedSyncAction,
    ClockifyIndex,
    CompleteTaskAction,
    ConflictAction,
    CreateProjectAction,
    CreateTaskAction,
    ReactivateTaskAction,
    RenameProjectAction,
    RenameTaskAction,
    SyncActionKind,
    SyncApplyError,
    SyncConflictError,
    SyncFailure,
    SyncPlan,
    SyncPlanStaleError,
    SyncPreparationError,
    SyncReport,
    SyncReportInventory,
    SyncReportScope,
    SyncScope,
    SyncSnapshot,
    apply_sync_plan,
    plan_digest,
    plan_jira_sync,
    prepare_jira_sync,
    run_jira_sync,
)
from jake_tools.jira import JiraIssue, JiraStatusCategory


def jira_issue(
    key: str,
    summary: str,
    *,
    status: str = "To Do",
    status_category: JiraStatusCategory = "To Do",
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


def build_index(
    projects: list[ClockifyProject],
    tasks: list[ClockifyTask],
) -> ClockifyIndex:
    return ClockifyIndex.build(projects, tasks)


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
        index=build_index([], []),
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
        index=build_index([project], tasks),
    )

    assert [(action.kind, action.jira_key) for action in plan.actions] == [
        (SyncActionKind.RENAME_PROJECT, "SF-131"),
        (SyncActionKind.RENAME_TASK, "SF-305"),
        (SyncActionKind.MARK_TASK_DONE, "SF-304"),
    ]


def test_plan_does_not_rename_project_for_whitespace_only_summary_drift() -> None:
    # A Jira summary that only differs from the Clockify project name by
    # leading/trailing whitespace should not be treated as a rename: both
    # the create and rename paths route through clockify_project_name_for_jira,
    # which strips the summary before naming or comparing.
    project = clockify_project("SF-131", "Production Vehicle")
    jira = [
        jira_issue(
            "SF-131",
            "  Production Vehicle  ",
            issue_type="Project / Phase",
        )
    ]

    plan = plan_jira_sync(
        active_issues=[],
        jira_issues=jira,
        index=build_index([project], []),
    )

    assert plan.actions == ()


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
        index=build_index([project], [task]),
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
        index=build_index([wrong_project], [task]),
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
        index=build_index([archived], [archived_task]),
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
        index=build_index([project], tasks),
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
        index=build_index([project], tasks),
    )

    assert len(plan.actions) == 1
    assert plan.actions[0].kind == SyncActionKind.CONFLICT
    assert "multiple active Clockify tasks" in plan.actions[0].message


def test_plan_and_apply_agree_on_duplicate_task_handling_through_shared_index() -> None:
    # Build ONE index and feed it to both plan_jira_sync and apply_sync_plan,
    # proving they see the same duplicate-task grouping instead of each
    # recomputing "group by Jira key, keep unique matches" independently and
    # possibly drifting apart.
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
    index = build_index([project], tasks)

    plan = plan_jira_sync(active_issues=[issue], jira_issues=[issue], index=index)

    assert plan.has_conflicts
    client = FakeClockifySyncClient()
    with pytest.raises(SyncApplyError, match="multiple active Clockify tasks"):
        apply_sync_plan(
            plan,
            clockify=client,
            workspace_id="workspace-1",
            client_id="client-1",
            index=index,
        )
    assert client.operations == []


def test_create_task_action_without_project_key_is_unrepresentable() -> None:
    with pytest.raises(ValidationError):
        CreateTaskAction(  # pyright: ignore[reportCallIssue]
            jira_key="SF-427",
            desired_name="SF-427 Evaluate PX4 external control methods",
        )


class FakeClockifySyncClient:
    def __init__(self) -> None:
        self.operations: list[str] = []
        self.project_note_override: str | None = None
        self._projects_by_id: dict[str, ClockifyProject] = {}
        self._tasks_by_id: dict[str, ClockifyTask] = {}
        self.task_read_override: ClockifyTask | None = None

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
        project = ClockifyProject(
            id=f"created-{key}",
            name=name,
            note=note,
            archived=False,
            billable=False,
            color="#689F38",
            public=True,
            clientId=client_id,
        )
        self._projects_by_id[project.id] = project
        return project

    def update_project_name(
        self,
        workspace_id: str,
        project: ClockifyProject,
        name: str,
    ) -> ClockifyProject:
        self.operations.append(f"rename_project:{project.id}")
        updated = project.model_copy(
            update={
                "name": name,
                "note": self.project_note_override or project.note,
            }
        )
        self._projects_by_id[updated.id] = updated
        return updated

    def create_task(
        self,
        workspace_id: str,
        project_id: str,
        *,
        name: str,
    ) -> ClockifyTask:
        self.operations.append(f"create_task:{project_id}")
        task = ClockifyTask(
            id=f"created-{name.split()[0]}",
            name=name,
            projectId=project_id,
            status="ACTIVE",
        )
        self._tasks_by_id[task.id] = task
        return task

    def update_task(
        self,
        workspace_id: str,
        task: ClockifyTask,
        *,
        name: str | None = None,
        status: TaskStatus | None = None,
    ) -> ClockifyTask:
        self.operations.append(f"update_task:{task.id}:{status or task.status}")
        updated = task.model_copy(
            update={
                "name": name if name is not None else task.name,
                "status": status if status is not None else task.status,
            }
        )
        self._tasks_by_id[updated.id] = updated
        return updated

    def get_project(self, workspace_id: str, project_id: str) -> ClockifyProject:
        return self._projects_by_id[project_id]

    def get_task(
        self,
        workspace_id: str,
        project_id: str,
        task_id: str,
    ) -> ClockifyTask:
        return self.task_read_override or self._tasks_by_id[task_id]


def test_apply_sync_plan_creates_project_before_dependent_task() -> None:
    client = FakeClockifySyncClient()
    plan = SyncPlan(
        actions=(
            CreateProjectAction(
                jira_key="SF-131",
                project_key="SF-131",
                desired_name="Production Vehicle",
            ),
            CreateTaskAction(
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
        index=build_index([], []),
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
            RenameProjectAction(
                jira_key="SF-131",
                project_id=project.id,
                desired_name="Production Vehicle",
            ),
            ReactivateTaskAction(
                jira_key="SF-427",
                task_id=reactivate_task.id,
                desired_name="SF-427 Evaluate PX4 external control methods",
            ),
            RenameTaskAction(
                jira_key="SF-305",
                task_id=rename_task.id,
                desired_name="SF-305 Wet test PX4 using Zoda",
            ),
            CompleteTaskAction(
                jira_key="SF-304",
                task_id=complete_task.id,
                desired_name=complete_task.name,
                jira_status="Done",
            ),
        )
    )

    result = apply_sync_plan(
        plan,
        clockify=client,
        workspace_id="workspace-1",
        client_id="client-1",
        index=build_index([project], [rename_task, reactivate_task, complete_task]),
    )

    assert result.applied[0].project is not None
    assert result.applied[0].project.name == "Production Vehicle"
    assert result.applied[1].task is not None
    assert result.applied[1].task.status == "ACTIVE"
    assert result.applied[2].task is not None
    assert result.applied[2].task.name == "SF-305 Wet test PX4 using Zoda"
    assert result.applied[3].task is not None
    assert result.applied[3].task.status == "DONE"


def test_apply_sync_plan_rejects_stale_task_reread() -> None:
    client = FakeClockifySyncClient()
    task = clockify_task("SF-304", "Bench test PX4", status="DONE")
    client.task_read_override = task
    plan = SyncPlan(
        actions=(
            ReactivateTaskAction(
                jira_key="SF-304",
                task_id=task.id,
                desired_name=task.name,
            ),
        )
    )

    with pytest.raises(SyncApplyError, match="expected 'ACTIVE'"):
        apply_sync_plan(
            plan,
            clockify=client,
            workspace_id="workspace-1",
            client_id="client-1",
            index=build_index([], [task]),
        )


def test_apply_sync_plan_rejects_renamed_project_with_changed_jira_note() -> None:
    client = FakeClockifySyncClient()
    client.project_note_override = "Jira: SF-999"
    project = clockify_project("SF-131", "Old project")
    plan = SyncPlan(
        actions=(
            RenameProjectAction(
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
            index=build_index([project], []),
        )


def test_apply_sync_plan_refuses_conflicts_before_mutating() -> None:
    client = FakeClockifySyncClient()
    plan = SyncPlan(
        actions=(
            ConflictAction(
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
            index=build_index([], []),
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
        self.requested_issue: str | None = None
        self.active_project_requests: list[str] = []

    def get_active_assigned_issues(self, project_key: str) -> list[JiraIssue]:
        assert project_key == "SF"
        self.active_project_requests.append(project_key)
        return self.active_issues

    def get_issue(self, key: str) -> JiraIssue:
        self.requested_issue = key
        return self.issues[key]

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
    assert snapshot.index.projects_by_jira_key["SF-131"] == project


def test_prepare_jira_sync_targets_exact_issue_regardless_of_assignee() -> None:
    project = clockify_project("SF-131", "Production Vehicle")
    done_task = clockify_task("SF-304", "Bench test PX4", status="DONE")
    unrelated_tasks = [
        clockify_task("SF-999", "Unrelated work", task_id="unrelated-1"),
        clockify_task("SF-999", "Unrelated work", task_id="unrelated-2"),
    ]
    issue = jira_issue(
        "SF-304",
        "Bench test PX4",
        status="In Progress",
        status_category="In Progress",
        assignee="David Htet",
    )
    parent = jira_issue(
        "SF-131",
        "Production Vehicle",
        issue_type="Project / Phase",
        parent_key=None,
        parent_summary=None,
    )
    unrelated = jira_issue("SF-999", "Unrelated work")
    clockify = FakeClockifyInventoryClient(
        clients=[ClockifyClientRecord(id="client-1", name="Sunfish Robotics")],
        projects=[project],
        active_tasks=unrelated_tasks,
        done_tasks=[done_task],
    )
    jira = FakeJiraInventoryClient(
        active_issues=[],
        issues=[issue, parent, unrelated],
    )

    snapshot = prepare_jira_sync(
        clockify=clockify,
        jira=jira,
        jira_project="SF",
        clockify_client="Sunfish Robotics",
        issue_keys=["sf-304"],
    )

    assert snapshot.scope == "issues"
    assert snapshot.requested_issue_keys == ("SF-304",)
    assert snapshot.active_issues == (issue,)
    assert jira.requested_issue == "SF-304"
    assert jira.requested_keys == ["SF-131", "SF-304"]
    assert jira.active_project_requests == []
    assert [(action.kind, action.jira_key) for action in snapshot.plan.actions] == [
        (SyncActionKind.REACTIVATE_TASK, "SF-304")
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


# --- SyncReport and plan_digest -------------------------------------------------


def _drift_clients() -> tuple[FakeClockifyInventoryClient, FakeJiraInventoryClient]:
    """Inventory where SF-427 needs reactivating and SF-304 marking done."""
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
        jira_issue("SF-304", "Bench test PX4", status="Done", status_category="Done"),
        active_issue,
    ]
    clockify = FakeClockifyInventoryClient(
        clients=[ClockifyClientRecord(id="client-1", name="Sunfish Robotics")],
        projects=[project],
        active_tasks=[active_task],
        done_tasks=[done_task],
    )
    # The sync fake's get_task reads what it holds; seed it with the
    # inventory so update_task's read-back sees the existing tasks.
    for task in (active_task, done_task):
        clockify._tasks_by_id[task.id] = task
    jira = FakeJiraInventoryClient(active_issues=[active_issue], issues=issues)
    return clockify, jira


def _snapshot_with_drift() -> SyncSnapshot:
    clockify, jira = _drift_clients()
    return prepare_jira_sync(
        clockify=clockify,
        jira=jira,
        jira_project="SF",
        clockify_client="Sunfish Robotics",
    )


def test_sync_report_separates_actions_from_conflicts_and_carries_digest() -> None:
    snapshot = _snapshot_with_drift()

    report = SyncReport.from_snapshot(
        snapshot, jira_project="SF", clockify_client="Sunfish Robotics"
    )

    assert report.mode == "preview"
    assert report.applied is False
    assert report.failure is None
    assert report.scope == SyncReportScope(
        kind=SyncScope.ASSIGNED_ACTIVE, jira_project="SF"
    )
    assert report.inventory == SyncReportInventory(
        active_issues=1, jira_issues=3, projects=1, tasks=2
    )
    assert [(action.kind, action.jira_key) for action in report.actions] == [
        (SyncActionKind.REACTIVATE_TASK, "SF-427"),
        (SyncActionKind.MARK_TASK_DONE, "SF-304"),
    ]
    assert report.conflicts == ()
    assert not report.has_conflicts
    assert report.plan_digest == plan_digest(snapshot)
    assert len(report.plan_digest) == 64
    # The report round-trips through JSON unchanged: the CLI prints exactly this.
    assert SyncReport.model_validate_json(report.model_dump_json()) == report


def test_sync_report_lists_conflicts_separately() -> None:
    orphan = jira_issue("SF-900", "No parent", parent_key=None, parent_summary=None)
    plan = plan_jira_sync(
        active_issues=[orphan], jira_issues=[orphan], index=build_index([], [])
    )
    snapshot = SyncSnapshot(
        workspace_id="workspace-1",
        client_id="client-1",
        scope=SyncScope.ISSUES,
        requested_issue_keys=("SF-900",),
        active_issues=(orphan,),
        jira_issues=(orphan,),
        projects=(),
        tasks=(),
        index=build_index([], []),
        plan=plan,
    )

    report = SyncReport.from_snapshot(
        snapshot, jira_project="SF", clockify_client="Sunfish Robotics"
    )

    assert report.actions == ()
    assert [conflict.jira_key for conflict in report.conflicts] == ["SF-900"]
    assert report.has_conflicts
    assert report.scope.issue_keys == ("SF-900",)


def test_plan_digest_is_stable_and_changes_with_the_plan() -> None:
    first = plan_digest(_snapshot_with_drift())
    second = plan_digest(_snapshot_with_drift())
    assert first == second

    snapshot = _snapshot_with_drift()
    changed_plan = SyncPlan(actions=snapshot.plan.actions[:1])
    assert plan_digest(snapshot.model_copy(update={"plan": changed_plan})) != first
    assert plan_digest(snapshot.model_copy(update={"workspace_id": "other"})) != first


def test_with_apply_outcome_flags_written_actions_and_records_failure() -> None:
    snapshot = _snapshot_with_drift()
    report = SyncReport.from_snapshot(
        snapshot, jira_project="SF", clockify_client="Sunfish Robotics"
    )
    first_action = snapshot.plan.actions[0]
    assert isinstance(first_action, ReactivateTaskAction)
    applied = [
        AppliedSyncAction(
            action=first_action,
            task=clockify_task("SF-427", "Evaluate PX4 external control methods"),
        )
    ]

    outcome = report.with_apply_outcome(
        applied,
        failure=SyncFailure(
            jira_key="SF-304", kind=SyncActionKind.MARK_TASK_DONE, message="boom"
        ),
    )

    assert outcome.mode == "apply"
    assert outcome.applied is True
    assert [(a.applied, a.verified) for a in outcome.actions] == [
        (True, True),
        (False, False),
    ]
    assert outcome.failure is not None and outcome.failure.jira_key == "SF-304"
    assert outcome.plan_digest == report.plan_digest


# --- SyncApplyError carries the partial outcome; run_jira_sync ------------------


class FailingSecondWriteClient(FakeClockifyInventoryClient):
    """Fails the second task update, as an HTTP 500 from Clockify would."""

    def update_task(
        self,
        workspace_id: str,
        task: ClockifyTask,
        *,
        name: str | None = None,
        status: TaskStatus | None = None,
    ) -> ClockifyTask:
        if any(op.startswith("update_task") for op in self.operations):
            raise ClockifyError("Clockify request failed for PUT /x: 500 Boom")
        return super().update_task(workspace_id, task, name=name, status=status)


def _failing_second_write_clients() -> tuple[
    FailingSecondWriteClient, FakeJiraInventoryClient
]:
    good, jira = _drift_clients()
    failing = FailingSecondWriteClient(
        clients=good.clients,
        projects=good.projects,
        active_tasks=good.active_tasks,
        done_tasks=good.done_tasks,
    )
    failing._tasks_by_id.update(good._tasks_by_id)
    return failing, jira


def _run(
    clockify: FakeClockifyInventoryClient,
    jira: FakeJiraInventoryClient,
    *,
    apply: bool = False,
    expected_digest: str | None = None,
) -> SyncReport:
    return run_jira_sync(
        clockify=clockify,
        jira=jira,
        jira_project="SF",
        clockify_client="Sunfish Robotics",
        apply=apply,
        expected_digest=expected_digest,
    )


def test_apply_sync_plan_error_carries_applied_actions_and_failing_action() -> None:
    clockify, jira = _failing_second_write_clients()
    snapshot = prepare_jira_sync(
        clockify=clockify,
        jira=jira,
        jira_project="SF",
        clockify_client="Sunfish Robotics",
    )

    with pytest.raises(SyncApplyError, match="500 Boom") as raised:
        apply_sync_plan(
            snapshot.plan,
            clockify=clockify,
            workspace_id=snapshot.workspace_id,
            client_id=snapshot.client_id,
            index=snapshot.index,
        )

    assert [item.action.jira_key for item in raised.value.applied] == ["SF-427"]
    assert raised.value.action is not None
    assert raised.value.action.jira_key == "SF-304"
    # The message is the underlying error's, unchanged, for the CLI's text mode.
    assert str(raised.value) == "Clockify request failed for PUT /x: 500 Boom"


def test_apply_sync_plan_conflict_refusal_has_no_failing_action() -> None:
    plan = SyncPlan(actions=(ConflictAction(jira_key="SF-1", message="dup"),))

    with pytest.raises(SyncApplyError) as raised:
        apply_sync_plan(
            plan,
            clockify=FakeClockifySyncClient(),
            workspace_id="workspace-1",
            client_id="client-1",
            index=build_index([], []),
        )

    assert raised.value.applied == ()
    assert raised.value.action is None


def test_run_jira_sync_preview_is_the_snapshot_report_and_writes_nothing() -> None:
    clockify, jira = _drift_clients()

    report = _run(clockify, jira)

    assert report == SyncReport.from_snapshot(
        _snapshot_with_drift(), jira_project="SF", clockify_client="Sunfish Robotics"
    )
    assert clockify.operations == []


def test_run_jira_sync_apply_with_matching_digest_applies_and_verifies() -> None:
    clockify, jira = _drift_clients()
    digest = _run(clockify, jira).plan_digest

    report = _run(clockify, jira, apply=True, expected_digest=digest)

    assert report.mode == "apply"
    assert report.failure is None
    assert [(a.jira_key, a.applied, a.verified) for a in report.actions] == [
        ("SF-427", True, True),
        ("SF-304", True, True),
    ]
    assert len(clockify.operations) == 2


def test_run_jira_sync_stale_digest_applies_nothing() -> None:
    clockify, jira = _drift_clients()

    with pytest.raises(SyncPlanStaleError) as raised:
        _run(clockify, jira, apply=True, expected_digest="0" * 64)

    assert raised.value.expected == "0" * 64
    assert raised.value.actual == plan_digest(_snapshot_with_drift())
    assert clockify.operations == []


def test_run_jira_sync_refuses_conflicts_before_writing() -> None:
    project = clockify_project("SF-131", "Production Vehicle")
    task = clockify_task("SF-304", "Bench test PX4")
    duplicate = task.model_copy(update={"id": "task-dup"})
    clockify = FakeClockifyInventoryClient(
        clients=[ClockifyClientRecord(id="client-1", name="Sunfish Robotics")],
        projects=[project],
        active_tasks=[task, duplicate],
        done_tasks=[],
    )
    issues = [
        jira_issue("SF-131", "Production Vehicle", issue_type="Project / Phase"),
        jira_issue("SF-304", "Bench test PX4", status="Done", status_category="Done"),
    ]
    jira = FakeJiraInventoryClient(active_issues=[], issues=issues)

    with pytest.raises(SyncConflictError, match="SF-304") as raised:
        _run(clockify, jira, apply=True)

    assert [c.jira_key for c in raised.value.report.conflicts] == ["SF-304"]
    assert clockify.operations == []


def test_run_jira_sync_partial_failure_returns_report_with_failure() -> None:
    clockify, jira = _failing_second_write_clients()

    report = _run(clockify, jira, apply=True)

    assert report.mode == "apply"
    assert report.failure == SyncFailure(
        jira_key="SF-304",
        kind=SyncActionKind.MARK_TASK_DONE,
        message="Clockify request failed for PUT /x: 500 Boom",
    )
    assert [(a.jira_key, a.applied) for a in report.actions] == [
        ("SF-427", True),
        ("SF-304", False),
    ]
