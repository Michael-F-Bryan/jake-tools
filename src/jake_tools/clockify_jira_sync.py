from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from enum import StrEnum
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .clockify import (
    ClockifyClientRecord,
    ClockifyError,
    ClockifyProject,
    ClockifyTask,
    ClockifyUser,
    TaskStatus,
    clockify_project_name_for_jira,
    clockify_task_name_for_jira,
    project_note_for,
)
from .jira import JiraIssue, normalise_jira_key

JIRA_REFERENCE_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")

# Sunfish's Jira convention: a "Project / Phase" issue becomes a Clockify
# project; every other issue type becomes a task under its parent.
PROJECT_PHASE_ISSUE_TYPE = "Project / Phase"


class SyncActionKind(StrEnum):
    CONFLICT = "CONFLICT"
    CREATE_PROJECT = "CREATE_PROJECT"
    RENAME_PROJECT = "RENAME_PROJECT"
    CREATE_TASK = "CREATE_TASK"
    REACTIVATE_TASK = "REACTIVATE_TASK"
    RENAME_TASK = "RENAME_TASK"
    MARK_TASK_DONE = "MARK_TASK_DONE"


class SyncScope(StrEnum):
    ASSIGNED_ACTIVE = "assigned-active"
    ISSUES = "issues"


class SyncActionBase(BaseModel):
    """The one field every sync action kind always has.

    Concrete subclasses each declare their own required fields (so an
    illegal action, such as a create-task with no parent project key,
    cannot be constructed at all) plus optional context fields with "" /
    None defaults for whatever their kind doesn't need. Those context
    fields are intentionally *not* inherited from a shared default here:
    pydantic (and pyright's dataclass-style checking of it) does not allow
    a subclass to turn an inherited defaulted field into a required one, so
    each subclass declares its own full field set instead of overriding.
    This base only carries jira_key, which every kind requires identically.
    """

    model_config = ConfigDict(frozen=True)

    jira_key: str


class ConflictAction(SyncActionBase):
    kind: Literal[SyncActionKind.CONFLICT] = SyncActionKind.CONFLICT
    message: str
    desired_name: str = ""
    current_name: str = ""
    project_key: str | None = None
    project_id: str | None = None
    task_id: str | None = None
    jira_status: str = ""


class CreateProjectAction(SyncActionBase):
    kind: Literal[SyncActionKind.CREATE_PROJECT] = SyncActionKind.CREATE_PROJECT
    project_key: str
    desired_name: str
    current_name: str = ""
    project_id: str | None = None
    task_id: str | None = None
    jira_status: str = ""
    message: str = ""


class RenameProjectAction(SyncActionBase):
    kind: Literal[SyncActionKind.RENAME_PROJECT] = SyncActionKind.RENAME_PROJECT
    project_id: str
    desired_name: str
    current_name: str = ""
    project_key: str | None = None
    task_id: str | None = None
    jira_status: str = ""
    message: str = ""


class CreateTaskAction(SyncActionBase):
    kind: Literal[SyncActionKind.CREATE_TASK] = SyncActionKind.CREATE_TASK
    project_key: str
    desired_name: str
    current_name: str = ""
    project_id: str | None = None
    task_id: str | None = None
    jira_status: str = ""
    message: str = ""


class ReactivateTaskAction(SyncActionBase):
    kind: Literal[SyncActionKind.REACTIVATE_TASK] = SyncActionKind.REACTIVATE_TASK
    task_id: str
    desired_name: str
    current_name: str = ""
    project_key: str | None = None
    project_id: str | None = None
    jira_status: str = ""
    message: str = ""


class RenameTaskAction(SyncActionBase):
    kind: Literal[SyncActionKind.RENAME_TASK] = SyncActionKind.RENAME_TASK
    task_id: str
    desired_name: str
    current_name: str = ""
    project_key: str | None = None
    project_id: str | None = None
    jira_status: str = ""
    message: str = ""


class CompleteTaskAction(SyncActionBase):
    kind: Literal[SyncActionKind.MARK_TASK_DONE] = SyncActionKind.MARK_TASK_DONE
    task_id: str
    desired_name: str
    jira_status: str
    current_name: str = ""
    project_key: str | None = None
    project_id: str | None = None
    message: str = ""


SyncAction = Annotated[
    ConflictAction
    | CreateProjectAction
    | RenameProjectAction
    | CreateTaskAction
    | ReactivateTaskAction
    | RenameTaskAction
    | CompleteTaskAction,
    Field(discriminator="kind"),
]


class SyncPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    actions: tuple[SyncAction, ...] = ()

    @property
    def has_conflicts(self) -> bool:
        return any(isinstance(action, ConflictAction) for action in self.actions)


class AppliedSyncAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: SyncAction
    project: ClockifyProject | None = None
    task: ClockifyTask | None = None


class SyncApplyError(RuntimeError):
    """An apply stopped before finishing the plan.

    ``applied`` is every action written and verified before the failure;
    ``action`` is the one that failed, or ``None`` when apply refused to
    start (a plan with conflicts).
    """

    def __init__(
        self,
        message: str,
        *,
        applied: Sequence[AppliedSyncAction] = (),
        action: SyncAction | None = None,
    ) -> None:
        super().__init__(message)
        self.applied: tuple[AppliedSyncAction, ...] = tuple(applied)
        self.action: SyncAction | None = action


class SyncPreparationError(RuntimeError):
    pass


class SyncApplyResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    applied: tuple[AppliedSyncAction, ...] = ()


class ClockifyIndex(BaseModel):
    """Clockify projects and tasks, indexed once and shared by plan and apply.

    "By Jira key" lookups only include keys with exactly one active match;
    keys referenced by more than one active project or task are conflicts,
    surfaced separately so plan and apply agree on which keys are ambiguous
    instead of each recomputing the grouping independently.
    """

    model_config = ConfigDict(frozen=True)

    projects_by_id: dict[str, ClockifyProject]
    tasks_by_id: dict[str, ClockifyTask]
    projects_by_jira_key: dict[str, ClockifyProject]
    tasks_by_jira_key: dict[str, ClockifyTask]
    conflicted_project_keys: frozenset[str]
    conflicted_task_keys: frozenset[str]

    @classmethod
    def build(
        cls,
        projects: Sequence[ClockifyProject],
        tasks: Sequence[ClockifyTask],
    ) -> ClockifyIndex:
        active_projects = [project for project in projects if not project.archived]
        active_project_ids = {project.id for project in active_projects}
        usable_tasks = [task for task in tasks if task.project_id in active_project_ids]

        project_groups = _projects_by_jira_key(active_projects)
        task_groups = _tasks_by_jira_key(usable_tasks)

        return cls(
            projects_by_id={project.id: project for project in projects},
            tasks_by_id={task.id: task for task in tasks},
            projects_by_jira_key={
                key: matches[0]
                for key, matches in project_groups.items()
                if len(matches) == 1
            },
            tasks_by_jira_key={
                key: matches[0]
                for key, matches in task_groups.items()
                if len(matches) == 1
            },
            conflicted_project_keys=frozenset(
                key for key, matches in project_groups.items() if len(matches) > 1
            ),
            conflicted_task_keys=frozenset(
                key for key, matches in task_groups.items() if len(matches) > 1
            ),
        )


class SyncSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    workspace_id: str
    client_id: str
    scope: SyncScope
    requested_issue_keys: tuple[str, ...] = ()
    active_issues: tuple[JiraIssue, ...]
    jira_issues: tuple[JiraIssue, ...]
    projects: tuple[ClockifyProject, ...]
    tasks: tuple[ClockifyTask, ...]
    index: ClockifyIndex
    plan: SyncPlan


class ClockifySyncClient(Protocol):
    def get_project(self, workspace_id: str, project_id: str) -> ClockifyProject: ...

    def get_task(
        self,
        workspace_id: str,
        project_id: str,
        task_id: str,
    ) -> ClockifyTask: ...

    def create_project(
        self,
        workspace_id: str,
        *,
        name: str,
        note: str,
        client_id: str,
    ) -> ClockifyProject: ...

    def update_project_name(
        self,
        workspace_id: str,
        project: ClockifyProject,
        name: str,
    ) -> ClockifyProject: ...

    def create_task(
        self,
        workspace_id: str,
        project_id: str,
        *,
        name: str,
    ) -> ClockifyTask: ...

    def update_task(
        self,
        workspace_id: str,
        task: ClockifyTask,
        *,
        name: str | None = None,
        status: TaskStatus | None = None,
    ) -> ClockifyTask: ...


class ClockifyInventoryClient(ClockifySyncClient, Protocol):
    def get_user(self) -> ClockifyUser: ...

    def get_clients(self, workspace_id: str) -> list[ClockifyClientRecord]: ...

    def get_projects(
        self,
        workspace_id: str,
        *,
        archived: bool,
    ) -> list[ClockifyProject]: ...

    def get_tasks(
        self,
        workspace_id: str,
        project_id: str,
        *,
        active: bool,
    ) -> list[ClockifyTask]: ...


class JiraInventoryClient(Protocol):
    def get_active_assigned_issues(self, project_key: str) -> list[JiraIssue]: ...

    def get_issue(self, key: str) -> JiraIssue: ...

    def get_issues(self, keys: Iterable[str]) -> list[JiraIssue]: ...


_ACTION_ORDER = {
    SyncActionKind.CONFLICT: 0,
    SyncActionKind.CREATE_PROJECT: 10,
    SyncActionKind.RENAME_PROJECT: 20,
    SyncActionKind.CREATE_TASK: 30,
    SyncActionKind.REACTIVATE_TASK: 40,
    SyncActionKind.RENAME_TASK: 50,
    SyncActionKind.MARK_TASK_DONE: 60,
}


def prepare_jira_sync(
    *,
    clockify: ClockifyInventoryClient,
    jira: JiraInventoryClient,
    jira_project: str,
    clockify_client: str,
    issue_keys: Sequence[str] = (),
) -> SyncSnapshot:
    user = clockify.get_user()
    if not user.active_workspace:
        raise SyncPreparationError("Clockify user has no active workspace")
    workspace_id = user.active_workspace

    clients = [
        client
        for client in clockify.get_clients(workspace_id)
        if client.name == clockify_client and not client.archived
    ]
    if len(clients) != 1:
        raise SyncPreparationError(
            f"Expected one active Clockify client named {clockify_client!r}, "
            f"found {len(clients)}"
        )
    client_id = clients[0].id

    projects = clockify.get_projects(workspace_id, archived=False)
    tasks_by_id: dict[str, ClockifyTask] = {}
    for project in projects:
        for active in (True, False):
            for task in clockify.get_tasks(
                workspace_id,
                project.id,
                active=active,
            ):
                tasks_by_id[task.id] = task
    tasks = list(tasks_by_id.values())
    index = ClockifyIndex.build(projects, tasks)

    requested_issue_keys = tuple(
        sorted({normalise_jira_key(key) for key in issue_keys})
    )
    if requested_issue_keys:
        selected_issues = [jira.get_issue(key) for key in requested_issue_keys]
        active_issues = [
            issue for issue in selected_issues if issue.status_category != "Done"
        ]
        scope = SyncScope.ISSUES
    else:
        selected_issues = []
        active_issues = jira.get_active_assigned_issues(jira_project)
        scope = SyncScope.ASSIGNED_ACTIVE

    jira_keys = {issue.key for issue in active_issues}
    jira_keys.update(issue.key for issue in selected_issues)
    jira_keys.update(
        issue.parent_key
        for issue in [*active_issues, *selected_issues]
        if issue.parent_key is not None
    )
    if scope == SyncScope.ASSIGNED_ACTIVE:
        # Every Jira key already referenced by an active Clockify project or
        # task, unique match or ambiguous conflict alike, so drift/conflict
        # detection can see them even if they are not otherwise active.
        jira_keys.update(index.projects_by_jira_key)
        jira_keys.update(index.conflicted_project_keys)
        jira_keys.update(index.tasks_by_jira_key)
        jira_keys.update(index.conflicted_task_keys)

    jira_by_key = {issue.key: issue for issue in jira.get_issues(sorted(jira_keys))}
    jira_by_key.update({issue.key: issue for issue in selected_issues})
    jira_issues = [jira_by_key[key] for key in sorted(jira_by_key)]
    plan = plan_jira_sync(
        active_issues=active_issues,
        jira_issues=jira_issues,
        index=index,
        managed_keys=jira_keys if scope == SyncScope.ISSUES else None,
    )
    return SyncSnapshot(
        workspace_id=workspace_id,
        client_id=client_id,
        scope=scope,
        requested_issue_keys=requested_issue_keys,
        active_issues=tuple(active_issues),
        jira_issues=tuple(jira_issues),
        projects=tuple(projects),
        tasks=tuple(tasks),
        index=index,
        plan=plan,
    )


def plan_jira_sync(
    *,
    active_issues: Sequence[JiraIssue],
    jira_issues: Sequence[JiraIssue],
    index: ClockifyIndex,
    managed_keys: set[str] | None = None,
) -> SyncPlan:
    jira_by_key = {issue.key: issue for issue in jira_issues}
    active_by_key = {issue.key: issue for issue in active_issues}

    actions: list[SyncAction] = []
    conflict_project_keys = (
        index.conflicted_project_keys
        if managed_keys is None
        else index.conflicted_project_keys & managed_keys
    )
    conflict_task_keys = (
        index.conflicted_task_keys
        if managed_keys is None
        else index.conflicted_task_keys & managed_keys
    )

    for key in sorted(conflict_project_keys):
        actions.append(
            ConflictAction(
                jira_key=key,
                message=f"multiple active Clockify projects reference {key}",
            )
        )
    for key in sorted(conflict_task_keys):
        actions.append(
            ConflictAction(
                jira_key=key,
                message=f"multiple active Clockify tasks reference {key}",
            )
        )

    project_by_key = index.projects_by_jira_key
    task_by_key = index.tasks_by_jira_key
    planned_project_keys: set[str] = set()

    def plan_project(key: str, summary: str) -> None:
        if key in project_by_key or key in planned_project_keys:
            return
        planned_project_keys.add(key)
        actions.append(
            CreateProjectAction(
                jira_key=key,
                desired_name=clockify_project_name_for_jira(summary),
                project_key=key,
            )
        )

    for key, project in sorted(project_by_key.items()):
        issue = jira_by_key.get(key)
        if issue is None:
            continue
        desired_name = clockify_project_name_for_jira(issue.summary)
        if desired_name != project.name:
            actions.append(
                RenameProjectAction(
                    jira_key=key,
                    current_name=project.name,
                    desired_name=desired_name,
                    project_key=key,
                    project_id=project.id,
                )
            )

    for issue in sorted(active_issues, key=lambda item: item.key):
        if issue.issue_type == PROJECT_PHASE_ISSUE_TYPE:
            if issue.key not in conflict_project_keys:
                plan_project(issue.key, issue.summary)
            continue

        if not issue.parent_key or not issue.parent_summary:
            actions.append(
                ConflictAction(
                    jira_key=issue.key,
                    message="active Jira task has no parent Project / Phase",
                )
            )
            continue
        if issue.parent_key in conflict_project_keys:
            continue

        plan_project(issue.parent_key, issue.parent_summary)
        if issue.key in conflict_task_keys:
            continue

        desired_name = clockify_task_name_for_jira(issue.key, issue.summary)
        task = task_by_key.get(issue.key)
        if task is None:
            actions.append(
                CreateTaskAction(
                    jira_key=issue.key,
                    desired_name=desired_name,
                    project_key=issue.parent_key,
                )
            )
            continue

        parent = project_by_key.get(issue.parent_key)
        if parent is None or task.project_id != parent.id:
            actions.append(
                ConflictAction(
                    jira_key=issue.key,
                    project_key=issue.parent_key,
                    task_id=task.id,
                    message=(
                        f"Clockify task {issue.key} belongs to a different active project"
                    ),
                )
            )
            continue

        if task.status == "DONE":
            actions.append(
                ReactivateTaskAction(
                    jira_key=issue.key,
                    current_name=task.name,
                    desired_name=desired_name,
                    project_key=issue.parent_key,
                    project_id=task.project_id,
                    task_id=task.id,
                )
            )
        elif task.name != desired_name:
            actions.append(
                RenameTaskAction(
                    jira_key=issue.key,
                    current_name=task.name,
                    desired_name=desired_name,
                    project_key=issue.parent_key,
                    project_id=task.project_id,
                    task_id=task.id,
                )
            )

    for key, task in sorted(task_by_key.items()):
        if key in conflict_task_keys or key in active_by_key:
            continue
        issue = jira_by_key.get(key)
        if issue is None:
            continue
        desired_name = clockify_task_name_for_jira(issue.key, issue.summary)
        if issue.status_category == "Done" and task.status == "ACTIVE":
            actions.append(
                CompleteTaskAction(
                    jira_key=key,
                    current_name=task.name,
                    desired_name=desired_name,
                    project_id=task.project_id,
                    task_id=task.id,
                    jira_status=issue.status,
                )
            )
        elif task.name != desired_name:
            actions.append(
                RenameTaskAction(
                    jira_key=key,
                    current_name=task.name,
                    desired_name=desired_name,
                    project_id=task.project_id,
                    task_id=task.id,
                )
            )

    ordered = sorted(
        actions,
        key=lambda action: (_ACTION_ORDER[action.kind], action.jira_key),
    )
    return SyncPlan(actions=tuple(ordered))


def apply_sync_plan(
    plan: SyncPlan,
    *,
    clockify: ClockifySyncClient,
    workspace_id: str,
    client_id: str,
    index: ClockifyIndex,
) -> SyncApplyResult:
    conflicts = [
        action for action in plan.actions if isinstance(action, ConflictAction)
    ]
    if conflicts:
        details = "; ".join(
            f"{action.jira_key}: {action.message}" for action in conflicts
        )
        raise SyncApplyError(f"Refusing to apply a plan with conflicts: {details}")

    projects_by_id = dict(index.projects_by_id)
    projects_by_key = dict(index.projects_by_jira_key)
    tasks_by_id = dict(index.tasks_by_id)
    applied: list[AppliedSyncAction] = []

    def apply_one(action: SyncAction) -> AppliedSyncAction:
        if isinstance(action, CreateProjectAction):
            note = project_note_for(action.jira_key)
            written_project = clockify.create_project(
                workspace_id,
                name=action.desired_name,
                note=note,
                client_id=client_id,
            )
            project = _write_and_verify_project(
                action,
                written_project,
                clockify=clockify,
                workspace_id=workspace_id,
                expected_note=note,
            )
            projects_by_id[project.id] = project
            projects_by_key[action.jira_key] = project
            return AppliedSyncAction(action=action, project=project)

        if isinstance(action, RenameProjectAction):
            current_project = projects_by_id.get(action.project_id)
            if current_project is None:
                raise SyncApplyError(
                    f"{action.jira_key}: Clockify project {action.project_id!r} is unavailable"
                )
            written_project = clockify.update_project_name(
                workspace_id, current_project, action.desired_name
            )
            updated_project = _write_and_verify_project(
                action,
                written_project,
                clockify=clockify,
                workspace_id=workspace_id,
                expected_note=current_project.note,
            )
            projects_by_id[updated_project.id] = updated_project
            projects_by_key[action.jira_key] = updated_project
            return AppliedSyncAction(action=action, project=updated_project)

        if isinstance(action, CreateTaskAction):
            parent_project = projects_by_key.get(action.project_key)
            if parent_project is None:
                raise SyncApplyError(
                    f"{action.jira_key}: parent Clockify project {action.project_key} is unavailable"
                )
            written_task = clockify.create_task(
                workspace_id, parent_project.id, name=action.desired_name
            )
            created_task = _write_and_verify_task(
                action,
                written_task,
                clockify=clockify,
                workspace_id=workspace_id,
                expected_status="ACTIVE",
            )
            tasks_by_id[created_task.id] = created_task
            return AppliedSyncAction(action=action, task=created_task)

        # The remaining kinds (RenameTaskAction, ReactivateTaskAction,
        # CompleteTaskAction) all mutate an existing task in place. Conflicts
        # were already rejected above, so this is never a ConflictAction;
        # the assert only exists to narrow the type for the type checker.
        assert isinstance(
            action, RenameTaskAction | ReactivateTaskAction | CompleteTaskAction
        )
        current_task = tasks_by_id.get(action.task_id)
        if current_task is None:
            raise SyncApplyError(
                f"{action.jira_key}: Clockify task {action.task_id!r} is unavailable"
            )
        expected_status = _expected_task_status(action, current_task)
        written_task = clockify.update_task(
            workspace_id,
            current_task,
            name=action.desired_name,
            status=expected_status,
        )
        updated_task = _write_and_verify_task(
            action,
            written_task,
            clockify=clockify,
            workspace_id=workspace_id,
            expected_status=expected_status,
        )
        tasks_by_id[updated_task.id] = updated_task
        return AppliedSyncAction(action=action, task=updated_task)

    for action in plan.actions:
        try:
            applied.append(apply_one(action))
        except (SyncApplyError, ClockifyError) as exc:
            # Same message as the underlying error, plus what had already
            # been written and verified, so a caller can report a partial
            # apply instead of losing it.
            raise SyncApplyError(str(exc), applied=applied, action=action) from exc

    return SyncApplyResult(applied=tuple(applied))


def _expected_task_status(
    action: RenameTaskAction | ReactivateTaskAction | CompleteTaskAction,
    current: ClockifyTask,
) -> TaskStatus:
    if isinstance(action, ReactivateTaskAction):
        return "ACTIVE"
    if isinstance(action, CompleteTaskAction):
        return "DONE"
    return current.status


def _write_and_verify_project(
    action: CreateProjectAction | RenameProjectAction,
    written: ClockifyProject,
    *,
    clockify: ClockifySyncClient,
    workspace_id: str,
    expected_note: str,
) -> ClockifyProject:
    """Verify a just-written project, then re-fetch and verify again.

    The second verification confirms Clockify actually persisted what it
    echoed back in the write response.
    """
    _verify_project(action, written, expected_note=expected_note)
    refetched = clockify.get_project(workspace_id, written.id)
    _verify_project(action, refetched, expected_note=expected_note)
    return refetched


def _write_and_verify_task(
    action: CreateTaskAction
    | RenameTaskAction
    | ReactivateTaskAction
    | CompleteTaskAction,
    written: ClockifyTask,
    *,
    clockify: ClockifySyncClient,
    workspace_id: str,
    expected_status: TaskStatus,
) -> ClockifyTask:
    """Verify a just-written task, then re-fetch and verify again.

    The second verification confirms Clockify actually persisted what it
    echoed back in the write response.
    """
    _verify_task(action, written, expected_status=expected_status)
    refetched = clockify.get_task(workspace_id, written.project_id, written.id)
    _verify_task(action, refetched, expected_status=expected_status)
    return refetched


def _verify_project(
    action: SyncAction,
    project: ClockifyProject,
    *,
    expected_note: str,
) -> None:
    if project.name != action.desired_name:
        raise SyncApplyError(
            f"{action.jira_key}: Clockify returned project name {project.name!r}, "
            f"expected {action.desired_name!r}"
        )
    if project.note != expected_note:
        raise SyncApplyError(
            f"{action.jira_key}: Clockify returned project note {project.note!r}, "
            f"expected {expected_note!r}"
        )


def _verify_task(
    action: SyncAction,
    task: ClockifyTask,
    *,
    expected_status: TaskStatus,
) -> None:
    if task.name != action.desired_name:
        raise SyncApplyError(
            f"{action.jira_key}: Clockify returned task name {task.name!r}, "
            f"expected {action.desired_name!r}"
        )
    if task.status != expected_status:
        raise SyncApplyError(
            f"{action.jira_key}: Clockify returned task status {task.status!r}, "
            f"expected {expected_status!r}"
        )


def _projects_by_jira_key(
    projects: Sequence[ClockifyProject],
) -> dict[str, list[ClockifyProject]]:
    grouped: defaultdict[str, list[ClockifyProject]] = defaultdict(list)
    for project in projects:
        match = JIRA_REFERENCE_PATTERN.search(project.note)
        if match:
            grouped[match.group(1)].append(project)
    return dict(grouped)


def _tasks_by_jira_key(
    tasks: Sequence[ClockifyTask],
) -> dict[str, list[ClockifyTask]]:
    grouped: defaultdict[str, list[ClockifyTask]] = defaultdict(list)
    for task in tasks:
        match = JIRA_REFERENCE_PATTERN.match(task.name)
        if match:
            grouped[match.group(1)].append(task)
    return dict(grouped)


# --- The typed report shared by the MCP tool and the CLI's --json -------------


class SyncReportScope(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: SyncScope
    jira_project: str
    issue_keys: tuple[str, ...] = ()


class SyncReportInventory(BaseModel):
    model_config = ConfigDict(frozen=True)

    active_issues: int
    jira_issues: int
    projects: int
    tasks: int


class SyncReportAction(BaseModel):
    """One planned (non-conflict) action, flattened to a single shape.

    ``applied`` and ``verified`` are both false in a preview. After an apply
    they record whether the action was written and whether the re-read of
    the Clockify record matched the plan.
    """

    model_config = ConfigDict(frozen=True)

    kind: SyncActionKind
    jira_key: str
    current_name: str = ""
    desired_name: str = ""
    project_key: str | None = None
    project_id: str | None = None
    task_id: str | None = None
    jira_status: str = ""
    message: str = ""
    applied: bool = False
    verified: bool = False

    @classmethod
    def from_action(cls, action: SyncAction) -> SyncReportAction:
        return cls(
            kind=action.kind,
            jira_key=action.jira_key,
            current_name=action.current_name,
            desired_name=action.desired_name,
            project_key=action.project_key,
            project_id=action.project_id,
            task_id=action.task_id,
            jira_status=action.jira_status,
            message=action.message,
        )


class SyncConflict(BaseModel):
    """Something the plan refuses to touch; any conflict blocks apply."""

    model_config = ConfigDict(frozen=True)

    jira_key: str
    message: str
    project_key: str | None = None
    project_id: str | None = None
    task_id: str | None = None

    @classmethod
    def from_action(cls, action: ConflictAction) -> SyncConflict:
        return cls(
            jira_key=action.jira_key,
            message=action.message,
            project_key=action.project_key,
            project_id=action.project_id,
            task_id=action.task_id,
        )


class SyncFailure(BaseModel):
    """Where an apply stopped.

    Every action before the one named here was applied and verified. The
    named action itself may have been partly written: the write can have
    landed in Clockify and the read-back verification failed afterwards, so
    inspect that record before retrying.
    """

    model_config = ConfigDict(frozen=True)

    jira_key: str | None = None
    kind: SyncActionKind | None = None
    message: str


class SyncReport(BaseModel):
    """What a preview or apply found and did.

    The MCP tool returns this model and ``jake-tools clockify jira-sync
    --json`` prints it, so "preview agrees with the CLI" holds by
    construction. ``plan_digest`` is :func:`plan_digest` over the snapshot the
    report was built from; an apply must present the digest of the plan it
    was shown, and a mismatch is ``plan_stale``.

    ``applied`` means an apply was attempted, not that it succeeded: check
    ``failure`` and each action's ``applied``/``verified`` flags for what
    was actually written.
    """

    model_config = ConfigDict(frozen=True)

    mode: Literal["preview", "apply"]
    workspace_id: str
    client_id: str
    jira_project: str
    clockify_client: str
    scope: SyncReportScope
    inventory: SyncReportInventory
    actions: tuple[SyncReportAction, ...]
    conflicts: tuple[SyncConflict, ...]
    plan_digest: str
    applied: bool = False
    failure: SyncFailure | None = None

    @classmethod
    def from_snapshot(
        cls,
        snapshot: SyncSnapshot,
        *,
        jira_project: str,
        clockify_client: str,
        mode: Literal["preview", "apply"] = "preview",
    ) -> SyncReport:
        return cls(
            mode=mode,
            workspace_id=snapshot.workspace_id,
            client_id=snapshot.client_id,
            jira_project=jira_project,
            clockify_client=clockify_client,
            scope=SyncReportScope(
                kind=snapshot.scope,
                jira_project=jira_project,
                issue_keys=snapshot.requested_issue_keys,
            ),
            inventory=SyncReportInventory(
                active_issues=len(snapshot.active_issues),
                jira_issues=len(snapshot.jira_issues),
                projects=len(snapshot.projects),
                tasks=len(snapshot.tasks),
            ),
            actions=tuple(
                SyncReportAction.from_action(action)
                for action in snapshot.plan.actions
                if not isinstance(action, ConflictAction)
            ),
            conflicts=tuple(
                SyncConflict.from_action(action)
                for action in snapshot.plan.actions
                if isinstance(action, ConflictAction)
            ),
            plan_digest=plan_digest(snapshot),
        )

    @property
    def has_conflicts(self) -> bool:
        return bool(self.conflicts)

    def with_apply_outcome(
        self,
        applied: Sequence[AppliedSyncAction],
        *,
        failure: SyncFailure | None = None,
    ) -> SyncReport:
        """The apply-mode report: flag each written action, record a failure.

        Every action ``apply_sync_plan`` returns (or had completed before it
        raised) was re-read and verified, so ``applied`` implies ``verified``.
        """
        written = {(item.action.kind, item.action.jira_key) for item in applied}
        return self.model_copy(
            update={
                "mode": "apply",
                "applied": True,
                "failure": failure,
                "actions": tuple(
                    action.model_copy(update={"applied": True, "verified": True})
                    if (action.kind, action.jira_key) in written
                    else action
                    for action in self.actions
                ),
            }
        )


def plan_digest(snapshot: SyncSnapshot) -> str:
    """SHA-256 over the canonical JSON of the plan and the IDs it depends on.

    The input is the workspace and client the plan targets plus every action
    (conflicts included), dumped in JSON mode with sorted keys. Actions carry
    the record IDs and current names they were planned against, so a changed
    snapshot that changes the plan changes the digest; an identical plan over
    a changed snapshot is still safe to apply.
    """
    canonical = json.dumps(
        {
            "workspace_id": snapshot.workspace_id,
            "client_id": snapshot.client_id,
            "actions": [_dump_action(action) for action in snapshot.plan.actions],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _dump_action(action: SyncAction) -> dict[str, object]:
    return action.model_dump(mode="json")


# --- Preview / apply, shared by the MCP tool and the CLI -----------------------


class SyncPlanStaleError(RuntimeError):
    """The plan recomputed for an apply is not the plan the caller previewed."""

    def __init__(self, *, expected: str, actual: str) -> None:
        super().__init__(
            "The Jira/Clockify sync plan changed since it was previewed "
            f"(expected digest {expected}, found {actual}); nothing was applied. "
            "Preview again and review the new plan."
        )
        self.expected = expected
        self.actual = actual


class SyncConflictError(RuntimeError):
    """Apply refused because the plan has conflicts; nothing was written.

    ``report`` is the preview report that blocked the apply.
    """

    def __init__(self, report: SyncReport) -> None:
        details = "; ".join(
            f"{conflict.jira_key}: {conflict.message}" for conflict in report.conflicts
        )
        super().__init__(f"Refusing to apply a plan with conflicts: {details}")
        self.report = report


def run_jira_sync(
    *,
    clockify: ClockifyInventoryClient,
    jira: JiraInventoryClient,
    jira_project: str,
    clockify_client: str,
    issue_keys: Sequence[str] = (),
    apply: bool = False,
    expected_digest: str | None = None,
    describe_failure: Callable[[SyncApplyError], str] = str,
) -> SyncReport:
    """Prepare the plan and either report it (preview) or apply it.

    Apply recomputes the plan. When ``expected_digest`` is given and differs
    from the recomputed plan's digest, raise :class:`SyncPlanStaleError`
    before writing anything; a plan with conflicts raises
    :class:`SyncConflictError`, also before writing. A failure part-way
    through an apply is not raised: the returned report carries the actions
    written and verified before it plus a :class:`SyncFailure` whose
    message is ``describe_failure(error)``. The default is the full message;
    a caller reporting to someone else passes a sanitiser.

    Errors reading Clockify or Jira (:class:`ClockifyError`,
    :class:`~jake_tools.jira.JiraError`, :class:`SyncPreparationError`)
    propagate unchanged.
    """
    snapshot = prepare_jira_sync(
        clockify=clockify,
        jira=jira,
        jira_project=jira_project,
        clockify_client=clockify_client,
        issue_keys=issue_keys,
    )
    report = SyncReport.from_snapshot(
        snapshot, jira_project=jira_project, clockify_client=clockify_client
    )
    if not apply:
        return report
    if expected_digest is not None and expected_digest != report.plan_digest:
        raise SyncPlanStaleError(expected=expected_digest, actual=report.plan_digest)
    if report.has_conflicts:
        raise SyncConflictError(report)

    try:
        result = apply_sync_plan(
            snapshot.plan,
            clockify=clockify,
            workspace_id=snapshot.workspace_id,
            client_id=snapshot.client_id,
            index=snapshot.index,
        )
    except SyncApplyError as exc:
        return report.with_apply_outcome(
            exc.applied,
            failure=SyncFailure(
                jira_key=exc.action.jira_key if exc.action else None,
                kind=exc.action.kind if exc.action else None,
                message=describe_failure(exc),
            ),
        )
    return report.with_apply_outcome(result.applied)
