from __future__ import annotations

import json
import re
import shlex
import subprocess
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from .clockify import (
    ClockifyClientRecord,
    ClockifyError,
    ClockifyProject,
    ClockifyTask,
    ClockifyUser,
    TaskStatus,
    clockify_task_name_for_jira,
    normalise_jira_key,
)

CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
JIRA_PROJECT_KEY_PATTERN = re.compile(r"^[A-Z][A-Z0-9]+$")
JIRA_REFERENCE_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")


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


class SyncActionKind(StrEnum):
    CONFLICT = "CONFLICT"
    CREATE_PROJECT = "CREATE_PROJECT"
    RENAME_PROJECT = "RENAME_PROJECT"
    CREATE_TASK = "CREATE_TASK"
    REACTIVATE_TASK = "REACTIVATE_TASK"
    RENAME_TASK = "RENAME_TASK"
    MARK_TASK_DONE = "MARK_TASK_DONE"


class SyncAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: SyncActionKind
    jira_key: str
    desired_name: str = ""
    current_name: str = ""
    project_key: str | None = None
    project_id: str | None = None
    task_id: str | None = None
    jira_status: str = ""
    message: str = ""


class SyncPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    actions: tuple[SyncAction, ...] = ()

    @property
    def has_conflicts(self) -> bool:
        return any(action.kind == SyncActionKind.CONFLICT for action in self.actions)


class SyncApplyError(RuntimeError):
    pass


class SyncPreparationError(RuntimeError):
    pass


class AppliedSyncAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: SyncAction
    project: ClockifyProject | None = None
    task: ClockifyTask | None = None


class SyncApplyResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    applied: tuple[AppliedSyncAction, ...] = ()


class SyncSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    workspace_id: str
    client_id: str
    active_issues: tuple[JiraIssue, ...]
    jira_issues: tuple[JiraIssue, ...]
    projects: tuple[ClockifyProject, ...]
    tasks: tuple[ClockifyTask, ...]
    plan: SyncPlan


class ClockifySyncClient(Protocol):
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

    active_issues = jira.get_active_assigned_issues(jira_project)
    jira_keys = {issue.key for issue in active_issues}
    jira_keys.update(
        issue.parent_key for issue in active_issues if issue.parent_key is not None
    )
    for project in projects:
        match = JIRA_REFERENCE_PATTERN.search(project.note)
        if match:
            jira_keys.add(match.group(1))
    for task in tasks:
        match = JIRA_REFERENCE_PATTERN.match(task.name)
        if match:
            jira_keys.add(match.group(1))

    jira_issues = jira.get_issues(sorted(jira_keys))
    plan = plan_jira_sync(
        active_issues=active_issues,
        jira_issues=jira_issues,
        projects=projects,
        tasks=tasks,
    )
    return SyncSnapshot(
        workspace_id=workspace_id,
        client_id=client_id,
        active_issues=tuple(active_issues),
        jira_issues=tuple(jira_issues),
        projects=tuple(projects),
        tasks=tuple(tasks),
        plan=plan,
    )


def plan_jira_sync(
    *,
    active_issues: Sequence[JiraIssue],
    jira_issues: Sequence[JiraIssue],
    projects: Sequence[ClockifyProject],
    tasks: Sequence[ClockifyTask],
) -> SyncPlan:
    jira_by_key = {issue.key: issue for issue in jira_issues}
    active_by_key = {issue.key: issue for issue in active_issues}
    active_projects = [project for project in projects if not project.archived]
    active_project_ids = {project.id for project in active_projects}
    usable_tasks = [task for task in tasks if task.project_id in active_project_ids]

    actions: list[SyncAction] = []
    project_groups = _projects_by_jira_key(active_projects)
    task_groups = _tasks_by_jira_key(usable_tasks)
    conflict_project_keys = {
        key for key, matches in project_groups.items() if len(matches) > 1
    }
    conflict_task_keys = {
        key for key, matches in task_groups.items() if len(matches) > 1
    }

    for key in sorted(conflict_project_keys):
        actions.append(
            SyncAction(
                kind=SyncActionKind.CONFLICT,
                jira_key=key,
                message=f"multiple active Clockify projects reference {key}",
            )
        )
    for key in sorted(conflict_task_keys):
        actions.append(
            SyncAction(
                kind=SyncActionKind.CONFLICT,
                jira_key=key,
                message=f"multiple active Clockify tasks reference {key}",
            )
        )

    project_by_key = {
        key: matches[0] for key, matches in project_groups.items() if len(matches) == 1
    }
    task_by_key = {
        key: matches[0] for key, matches in task_groups.items() if len(matches) == 1
    }
    planned_project_keys: set[str] = set()

    def plan_project(key: str, summary: str) -> None:
        if key in project_by_key or key in planned_project_keys:
            return
        planned_project_keys.add(key)
        actions.append(
            SyncAction(
                kind=SyncActionKind.CREATE_PROJECT,
                jira_key=key,
                desired_name=summary,
                project_key=key,
            )
        )

    for key, project in sorted(project_by_key.items()):
        issue = jira_by_key.get(key)
        if issue and issue.summary != project.name:
            actions.append(
                SyncAction(
                    kind=SyncActionKind.RENAME_PROJECT,
                    jira_key=key,
                    current_name=project.name,
                    desired_name=issue.summary,
                    project_key=key,
                    project_id=project.id,
                )
            )

    for issue in sorted(active_issues, key=lambda item: item.key):
        if issue.issue_type == "Project / Phase":
            if issue.key not in conflict_project_keys:
                plan_project(issue.key, issue.summary)
            continue

        if not issue.parent_key or not issue.parent_summary:
            actions.append(
                SyncAction(
                    kind=SyncActionKind.CONFLICT,
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
                SyncAction(
                    kind=SyncActionKind.CREATE_TASK,
                    jira_key=issue.key,
                    desired_name=desired_name,
                    project_key=issue.parent_key,
                )
            )
            continue

        parent = project_by_key.get(issue.parent_key)
        if parent is None or task.project_id != parent.id:
            actions.append(
                SyncAction(
                    kind=SyncActionKind.CONFLICT,
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
                SyncAction(
                    kind=SyncActionKind.REACTIVATE_TASK,
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
                SyncAction(
                    kind=SyncActionKind.RENAME_TASK,
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
                SyncAction(
                    kind=SyncActionKind.MARK_TASK_DONE,
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
                SyncAction(
                    kind=SyncActionKind.RENAME_TASK,
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
    projects: Sequence[ClockifyProject],
    tasks: Sequence[ClockifyTask],
) -> SyncApplyResult:
    conflicts = [
        action for action in plan.actions if action.kind == SyncActionKind.CONFLICT
    ]
    if conflicts:
        details = "; ".join(
            f"{action.jira_key}: {action.message}" for action in conflicts
        )
        raise SyncApplyError(f"Refusing to apply a plan with conflicts: {details}")

    projects_by_id = {project.id: project for project in projects}
    project_groups = _projects_by_jira_key(
        [project for project in projects if not project.archived]
    )
    projects_by_key = {
        key: matches[0] for key, matches in project_groups.items() if len(matches) == 1
    }
    tasks_by_id = {task.id: task for task in tasks}
    applied: list[AppliedSyncAction] = []

    for action in plan.actions:
        if action.kind == SyncActionKind.CREATE_PROJECT:
            project = clockify.create_project(
                workspace_id,
                name=action.desired_name,
                note=f"Jira: {action.jira_key}",
                client_id=client_id,
            )
            _verify_project(
                action,
                project,
                expected_note=f"Jira: {action.jira_key}",
            )
            projects_by_id[project.id] = project
            projects_by_key[action.jira_key] = project
            applied.append(AppliedSyncAction(action=action, project=project))
            continue

        if action.kind == SyncActionKind.RENAME_PROJECT:
            project = _require_project(action, projects_by_id)
            updated = clockify.update_project_name(
                workspace_id,
                project,
                action.desired_name,
            )
            _verify_project(action, updated, expected_note=project.note)
            projects_by_id[updated.id] = updated
            projects_by_key[action.jira_key] = updated
            applied.append(AppliedSyncAction(action=action, project=updated))
            continue

        if action.kind == SyncActionKind.CREATE_TASK:
            if not action.project_key:
                raise SyncApplyError(
                    f"{action.jira_key}: create-task action has no parent project key"
                )
            project = projects_by_key.get(action.project_key)
            if project is None:
                raise SyncApplyError(
                    f"{action.jira_key}: parent Clockify project {action.project_key} is unavailable"
                )
            task = clockify.create_task(
                workspace_id,
                project.id,
                name=action.desired_name,
            )
            _verify_task(action, task, expected_status="ACTIVE")
            tasks_by_id[task.id] = task
            applied.append(AppliedSyncAction(action=action, task=task))
            continue

        task = _require_task(action, tasks_by_id)
        expected_status: TaskStatus
        if action.kind == SyncActionKind.REACTIVATE_TASK:
            expected_status = "ACTIVE"
        elif action.kind == SyncActionKind.MARK_TASK_DONE:
            expected_status = "DONE"
        elif action.kind == SyncActionKind.RENAME_TASK:
            expected_status = task.status
        else:
            raise SyncApplyError(f"Unsupported sync action: {action.kind}")

        updated = clockify.update_task(
            workspace_id,
            task,
            name=action.desired_name,
            status=expected_status,
        )
        _verify_task(action, updated, expected_status=expected_status)
        tasks_by_id[updated.id] = updated
        applied.append(AppliedSyncAction(action=action, task=updated))

    return SyncApplyResult(applied=tuple(applied))


def _require_project(
    action: SyncAction,
    projects_by_id: dict[str, ClockifyProject],
) -> ClockifyProject:
    if not action.project_id or action.project_id not in projects_by_id:
        raise SyncApplyError(
            f"{action.jira_key}: Clockify project {action.project_id!r} is unavailable"
        )
    return projects_by_id[action.project_id]


def _require_task(
    action: SyncAction,
    tasks_by_id: dict[str, ClockifyTask],
) -> ClockifyTask:
    if not action.task_id or action.task_id not in tasks_by_id:
        raise SyncApplyError(
            f"{action.jira_key}: Clockify task {action.task_id!r} is unavailable"
        )
    return tasks_by_id[action.task_id]


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
