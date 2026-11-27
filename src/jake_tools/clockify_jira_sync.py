from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from enum import StrEnum
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .clockify import (
    ClockifyClientRecord,
    ClockifyError,
    ClockifyProject,
    ClockifyTask,
    ClockifyUser,
    JsonObject,
    TaskRateKind,
    TaskStatus,
    clockify_project_name_for_jira,
    clockify_task_name_for_jira,
    jira_key_of_project_note,
    jira_key_of_task_name,
    project_note_for,
)
from .clockify_relocation import (
    ENTRY_PLACEMENT_FIELDS,
    TASK_IDENTITY_FIELDS,
    TASK_RATE_FIELDS,
    changed_fields,
    entry_blocker,
    entry_move_payload,
    record_fingerprint,
    relocation_baseline,
    relocation_marker,
    retired_task_name,
    staged_task_name,
    task_create_payload,
    task_rate_amount,
    task_update_payload,
)
from .jira import JiraIssue, normalise_jira_key

# Sunfish's Jira convention: a "Project / Phase" issue becomes a Clockify
# project; every other issue type becomes a task under its ancestor phase.
PROJECT_PHASE_ISSUE_TYPE = "Project / Phase"


class SyncActionKind(StrEnum):
    CONFLICT = "CONFLICT"
    CREATE_PROJECT = "CREATE_PROJECT"
    RENAME_PROJECT = "RENAME_PROJECT"
    RELOCATE_TASK = "RELOCATE_TASK"
    CREATE_TASK = "CREATE_TASK"
    REACTIVATE_TASK = "REACTIVATE_TASK"
    RENAME_TASK = "RENAME_TASK"
    MARK_TASK_DONE = "MARK_TASK_DONE"


class SyncScope(StrEnum):
    # Retain the report value: active covers planned or underway work.
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


class RelocationEntry(BaseModel):
    """One time entry a relocation carries, as the preview read it.

    ``fingerprint`` is :func:`~jake_tools.clockify_relocation.record_fingerprint`
    of the whole entry record, so the digest binds every field: an entry
    changed after the preview is refused, never overwritten. ``moved`` means an
    earlier, interrupted apply already re-pointed it at the staging task.
    """

    model_config = ConfigDict(frozen=True)

    entry_id: str
    user_id: str
    fingerprint: str
    moved: bool = False


class RelocateTaskAction(SyncActionBase):
    """Move a task, with its time entries, to its Jira-derived project.

    Clockify cannot re-parent a task, so this is a staged replacement (see
    :mod:`jake_tools.clockify_relocation`). ``staged_task_id`` and
    ``source_retired`` describe an interrupted relocation being resumed.
    The planner emits it uninspected; :func:`prepare_jira_sync` reads the
    task and entry records, fills the fingerprints and sets ``inspected``.
    Apply refuses an uninspected relocation.
    """

    kind: Literal[SyncActionKind.RELOCATE_TASK] = SyncActionKind.RELOCATE_TASK
    project_key: str
    desired_name: str
    desired_status: TaskStatus
    source_task_id: str
    source_project_id: str
    staged_task_id: str | None = None
    source_retired: bool = False
    inspected: bool = False
    source_fingerprint: str = ""
    staged_fingerprint: str = ""
    entries: tuple[RelocationEntry, ...] = ()
    # The relocation_baseline the staging task's marker carries (or, with
    # ``rebaseline``, will carry: a person removed it to accept the records).
    baseline: str = ""
    rebaseline: bool = False
    current_name: str = ""
    # The destination project, when it already exists.
    project_id: str | None = None
    # The task being moved (the source).
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
    """Complete a task: an existing one by ``task_id``, or, with
    ``after_create``, the one the plan's CREATE_TASK for the same Jira key
    writes first (Clockify creates tasks active, so a Done issue's task
    needs both writes). Exactly one of the two identifies the task.
    """

    kind: Literal[SyncActionKind.MARK_TASK_DONE] = SyncActionKind.MARK_TASK_DONE
    task_id: str | None = None
    after_create: bool = False
    desired_name: str
    jira_status: str
    current_name: str = ""
    project_key: str | None = None
    project_id: str | None = None
    message: str = ""

    @model_validator(mode="after")
    def _identifies_one_task(self) -> CompleteTaskAction:
        if (self.task_id is None) != self.after_create:
            raise ValueError(
                "a completion names an existing task_id, or follows the "
                "creation of its task, never both or neither"
            )
        return self


SyncAction = Annotated[
    ConflictAction
    | CreateProjectAction
    | RenameProjectAction
    | RelocateTaskAction
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
    start (a plan with conflicts). ``recovery`` is this package's own account
    (record IDs and steps only) of what a multi-step action had verified and
    which write may be uncertain.
    """

    def __init__(
        self,
        message: str,
        *,
        applied: Sequence[AppliedSyncAction] = (),
        action: SyncAction | None = None,
        recovery: str = "",
    ) -> None:
        super().__init__(message)
        self.applied: tuple[AppliedSyncAction, ...] = tuple(applied)
        self.action: SyncAction | None = action
        self.recovery = recovery


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
    # Tasks named as a relocation's staging task, by the Jira key they carry.
    # They do not claim the key, so they are not in ``tasks_by_jira_key``.
    staged_tasks_by_jira_key: dict[str, tuple[ClockifyTask, ...]] = Field(
        default_factory=dict
    )

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
        staged_groups: defaultdict[str, list[ClockifyTask]] = defaultdict(list)
        for task in usable_tasks:
            marker = relocation_marker(task.name)
            if marker is not None and marker.state == "staged":
                staged_groups[marker.jira_key].append(task)

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
            staged_tasks_by_jira_key={
                key: tuple(matches) for key, matches in staged_groups.items()
            },
        )


class SyncSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    workspace_id: str
    client_id: str
    scope: SyncScope
    requested_issue_keys: tuple[str, ...] = ()
    # What the scope selects for planning: the assigned planned or underway
    # work, or exactly the requested issues whatever their status.
    selected_issues: tuple[JiraIssue, ...]
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

    # Whole-record access, for relocation (see ClockifyClient).
    def get_workspace_users(self, workspace_id: str) -> list[ClockifyUser]: ...

    def get_task_time_entries(
        self,
        workspace_id: str,
        user_id: str,
        *,
        project_id: str,
        task_id: str,
    ) -> list[JsonObject]: ...

    def get_time_entry_record(self, workspace_id: str, entry_id: str) -> JsonObject: ...

    def update_time_entry(
        self, workspace_id: str, entry_id: str, payload: JsonObject
    ) -> JsonObject: ...

    def get_task_record(
        self, workspace_id: str, project_id: str, task_id: str
    ) -> JsonObject: ...

    def create_task_record(
        self, workspace_id: str, project_id: str, payload: JsonObject
    ) -> JsonObject: ...

    def update_task_record(
        self, workspace_id: str, project_id: str, task_id: str, payload: JsonObject
    ) -> JsonObject: ...

    def update_task_rate(
        self,
        workspace_id: str,
        project_id: str,
        task_id: str,
        *,
        kind: TaskRateKind,
        amount: int,
    ) -> JsonObject: ...


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
    SyncActionKind.RELOCATE_TASK: 25,
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
        # Explicit keys override discovery filters (assignee, status) only.
        selected_issues = [jira.get_issue(key) for key in requested_issue_keys]
        scope = SyncScope.ISSUES
    else:
        selected_issues = jira.get_active_assigned_issues(jira_project)
        scope = SyncScope.ASSIGNED_ACTIVE

    jira_keys = {issue.key for issue in selected_issues}
    jira_keys.update(
        issue.parent_key for issue in selected_issues if issue.parent_key is not None
    )
    if scope == SyncScope.ASSIGNED_ACTIVE:
        # Every Jira key already referenced by an active Clockify project or
        # task, unique match or ambiguous conflict alike, so drift/conflict
        # detection can see them even if they are not otherwise active.
        jira_keys.update(index.projects_by_jira_key)
        jira_keys.update(index.conflicted_project_keys)
        jira_keys.update(index.tasks_by_jira_key)
        jira_keys.update(index.conflicted_task_keys)
        # An interrupted relocation must be resumed or reported even once its
        # issue leaves discovery (a retired source no longer claims the key).
        jira_keys.update(index.staged_tasks_by_jira_key)

    jira_by_key = {issue.key: issue for issue in jira.get_issues(sorted(jira_keys))}
    jira_by_key.update({issue.key: issue for issue in selected_issues})
    # Ancestors provide placement, not additional task selection. Fetch each
    # key at most once, including unavailable keys and cyclic hierarchies.
    frontier = list(selected_issues)
    if scope == SyncScope.ASSIGNED_ACTIVE:
        selected_keys = {issue.key for issue in selected_issues}
        frontier.extend(
            jira_by_key[key]
            for key in sorted(index.staged_tasks_by_jira_key)
            if key in jira_by_key and key not in selected_keys
        )
    visited: set[str] = set()
    while frontier:
        visited.update(issue.key for issue in frontier)
        parent_keys = {
            issue.parent_key
            for issue in frontier
            if issue.issue_type != PROJECT_PHASE_ISSUE_TYPE
            and issue.parent_key is not None
        }
        missing_keys = parent_keys - jira_keys
        jira_keys.update(parent_keys)
        if missing_keys:
            jira_by_key.update(
                (issue.key, issue) for issue in jira.get_issues(sorted(missing_keys))
            )
        frontier = [
            jira_by_key[key]
            for key in sorted(parent_keys - visited)
            if key in jira_by_key
        ]
    jira_issues = [jira_by_key[key] for key in sorted(jira_by_key)]
    plan = plan_jira_sync(
        selected_issues=selected_issues,
        jira_issues=jira_issues,
        index=index,
        managed_keys=jira_keys if scope == SyncScope.ISSUES else None,
    )
    plan = _inspect_relocations(
        plan, clockify=clockify, workspace_id=workspace_id, index=index
    )
    return SyncSnapshot(
        workspace_id=workspace_id,
        client_id=client_id,
        scope=scope,
        requested_issue_keys=requested_issue_keys,
        selected_issues=tuple(selected_issues),
        jira_issues=tuple(jira_issues),
        projects=tuple(projects),
        tasks=tuple(tasks),
        index=index,
        plan=plan,
    )


def _ordered(actions: Iterable[SyncAction]) -> SyncPlan:
    return SyncPlan(
        actions=tuple(
            sorted(
                actions,
                key=lambda action: (_ACTION_ORDER[action.kind], action.jira_key),
            )
        )
    )


def _inspect_relocations(
    plan: SyncPlan,
    *,
    clockify: ClockifySyncClient,
    workspace_id: str,
    index: ClockifyIndex,
) -> SyncPlan:
    """Read what each planned relocation would touch, and bind it.

    Every relocation gets the fingerprints of its source (and staging) task
    and of every time entry on them, so the digest covers the whole records
    and an apply refuses any that changed. One that cannot be moved safely
    becomes a conflict naming the reason.
    """
    if not any(isinstance(action, RelocateTaskAction) for action in plan.actions):
        return plan
    users = clockify.get_workspace_users(workspace_id)
    inspected = [
        _inspect_relocation(
            action,
            clockify=clockify,
            workspace_id=workspace_id,
            users=users,
            index=index,
        )
        if isinstance(action, RelocateTaskAction)
        else action
        for action in plan.actions
    ]
    return _ordered(inspected)


def _task_entries(
    clockify: ClockifySyncClient,
    workspace_id: str,
    users: Sequence[ClockifyUser],
    *,
    project_id: str,
    task_id: str,
) -> list[tuple[str, JsonObject]]:
    """Every time entry on a task, per user, each read whole by ID.

    Clockify v1 lists entries per user only, so this scans every listed
    member. It is a sequence of offset-paginated reads, not a snapshot: it
    assumes nobody else writes these entries meanwhile.
    """
    found: list[tuple[str, JsonObject]] = []
    for user in users:
        for listed in clockify.get_task_time_entries(
            workspace_id, user.id, project_id=project_id, task_id=task_id
        ):
            if listed.get("taskId") != task_id or listed.get("projectId") != project_id:
                continue
            entry_id = str(listed["id"])
            found.append(
                (user.id, clockify.get_time_entry_record(workspace_id, entry_id))
            )
    return found


def _inspect_relocation(
    action: RelocateTaskAction,
    *,
    clockify: ClockifySyncClient,
    workspace_id: str,
    users: Sequence[ClockifyUser],
    index: ClockifyIndex,
) -> SyncAction:
    key = action.jira_key

    def conflict(reasons: Sequence[str]) -> ConflictAction:
        return ConflictAction(
            jira_key=key,
            project_key=action.project_key,
            project_id=action.project_id,
            task_id=action.source_task_id,
            message=(
                f"cannot relocate {key} to {action.project_key}: " + "; ".join(reasons)
            ),
        )

    source = clockify.get_task_record(
        workspace_id, action.source_project_id, action.source_task_id
    )
    source_entries = _task_entries(
        clockify,
        workspace_id,
        users,
        project_id=action.source_project_id,
        task_id=action.source_task_id,
    )
    staged: JsonObject | None = None
    staged_entries: list[tuple[str, JsonObject]] = []
    if action.staged_task_id is not None:
        assert action.project_id is not None
        staged = clockify.get_task_record(
            workspace_id, action.project_id, action.staged_task_id
        )
        staged_entries = _task_entries(
            clockify,
            workspace_id,
            users,
            project_id=action.project_id,
            task_id=action.staged_task_id,
        )

    reasons = [
        reason
        for _, entry in source_entries
        if (reason := entry_blocker(entry)) is not None
    ]
    seen: set[str] = set()
    for _, entry in [*source_entries, *staged_entries]:
        entry_id = str(entry["id"])
        if entry_id in seen:
            reasons.append(f"time entry {entry_id} was listed more than once")
        seen.add(entry_id)
    if action.source_retired and source_entries:
        reasons.append(
            f"retired source task {action.source_task_id} still has "
            f"{len(source_entries)} time entries"
        )
    baseline = relocation_baseline(
        source, [entry for _, entry in [*source_entries, *staged_entries]]
    )
    rebaseline = False
    if staged is not None:
        marker = relocation_marker(str(staged.get("name", "")))
        if marker is None or marker.state != "staged":
            reasons.append(
                f"staging task {action.staged_task_id} no longer carries its marker"
            )
        elif marker.baseline is None:
            rebaseline = True
        elif marker.baseline != baseline:
            moved_ids = sorted(str(entry["id"]) for _, entry in staged_entries)
            suspects = (
                f"time entries {', '.join(moved_ids)} on the staging task"
                if moved_ids
                else "the source task and its time entries"
            )
            reasons.append(
                f"the records no longer match the baseline staging task "
                f"{action.staged_task_id} recorded before the first entry moved "
                "(a move, or a write whose outcome was uncertain, may have "
                "changed fields Clockify cannot set back, such as rates); "
                f"inspect {suspects} in Clockify, and to accept their current "
                f"values remove ' #{marker.baseline}' from the staging task's "
                "name, then preview again"
            )
    if reasons:
        return conflict(reasons)

    entries = sorted(
        [
            RelocationEntry(
                entry_id=str(entry["id"]),
                user_id=user_id,
                fingerprint=record_fingerprint(entry),
                moved=moved,
            )
            for moved, found in ((False, source_entries), (True, staged_entries))
            for user_id, entry in found
        ],
        key=lambda entry: entry.entry_id,
    )
    source_project = index.projects_by_id.get(action.source_project_id)
    source_label = (
        f"{source_project.name!r} ({action.source_project_id})"
        if source_project
        else action.source_project_id
    )
    message = (
        f"move task {action.source_task_id} from project {source_label} to the "
        f"project for {action.project_key}"
        + ("" if action.project_id else " (created first)")
        + f"; {len(entries)} time entries follow it"
    )
    if action.staged_task_id is not None:
        moved = sum(entry.moved for entry in entries)
        message += (
            f"; resumes an interrupted relocation through staging task "
            f"{action.staged_task_id} ({moved} entries already moved"
            + (", source already retired)" if action.source_retired else ")")
        )
    if rebaseline:
        message += (
            "; first records a new baseline in the staging task's name, "
            "accepting the current records as a person decided"
        )
    return action.model_copy(
        update={
            "inspected": True,
            "baseline": baseline,
            "rebaseline": rebaseline,
            "source_fingerprint": record_fingerprint(source),
            "staged_fingerprint": record_fingerprint(staged) if staged else "",
            "entries": tuple(entries),
            "message": message,
        }
    )


def _project_phase_ancestor(
    issue: JiraIssue, jira_by_key: dict[str, JiraIssue]
) -> JiraIssue:
    visited: set[str] = set()
    while issue.issue_type != PROJECT_PHASE_ISSUE_TYPE:
        if issue.key in visited:
            raise ValueError(f"cycle in Jira ancestry at {issue.key}")
        visited.add(issue.key)
        if not issue.parent_key:
            raise ValueError(f"{issue.key} has no parent Project / Phase")
        parent = jira_by_key.get(issue.parent_key)
        if parent is None:
            raise ValueError(f"Jira ancestor {issue.parent_key} is unavailable")
        issue = parent
    return issue


def plan_jira_sync(
    *,
    selected_issues: Sequence[JiraIssue],
    jira_issues: Sequence[JiraIssue],
    index: ClockifyIndex,
    managed_keys: set[str] | None = None,
) -> SyncPlan:
    jira_by_key = {issue.key: issue for issue in jira_issues}
    selected_by_key = {issue.key: issue for issue in selected_issues}

    actions: list[SyncAction] = []
    conflict_project_keys = (
        index.conflicted_project_keys
        if managed_keys is None
        else index.conflicted_project_keys & managed_keys
    )
    # With explicit issues, managed_keys also holds the ancestors fetched for
    # placement. Their phase projects are guarded (the task lands there), but
    # only the selected keys' own tasks are reconciled, so only theirs conflict.
    conflict_task_keys = (
        index.conflicted_task_keys
        if managed_keys is None
        else index.conflicted_task_keys & selected_by_key.keys()
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

    def plan_completion(
        issue: JiraIssue, task: ClockifyTask, project_key: str | None
    ) -> None:
        # Only the Done category closes a task, and it never reopens one.
        desired_name = clockify_task_name_for_jira(issue.key, issue.summary)
        if task.status == "ACTIVE":
            actions.append(
                CompleteTaskAction(
                    jira_key=issue.key,
                    current_name=task.name,
                    desired_name=desired_name,
                    project_key=project_key,
                    project_id=task.project_id,
                    task_id=task.id,
                    jira_status=issue.status,
                )
            )
        elif task.name != desired_name:
            actions.append(
                RenameTaskAction(
                    jira_key=issue.key,
                    current_name=task.name,
                    desired_name=desired_name,
                    project_key=project_key,
                    project_id=task.project_id,
                    task_id=task.id,
                )
            )

    for issue in sorted(selected_issues, key=lambda item: item.key):
        if issue.issue_type == PROJECT_PHASE_ISSUE_TYPE:
            if issue.key not in conflict_project_keys:
                plan_project(issue.key, issue.summary)
            continue

        try:
            phase = _project_phase_ancestor(issue, jira_by_key)
        except ValueError as exc:
            actions.append(
                ConflictAction(
                    jira_key=issue.key,
                    message=str(exc),
                )
            )
            continue
        if phase.key in conflict_project_keys:
            if managed_keys is not None:
                # Say why this requested issue produced no task action.
                actions.append(
                    ConflictAction(
                        jira_key=issue.key,
                        project_key=phase.key,
                        message=(
                            f"cannot place {issue.key}: multiple active Clockify "
                            f"projects reference its Project / Phase {phase.key}"
                        ),
                    )
                )
            continue

        plan_project(phase.key, phase.summary)
        if issue.key in conflict_task_keys:
            continue

        desired_name = clockify_task_name_for_jira(issue.key, issue.summary)
        completed = issue.status_category == "Done"
        task = task_by_key.get(issue.key)
        parent = project_by_key.get(phase.key)
        staged = index.staged_tasks_by_jira_key.get(issue.key, ())
        misplaced = task is not None and (
            parent is None or task.project_id != parent.id
        )
        if staged or misplaced:
            # Jira decides placement: a task under another project moves to
            # the phase's project, or resumes an interrupted move.
            actions.append(
                _plan_relocation(
                    issue,
                    phase_key=phase.key,
                    task=task,
                    parent=parent,
                    staged=staged,
                    index=index,
                    desired_name=desired_name,
                    desired_status="DONE" if completed else "ACTIVE",
                )
            )
            continue
        if task is None:
            actions.append(
                CreateTaskAction(
                    jira_key=issue.key,
                    desired_name=desired_name,
                    project_key=phase.key,
                )
            )
            if completed:
                # Clockify creates tasks active: a finished issue needs the
                # second write, and the preview shows both.
                actions.append(
                    CompleteTaskAction(
                        jira_key=issue.key,
                        desired_name=desired_name,
                        project_key=phase.key,
                        after_create=True,
                        jira_status=issue.status,
                        message="completes the task created for this issue",
                    )
                )
            continue
        assert task is not None

        if completed:
            plan_completion(issue, task, phase.key)
        elif task.status == "DONE":
            actions.append(
                ReactivateTaskAction(
                    jira_key=issue.key,
                    current_name=task.name,
                    desired_name=desired_name,
                    project_key=phase.key,
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
                    project_key=phase.key,
                    project_id=task.project_id,
                    task_id=task.id,
                )
            )

    # Explicit issues reconcile exactly the selected tasks (above): a fetched
    # ancestor is placement, not selection, so its own task is left alone.
    # An interrupted relocation whose issue is no longer selected is still
    # finished (Jira's Done status applied on promotion) or reported.
    staged_keys = (
        index.staged_tasks_by_jira_key.keys() - selected_by_key.keys()
        if managed_keys is None
        else set()
    )
    for key in sorted(staged_keys - conflict_task_keys):
        issue = jira_by_key.get(key)
        if issue is None:
            actions.append(
                ConflictAction(
                    jira_key=key,
                    message=(
                        f"cannot resume the relocation of {key}: Jira did not "
                        "return the issue its staging task names"
                    ),
                )
            )
            continue
        try:
            phase = _project_phase_ancestor(issue, jira_by_key)
        except ValueError as exc:
            actions.append(
                ConflictAction(
                    jira_key=key,
                    message=f"cannot resume the relocation of {key}: {exc}",
                )
            )
            continue
        if phase.key in conflict_project_keys:
            continue
        actions.append(
            _plan_relocation(
                issue,
                phase_key=phase.key,
                task=task_by_key.get(key),
                parent=project_by_key.get(phase.key),
                staged=index.staged_tasks_by_jira_key[key],
                index=index,
                desired_name=clockify_task_name_for_jira(issue.key, issue.summary),
                desired_status="DONE" if issue.status_category == "Done" else "ACTIVE",
            )
        )

    other_tasks = task_by_key if managed_keys is None else {}
    for key, task in sorted(other_tasks.items()):
        if key in conflict_task_keys or key in selected_by_key or key in staged_keys:
            continue
        issue = jira_by_key.get(key)
        if issue is None:
            continue
        # Falling out of the selected scope (e.g. backlog deferral) is not
        # completion. Only the Done category closes an existing task.
        if issue.status_category == "Done":
            plan_completion(issue, task, None)
            continue
        desired_name = clockify_task_name_for_jira(issue.key, issue.summary)
        if task.name != desired_name:
            actions.append(
                RenameTaskAction(
                    jira_key=key,
                    current_name=task.name,
                    desired_name=desired_name,
                    project_id=task.project_id,
                    task_id=task.id,
                )
            )

    return _ordered(actions)


def _plan_relocation(
    issue: JiraIssue,
    *,
    phase_key: str,
    task: ClockifyTask | None,
    parent: ClockifyProject | None,
    staged: tuple[ClockifyTask, ...],
    index: ClockifyIndex,
    desired_name: str,
    desired_status: TaskStatus,
) -> SyncAction:
    """A relocation for ``issue``'s task, or why it cannot be one.

    Only these shapes are relocations: the key's one task sits outside the
    phase's project (fresh), optionally with exactly one staging task in the
    phase's project whose marker names that task (resumed), or no task claims
    the key and exactly one staging task names a source that was retired into
    it (resumed before promotion). Every other shape is a conflict.
    """
    key = issue.key

    def conflict(message: str) -> ConflictAction:
        return ConflictAction(
            jira_key=key,
            project_key=phase_key,
            task_id=task.id if task else None,
            message=f"cannot relocate {key}: {message}; resolve it in Clockify, then preview again",
        )

    def relocation(
        source: ClockifyTask,
        *,
        staged_task_id: str | None = None,
        source_retired: bool = False,
    ) -> RelocateTaskAction:
        return RelocateTaskAction(
            jira_key=key,
            project_key=phase_key,
            project_id=parent.id if parent else None,
            task_id=source.id,
            current_name=source.name,
            desired_name=desired_name,
            desired_status=desired_status,
            jira_status=issue.status,
            source_task_id=source.id,
            source_project_id=source.project_id,
            staged_task_id=staged_task_id,
            source_retired=source_retired,
        )

    if len(staged) > 1:
        ids = ", ".join(sorted(t.id for t in staged))
        return conflict(f"multiple Clockify staging tasks carry {key} ({ids})")
    if not staged:
        assert task is not None
        return relocation(task)

    stage = staged[0]
    marker = relocation_marker(stage.name)
    assert marker is not None
    if parent is None or stage.project_id != parent.id:
        return conflict(
            f"staging task {stage.id} is not in the Clockify project for "
            f"its Project / Phase {phase_key}"
        )
    if task is not None:
        if task.project_id == parent.id:
            return conflict(
                f"task {task.id} is already in place, but staging task {stage.id} "
                "also carries the key"
            )
        if marker.other_task_id != task.id:
            return conflict(
                f"staging task {stage.id} names source task {marker.other_task_id}, "
                f"but the task carrying the key is {task.id}"
            )
        return relocation(task, staged_task_id=stage.id)

    source = index.tasks_by_id.get(marker.other_task_id)
    source_marker = relocation_marker(source.name) if source else None
    if (
        source is None
        or source_marker is None
        or source_marker.state != "retired"
        or source_marker.jira_key != key
        or source_marker.other_task_id != stage.id
    ):
        return conflict(
            f"staging task {stage.id} names source task {marker.other_task_id}, "
            "which is not a task retired into it"
        )
    return relocation(source, staged_task_id=stage.id, source_retired=True)


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
    created_task_ids: dict[str, str] = {}
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

        if isinstance(action, RelocateTaskAction):
            destination = projects_by_key.get(action.project_key)
            if destination is None or (
                action.project_id is not None and destination.id != action.project_id
            ):
                raise SyncApplyError(
                    f"{action.jira_key}: destination Clockify project "
                    f"{action.project_key} is unavailable"
                )
            relocated = _apply_relocation(
                action,
                clockify=clockify,
                workspace_id=workspace_id,
                destination=destination,
            )
            tasks_by_id[relocated.id] = relocated
            return AppliedSyncAction(action=action, task=relocated)

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
            created_task_ids[action.jira_key] = created_task.id
            return AppliedSyncAction(action=action, task=created_task)

        # The remaining kinds (RenameTaskAction, ReactivateTaskAction,
        # CompleteTaskAction) all mutate an existing task in place. Conflicts
        # were already rejected above, so this is never a ConflictAction;
        # the assert only exists to narrow the type for the type checker.
        assert isinstance(
            action, RenameTaskAction | ReactivateTaskAction | CompleteTaskAction
        )
        task_id = action.task_id or created_task_ids.get(action.jira_key)
        current_task = tasks_by_id.get(task_id) if task_id else None
        if current_task is None:
            raise SyncApplyError(
                f"{action.jira_key}: Clockify task {task_id!r} is unavailable"
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
            # apply instead of losing it. A relocation's own error carries
            # its recovery account; its cause is still the upstream error.
            recovery = exc.recovery if isinstance(exc, SyncApplyError) else ""
            cause = (
                exc.__cause__
                if isinstance(exc, SyncApplyError) and exc.recovery and exc.__cause__
                else exc
            )
            raise SyncApplyError(
                str(exc), applied=applied, action=action, recovery=recovery
            ) from cause

    return SyncApplyResult(applied=tuple(applied))


class _RelocationProgress:
    """What a relocation has verified so far, for an honest failure report.

    ``pending`` names a write that was issued and has not yet been verified:
    it may not have landed, or may have landed and not read back as planned.
    """

    def __init__(self, action: RelocateTaskAction) -> None:
        self.action = action
        self.step = "reading the source task"
        self.pending: str | None = None
        self.staged_task_id = action.staged_task_id
        self.moved = [entry.entry_id for entry in action.entries if entry.moved]
        self.retired = action.source_retired

    def at(self, step: str) -> None:
        self.step = step

    def write(self, what: str) -> None:
        self.step = what
        self.pending = what

    def verified(self) -> None:
        self.pending = None

    def describe(self) -> str:
        action = self.action
        done: list[str] = []
        if self.staged_task_id is not None and not (
            self.pending and self.pending.startswith("creating")
        ):
            done.append(f"staging task {self.staged_task_id} is in the destination")
        if self.moved:
            done.append(f"time entries moved and verified: {', '.join(self.moved)}")
        if self.retired:
            done.append(f"source task {action.source_task_id} is retired")
        verified = "; ".join(done) if done else "nothing"
        unverified = (
            f" Unverified write: {self.pending}. It may not have landed, or may "
            "have landed and not read back as planned."
            if self.pending
            else ""
        )
        return (
            f"Relocation of {action.jira_key} (source task {action.source_task_id} "
            f"in project {action.source_project_id}) stopped while {self.step}. "
            f"Verified so far: {verified}.{unverified} Nothing was rolled back. "
            "Preview again before retrying: the preview re-reads the source task, "
            f"any '[relocating {action.jira_key} from {action.source_task_id} …]' "
            "staging task and every time entry, and resumes only if they still "
            "match the baseline that staging task recorded before any entry moved."
        )


def _apply_relocation(
    action: RelocateTaskAction,
    *,
    clockify: ClockifySyncClient,
    workspace_id: str,
    destination: ClockifyProject,
) -> ClockifyTask:
    """Stage, move entries, retire, promote; each write read back whole.

    Assumes nobody else writes these records while it runs. A failure stops
    it with :class:`SyncApplyError` whose ``recovery`` says what was verified;
    nothing is rolled back (a rollback would be one more uncertain write, and
    Clockify cannot set every field back).
    """
    progress = _RelocationProgress(action)
    try:
        return _run_relocation(
            action,
            progress,
            clockify=clockify,
            workspace_id=workspace_id,
            destination=destination,
        )
    except (SyncApplyError, ClockifyError) as exc:
        raise SyncApplyError(str(exc), recovery=progress.describe()) from exc


def _run_relocation(
    action: RelocateTaskAction,
    progress: _RelocationProgress,
    *,
    clockify: ClockifySyncClient,
    workspace_id: str,
    destination: ClockifyProject,
) -> ClockifyTask:
    key = action.jira_key
    ws = workspace_id
    if not action.inspected or not action.baseline:
        raise SyncApplyError(
            f"{key}: the relocation was planned without reading its records; "
            "preview again"
        )

    def stale(what: str) -> SyncApplyError:
        return SyncApplyError(
            f"{key}: {what} changed since the preview and was not written; "
            "preview again"
        )

    source = clockify.get_task_record(
        ws, action.source_project_id, action.source_task_id
    )
    if record_fingerprint(source) != action.source_fingerprint:
        raise stale(f"Clockify task {action.source_task_id}")
    # Every record the baseline covers is as previewed before the first write.
    for entry in action.entries:
        progress.at(f"reading time entry {entry.entry_id}")
        if (
            record_fingerprint(clockify.get_time_entry_record(ws, entry.entry_id))
            != entry.fingerprint
        ):
            raise stale(f"time entry {entry.entry_id}")

    staged_name = staged_task_name(
        key, action.source_task_id, action.desired_name, baseline=action.baseline
    )
    if action.staged_task_id is None:
        progress.write(f"creating the staging task in project {destination.id}")
        created = clockify.create_task_record(
            ws, destination.id, task_create_payload(source, name=staged_name)
        )
        staged_id = str(created["id"])
        progress.staged_task_id = staged_id
        progress.write(f"creating staging task {staged_id} in project {destination.id}")
        staged = clockify.get_task_record(ws, destination.id, staged_id)
        if (
            staged.get("name") != staged_name
            or staged.get("projectId") != destination.id
        ):
            raise SyncApplyError(
                f"{key}: Clockify returned staging task {staged_id} as "
                f"{staged.get('name')!r} in project {staged.get('projectId')}, "
                f"expected {staged_name!r} in {destination.id}"
            )
        progress.verified()
    else:
        staged_id = action.staged_task_id
        progress.at(f"reading staging task {staged_id}")
        staged = clockify.get_task_record(ws, destination.id, staged_id)
        if record_fingerprint(staged) != action.staged_fingerprint:
            raise stale(f"Clockify staging task {staged_id}")
        if action.rebaseline:
            # A person accepted the current records: record that baseline
            # before anything else is written.
            name = str(staged["name"]).split("] ", 1)[1]
            rebased = staged_task_name(
                key, action.source_task_id, name, baseline=action.baseline
            )
            progress.write(f"recording a new baseline on staging task {staged_id}")
            clockify.update_task_record(
                ws,
                destination.id,
                staged_id,
                task_update_payload(staged, name=rebased, status=str(staged["status"])),
            )
            after = clockify.get_task_record(ws, destination.id, staged_id)
            _verify_task_record(
                key, after, staged, name=rebased, status=_task_status(staged)
            )
            staged = after
            progress.verified()

    staged = _copy_task_fields(
        action,
        progress,
        source,
        staged,
        clockify=clockify,
        ws=ws,
        project_id=destination.id,
    )

    for entry in action.entries:
        if entry.moved:
            continue
        progress.at(f"reading time entry {entry.entry_id}")
        current = clockify.get_time_entry_record(ws, entry.entry_id)
        if record_fingerprint(current) != entry.fingerprint:
            raise stale(f"time entry {entry.entry_id}")
        progress.write(
            f"moving time entry {entry.entry_id} to staging task {staged_id}"
        )
        clockify.update_time_entry(
            ws,
            entry.entry_id,
            entry_move_payload(current, project_id=destination.id, task_id=staged_id),
        )
        after = clockify.get_time_entry_record(ws, entry.entry_id)
        if after.get("projectId") != destination.id or after.get("taskId") != staged_id:
            raise SyncApplyError(
                f"{key}: time entry {entry.entry_id} read back in project "
                f"{after.get('projectId')} task {after.get('taskId')}, expected "
                f"{destination.id} task {staged_id}"
            )
        if changed := changed_fields(current, after, ignore=ENTRY_PLACEMENT_FIELDS):
            original = json.dumps(
                {field: current.get(field) for field in changed}, sort_keys=True
            )[:500]
            raise SyncApplyError(
                f"{key}: time entry {entry.entry_id} moved to staging task "
                f"{staged_id} but read back with changed {', '.join(changed)}; "
                f"original values {original} (preview fingerprint "
                f"{entry.fingerprint}). It was not changed back: the next "
                "preview reports it against the staging task's baseline"
            )
        progress.moved.append(entry.entry_id)
        progress.verified()

    if not action.source_retired:
        progress.at("re-scanning the source task for time entries")
        users = clockify.get_workspace_users(ws)
        remaining = sorted(
            str(entry["id"])
            for _, entry in _task_entries(
                clockify,
                ws,
                users,
                project_id=action.source_project_id,
                task_id=action.source_task_id,
            )
        )
        if remaining:
            raise SyncApplyError(
                f"{key}: source task {action.source_task_id} still has time "
                f"entries the preview did not include ({', '.join(remaining)}); "
                "it was not retired"
            )
        retired_name = retired_task_name(key, staged_id, str(source["name"]))
        progress.write(f"retiring source task {action.source_task_id}")
        clockify.update_task_record(
            ws,
            action.source_project_id,
            action.source_task_id,
            task_update_payload(source, name=retired_name, status="DONE"),
        )
        retired = clockify.get_task_record(
            ws, action.source_project_id, action.source_task_id
        )
        _verify_task_record(key, retired, source, name=retired_name, status="DONE")
        progress.retired = True
        progress.verified()

    progress.write(f"promoting staging task {staged_id} to {action.desired_name!r}")
    clockify.update_task_record(
        ws,
        destination.id,
        staged_id,
        task_update_payload(
            staged, name=action.desired_name, status=action.desired_status
        ),
    )
    promoted = clockify.get_task_record(ws, destination.id, staged_id)
    _verify_task_record(
        key, promoted, staged, name=action.desired_name, status=action.desired_status
    )
    try:
        task = ClockifyTask.model_validate(promoted)
    except ValidationError as exc:
        raise SyncApplyError(
            f"{key}: Clockify returned promoted task {staged_id} in an unreadable shape"
        ) from exc
    progress.verified()
    return task


def _task_status(task: JsonObject) -> TaskStatus:
    return "DONE" if task.get("status") == "DONE" else "ACTIVE"


def _copy_task_fields(
    action: RelocateTaskAction,
    progress: _RelocationProgress,
    source: JsonObject,
    staged: JsonObject,
    *,
    clockify: ClockifySyncClient,
    ws: str,
    project_id: str,
) -> JsonObject:
    """Make the staging task carry the source's task fields, then prove it.

    Create cannot set billable, and rates have their own endpoints (set from
    now on, never retroactively). Any field still different afterwards,
    unknown ones included, stops the relocation before an entry moves.
    """
    staged_id = str(staged["id"])
    rate_fields = {field for field, _ in TASK_RATE_FIELDS}
    if set(changed_fields(source, staged, ignore=TASK_IDENTITY_FIELDS)) - rate_fields:
        progress.write(f"copying task fields to staging task {staged_id}")
        clockify.update_task_record(
            ws,
            project_id,
            staged_id,
            task_update_payload(
                source, name=str(staged["name"]), status=str(staged["status"])
            ),
        )
        staged = clockify.get_task_record(ws, project_id, staged_id)
    for field, kind in TASK_RATE_FIELDS:
        amount = task_rate_amount(source, field)
        if amount is not None and staged.get(field) != source.get(field):
            progress.write(f"copying the task {kind} to staging task {staged_id}")
            clockify.update_task_rate(
                ws, project_id, staged_id, kind=kind, amount=amount
            )
            staged = clockify.get_task_record(ws, project_id, staged_id)
    if changed := changed_fields(source, staged, ignore=TASK_IDENTITY_FIELDS):
        raise SyncApplyError(
            f"{action.jira_key}: staging task {staged_id} still differs from "
            f"source task {action.source_task_id} in {', '.join(changed)}; "
            "no further time entry was moved"
        )
    progress.verified()
    return staged


def _verify_task_record(
    key: str,
    task: JsonObject,
    before: JsonObject,
    *,
    name: str,
    status: TaskStatus,
) -> None:
    if task.get("name") != name or task.get("status") != status:
        raise SyncApplyError(
            f"{key}: Clockify returned task {task.get('id')} as "
            f"{task.get('name')!r} {task.get('status')!r}, expected {name!r} {status!r}"
        )
    if changed := changed_fields(before, task, ignore=TASK_IDENTITY_FIELDS):
        raise SyncApplyError(
            f"{key}: Clockify task {task.get('id')} read back with changed "
            f"{', '.join(changed)}"
        )


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
            f"{action.jira_key}: Clockify returned task {task.id} with name "
            f"{task.name!r}, expected {action.desired_name!r}"
        )
    if task.status != expected_status:
        raise SyncApplyError(
            f"{action.jira_key}: Clockify returned task {task.id} with status "
            f"{task.status!r}, expected {expected_status!r}"
        )


def _projects_by_jira_key(
    projects: Sequence[ClockifyProject],
) -> dict[str, list[ClockifyProject]]:
    grouped: defaultdict[str, list[ClockifyProject]] = defaultdict(list)
    for project in projects:
        key = jira_key_of_project_note(project.note)
        if key:
            grouped[key].append(project)
    return dict(grouped)


def _tasks_by_jira_key(
    tasks: Sequence[ClockifyTask],
) -> dict[str, list[ClockifyTask]]:
    grouped: defaultdict[str, list[ClockifyTask]] = defaultdict(list)
    for task in tasks:
        key = jira_key_of_task_name(task.name)
        if key:
            grouped[key].append(task)
    return dict(grouped)


# --- The typed report shared by the MCP tool and the CLI's --json -------------


class SyncReportScope(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: SyncScope
    jira_project: str
    issue_keys: tuple[str, ...] = ()


class SyncReportInventory(BaseModel):
    """Counts behind a report.

    ``active_issues`` is the planned or underway issues the scope selected:
    all of the discovered issues, or the requested ones whose Jira status
    category is not Done. A requested Done issue is still planned (and
    created, then completed, when missing) without being counted here.
    """

    model_config = ConfigDict(frozen=True)

    active_issues: int
    jira_issues: int
    projects: int
    tasks: int


class SyncRelocation(BaseModel):
    """What a RELOCATE_TASK moves: the source, and the entries that follow it.

    ``time_entry_ids`` are every entry the relocation carries (the digest
    binds a fingerprint of each whole record); ``moved_entry_ids`` are the
    ones an earlier, interrupted apply already moved to ``staged_task_id``.
    """

    model_config = ConfigDict(frozen=True)

    source_project_id: str
    source_task_id: str
    desired_status: TaskStatus
    staged_task_id: str | None = None
    source_retired: bool = False
    time_entry_ids: tuple[str, ...] = ()
    moved_entry_ids: tuple[str, ...] = ()

    @classmethod
    def from_action(cls, action: RelocateTaskAction) -> SyncRelocation:
        return cls(
            source_project_id=action.source_project_id,
            source_task_id=action.source_task_id,
            desired_status=action.desired_status,
            staged_task_id=action.staged_task_id,
            source_retired=action.source_retired,
            time_entry_ids=tuple(entry.entry_id for entry in action.entries),
            moved_entry_ids=tuple(
                entry.entry_id for entry in action.entries if entry.moved
            ),
        )


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
    relocation: SyncRelocation | None = None
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
            relocation=(
                SyncRelocation.from_action(action)
                if isinstance(action, RelocateTaskAction)
                else None
            ),
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
    inspect that record before retrying. ``recovery`` is set for a
    multi-step action (a relocation): what it had verified, which write may
    be uncertain, and how the next preview resumes.
    """

    model_config = ConfigDict(frozen=True)

    jira_key: str | None = None
    kind: SyncActionKind | None = None
    message: str
    recovery: str = ""


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
                active_issues=sum(
                    snapshot.scope == SyncScope.ASSIGNED_ACTIVE
                    or issue.status_category != "Done"
                    for issue in snapshot.selected_issues
                ),
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
        written = {(item.action.kind, item.action.jira_key): item for item in applied}
        return self.model_copy(
            update={
                "mode": "apply",
                "applied": True,
                "failure": failure,
                "actions": tuple(
                    _with_written_record(action, written[key])
                    if (key := (action.kind, action.jira_key)) in written
                    else action
                    for action in self.actions
                ),
            }
        )


def _with_written_record(
    action: SyncReportAction, item: AppliedSyncAction
) -> SyncReportAction:
    """Flag the action written, and name the record it wrote.

    A create has no ID in the plan; the report gives the verified record's, so
    a later failure (a task created but not yet completed) leaves a known
    identity to inspect.
    """
    update: dict[str, object] = {"applied": True, "verified": True}
    if item.task is not None:
        update.update(task_id=item.task.id, project_id=item.task.project_id)
    elif item.project is not None:
        update["project_id"] = item.project.id
    return action.model_copy(update=update)


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
                recovery=exc.recovery,
            ),
        )
    return report.with_apply_outcome(result.applied)
