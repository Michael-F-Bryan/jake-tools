from __future__ import annotations

import functools
import json
from typing import Any, cast

import click
from pydantic import BaseModel

from ..clockify import (
    CLOCKIFY_API_ROOT,
    ClockifyClient,
    ClockifyError,
    ClockifyUser,
    JiraIssueRef,
)
from ..clockify_jira_sync import (
    ClockifyInventoryClient,
    JiraInventoryClient,
    SyncAction,
    SyncActionKind,
    SyncApplyError,
    SyncPreparationError,
    SyncSnapshot,
    apply_sync_plan,
    prepare_jira_sync,
)
from ..jira import JiraClient, JiraError


class ClockifyOptions(BaseModel):
    """Clockify credentials, resolved once per invocation (flag -> env -> default).

    Set by the ``clockify`` group callback and carried on ``ctx.obj`` — the
    one seam Click's own dispatch requires, since group-level flags are
    parsed before any subcommand runs. It holds nothing but the flag values
    themselves; every subcommand builds its own client from it explicitly.
    """

    api_key: str | None
    api_base_url: str = CLOCKIFY_API_ROOT

    def inventory_client(self) -> ClockifyInventoryClient:
        if self.api_key is None:
            raise ClockifyError(
                "Clockify API key is required. Set CLOCKIFY_API_KEY or pass --api-key."
            )
        return ClockifyClient(api_key=self.api_key, base_url=self.api_base_url)


class JiraOptions(BaseModel):
    """Jira REST credentials for the ``jira-sync`` command."""

    base_url: str | None
    email: str | None
    api_token: str | None

    def inventory_client(self) -> JiraInventoryClient:
        if self.base_url is None or self.email is None or self.api_token is None:
            missing = [
                name
                for name, value in (
                    ("JIRA_BASE_URL", self.base_url),
                    ("JIRA_EMAIL", self.email),
                    ("JIRA_API_TOKEN", self.api_token),
                )
                if value is None
            ]
            raise JiraError(
                f"Jira configuration is required. Set {', '.join(missing)}."
            )
        return JiraClient(
            base_url=self.base_url,
            email=self.email,
            api_token=self.api_token,
        )


def jira_options(func: Any) -> Any:
    """Inject a :class:`JiraOptions` built from the ``--jira-*`` flags."""

    @click.option(
        "--jira-base-url",
        envvar="JIRA_BASE_URL",
        default=None,
        help="Jira site URL or hostname. Defaults to JIRA_BASE_URL.",
    )
    @click.option(
        "--jira-email",
        envvar="JIRA_EMAIL",
        default=None,
        help="Atlassian account email. Defaults to JIRA_EMAIL.",
    )
    @click.option(
        "--jira-api-token",
        envvar="JIRA_API_TOKEN",
        default=None,
        help="Atlassian API token. Defaults to JIRA_API_TOKEN.",
    )
    @click.pass_context
    @functools.wraps(func)
    def wrapper(ctx: click.Context, *args: Any, **kwargs: Any) -> Any:
        options = JiraOptions(
            base_url=kwargs.pop("jira_base_url"),
            email=kwargs.pop("jira_email"),
            api_token=kwargs.pop("jira_api_token"),
        )
        return ctx.invoke(func, *args, jira_options=options, **kwargs)

    return wrapper


def _clockify_options(ctx: click.Context) -> ClockifyOptions:
    return cast(ClockifyOptions, ctx.obj)


@click.group()
@click.option(
    "--api-key",
    envvar="CLOCKIFY_API_KEY",
    default=None,
    help="Clockify API key. Defaults to CLOCKIFY_API_KEY.",
)
@click.option(
    "--api-base-url",
    envvar="CLOCKIFY_API_BASE_URL",
    default=CLOCKIFY_API_ROOT,
    show_default=True,
    help="Clockify API base URL. Defaults to CLOCKIFY_API_BASE_URL.",
)
@click.pass_context
def clockify(ctx: click.Context, api_key: str | None, api_base_url: str) -> None:
    """Work with Clockify time-tracking data."""
    ctx.obj = ClockifyOptions(api_key=api_key, api_base_url=api_base_url)


@clockify.command("jira-name")
@click.argument("key")
@click.argument("summary")
@click.option(
    "--kind",
    type=click.Choice(["all", "project", "task", "note"]),
    default="all",
    show_default=True,
    help="Which Clockify name to emit.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def jira_name(key: str, summary: str, kind: str, as_json: bool) -> None:
    """Render Clockify names for a Jira-backed item.

    Jira project/phase work becomes a Clockify project named from the summary
    only. Individual Jira tickets remain Clockify tasks prefixed with the
    ticket key.
    """
    try:
        issue = JiraIssueRef(key=key, summary=summary)
        values = {
            "project": issue.project_name,
            "task": issue.task_name,
            "note": issue.project_note,
        }
    except (ClockifyError, JiraError) as exc:
        raise click.ClickException(str(exc)) from exc

    selected = values if kind == "all" else {kind: values[kind]}
    if as_json:
        click.echo(json.dumps(selected, indent=2))
        return

    for label, value in selected.items():
        click.echo(f"{label}: {value}")


@clockify.command("jira-sync")
@jira_options
@click.option(
    "--apply/--dry-run",
    "apply_changes",
    default=False,
    help="Apply the plan or only report it. Defaults to a dry run.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.option(
    "--jira-project",
    default="SF",
    show_default=True,
    help="Jira project key to reconcile.",
)
@click.option(
    "--clockify-client",
    default="Sunfish Robotics",
    show_default=True,
    help="Exact active Clockify client name for newly created projects.",
)
@click.option(
    "--issue",
    "issue_keys",
    multiple=True,
    help=(
        "Reconcile an exact Jira issue regardless of assignee. Repeat for multiple "
        "issues; otherwise sync active work assigned to currentUser()."
    ),
)
@click.pass_context
def jira_sync(
    ctx: click.Context,
    jira_options: JiraOptions,
    apply_changes: bool,
    as_json: bool,
    jira_project: str,
    clockify_client: str,
    issue_keys: tuple[str, ...],
) -> None:
    """Reconcile Jira work with Clockify.

    Dry runs are read-only. Use --apply to create, rename, reactivate, or complete
    Clockify records from the emitted plan. By default, only active Jira work
    assigned to currentUser() is selected. --issue selects exact work items
    regardless of assignee. The command never deletes records.
    """
    try:
        clockify_api = _clockify_options(ctx).inventory_client()
        jira_api = jira_options.inventory_client()
        snapshot = prepare_jira_sync(
            clockify=clockify_api,
            jira=jira_api,
            jira_project=jira_project,
            clockify_client=clockify_client,
            issue_keys=issue_keys,
        )
        result = None
        if apply_changes:
            result = apply_sync_plan(
                snapshot.plan,
                clockify=clockify_api,
                workspace_id=snapshot.workspace_id,
                client_id=snapshot.client_id,
                index=snapshot.index,
            )
    except (ClockifyError, JiraError, SyncPreparationError, SyncApplyError) as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        payload = {
            "mode": "apply" if apply_changes else "dry-run",
            "workspaceId": snapshot.workspace_id,
            "jiraProject": jira_project,
            "clockifyClient": clockify_client,
            "scope": _scope_payload(snapshot, jira_project=jira_project),
            "inventory": _inventory_payload(snapshot),
            "actions": [_action_payload(action) for action in snapshot.plan.actions],
        }
        if result is not None:
            payload["applied"] = len(result.applied)
            # Every applied action re-fetches and verifies the Clockify record;
            # apply_sync_plan raises before returning if verification fails, so
            # reaching this point means everything applied was verified.
            payload["verified"] = True
        click.echo(json.dumps(payload, indent=2))
    else:
        _emit_sync_plan(snapshot, applied=apply_changes)

    if snapshot.plan.has_conflicts:
        ctx.exit(1)


@clockify.command()
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_context
def whoami(ctx: click.Context, as_json: bool) -> None:
    """Show the Clockify user for the configured API key."""
    try:
        client = _clockify_options(ctx).inventory_client()
        user = client.get_user()
    except ClockifyError as exc:
        raise click.ClickException(str(exc)) from exc

    _emit_user(user, as_json=as_json)


def _action_payload(action: SyncAction) -> dict[str, object]:
    return {
        "kind": action.kind.value,
        "jiraKey": action.jira_key,
        "currentName": action.current_name,
        "desiredName": action.desired_name,
        "projectKey": action.project_key,
        "projectId": action.project_id,
        "taskId": action.task_id,
        "jiraStatus": action.jira_status,
        "message": action.message,
    }


def _scope_payload(snapshot: SyncSnapshot, *, jira_project: str) -> dict[str, object]:
    return {
        "kind": snapshot.scope.value,
        "jiraProject": jira_project,
        "issueKeys": list(snapshot.requested_issue_keys),
    }


def _inventory_payload(snapshot: SyncSnapshot) -> dict[str, int]:
    return {
        "activeIssues": len(snapshot.active_issues),
        "jiraIssues": len(snapshot.jira_issues),
        "projects": len(snapshot.projects),
        "tasks": len(snapshot.tasks),
    }


def _emit_sync_plan(snapshot: SyncSnapshot, *, applied: bool) -> None:
    actions = snapshot.plan.actions
    if not actions:
        click.echo(f"No Clockify changes required for {_scope_description(snapshot)}.")
        return

    label = "Applied" if applied else "Dry run"
    suffix = "change" if len(actions) == 1 else "changes"
    click.echo(f"{label}: {len(actions)} {suffix}")
    for action in actions:
        click.echo(_describe_action(action))


def _scope_description(snapshot: SyncSnapshot) -> str:
    if snapshot.requested_issue_keys:
        return ", ".join(snapshot.requested_issue_keys)
    return "active Jira issues assigned to currentUser()"


def _describe_action(action: SyncAction) -> str:
    prefix = f"{action.kind.value} {action.jira_key}"
    if action.kind in {SyncActionKind.RENAME_PROJECT, SyncActionKind.RENAME_TASK}:
        return f"{prefix} — {action.current_name!r} → {action.desired_name!r}"
    if action.kind == SyncActionKind.MARK_TASK_DONE:
        return f"{prefix} — Jira status: {action.jira_status}"
    if action.kind == SyncActionKind.REACTIVATE_TASK:
        return f"{prefix} — reactivate as {action.desired_name!r}"
    if action.kind == SyncActionKind.CREATE_TASK:
        return f"{prefix} — create under {action.project_key}: {action.desired_name}"
    if action.kind == SyncActionKind.CREATE_PROJECT:
        return f"{prefix} — create project {action.desired_name!r}"
    if action.kind == SyncActionKind.CONFLICT:
        return f"{prefix} — {action.message}"
    return prefix


def _emit_user(user: ClockifyUser, *, as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps(user.model_dump(mode="json"), indent=2))
        return

    click.echo(f"ID: {user.id}")
    click.echo(f"Name: {user.name}")
    click.echo(f"Email: {user.email}")
    click.echo(f"Active workspace: {user.active_workspace}")
    click.echo(f"Default workspace: {user.default_workspace}")
