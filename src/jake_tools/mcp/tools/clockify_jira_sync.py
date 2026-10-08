"""``clockify_jira_sync``: preview, then apply, the Jira to Clockify plan.

The handler validates arguments and credentials, builds the clients from
:class:`~jake_tools.config.Config` per call, and runs
:func:`jake_tools.clockify_jira_sync.run_jira_sync` (the same function the
CLI calls) on a worker thread. Everything else is the domain module's job.
"""

from __future__ import annotations

import functools

import anyio.to_thread
from mcp.server.fastmcp import FastMCP

from ...clockify import ClockifyClient, ClockifyError
from ...clockify_jira_sync import (
    SyncConflictError,
    SyncPlanStaleError,
    SyncPreparationError,
    SyncReport,
    run_jira_sync,
)
from ...config import Config
from ...jira import JiraClient, JiraError, normalise_jira_key
from ..errors import ToolError
from ..server import structured_tool

DESCRIPTION = """\
Reconcile Jira work into Clockify projects and tasks.

Preview by default (apply=false): reads Jira and Clockify and returns a
SyncReport - scope, inventory counts, the planned actions (create/rename
project, create/reactivate/rename task, mark task done), any conflicts, and
a plan_digest. Nothing is written.

To apply, call again with apply=true and plan_digest set to the digest from
the preview you reviewed. The plan is recomputed; if it no longer matches
the digest the result is a plan_stale error and nothing is written. Plans
with conflicts are never applied (invalid_argument listing the conflicting
Jira keys). Every written record is re-read and verified; each action in the
returned report has applied/verified flags. If a write fails part-way, the
result is still a report: actions written before the failure are marked
applied, and `failure` names the action that failed and why.

issues: optional Jira keys (e.g. ["SF-304"]) to reconcile regardless of
assignee; omitted means active Jira work assigned to the credentialled user.

Never deletes anything in Clockify or Jira, and never writes to Jira.
"""


def register(app: FastMCP, config: Config) -> None:
    @structured_tool(app, name="clockify_jira_sync", description=DESCRIPTION)
    async def clockify_jira_sync(
        issues: list[str] | None = None,
        apply: bool = False,
        plan_digest: str | None = None,
    ) -> SyncReport:
        if apply and not plan_digest:
            raise ToolError(
                "invalid_argument",
                "apply=true requires plan_digest from a preview of the same plan",
                detail={"argument": "plan_digest"},
            )
        issue_keys = _normalise_issue_keys(issues or [])
        clockify, jira = _build_clients(config)

        run = functools.partial(
            run_jira_sync,
            clockify=clockify,
            jira=jira,
            jira_project=config.jira_project.value,
            clockify_client=config.clockify_client.value,
            issue_keys=issue_keys,
            apply=apply,
            expected_digest=plan_digest if apply else None,
        )
        try:
            return await anyio.to_thread.run_sync(run)
        except SyncPlanStaleError as exc:
            raise ToolError(
                "plan_stale",
                str(exc),
                detail={"expected": exc.expected, "actual": exc.actual},
            ) from exc
        except SyncConflictError as exc:
            raise ToolError(
                "invalid_argument",
                "The plan has conflicts and cannot be applied; resolve them in "
                "Clockify or Jira and preview again. Nothing was written.",
                detail={
                    "conflicts": [c.jira_key for c in exc.report.conflicts],
                },
            ) from exc
        except (ClockifyError, SyncPreparationError) as exc:
            raise _upstream_error("Clockify", exc, config) from exc
        except JiraError as exc:
            raise _upstream_error("Jira", exc, config) from exc


def _normalise_issue_keys(issues: list[str]) -> tuple[str, ...]:
    keys: list[str] = []
    for key in issues:
        try:
            keys.append(normalise_jira_key(key))
        except JiraError as exc:
            raise ToolError(
                "invalid_argument",
                f"Invalid Jira issue key: {key!r}",
                detail={"argument": "issues", "key": key},
            ) from exc
    return tuple(keys)


def _build_clients(config: Config) -> tuple[ClockifyClient, JiraClient]:
    api_key = config.clockify_api_key.value
    if not api_key:
        raise ToolError(
            "missing_credentials",
            "Clockify credentials are not configured: set CLOCKIFY_API_KEY "
            "in the server's environment.",
            detail={"variables": ["CLOCKIFY_API_KEY"]},
        )
    missing = [
        setting.name
        for setting in (config.jira_base_url, config.jira_email, config.jira_api_token)
        if not setting.value
    ]
    base_url = config.jira_base_url.value
    email = config.jira_email.value
    token = config.jira_api_token.value
    if not (base_url and email and token):
        raise ToolError(
            "missing_credentials",
            f"Jira credentials are not configured: set {', '.join(missing)} "
            "in the server's environment.",
            detail={"variables": missing},
        )
    try:
        clockify = ClockifyClient(
            api_key=api_key, base_url=config.clockify_api_base_url.value
        )
    except ClockifyError as exc:
        raise _upstream_error("Clockify", exc, config) from exc
    try:
        jira = JiraClient(base_url=base_url, email=email, api_token=token)
    except JiraError as exc:
        raise _upstream_error("Jira", exc, config) from exc
    return clockify, jira


def _upstream_error(integration: str, exc: Exception, config: Config) -> ToolError:
    """An ``upstream_error`` naming the integration, with secrets scrubbed.

    Client messages carry the method, path, status and a slice of the
    response body. None of them should contain a credential, but a Jira base
    URL counts as a secret here and an upstream could echo anything, so every
    configured secret value is redacted before the message leaves.
    """
    message = str(exc)
    for setting in (
        config.clockify_api_key,
        config.jira_base_url,
        config.jira_email,
        config.jira_api_token,
    ):
        if setting.value:
            message = message.replace(setting.value, f"[{setting.name}]")
    return ToolError(
        "upstream_error",
        f"{integration} error: {message}",
        detail={"integration": integration.lower()},
    )
