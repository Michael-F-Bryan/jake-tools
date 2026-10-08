"""``clockify_jira_sync``: preview, then apply, the Jira to Clockify plan.

The handler validates arguments and credentials, builds the clients from
:class:`~jake_tools.config.Config` per call, and runs
:func:`jake_tools.clockify_jira_sync.run_jira_sync` (the same function the
CLI calls) on a worker thread. Everything else is the domain module's job.
"""

from __future__ import annotations

import functools
from urllib.parse import urlsplit

import anyio.to_thread
from mcp.server.fastmcp import FastMCP

from ...clockify import ClockifyClient, ClockifyError
from ...clockify_jira_sync import (
    SyncApplyError,
    SyncConflictError,
    SyncPlanStaleError,
    SyncPreparationError,
    SyncReport,
    run_jira_sync,
)
from ...config import Config
from ...http import UpstreamError
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
            describe_failure=functools.partial(_safe_message, config=config),
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
        except (ClockifyError, SyncPreparationError, JiraError) as exc:
            raise ToolError(
                "upstream_error",
                _safe_message(exc, config),
                detail={"integration": _integration(exc).lower()},
            ) from exc


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
    base_url = config.jira_base_url.value
    email = config.jira_email.value
    token = config.jira_api_token.value
    if not (base_url and email and token):
        missing = [
            name
            for name, value in (
                ("JIRA_BASE_URL", base_url),
                ("JIRA_EMAIL", email),
                ("JIRA_API_TOKEN", token),
            )
            if not value
        ]
        raise ToolError(
            "missing_credentials",
            f"Jira credentials are not configured: set {', '.join(missing)} "
            "in the server's environment.",
            detail={"variables": missing},
        )
    # Constructor errors are configuration problems; their text may quote a
    # configured value, so it is never passed on.
    try:
        clockify = ClockifyClient(
            api_key=api_key, base_url=config.clockify_api_base_url.value
        )
    except ClockifyError as exc:
        raise ToolError(
            "upstream_error",
            "Clockify error: the client rejected its configuration "
            "(check CLOCKIFY_API_KEY and CLOCKIFY_API_BASE_URL)",
            detail={"integration": "clockify"},
        ) from exc
    try:
        jira = JiraClient(base_url=base_url, email=email, api_token=token)
    except JiraError as exc:
        raise ToolError(
            "upstream_error",
            "Jira error: the client rejected its configuration "
            "(check JIRA_BASE_URL, JIRA_EMAIL and JIRA_API_TOKEN)",
            detail={"integration": "jira"},
        ) from exc
    return clockify, jira


# Shorter values are not redacted: replacing every occurrence of a one- or
# two-character "secret" mangles the message without protecting anything.
_MIN_REDACTED_LENGTH = 6


def _integration(exc: BaseException) -> str:
    return "Jira" if isinstance(exc, JiraError) else "Clockify"


def _safe_message(exc: BaseException, config: Config) -> str:
    """A message about ``exc`` that is safe to hand to the MCP client.

    Raw ``requests`` exception text names the host, and HTTP errors carry a
    slice of the response body, so neither is ever passed on: an
    :class:`UpstreamError` with a ``safe_summary`` is reported by that
    summary alone (operation plus HTTP status or exception class). Anything
    else is this package's own text. Every configured secret is redacted on
    top, as defence in depth.

    A :class:`SyncApplyError` is described by the error that caused it.
    """
    if isinstance(exc, SyncApplyError) and exc.__cause__ is not None:
        exc = exc.__cause__
    if isinstance(exc, UpstreamError) and exc.safe_summary is not None:
        text = exc.safe_summary
    else:
        text = str(exc)
    return f"{_integration(exc)} error: {_redact(text, config)}"


def _redact(text: str, config: Config) -> str:
    for value, name in sorted(
        _secret_forms(config), key=lambda item: len(item[0]), reverse=True
    ):
        text = text.replace(value, f"[{name}]")
    return text


def _secret_forms(config: Config) -> set[tuple[str, str]]:
    """Every spelling of a configured secret that could appear in a message."""
    forms: set[tuple[str, str]] = set()
    for setting in (
        config.clockify_api_key,
        config.jira_base_url,
        config.jira_email,
        config.jira_api_token,
    ):
        if setting.value:
            forms.add((setting.value, setting.name))
            forms.add((setting.value.strip(), setting.name))
    if config.jira_base_url.value:
        url = config.jira_base_url.value.strip().rstrip("/")
        if "://" not in url:
            url = f"https://{url}"
        parts = urlsplit(url)
        for form in (url, parts.netloc, parts.hostname or ""):
            forms.add((form, "JIRA_BASE_URL"))
    return {
        (value, name) for value, name in forms if len(value) >= _MIN_REDACTED_LENGTH
    }
