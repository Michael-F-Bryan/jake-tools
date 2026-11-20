"""The operator flags shared by ``serve`` and ``doctor``.

A flag is the highest-precedence way to set one non-secret setting for one
invocation; it overrides the environment variable of the same setting
(``flag > env > default``). The table below is deliberately explicit: one row
per flag, naming the setting it overrides (the key :func:`~jake_tools.config.
load_config` takes in ``overrides``) and the environment variable that sets the
same thing, shown in ``--help``. Secrets have no flag, so they never reach a
process listing; ``NAME`` or ``NAME_FILE`` in the environment is the only way.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

import click

from ..claude import EffortLevel


@dataclass(frozen=True)
class Flag:
    flag: str
    setting: str
    env: str
    help: str
    type: Any = str
    multiple: bool = False
    is_bool: bool = False

    @property
    def param(self) -> str:
        return self.flag.split("/")[0].removeprefix("--").replace("-", "_")


FLAGS: tuple[Flag, ...] = (
    Flag(
        "--runs-dir",
        "claude.runs_dir",
        "JAKE_TOOLS_RUNS_DIR",
        "Directory for delegated-task run state.",
        click.Path(path_type=Path),
    ),
    Flag(
        "--session-store",
        "session_store.path",
        "JAKE_TOOLS_SESSION_STORE",
        "Hermes session store (state.db).",
        click.Path(path_type=Path),
    ),
    Flag(
        "--jira-project",
        "clockify.jira_project",
        "JAKE_TOOLS_JIRA_PROJECT",
        "Jira project key for Clockify/Jira sync.",
    ),
    Flag(
        "--clockify-client",
        "clockify.client",
        "JAKE_TOOLS_CLOCKIFY_CLIENT",
        "Clockify client name.",
    ),
    Flag(
        "--clockify-api-base-url",
        "clockify.api_base_url",
        "CLOCKIFY_API_BASE_URL",
        "Clockify API base URL.",
    ),
    Flag(
        "--clockify-workspace",
        "clockify.workspaces",
        "JAKE_TOOLS_CLOCKIFY_WORKSPACES",
        "Allowed Clockify workspace ID; repeat for several.",
        multiple=True,
    ),
    Flag(
        "--claude-model",
        "claude.model",
        "JAKE_TOOLS_CLAUDE_MODEL",
        "Default model for delegated tasks.",
    ),
    Flag(
        "--claude-effort",
        "claude.effort",
        "JAKE_TOOLS_CLAUDE_EFFORT",
        "Default effort for delegated tasks.",
        click.Choice(get_args(EffortLevel)),
    ),
    Flag(
        "--claude-max-turns",
        "claude.max_turns",
        "JAKE_TOOLS_CLAUDE_MAX_TURNS",
        "Default turn limit for delegated tasks.",
        int,
    ),
    Flag(
        "--claude-timeout-seconds",
        "claude.timeout_seconds",
        "JAKE_TOOLS_CLAUDE_TIMEOUT_SECONDS",
        "Default wall-clock limit for delegated tasks.",
        float,
    ),
    Flag(
        "--claude-max-budget-usd",
        "claude.max_budget_usd",
        "JAKE_TOOLS_CLAUDE_MAX_BUDGET_USD",
        "Default spend limit for delegated tasks.",
        float,
    ),
    Flag(
        "--claude-max-live-tasks",
        "claude.max_live_tasks",
        "JAKE_TOOLS_CLAUDE_MAX_LIVE_TASKS",
        "Delegated tasks this process lets run at once.",
        int,
    ),
    Flag(
        "--claude-account-context",
        "claude.account_context",
        "JAKE_TOOLS_CLAUDE_ACCOUNT_CONTEXT",
        "Non-secret label of the Claude account context.",
    ),
    Flag(
        "--hindsight-api-url",
        "hindsight.api_url",
        "JAKE_TOOLS_HINDSIGHT_API_URL",
        "Hindsight server delegated workers share memory through (set with the bank).",
    ),
    Flag(
        "--hindsight-bank-id",
        "hindsight.bank_id",
        "JAKE_TOOLS_HINDSIGHT_BANK_ID",
        "Hindsight bank delegated workers share with their parent agent.",
    ),
    Flag(
        "--hindsight-access",
        "hindsight.access",
        "JAKE_TOOLS_HINDSIGHT_ACCESS",
        "Worker memory access: read, or readwrite (default).",
    ),
    Flag(
        "--transcription-api-base-url",
        "transcription.api_base_url",
        "TRANSCRIPTION_API_BASE_URL",
        "Transcription service base URL.",
    ),
    Flag(
        "--transcription-timeout-seconds",
        "transcription.timeout_seconds",
        "JAKE_TOOLS_TRANSCRIPTION_TIMEOUT_SECONDS",
        "How long one transcription call may take.",
        float,
    ),
    Flag(
        "--microsoft-tenant-id",
        "microsoft.tenant_id",
        "JAKE_TOOLS_MICROSOFT_TENANT_ID",
        "Microsoft tenant ID.",
    ),
    Flag(
        "--microsoft-client-id",
        "microsoft.client_id",
        "JAKE_TOOLS_MICROSOFT_CLIENT_ID",
        "Microsoft application (client) ID.",
    ),
    Flag(
        "--microsoft-account",
        "microsoft.account",
        "JAKE_TOOLS_MICROSOFT_ACCOUNT",
        "Sign-in name the delegated grant must belong to.",
    ),
    Flag(
        "--microsoft-account-object-id",
        "microsoft.account_object_id",
        "JAKE_TOOLS_MICROSOFT_ACCOUNT_OBJECT_ID",
        "Optional immutable pin: the account's object ID.",
    ),
    Flag(
        "--files-sharepoint-host",
        "files.sharepoint_hosts",
        "JAKE_TOOLS_FILES_SHAREPOINT_HOSTS",
        "SharePoint host sharepoint_resolve accepts; repeat for several.",
        multiple=True,
    ),
    Flag(
        "--artifacts/--no-artifacts",
        "artifacts.enabled",
        "JAKE_TOOLS_ARTIFACTS_ENABLED",
        "Expose artifact_publish and artifact_read.",
        is_bool=True,
    ),
)


def _option(flag: Flag) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    names = (flag.flag, flag.param)
    help_text = f"{flag.help} Overrides {flag.env}."
    if flag.is_bool:
        return click.option(*names, default=None, help=help_text)
    return click.option(
        *names,
        type=flag.type,
        multiple=flag.multiple,
        default=None,
        help=help_text,
    )


def config_options(func: Callable[..., Any]) -> Callable[..., Any]:
    """Add every flag; call ``func`` with the supplied ones as ``overrides``."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        overrides: dict[str, Any] = {}
        for flag in FLAGS:
            value = kwargs.pop(flag.param)
            if value is None or value == ():
                continue
            overrides[flag.setting] = tuple(value) if flag.multiple else value
        return func(*args, overrides=overrides, **kwargs)

    decorated: Callable[..., Any] = wrapper
    for flag in reversed(FLAGS):
        decorated = _option(flag)(decorated)
    return decorated
