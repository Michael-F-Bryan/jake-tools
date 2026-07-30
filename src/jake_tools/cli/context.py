"""The typed seam between Click commands and the dependencies they construct.

Every external boundary a command needs — the Claude agent, the Clockify and
Jira clients, the newsletter client — is built by a factory carried on
:class:`AppContext`. Production commands get the defaults below; tests inject
fakes by passing ``obj=AppContext(...)`` to :class:`click.testing.CliRunner`
instead of monkeypatching module attributes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import click

from ..claude import AgentSpec, ClaudeAgent
from ..clockify import CLOCKIFY_API_ROOT, ClockifyClient, ClockifyError
from ..clockify_jira_sync import ClockifyInventoryClient, JiraInventoryClient
from ..jira import JiraClient, JiraError
from ..newsletters import NewsletterClient, NewsletterOperations
from ..transcripts.bundle.control import BundleExecutor

AgentFactory = Callable[[AgentSpec], ClaudeAgent]


@dataclass(frozen=True)
class ClockifyConfig:
    """Clockify credentials, resolved once (flag -> env -> default)."""

    api_key: str | None
    api_base_url: str = CLOCKIFY_API_ROOT


@dataclass(frozen=True)
class JiraConfig:
    """Jira REST credentials, resolved once by the command boundary."""

    base_url: str | None
    email: str | None
    api_token: str | None


ClockifyClientFactory = Callable[[ClockifyConfig], ClockifyInventoryClient]
JiraClientFactory = Callable[[JiraConfig], JiraInventoryClient]
NewsletterClientFactory = Callable[[], NewsletterOperations]


def _default_agent_factory(spec: AgentSpec) -> ClaudeAgent:
    return ClaudeAgent(defaults=spec)


def _default_clockify_client_factory(config: ClockifyConfig) -> ClockifyInventoryClient:
    if config.api_key is None:
        raise ClockifyError(
            "Clockify API key is required. Set CLOCKIFY_API_KEY or pass --api-key."
        )
    return ClockifyClient(api_key=config.api_key, base_url=config.api_base_url)


def _default_jira_client_factory(config: JiraConfig) -> JiraInventoryClient:
    if config.base_url is None or config.email is None or config.api_token is None:
        missing = [
            name
            for name, value in (
                ("JIRA_BASE_URL", config.base_url),
                ("JIRA_EMAIL", config.email),
                ("JIRA_API_TOKEN", config.api_token),
            )
            if value is None
        ]
        raise JiraError(f"Jira configuration is required. Set {', '.join(missing)}.")
    return JiraClient(
        base_url=config.base_url,
        email=config.email,
        api_token=config.api_token,
    )


@dataclass(frozen=True)
class AppContext:
    """Factories for every real dependency a CLI command builds.

    Carried on Click's ``ctx.obj``. ``clockify_config`` starts unset; the
    ``clockify`` group callback is the one place that resolves and attaches
    it, once per invocation. ``bundle_executors`` is the M2 ``resume``
    dispatch seam: v1 ships none (the empty default), so every
    ``next_action.kind`` is an explicit "no executor registered" error
    until a real transform is injected here -- production code never
    populates this map either, only tests exercising the dispatch path.
    """

    agent_factory: AgentFactory = _default_agent_factory
    clockify_client_factory: ClockifyClientFactory = _default_clockify_client_factory
    jira_client_factory: JiraClientFactory = _default_jira_client_factory
    newsletter_client_factory: NewsletterClientFactory = NewsletterClient
    clockify_config: ClockifyConfig | None = None
    bundle_executors: Mapping[str, BundleExecutor] = MappingProxyType({})

    def with_clockify_config(self, config: ClockifyConfig) -> AppContext:
        return replace(self, clockify_config=config)


def app_context(ctx: click.Context) -> AppContext:
    """The typed context for ``ctx``, defaulting when nothing was injected."""
    return ctx.obj if isinstance(ctx.obj, AppContext) else AppContext()
