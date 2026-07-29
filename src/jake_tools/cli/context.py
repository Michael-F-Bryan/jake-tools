"""The typed seam between Click commands and the dependencies they construct.

Every external boundary a command needs — the Claude agent, the Clockify and
Jira clients, the newsletter client — is built by a factory carried on
:class:`AppContext`. Production commands get the defaults below; tests inject
fakes by passing ``obj=AppContext(...)`` to :class:`click.testing.CliRunner`
instead of monkeypatching module attributes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import click

from ..claude import AgentSpec, ClaudeAgent
from ..clockify import CLOCKIFY_API_ROOT, ClockifyClient, ClockifyError
from ..clockify_jira_sync import ClockifyInventoryClient, JiraInventoryClient
from ..jira import AcliJiraClient
from ..newsletters import NewsletterClient, NewsletterOperations

AgentFactory = Callable[[AgentSpec], ClaudeAgent]


@dataclass(frozen=True)
class ClockifyConfig:
    """Clockify credentials, resolved once (flag -> env -> default)."""

    api_key: str | None
    api_base_url: str = CLOCKIFY_API_ROOT


ClockifyClientFactory = Callable[[ClockifyConfig], ClockifyInventoryClient]
JiraClientFactory = Callable[[], JiraInventoryClient]
NewsletterClientFactory = Callable[[], NewsletterOperations]


def _default_agent_factory(spec: AgentSpec) -> ClaudeAgent:
    return ClaudeAgent(defaults=spec)


def _default_clockify_client_factory(config: ClockifyConfig) -> ClockifyInventoryClient:
    if config.api_key is None:
        raise ClockifyError(
            "Clockify API key is required. Set CLOCKIFY_API_KEY or pass --api-key."
        )
    return ClockifyClient(api_key=config.api_key, base_url=config.api_base_url)


@dataclass(frozen=True)
class AppContext:
    """Factories for every real dependency a CLI command builds.

    Carried on Click's ``ctx.obj``. ``clockify_config`` starts unset; the
    ``clockify`` group callback is the one place that resolves and attaches
    it, once per invocation.
    """

    agent_factory: AgentFactory = _default_agent_factory
    clockify_client_factory: ClockifyClientFactory = _default_clockify_client_factory
    jira_client_factory: JiraClientFactory = AcliJiraClient
    newsletter_client_factory: NewsletterClientFactory = NewsletterClient
    clockify_config: ClockifyConfig | None = None

    def with_clockify_config(self, config: ClockifyConfig) -> AppContext:
        return replace(self, clockify_config=config)


def app_context(ctx: click.Context) -> AppContext:
    """The typed context for ``ctx``, defaulting when nothing was injected."""
    return ctx.obj if isinstance(ctx.obj, AppContext) else AppContext()
