"""Behaviour of the shared ``@agent`` Click decorator (cli/options.py)."""

from __future__ import annotations

from typing import get_args

import click
from click.testing import CliRunner

from jake_tools.claude import AgentSpec, ClaudeAgent, EffortLevel
from jake_tools.cli.context import AppContext
from jake_tools.cli.options import agent


@click.command()
@agent
def _probe(agent: ClaudeAgent) -> None:
    click.echo(f"model={agent.defaults.model} effort={agent.defaults.effort}")


def test_agent_decorator_builds_a_real_agent_from_flags_by_default() -> None:
    runner = CliRunner()

    result = runner.invoke(_probe, ["--model", "claude-haiku-4-5", "--effort", "high"])

    assert result.exit_code == 0
    assert result.output == "model=claude-haiku-4-5 effort=high\n"


def test_agent_decorator_routes_model_and_effort_through_an_injected_factory() -> None:
    """An injected agent factory must still see --model/--effort.

    Regression guard: the old ``isinstance(ctx.obj, dict)`` seam returned an
    injected agent as-is, so a test (or operator) passing --model alongside a
    fake agent had the flag silently ignored.
    """
    recorded: list[AgentSpec] = []

    def recording_factory(spec: AgentSpec) -> ClaudeAgent:
        recorded.append(spec)
        return ClaudeAgent(defaults=spec)

    app = AppContext(agent_factory=recording_factory)
    runner = CliRunner()

    result = runner.invoke(
        _probe,
        ["--model", "claude-opus-4", "--effort", "xhigh"],
        obj=app,
    )

    assert result.exit_code == 0
    assert recorded == [AgentSpec(model="claude-opus-4", effort="xhigh")]
    assert result.output == "model=claude-opus-4 effort=xhigh\n"


def test_effort_choices_are_derived_from_the_claude_sdk_effort_level() -> None:
    option = next(param for param in _probe.params if param.name == "effort")

    assert isinstance(option.type, click.Choice)
    assert tuple(option.type.choices) == get_args(EffortLevel)
