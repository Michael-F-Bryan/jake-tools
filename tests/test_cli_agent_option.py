"""Behaviour of the shared ``agent_options`` Click decorator (cli/options.py)."""

from __future__ import annotations

from typing import get_args

import click
from click.testing import CliRunner

from jake_tools.claude import AgentSpec, EffortLevel
from jake_tools.cli.options import AgentOptions, agent_options


@click.command()
@agent_options
def _probe(agent_options: AgentOptions) -> None:
    agent = agent_options.agent()
    click.echo(f"model={agent.defaults.model} effort={agent.defaults.effort}")


def test_agent_options_decorator_builds_a_real_agent_from_flags_by_default() -> None:
    runner = CliRunner()

    result = runner.invoke(_probe, ["--model", "claude-haiku-4-5", "--effort", "high"])

    assert result.exit_code == 0
    assert result.output == "model=claude-haiku-4-5 effort=high\n"


def test_handler_built_agent_carries_the_flag_values() -> None:
    """The handler builds the real agent itself from ``AgentOptions``.

    Regression guard: the old ``@agent`` decorator resolved the agent through
    an injected factory on ``ctx.obj``, and the concern was that a factory
    could silently ignore ``--model``/``--effort``. There is no factory
    seam any more — the handler calls ``agent_options.spec()``/``.agent()``
    directly — so the equivalent guard is that those flag values reach the
    spec the handler actually builds.
    """
    captured: list[AgentSpec] = []

    @click.command()
    @agent_options
    def probe(agent_options: AgentOptions) -> None:
        spec = agent_options.spec()
        captured.append(spec)
        click.echo(f"model={spec.model} effort={spec.effort}")

    runner = CliRunner()

    result = runner.invoke(probe, ["--model", "claude-opus-4", "--effort", "xhigh"])

    assert result.exit_code == 0
    assert captured == [AgentSpec(model="claude-opus-4", effort="xhigh")]
    assert result.output == "model=claude-opus-4 effort=xhigh\n"


def test_effort_choices_are_derived_from_the_claude_sdk_effort_level() -> None:
    option = next(param for param in _probe.params if param.name == "effort")

    assert isinstance(option.type, click.Choice)
    assert tuple(option.type.choices) == get_args(EffortLevel)
