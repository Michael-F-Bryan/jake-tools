import asyncio
import functools
from collections.abc import Callable, Coroutine
from typing import Any, cast, get_args

import click
from pydantic import BaseModel

from ..claude import DEFAULT_MODEL, AgentSpec, ClaudeAgent, EffortLevel

F = Callable[..., Any]


def coro[**P, R](func: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, R]:
    """Run an async Click callback to completion.

    Click callbacks must be synchronous, so this is the one place where the
    async agent boundary meets the sync CLI shell. Apply it closest to the
    function, below the option decorators.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return asyncio.run(func(*args, **kwargs))

    return wrapper


class AgentOptions(BaseModel):
    """Which model/effort to run the shared Claude agent seam with.

    Every LLM-calling command builds its :class:`~..claude.ClaudeAgent` from
    one of these instead of resolving it through shared context state, per
    the CLI-options memo
    (``_working/transcription-workflow-interview/plans/memo-cli-options.md``).
    """

    model: str
    effort: EffortLevel | None

    def spec(self) -> AgentSpec:
        return AgentSpec(model=self.model, effort=self.effort)

    def agent(self) -> ClaudeAgent:
        return ClaudeAgent(defaults=self.spec())


def agent_options(func: F) -> F:
    """Inject an :class:`AgentOptions` built from ``--model``/``--effort``.

    Stacks the two flags, pops their parsed values, builds the typed options
    object, and forwards it via ``ctx.invoke`` — the same shape as the
    ``transcript_options.py`` decorators. The handler constructs the real
    :class:`~..claude.ClaudeAgent` from ``agent_options.agent()`` at the top
    of its body, so a fake agent for tests is created by monkeypatching
    ``AgentOptions.agent`` (or the specific dependency it returns), not by
    injecting a factory through shared context state.
    """

    @click.option(
        "--model",
        default=DEFAULT_MODEL,
        show_default=True,
        help="Claude model for this command's agent calls.",
    )
    @click.option(
        "--effort",
        type=click.Choice(get_args(EffortLevel)),
        default=None,
        help="Reasoning effort. Higher costs more and takes longer.",
    )
    @click.pass_context
    @functools.wraps(func)
    def wrapper(ctx: click.Context, *args: Any, **kwargs: Any) -> Any:
        options = AgentOptions(
            model=kwargs.pop("model"),
            effort=cast(EffortLevel | None, kwargs.pop("effort")),
        )
        return ctx.invoke(func, *args, agent_options=options, **kwargs)

    return cast(F, wrapper)
