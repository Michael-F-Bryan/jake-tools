import asyncio
import functools
from collections.abc import Callable, Coroutine
from typing import Any, cast

import click
from claude_agent_sdk import EffortLevel

from ..claude import DEFAULT_MODEL, AgentSpec, ClaudeAgent


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


def agent[F: Callable[..., Any]](func: F) -> F:
    """Inject a :class:`ClaudeAgent` as the decorated command's first argument."""

    @click.option(
        "--model",
        default=DEFAULT_MODEL,
        show_default=True,
        help="Claude model for this command's agent calls.",
    )
    @click.option(
        "--effort",
        type=click.Choice(["low", "medium", "high", "xhigh", "max"]),
        default=None,
        help="Reasoning effort. Higher costs more and takes longer.",
    )
    @click.pass_context
    @functools.wraps(func)
    def wrapper(
        ctx: click.Context,
        model: str,
        effort: str | None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        injected = ctx.obj.get("agent") if isinstance(ctx.obj, dict) else None
        instance = injected or ClaudeAgent(
            defaults=AgentSpec(model=model, effort=cast(EffortLevel | None, effort))
        )
        return ctx.invoke(func, instance, *args, **kwargs)

    return cast(F, wrapper)
