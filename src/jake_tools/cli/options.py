import functools
from collections.abc import Callable
from typing import Any, cast

import click

from ..hermes import DEFAULT_MODEL, AgentSpec, Hermes


def hermes[F: Callable[..., Any]](func: F) -> F:
    """
    Inject a Hermes instance into the decorated function.
    """

    @click.option(
        "--default-model",
        default=DEFAULT_MODEL,
        help="The default model to use for the Hermes instance.",
    )
    @click.option(
        "--provider",
        default="",
        help="Override the LLM provider (e.g. 'openai-codex', 'openrouter').",
    )
    @click.pass_context
    @functools.wraps(func)
    def wrapper(
        ctx: click.Context,
        default_model: str,
        provider: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        injected = None
        if isinstance(ctx.obj, dict):
            injected = ctx.obj.get("hermes")

        instance = injected or Hermes(
            defaults=AgentSpec(model=default_model, provider=provider or "")
        )
        return ctx.invoke(func, instance, *args, **kwargs)

    return cast(F, wrapper)
