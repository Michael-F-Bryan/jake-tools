from collections.abc import Callable
import functools
from typing import Any, cast

import click

from ..hermes import DEFAULT_MODEL, Hermes


def hermes[F: Callable[..., Any]](func: F) -> F:
    """
    Inject a Hermes instance into the decorated function.
    """

    @click.option(
        "--default-model",
        default=DEFAULT_MODEL,
        help="The default model to use for the Hermes instance.",
    )
    @click.pass_context
    @functools.wraps(func)
    def wrapper(
        ctx: click.Context, default_model: str, *args: Any, **kwargs: Any
    ) -> Any:
        instance = Hermes(default_model=default_model)
        return ctx.invoke(func, instance, *args, **kwargs)

    return cast(F, wrapper)
