from collections.abc import Callable
import functools
from typing import Any, cast

import click

from ..hermes import Hermes


def hermes[F: Callable[..., Any]](func: F) -> F:
    """
    Inject a Hermes instance into the decorated function.
    """

    @click.option(
        "--default-model",
        default="gpt-5.4-mini",
        help="The default model to use for the Hermes instance.",
    )
    @functools.wraps(func)
    def wrapper(default_model: str, *args: Any, **kwargs: Any) -> Any:
        hermes = Hermes(default_model=default_model)
        return func(hermes, *args, **kwargs)

    return cast(F, wrapper)
