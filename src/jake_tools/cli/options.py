from collections.abc import Callable
import functools
from typing import Any, cast

from ..hermes import Hermes


def hermes[F: Callable[..., Any]](func: F) -> F:
    """
    Inject a Hermes instance into the decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(hermes=Hermes(), *args, **kwargs)

    return cast(F, wrapper)
