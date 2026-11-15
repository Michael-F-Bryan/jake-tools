"""newsletter_list: one bounded page of CSU Weekly Newsletter entries."""

from __future__ import annotations

import functools
from typing import Annotated

from pydantic import Field

from ..config import Config
from ..newsletter_entries import NewsletterListResult, list_entries
from . import Handler, Operation
from ._newsletter import call_newsletter

DESCRIPTION = (
    "Read-only. List CSU Weekly Newsletter entries newest first, at most `limit` "
    "(1..100, default 10) per call. Coverage is explicit: `complete` is true only when "
    "this page reaches the end of the list; otherwise call again with `next_cursor` "
    "and the same `limit`. Bodies are omitted unless include_body=true (use "
    "newsletter_get for one entry with its attachments). Paging is not a snapshot: "
    "entries created or removed between calls can shift it. Uses the server's "
    "configured Microsoft account; never another."
)


def build(config: Config) -> Handler:
    async def newsletter_list(
        limit: Annotated[int, Field(ge=1, le=100)] = 10,
        cursor: Annotated[str | None, Field(min_length=1, max_length=4096)] = None,
        include_body: bool = False,
    ) -> NewsletterListResult:
        return await call_newsletter(
            config,
            functools.partial(
                list_entries, limit=limit, cursor=cursor, include_body=include_body
            ),
        )

    return newsletter_list


OPERATION = Operation(
    name="newsletter_list",
    capability="newsletter",
    skill="newsletters",
    description=DESCRIPTION,
    build=build,
)
