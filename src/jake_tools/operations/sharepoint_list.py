"""sharepoint_list: one bounded page of a folder's children."""

from __future__ import annotations

import functools
from typing import Annotated

from pydantic import Field

from ..config import Config
from ..file_transfers import FolderListing, list_folder
from . import Handler, Operation
from ._files import DriveId, ItemId, call_files

DESCRIPTION = (
    "Read-only. List a folder's files and sub-folders, at most `limit` (1..200, "
    "default 50) per call, each with `drive_id`, `item_id`, name, kind, web URL, "
    "size and version (`etag`, `ctag`, modified, modified_by). Coverage is "
    "explicit: `complete` is true only when this page reaches the end of the "
    "folder; otherwise call again with the same drive_id, item_id and limit and the "
    "returned `next_cursor`. The service's order is not promised and paging is not "
    "a snapshot: items added or removed between calls can shift it. Uses the "
    "server's configured Microsoft account; never another."
)


def build(config: Config) -> Handler:
    async def sharepoint_list(
        drive_id: DriveId,
        item_id: ItemId,
        limit: Annotated[int, Field(ge=1, le=200)] = 50,
        cursor: Annotated[str | None, Field(min_length=1, max_length=4096)] = None,
    ) -> FolderListing:
        return await call_files(
            config,
            functools.partial(
                list_folder,
                drive_id=drive_id,
                item_id=item_id,
                limit=limit,
                cursor=cursor,
            ),
        )

    return sharepoint_list


OPERATION = Operation(
    name="sharepoint_list",
    skill="sharepoint-files",
    description=DESCRIPTION,
    build=build,
    omit_none=True,
)
