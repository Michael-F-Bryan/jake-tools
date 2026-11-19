"""sharepoint_get: one drive item and the version of it that exists now."""

from __future__ import annotations

import functools

from ..config import Config
from ..file_transfers import get_item
from ..sharepoint_files import RemoteItem
from . import Handler, Operation
from ._files import DriveId, ItemId, call_files

DESCRIPTION = (
    "Read-only. Read one drive item by `drive_id` and `item_id`: name, kind, web URL, "
    "size and its current version (`etag`, `ctag`, modified, modified_by). Use it "
    "to re-check an item before replacing it, or to look at an item after an upload "
    "whose outcome was uncertain. Uses the server's configured Microsoft account; "
    "never another."
)


def build(config: Config) -> Handler:
    async def sharepoint_get(drive_id: DriveId, item_id: ItemId) -> RemoteItem:
        return await call_files(
            config,
            functools.partial(get_item, drive_id=drive_id, item_id=item_id),
        )

    return sharepoint_get


OPERATION = Operation(
    name="sharepoint_get",
    skill="sharepoint-files",
    description=DESCRIPTION,
    build=build,
    omit_none=True,
)
