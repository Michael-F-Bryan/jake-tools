"""sharepoint_upload: add a NEW file to a folder, with explicit conflict behaviour."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from ..config import Config
from ..file_transfers import UploadResult, check_name, read_source, upload_new
from ..sharepoint_files import DriveClient
from . import Handler, Operation
from ._files import DriveId, ItemId, call_files, host_path

DESCRIPTION = (
    "WRITE ACTION: adds a NEW file to a SharePoint/Teams folder that other people can "
    "see. Only call with a file and a destination the user approved in this "
    "conversation. `source_path` is an absolute path on the jake-tools host (1 byte "
    "to 50 MiB); `parent_id` is a folder's item_id; `name` defaults to the source "
    "file's name. `on_conflict` says what happens if the folder already has that "
    "name: `fail` (default; nothing is written) or `rename` (SharePoint keeps both "
    "and picks a new name, reported as `renamed`). It never overwrites an existing "
    "file: use sharepoint_replace for that. The upload is then read back and its "
    "bytes compared with the source. Success means verified=true, with the exact "
    "item and etag. On failure the error detail says whether nothing was written "
    "(`not_applied`), the write may have happened (`write_uncertain`), or it "
    "happened but the readback differs or was unavailable; nothing is retried "
    "automatically and a repeat can create a second file. The file is transported "
    "unchanged. Acts as the server's configured Microsoft account."
)


def build(config: Config) -> Handler:
    async def sharepoint_upload(
        drive_id: DriveId,
        parent_id: ItemId,
        source_path: str,
        name: Annotated[str | None, Field(min_length=1, max_length=255)] = None,
        on_conflict: Literal["fail", "rename"] = "fail",
    ) -> UploadResult:
        source = host_path(source_path, "source_path")

        def run(client: DriveClient) -> UploadResult:
            target = check_name(name if name is not None else Path(source).name)
            return upload_new(
                client,
                drive_id=drive_id,
                parent_id=parent_id,
                name=target,
                content=read_source(source),
                conflict=on_conflict,
            )

        return await call_files(config, run, writes=True)

    return sharepoint_upload


OPERATION = Operation(
    name="sharepoint_upload",
    skill="sharepoint-files",
    description=DESCRIPTION,
    build=build,
    omit_none=True,
)
