"""sharepoint_replace: replace an existing file only if it is still at the expected version."""

from __future__ import annotations

from ..config import Config
from ..file_transfers import UploadResult, read_source, replace_file
from ..sharepoint_files import DriveClient
from . import Handler, Operation
from ._files import DriveId, ETag, ItemId, call_files, host_path

DESCRIPTION = (
    "WRITE ACTION: replaces the content of an EXISTING SharePoint/Teams file that other "
    "people use, but only if it is still at the version you last saw. Only call with a "
    "file the user approved replacing in this conversation. `expected_etag` is the "
    "`etag` exactly as sharepoint_get, sharepoint_list or sharepoint_download "
    "showed it. If the file is at a different version the call fails with `plan_stale` "
    "and writes nothing, and the error shows the current version: never overwrite "
    "another person's edits, and never retry with the new etag without reconciling "
    "their changes first. Otherwise the etag is also sent as If-Match when the "
    "upload starts (Microsoft answers 412 if it is stale) and the upload is read "
    "back and compared byte for byte. That check is made once, when the upload "
    "starts: it is not a lock, and an edit landing between that and the commit is "
    "not detected. `source_path` is an absolute path on the jake-tools host (1 byte "
    "to 50 MiB). Success means verified=true, with the exact item and new etag. On "
    "failure the error detail says whether nothing was written (`not_applied`), the "
    "write may have happened (`write_uncertain`), or it happened but the readback "
    "differs or was unavailable; nothing is retried automatically. The file is "
    "transported unchanged. Acts as the server's configured Microsoft account."
)


def build(config: Config) -> Handler:
    async def sharepoint_replace(
        drive_id: DriveId, item_id: ItemId, source_path: str, expected_etag: ETag
    ) -> UploadResult:
        source = host_path(source_path, "source_path")

        def run(client: DriveClient) -> UploadResult:
            return replace_file(
                client,
                drive_id=drive_id,
                item_id=item_id,
                content=read_source(source),
                expected_etag=expected_etag,
            )

        return await call_files(config, run, writes=True)

    return sharepoint_replace


OPERATION = Operation(
    name="sharepoint_replace",
    skill="sharepoint-files",
    description=DESCRIPTION,
    build=build,
    omit_none=True,
)
