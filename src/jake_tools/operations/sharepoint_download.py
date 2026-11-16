"""sharepoint_download: save one file's current version to a new local path."""

from __future__ import annotations

from ..config import Config
from ..file_transfers import DownloadResult, download_file
from ..sharepoint_files import DriveClient
from . import Handler, Operation
from ._files import DriveId, ItemId, call_files, host_path

DESCRIPTION = (
    "Read-only on SharePoint. Save one file's bytes to `destination`, an absolute "
    "path on the jake-tools host that must NOT already exist (a local file is never "
    "overwritten). Returns the item and the exact version that was read (`etag`, "
    "`ctag`, modified, modified_by), the saved size and sha256, and what was "
    "compared. The save is accepted only if the item's etag did not change while it "
    "was read and the byte count equals the item's size; otherwise nothing is "
    "saved. At most 50 MiB. The file is transported unchanged, never converted or "
    "edited. Replacing existing files is not supported. "
    "Uses the server's configured Microsoft account; never another."
)


def build(config: Config) -> Handler:
    async def sharepoint_download(
        drive_id: DriveId, item_id: ItemId, destination: str
    ) -> DownloadResult:
        target = host_path(destination, "destination")

        def run(client: DriveClient) -> DownloadResult:
            return download_file(
                client, drive_id=drive_id, item_id=item_id, destination=target
            )

        return await call_files(config, run)

    return sharepoint_download


OPERATION = Operation(
    name="sharepoint_download",
    skill="sharepoint-files",
    description=DESCRIPTION,
    build=build,
    omit_none=True,
)
