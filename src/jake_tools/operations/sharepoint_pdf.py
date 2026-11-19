"""sharepoint_pdf: save Microsoft's PDF rendering of a file to a new local path."""

from __future__ import annotations

from ..config import Config
from ..file_transfers import DownloadResult, download_pdf
from ..sharepoint_files import DriveClient
from . import Handler, Operation
from ._files import DriveId, ItemId, call_files, host_path

DESCRIPTION = (
    "Read-only on SharePoint. Save Microsoft's own PDF rendering of a file (Graph "
    "`content?format=pdf`) to `destination`, an absolute path on the jake-tools host "
    "that must NOT already exist. Only the source types Microsoft documents for PDF "
    "conversion are attempted (Office documents, OpenDocument, RTF, HTML/Markdown, "
    "email, EPUB, TIFF and a few others); any other file is refused with "
    "reason=unsupported_conversion and the supported list, before any content is "
    "requested, and a rendering Microsoft refuses (its documented limit is a 2 MB "
    "source) is reported the same way. The result is checked to be a PDF and the "
    "source's etag to be unchanged while it was rendered; the rendering is not "
    "compared with the original, and fidelity (layout, fonts, tracked changes) is "
    "Microsoft's and was not checked. Uses the server's configured Microsoft "
    "account; never another."
)


def build(config: Config) -> Handler:
    async def sharepoint_pdf(
        drive_id: DriveId, item_id: ItemId, destination: str
    ) -> DownloadResult:
        target = host_path(destination, "destination")

        def run(client: DriveClient) -> DownloadResult:
            return download_pdf(
                client, drive_id=drive_id, item_id=item_id, destination=target
            )

        return await call_files(config, run)

    return sharepoint_pdf


OPERATION = Operation(
    name="sharepoint_pdf",
    skill="sharepoint-files",
    description=DESCRIPTION,
    build=build,
    omit_none=True,
)
