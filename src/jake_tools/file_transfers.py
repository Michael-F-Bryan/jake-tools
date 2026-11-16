"""SharePoint/Teams file transport: resolve, list, download, upload, replace.

:class:`~jake_tools.sharepoint_files.DriveClient` is the HTTP; this module owns
what a caller can trust:

* **Coverage.** A folder listing is one bounded page plus the cursor to the
  next, never a silently truncated folder.
* **Identity.** Every item carries its drive, item ID, URL and ``etag``/``ctag``
  version. A download is accepted only if the item's version did not change
  while it was read and the bytes are the size the item reported.
* **Verification.** A write is accepted only after a separate readback of the
  exact item: the bytes are downloaded and compared with what was sent.
* **No silent overwrite.** A new file chooses ``fail`` or ``rename`` on a name
  clash, never ``replace``. Replacing needs the ``etag`` the caller last saw: a
  different current version is refused before anything is sent, and the same
  ``etag`` is sent as ``If-Match`` when the upload session is created
  (documented to answer ``412`` if stale). That check is made once, at session
  creation; the upstream documents no check at the final commit, so the window
  between them is not closed. The readback proves what is there, not that
  nobody else wrote in that window.
* **Honest outcomes.** :class:`FileTransferError` says whether nothing was
  written (``not_applied``), the write may or may not have happened
  (``write_uncertain``), or it happened and the readback disagrees
  (``verification_failed``) or could not be made (``verification_unavailable``).
  Nothing is retried.

This transports files made elsewhere; it never edits their content.
"""

from __future__ import annotations

import contextlib
import hashlib
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from .http import UpstreamError
from .sharepoint_files import (
    ConflictBehavior,
    DriveClient,
    FilesError,
    RemoteItem,
    TooLargeError,
)

MAX_TRANSFER_BYTES = 50 * 1024 * 1024
"""Jake's bound for one upload or download: one request, under the upload
route's documented 60 MiB per-request ceiling, keeping a call's memory known."""
MAX_NAME_CHARS = 255
FORBIDDEN_NAME_CHARACTERS = frozenset('"*:<>?/\\|')
"""The characters OneDrive and SharePoint do not allow in a file name
(support.microsoft.com, "Restrictions and limitations in OneDrive and SharePoint")."""

PDF_SOURCE_EXTENSIONS = frozenset(
    [
        "doc",
        "docx",
        "dot",
        "dotx",
        "dotm",
        "dsn",
        "dwg",
        "eml",
        "epub",
        "fluidframework",
        "form",
        "htm",
        "html",
        "loop",
        "loot",
        "markdown",
        "md",
        "msg",
        "note",
        "odp",
        "ods",
        "odt",
        "page",
        "pps",
        "ppsx",
        "ppt",
        "pptx",
        "pulse",
        "rtf",
        "task",
        "tif",
        "tiff",
        "wbtx",
        "whiteboard",
        "xls",
        "xlsm",
        "xlsx",
    ]
)
"""Verbatim from ``driveitem-get-content-format`` for ``format=pdf``. The same
page notes a 2 MB limit for conversion, which is reported if the upstream
refuses, not enforced here."""

WriteOutcome = Literal[
    "not_applied",
    "write_uncertain",
    "verification_failed",
    "verification_unavailable",
]
WriteReason = Literal[
    "name_exists", "version_conflict", "locked", "rejected", "no_upload_location"
]


class FileInputError(ValueError):
    """The request cannot be carried out; nothing was changed or sent."""

    def __init__(
        self,
        argument: str,
        message: str,
        *,
        reason: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.argument = argument
        self.reason = reason
        self.extra = detail or {}


class VersionConflictError(RuntimeError):
    """The item is not at the version the caller expected. Nothing was written."""

    def __init__(self, message: str, *, expected: str, current: RemoteItem) -> None:
        super().__init__(message)
        self.expected = expected
        self.current = current


class FileReadError(RuntimeError):
    """A read could not be trusted (the item changed under it, wrong size,
    not a PDF) or was refused by the upstream. Nothing was changed."""

    def __init__(self, reason: str, message: str, *, item: RemoteItem | None) -> None:
        super().__init__(message)
        self.reason = reason
        self.item = item


class Observation(BaseModel):
    """What an immediate look at the target showed after an uncertain write."""

    state: Literal["found_identical", "found_different", "not_found", "unreadable"]
    item: RemoteItem | None = None


class FileTransferError(RuntimeError):
    """A write did not complete or could not be confirmed."""

    def __init__(
        self,
        outcome: WriteOutcome,
        message: str,
        *,
        reason: WriteReason | None = None,
        drive_id: str,
        item: RemoteItem | None = None,
        name: str | None = None,
        mismatches: list[str] | None = None,
        observed: Observation | None = None,
        changed_since_write: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.outcome: WriteOutcome = outcome
        self.reason = reason
        self.drive_id = drive_id
        self.item = item
        self.name = name
        self.mismatches = mismatches or []
        self.observed = observed
        self.changed_since_write = changed_since_write

    @property
    def retry_safe(self) -> bool:
        """True only when the upstream definitely wrote nothing."""
        return self.outcome == "not_applied"

    @property
    def next_step(self) -> str:
        match self.outcome:
            case "not_applied":
                if self.reason == "name_exists":
                    return (
                        "Nothing was written: a file with that name already exists. "
                        "Pick another name, use on_conflict=rename, or read the "
                        "existing file and replace it with sharepoint_replace."
                    )
                if self.reason == "version_conflict":
                    return (
                        "Nothing was written: the file is no longer at the version "
                        "you expected, so someone else changed it. Download the "
                        "current version, reconcile their edits outside this tool, "
                        "then call sharepoint_replace with the new etag."
                    )
                if self.reason == "locked":
                    return (
                        "Nothing was written: the file is locked (open for editing "
                        "or checked out). Ask the person, then try again."
                    )
                return "Nothing was written. Fix the cause and call again."
            case "write_uncertain":
                return (
                    "The upload may or may not have been applied. Do NOT repeat it "
                    "blindly: call sharepoint_get on the item (or sharepoint_list on "
                    "the folder), compare, and tell the user. `observed` is what an "
                    "immediate check showed."
                )
            case "verification_failed":
                return (
                    "The upload was acknowledged but the readback differs (see "
                    "mismatches). Inspect the item with sharepoint_get or "
                    "sharepoint_download before changing it again; do not repeat "
                    "the write blindly."
                )
            case "verification_unavailable":
                return (
                    "The upload was acknowledged but could not be read back. Check "
                    "it with sharepoint_get or sharepoint_download before reporting "
                    "it done."
                )

    def detail(self) -> dict[str, object]:
        detail: dict[str, object] = {
            "outcome": self.outcome,
            "drive_id": self.drive_id,
            "retry_safe": self.retry_safe,
            "next_step": self.next_step,
        }
        if self.reason:
            detail["reason"] = self.reason
        if self.name:
            detail["name"] = self.name
        if self.item:
            detail["item"] = self.item.model_dump(mode="json", exclude_none=True)
        if self.mismatches:
            detail["mismatches"] = self.mismatches
        if self.observed:
            detail["observed"] = self.observed.model_dump(
                mode="json", exclude_none=True
            )
        if self.changed_since_write is not None:
            detail["changed_since_write"] = self.changed_since_write
        return detail


# -- results ---------------------------------------------------------------


class ResolveResult(BaseModel):
    item: RemoteItem
    resolved_by: Literal["link", "channel"]


class FolderListing(BaseModel):
    """One page of a folder. Order is the service's, not promised.

    ``complete`` is true only when this page reaches the end of the folder.
    When false, call again with ``next_cursor``, the same folder and the same
    ``limit``. Paging is not a snapshot.
    """

    drive_id: str
    folder_id: str
    entries: list[RemoteItem]
    count: int
    limit: int
    next_cursor: str | None = None
    complete: bool


class DownloadResult(BaseModel):
    """A file saved locally. ``item`` is the version that was read."""

    item: RemoteItem
    path: str
    size_bytes: int
    sha256: str
    compared: list[str]
    format: Literal["original", "pdf"] = "original"


class UploadResult(BaseModel):
    """A write the readback confirmed. ``item`` is the item as read back."""

    operation: Literal["create", "replace"]
    verified: Literal[True] = True
    item: RemoteItem
    requested_name: str | None = None
    renamed: bool = False
    expected_etag: str | None = None
    etag_at_write: str | None = None
    changed_since_write: bool
    """True when the item's ``etag`` at readback is not the one the upload
    reported, i.e. something touched it after this write (the bytes still match)."""
    size_bytes: int
    sha256: str
    compared: list[str]


# -- locating ----------------------------------------------------------------


def check_link(url: str, hosts: tuple[str, ...]) -> str:
    parts = urllib.parse.urlsplit(url.strip())
    allowed = {host.lower() for host in hosts}
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username
        or parts.password
    ):
        raise FileInputError("url", "url must be an https SharePoint link")
    if parts.hostname.lower() not in allowed:
        raise FileInputError(
            "url",
            "that link is not on a SharePoint host this server is configured for",
            reason="host_not_allowed",
            detail={"allowed_hosts": sorted(allowed)},
        )
    return url.strip()


def resolve_link(
    client: DriveClient, url: str, *, hosts: tuple[str, ...]
) -> ResolveResult:
    return ResolveResult(
        item=client.resolve_link(check_link(url, hosts)), resolved_by="link"
    )


def resolve_channel(
    client: DriveClient, *, team_id: str, channel_id: str
) -> ResolveResult:
    return ResolveResult(
        item=client.channel_folder(team_id, channel_id), resolved_by="channel"
    )


def get_item(client: DriveClient, *, drive_id: str, item_id: str) -> RemoteItem:
    return client.get_item(drive_id, item_id)


def list_folder(
    client: DriveClient,
    *,
    drive_id: str,
    item_id: str,
    limit: int,
    cursor: str | None,
) -> FolderListing:
    page = client.list_children(drive_id, item_id, limit=limit, cursor=cursor)
    return FolderListing(
        drive_id=drive_id,
        folder_id=item_id,
        entries=page.items,
        count=len(page.items),
        limit=limit,
        next_cursor=page.next_cursor,
        complete=page.next_cursor is None,
    )


# -- reading -----------------------------------------------------------------


def download_file(
    client: DriveClient, *, drive_id: str, item_id: str, destination: Path
) -> DownloadResult:
    return _read_to(client, drive_id, item_id, destination, pdf=False)


def download_pdf(
    client: DriveClient, *, drive_id: str, item_id: str, destination: Path
) -> DownloadResult:
    return _read_to(client, drive_id, item_id, destination, pdf=True)


def _read_to(
    client: DriveClient,
    drive_id: str,
    item_id: str,
    destination: Path,
    *,
    pdf: bool,
) -> DownloadResult:
    check_destination(destination)
    before = client.get_item(drive_id, item_id)
    if before.kind != "file":
        raise FileInputError(
            "item_id",
            f"{before.name!r} is a {before.kind}, not a file; use sharepoint_list "
            "for a folder",
            reason="not_a_file",
        )
    if pdf:
        extension = before.name.rsplit(".", 1)[-1].lower() if "." in before.name else ""
        if extension not in PDF_SOURCE_EXTENSIONS:
            raise FileInputError(
                "item_id",
                f"{before.name!r} cannot be rendered as PDF: "
                f"{'.' + extension if extension else 'it has no extension'} is not a "
                "source type Microsoft documents for format=pdf",
                reason="unsupported_conversion",
                detail={"supported_extensions": sorted(PDF_SOURCE_EXTENSIONS)},
            )
    if before.size_bytes is not None and before.size_bytes > MAX_TRANSFER_BYTES:
        raise FileInputError(
            "item_id",
            f"{before.name!r} is {before.size_bytes} bytes; this tool transfers at "
            f"most {MAX_TRANSFER_BYTES // (1024 * 1024)} MiB",
            reason="too_large",
        )
    try:
        content = client.download(
            drive_id,
            item_id,
            max_bytes=MAX_TRANSFER_BYTES,
            format="pdf" if pdf else None,
        )
    except FilesError as exc:
        if pdf and exc.status in (400, 406, 415, 422, 501):
            raise FileInputError(
                "item_id",
                f"Microsoft refused to render {before.name!r} as PDF ({exc}). Its "
                "documented limits include a 2 MB source size; render it in "
                "SharePoint or Word instead.",
                reason="unsupported_conversion",
                detail={"http_status": exc.status},
            ) from exc
        raise
    after = client.get_item(drive_id, item_id)
    compared = [f"etag unchanged ({before.etag})"]
    if after.etag != before.etag:
        raise FileReadError(
            "changed_during_download",
            f"{before.name!r} changed while it was being read (etag "
            f"{before.etag} -> {after.etag}); nothing was saved. Read it again.",
            item=after,
        )
    if pdf:
        if not content.startswith(b"%PDF-"):
            raise FileReadError(
                "not_a_pdf",
                "Microsoft's rendering of the file is not a PDF; nothing was saved.",
                item=before,
            )
        compared.append("starts with %PDF-")
    else:
        if before.size_bytes is not None and len(content) != before.size_bytes:
            raise FileReadError(
                "size_mismatch",
                f"{before.name!r} is {before.size_bytes} bytes but {len(content)} "
                "were received; nothing was saved.",
                item=before,
            )
        compared.append(f"size equals the item's size ({len(content)} bytes)")
    save_new(destination, content)
    return DownloadResult(
        item=before,
        path=str(destination),
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        compared=compared,
        format="pdf" if pdf else "original",
    )


def check_destination(destination: Path) -> None:
    if not destination.is_absolute():
        raise FileInputError(
            "destination", "destination must be an absolute path on the jake-tools host"
        )
    if destination.exists() or destination.is_symlink():
        raise FileInputError(
            "destination",
            f"{destination} already exists; downloads never overwrite a local file",
            reason="destination_exists",
        )
    if not destination.parent.is_dir():
        raise FileInputError(
            "destination", f"the folder {destination.parent} does not exist"
        )


def save_new(destination: Path, content: bytes) -> None:
    try:
        with destination.open("xb") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise FileInputError(
            "destination",
            f"{destination} already exists; downloads never overwrite a local file",
            reason="destination_exists",
        ) from exc
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise FileInputError(
            "destination", f"cannot write {destination}: {exc.strerror}"
        ) from exc


# -- writing -----------------------------------------------------------------


def read_source(path: Path) -> bytes:
    if not path.is_absolute():
        raise FileInputError(
            "source_path", "source_path must be an absolute path on the jake-tools host"
        )
    if not path.is_file():
        raise FileInputError(
            "source_path", f"{path} is not a file readable by this process"
        )
    size = path.stat().st_size
    if size == 0:
        raise FileInputError(
            "source_path", f"{path.name!r} is empty; nothing to upload"
        )
    if size > MAX_TRANSFER_BYTES:
        raise FileInputError(
            "source_path",
            f"{path.name!r} exceeds {MAX_TRANSFER_BYTES // (1024 * 1024)} MiB, the "
            "most this tool uploads",
            reason="too_large",
        )
    try:
        return path.read_bytes()
    except OSError as exc:
        raise FileInputError(
            "source_path", f"{path} cannot be read: {exc.strerror}"
        ) from exc


def check_name(name: str) -> str:
    bad = sorted(set(name) & FORBIDDEN_NAME_CHARACTERS)
    if (
        not name
        or len(name) > MAX_NAME_CHARS
        or name != name.strip()
        or name.endswith(".")
        or name.startswith("~$")
        or bad
    ):
        raise FileInputError(
            "name",
            f"{name!r} is not a usable SharePoint file name"
            + (f" (contains {' '.join(bad)})" if bad else "")
            + "; names are 1..255 characters, with no leading/trailing space or "
            "trailing dot",
        )
    return name


def upload_new(
    client: DriveClient,
    *,
    drive_id: str,
    parent_id: str,
    name: str,
    content: bytes,
    conflict: ConflictBehavior,
) -> UploadResult:
    parent = client.get_item(drive_id, parent_id)
    if parent.kind != "folder":
        raise FileInputError(
            "parent_id",
            f"{parent.name!r} is a {parent.kind}, not a folder",
            reason="not_a_folder",
        )
    written = _write(
        client,
        drive_id=drive_id,
        name=name,
        content=content,
        open_session=lambda: client.create_upload_session(
            drive_id, parent_id=parent_id, name=name, conflict=conflict
        ),
        look=lambda: client.item_by_name(drive_id, parent_id, name),
    )
    return _verify(
        client,
        written,
        operation="create",
        content=content,
        requested_name=name,
        expected_etag=None,
        parent_id=parent_id,
    )


def replace_file(
    client: DriveClient,
    *,
    drive_id: str,
    item_id: str,
    content: bytes,
    expected_etag: str,
) -> UploadResult:
    current = client.get_item(drive_id, item_id)
    if current.kind != "file":
        raise FileInputError(
            "item_id",
            f"{current.name!r} is a {current.kind}; only a file can be replaced",
            reason="not_a_file",
        )
    if current.etag != expected_etag:
        raise VersionConflictError(
            f"{current.name!r} is at etag {current.etag}, not the {expected_etag} "
            "you expected: it was changed since you read it. Nothing was written.",
            expected=expected_etag,
            current=current,
        )
    try:
        written = _write(
            client,
            drive_id=drive_id,
            name=current.name,
            content=content,
            open_session=lambda: client.create_upload_session(
                drive_id, item_id=item_id, if_match=expected_etag
            ),
            look=lambda: client.get_item(drive_id, item_id),
        )
    except FileTransferError as exc:
        if exc.reason == "version_conflict":
            # The upstream refused the If-Match: show who has the file now.
            with contextlib.suppress(UpstreamError):
                exc.item = client.get_item(drive_id, item_id)
        raise
    return _verify(
        client,
        written,
        operation="replace",
        content=content,
        requested_name=None,
        expected_etag=expected_etag,
        parent_id=None,
    )


def _write(
    client: DriveClient,
    *,
    drive_id: str,
    name: str,
    content: bytes,
    open_session: Callable[[], str],
    look: Callable[[], RemoteItem | None],
) -> RemoteItem:
    """Open the session, upload, and classify a failure at each step."""
    try:
        upload_url = open_session()
    except FilesError as exc:
        # No content was sent, so nothing can have been written, whatever the
        # status: a 5xx here is a failed start, not an uncertain write.
        raise FileTransferError(
            "not_applied",
            f"could not start the upload: {exc}",
            reason=_reason(exc, at_session=True),
            drive_id=drive_id,
            name=name,
        ) from exc
    try:
        return client.upload_content(upload_url, content, drive_id=drive_id)
    except FilesError as exc:
        if exc.rejected:
            raise FileTransferError(
                "not_applied",
                f"the upload was rejected: {exc}",
                reason=_reason(exc, at_session=False),
                drive_id=drive_id,
                name=name,
            ) from exc
        raise FileTransferError(
            "write_uncertain",
            f"the upload has an unknown outcome: {exc}",
            drive_id=drive_id,
            name=name,
            observed=_observe(look, content, client, drive_id),
        ) from exc


def _reason(exc: FilesError, *, at_session: bool) -> WriteReason | None:
    if exc.status == 412:
        return "version_conflict"
    if exc.status == 409 or exc.code == "nameAlreadyExists":
        return "name_exists"
    if exc.status == 423 or exc.code == "resourceLocked":
        return "locked"
    return "rejected" if exc.rejected else None


def _observe(
    look: Callable[[], RemoteItem | None],
    content: bytes,
    client: DriveClient,
    drive_id: str,
) -> Observation:
    """Best-effort look at the target after an unknown outcome. Never raises
    and changes nothing."""
    try:
        item = look()
        if item is None:
            return Observation(state="not_found")
        if item.kind != "file":
            return Observation(state="found_different", item=item)
        found = client.download(drive_id, item.item_id, max_bytes=len(content) + 1)
    except TooLargeError:
        return Observation(state="found_different", item=item)
    except UpstreamError, OSError:
        return Observation(state="unreadable")
    return Observation(
        state="found_identical" if found == content else "found_different", item=item
    )


def _verify(
    client: DriveClient,
    written: RemoteItem,
    *,
    operation: Literal["create", "replace"],
    content: bytes,
    requested_name: str | None,
    expected_etag: str | None,
    parent_id: str | None,
) -> UploadResult:
    drive_id = written.drive_id
    found: bytes | None
    try:
        item = client.get_item(drive_id, written.item_id)
        try:
            found = client.download(
                drive_id, written.item_id, max_bytes=len(content) + 1
            )
        except TooLargeError:
            found = None  # longer than the upload, so it cannot equal it
    except UpstreamError as exc:
        raise FileTransferError(
            "verification_unavailable",
            f"the upload was acknowledged but could not be read back: {exc}",
            drive_id=drive_id,
            item=written,
        ) from exc
    digest = hashlib.sha256(content).hexdigest()
    compared: list[str] = []
    mismatches: list[str] = []
    _check(
        compared,
        mismatches,
        f"content ({len(content)} bytes, sha256)",
        found == content,
    )
    _check(compared, mismatches, "size", item.size_bytes == len(content))
    _check(compared, mismatches, "is a file", item.kind == "file")
    if parent_id is not None:
        _check(compared, mismatches, "parent folder", item.parent_id == parent_id)
    if requested_name is not None and item.name != requested_name:
        compared.append(f"name differs ({requested_name!r} -> {item.name!r}): renamed")
    changed = item.etag != written.etag
    if mismatches:
        raise FileTransferError(
            "verification_failed",
            "the readback differs from what was uploaded: " + "; ".join(mismatches),
            drive_id=drive_id,
            item=item,
            mismatches=mismatches,
            changed_since_write=changed,
        )
    return UploadResult(
        operation=operation,
        item=item,
        requested_name=requested_name,
        renamed=requested_name is not None and item.name != requested_name,
        expected_etag=expected_etag,
        etag_at_write=written.etag,
        changed_since_write=changed,
        size_bytes=len(content),
        sha256=digest,
        compared=compared,
    )


def _check(compared: list[str], mismatches: list[str], name: str, ok: bool) -> None:
    compared.append(name)
    if not ok:
        mismatches.append(name)
