"""Microsoft Graph drive-item client for SharePoint and Teams files.

One bounded HTTP client over the documented drive-item contracts; the
workflows that decide what a caller can trust live in
:mod:`jake_tools.file_transfers`. Contracts relied on (Microsoft Learn,
Graph v1.0):

* ``GET /shares/{u!<base64url of the link>}/driveItem`` resolves a link
  (``shares-get``);
* ``GET /teams/{team}/channels/{channel}/filesFolder`` is a channel's folder
  (``channel-get-filesfolder``);
* ``GET /drives/{drive}/items/{item}`` and ``.../children`` read an item and
  page its children (``@odata.nextLink``);
* ``GET .../content[?format=pdf]`` answers ``302`` with a pre-authenticated
  ``Location`` that "doesn't require an ``Authorization`` header"
  (``driveitem-get-content``, ``driveitem-get-content-format``);
* ``POST .../createUploadSession`` is the one upload route: it documents
  ``@microsoft.graph.conflictBehavior`` (``fail`` by default, ``rename``), and
  the content is then ``PUT`` to the returned ``uploadUrl`` without an
  ``Authorization`` header (``driveitem-createuploadsession``). Only new files
  are uploaded: replacing an existing item needs a commit-time version
  condition, and none is documented for SharePoint (see ``file_transfers``).

No request is ever retried and redirects are never followed implicitly: the
two that exist (a download's ``Location``) are followed by hand, without the
bearer token. Pre-authenticated URLs are secrets; they, the link being
resolved and response bodies never appear in an error.
"""

from __future__ import annotations

import base64
import re
import urllib.parse
from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .http import HttpSession, UpstreamError
from .microsoft_auth import require_secure_url
from .newsletters import GRAPH_RESOURCE, GRAPH_ROOT

ItemKind = Literal["file", "folder", "other"]
ConflictBehavior = Literal["fail", "rename"]
"""``replace`` is deliberately absent: it would overwrite whatever is there."""

CONTENT_TIMEOUT_SECONDS = (10, 120)
METADATA_TIMEOUT_SECONDS = 30
_SAFE_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_.]{0,60}")


class FilesError(UpstreamError):
    """A Graph request failed.

    ``status`` is the HTTP status when the upstream answered and ``code`` its
    short error code (``nameAlreadyExists``). ``message`` and ``safe_summary``
    are the same host-free, URL-free sentence.
    """

    def __init__(
        self,
        what: str,
        *,
        status: int | None = None,
        code: str | None = None,
        detail: str | None = None,
    ) -> None:
        facts = ", ".join(
            part
            for part in (
                f"HTTP {status}" if status else None,
                code,
                detail,
            )
            if part
        )
        message = f"{what}: {facts or 'request failed'}"
        super().__init__(message, safe_summary=message)
        self.what = what
        self.status = status
        self.code = code

    @property
    def rejected(self) -> bool:
        """True when the upstream definitely did not apply the request (a 4xx
        other than 408). Anything else (no answer, a timeout, a 5xx or an
        unreadable 2xx) leaves a write's effect uncertain."""
        return (
            self.status is not None and 400 <= self.status < 500 and self.status != 408
        )


class TooLargeError(FilesError):
    """The content is longer than the caller allowed for. In a readback this
    is evidence (the remote bytes cannot equal a shorter upload), not an
    outage."""


class RemoteItem(BaseModel):
    """A drive item and the version of it that was seen.

    ``etag`` changes with any change to the item; ``ctag`` only with a change
    to its content.
    """

    model_config = ConfigDict(frozen=True)

    drive_id: str
    item_id: str
    name: str
    kind: ItemKind
    web_url: str = ""
    size_bytes: int | None = None
    etag: str | None = None
    ctag: str | None = None
    modified: str | None = None
    modified_by: str | None = None
    mime_type: str | None = None
    parent_id: str | None = None
    child_count: int | None = None


class ChildrenPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: list[RemoteItem]
    next_cursor: str | None = None


class _Facet(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mime_type: str | None = Field(default=None, alias="mimeType")
    child_count: int | None = Field(default=None, alias="childCount")


class _ParentReference(BaseModel):
    drive_id: str | None = Field(default=None, alias="driveId")
    id: str | None = None


class _User(BaseModel):
    display_name: str | None = Field(default=None, alias="displayName")


class _Identity(BaseModel):
    user: _User | None = None


class _DriveItem(BaseModel):
    """The Graph ``driveItem`` fields used. An item without an ID is not one."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(min_length=1)
    name: str = ""
    web_url: str = Field(default="", alias="webUrl")
    size: int | None = None
    etag: str | None = Field(default=None, alias="eTag")
    ctag: str | None = Field(default=None, alias="cTag")
    modified: str | None = Field(default=None, alias="lastModifiedDateTime")
    modified_by: _Identity | None = Field(default=None, alias="lastModifiedBy")
    parent: _ParentReference | None = Field(default=None, alias="parentReference")
    file: _Facet | None = None
    folder: _Facet | None = None

    def to_remote(self, default_drive_id: str | None) -> RemoteItem | None:
        """``None`` when the reply does not say which drive the item is in."""
        drive_id = (self.parent.drive_id if self.parent else None) or default_drive_id
        if not drive_id:
            return None
        kind: ItemKind = (
            "file" if self.file is not None else "folder" if self.folder else "other"
        )
        facet = self.file or self.folder
        return RemoteItem(
            drive_id=drive_id,
            item_id=self.id,
            name=self.name,
            kind=kind,
            web_url=self.web_url,
            size_bytes=self.size,
            etag=self.etag,
            ctag=self.ctag,
            modified=self.modified,
            modified_by=(
                self.modified_by.user.display_name
                if self.modified_by and self.modified_by.user
                else None
            ),
            mime_type=facet.mime_type if facet else None,
            parent_id=self.parent.id if self.parent else None,
            child_count=self.folder.child_count if self.folder else None,
        )


class _Children(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    value: list[_DriveItem]
    """Required: a reply without the collection is not an empty folder."""
    next_link: str | None = Field(default=None, alias="@odata.nextLink")


class _UploadSession(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    upload_url: str = Field(min_length=1, alias="uploadUrl")


def encode_sharing_url(url: str) -> str:
    """The ``u!`` share token for ``url`` (``shares-get``, "Encoding sharing URLs")."""
    encoded = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
    return f"u!{encoded}"


class DriveClient:
    """The drive-item calls. ``token_provider(resource)`` yields a Graph token."""

    def __init__(
        self,
        *,
        token_provider: Callable[[str], str],
        graph_root: str = GRAPH_ROOT,
        session: HttpSession | None = None,
    ) -> None:
        self._token_provider = token_provider
        self._graph_root = graph_root.rstrip("/")
        self._session = session or requests.Session()

    # -- locating -------------------------------------------------------

    def resolve_link(self, url: str) -> RemoteItem:
        path = f"/shares/{encode_sharing_url(url)}/driveItem"
        return self._item(self._get_json("resolve link", path), None, "resolve link")

    def channel_folder(self, team_id: str, channel_id: str) -> RemoteItem:
        path = f"/teams/{_q(team_id)}/channels/{_q(channel_id)}/filesFolder"
        return self._item(
            self._get_json("get channel folder", path), None, "get channel folder"
        )

    def get_item(self, drive_id: str, item_id: str) -> RemoteItem:
        data = self._get_json("get item", self._item_path(drive_id, item_id))
        return self._item(data, drive_id, "get item")

    def item_by_name(
        self, drive_id: str, parent_id: str, name: str
    ) -> RemoteItem | None:
        """The child ``name`` of ``parent_id``, or ``None`` if there is none."""
        path = f"{self._item_path(drive_id, parent_id)}:/{_q(name)}"
        try:
            data = self._get_json("get item by name", path)
        except FilesError as exc:
            if exc.status == 404:
                return None
            raise
        return self._item(data, drive_id, "get item by name")

    def list_children(
        self, drive_id: str, item_id: str, *, limit: int, cursor: str | None = None
    ) -> ChildrenPage:
        """At most ``limit`` children, and where the next page starts.

        ``cursor`` is the ``$skiptoken`` of an earlier page's ``next_cursor``.
        A next link the client cannot turn back into a cursor is an error,
        never a silently shorter folder.
        """
        params = {"$top": str(limit)}
        if cursor is not None:
            params["$skiptoken"] = cursor
        query = urllib.parse.urlencode(params, safe="$")
        data = self._get_json(
            "list folder", f"{self._item_path(drive_id, item_id)}/children?{query}"
        )
        response = _parse(_Children, data, "list folder")
        items = [_remote(child, drive_id, "list folder") for child in response.value]
        return ChildrenPage(items=items, next_cursor=_next_cursor(response.next_link))

    # -- content --------------------------------------------------------

    def download(
        self,
        drive_id: str,
        item_id: str,
        *,
        max_bytes: int,
        format: str | None = None,
    ) -> bytes:
        """The item's bytes (or its ``format`` rendering), at most ``max_bytes``.

        The ``/content`` answer is a ``302``; its ``Location`` is fetched by
        hand and without the bearer token.
        """
        what = f"render {format}" if format else "download"
        path = f"{self._item_path(drive_id, item_id)}/content"
        if format is not None:
            path += f"?format={urllib.parse.quote(format, safe='')}"
        redirect = self._send(
            "GET", self._graph_root + path, what=what, redirect_ok=True
        )
        location = redirect.headers.get("Location")
        if not 300 <= redirect.status_code < 400 or not location:
            raise FilesError(
                what,
                status=redirect.status_code,
                detail="expected a redirect to the content",
            )
        try:
            target = require_secure_url(location, "download location")
        except ValueError as exc:
            raise FilesError(what, detail="the content location is not https") from exc
        response = self._send(
            "GET",
            target,
            what=what,
            auth=False,
            stream=True,
            headers={"Accept": "*/*"},
            timeout=CONTENT_TIMEOUT_SECONDS,
        )
        try:
            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise TooLargeError(what, detail=f"content exceeds {max_bytes} bytes")
            chunks: list[bytes] = []
            total = 0
            try:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise TooLargeError(
                            what, detail=f"content exceeds {max_bytes} bytes"
                        )
                    chunks.append(chunk)
            except requests.RequestException as exc:
                raise FilesError(what, detail=type(exc).__name__) from exc
            return b"".join(chunks)
        finally:
            response.close()

    def create_upload_session(
        self,
        drive_id: str,
        *,
        parent_id: str,
        name: str,
        conflict: ConflictBehavior,
    ) -> str:
        """Open an upload session for a NEW file and return its ``uploadUrl``
        (a secret). Nothing is written until content is uploaded."""
        path = (
            f"{self._item_path(drive_id, parent_id)}:/{_q(name)}:/createUploadSession"
        )
        body = {"item": {"@microsoft.graph.conflictBehavior": conflict, "name": name}}
        response = self._send(
            "POST",
            self._graph_root + path,
            what="create upload session",
            json=body,
            headers={"Content-Type": "application/json"},
        )
        session = _parse(
            _UploadSession,
            _json_object(response, "create upload session"),
            "create upload session",
        )
        try:
            return require_secure_url(session.upload_url, "upload location")
        except ValueError as exc:
            raise FilesError(
                "create upload session",
                status=response.status_code,
                detail="the upload location is not https",
            ) from exc

    def upload_content(
        self, upload_url: str, content: bytes, *, drive_id: str
    ) -> RemoteItem:
        """``PUT`` the whole file as one fragment; ``200``/``201`` is committed.

        The upload URL is pre-authenticated, so no bearer token is sent.
        """
        response = self._send(
            "PUT",
            upload_url,
            what="upload content",
            auth=False,
            data=content,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Range": f"bytes 0-{len(content) - 1}/{len(content)}",
            },
            timeout=CONTENT_TIMEOUT_SECONDS,
        )
        if response.status_code not in (200, 201):
            # Anything else (a 202 "more expected") is not a committed file.
            raise FilesError(
                "upload content",
                status=response.status_code,
                detail="the upload was not reported complete",
            )
        data = _json_object(response, "upload content")
        return self._item(data, drive_id, "upload content")

    # -- plumbing -------------------------------------------------------

    def _item_path(self, drive_id: str, item_id: str) -> str:
        return f"/drives/{_q(drive_id)}/items/{_q(item_id)}"

    def _get_json(self, what: str, path: str) -> dict[str, Any]:
        response = self._send("GET", self._graph_root + path, what=what)
        return _json_object(response, what)

    def _item(
        self, data: dict[str, Any], default_drive_id: str | None, what: str
    ) -> RemoteItem:
        return _remote(_parse(_DriveItem, data, what), default_drive_id, what)

    def _send(
        self,
        method: str,
        url: str,
        *,
        what: str,
        auth: bool = True,
        headers: Mapping[str, str] | None = None,
        json: Mapping[str, object] | None = None,
        data: bytes | None = None,
        stream: bool = False,
        redirect_ok: bool = False,
        timeout: float | tuple[float, float] = METADATA_TIMEOUT_SECONDS,
    ) -> Any:
        sent = {"Accept": "application/json", **(headers or {})}
        if auth:
            sent["Authorization"] = f"Bearer {self._token_provider(GRAPH_RESOURCE)}"
        try:
            response = self._session.request(
                method,
                url,
                headers=sent,
                json=json,
                data=data,
                stream=stream,
                timeout=timeout,
                # A redirect would re-send the request (and its bearer token or
                # content) somewhere else on its own: the callers that expect
                # one read its Location themselves.
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            # The exception text carries the URL, which can be a secret.
            raise FilesError(what, detail=type(exc).__name__) from exc
        redirected = 300 <= response.status_code < 400
        if response.status_code >= 400 or (redirected and not redirect_ok):
            code = _error_code(response)
            response.close()
            raise FilesError(what, status=response.status_code, code=code)
        return response


def _q(value: str) -> str:
    """A path segment: ``!``, ``:``, ``@`` and ``=`` are legal in IDs, ``/`` is not."""
    return urllib.parse.quote(value, safe="!:@=")


def _error_code(response: Any) -> str | None:
    try:
        error = response.json().get("error")
        code = error.get("code") if isinstance(error, dict) else None
    except ValueError, AttributeError:
        return None
    return code if isinstance(code, str) and _SAFE_CODE.fullmatch(code) else None


def _json_object(response: Any, what: str) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise FilesError(
            what, status=response.status_code, detail="unreadable response"
        ) from exc
    if not isinstance(data, dict):
        raise FilesError(
            what, status=response.status_code, detail="unexpected response"
        )
    return cast(dict[str, Any], data)


def _parse[ModelT: BaseModel](
    model: type[ModelT], data: dict[str, Any], what: str
) -> ModelT:
    """A wrong shape is an upstream failure that carries no payload text."""
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise FilesError(what, status=200, detail="unexpected response shape") from exc


def _remote(item: _DriveItem, default_drive_id: str | None, what: str) -> RemoteItem:
    remote = item.to_remote(default_drive_id)
    if remote is None:
        raise FilesError(what, status=200, detail="the reply does not name the drive")
    return remote


def _next_cursor(next_link: str | None) -> str | None:
    if not next_link:
        return None
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(next_link).query)
    token = query.get("$skiptoken")
    if not token or not token[0]:
        raise FilesError(
            "list folder", detail="more pages exist but no continuation token was given"
        )
    return token[0]
