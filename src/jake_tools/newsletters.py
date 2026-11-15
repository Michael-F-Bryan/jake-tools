from __future__ import annotations

import html
import json
import re
import subprocess
import time
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .http import HttpSession, UpstreamError

CSU_TENANT_ID = "0a3a5574-cfda-4314-952e-c0b3e1dcac6d"
CSU_SITE_HOST = "csuses.sharepoint.com"
CSU_SITE_PATH = "/sites/leadershipteam"
CSU_SITE_ID = "csuses.sharepoint.com,06fe37df-b7a9-4035-bd7e-b0a93a2c08f9,f30de658-f81a-4bf9-a5f0-59ba192d60d7"
CSU_WEEKLY_NEWSLETTER_LIST_ID = "b07a8e3d-fa7e-4191-8733-549140a576db"
GRAPH_ROOT = "https://graph.microsoft.com/v1.0"
SHAREPOINT_ROOT = f"https://{CSU_SITE_HOST}{CSU_SITE_PATH}/_api"
GRAPH_RESOURCE = "https://graph.microsoft.com"
SHAREPOINT_RESOURCE = f"https://{CSU_SITE_HOST}"

JsonObject = dict[str, object]


@dataclass(frozen=True)
class NewsletterAttachment:
    path: Path

    @property
    def filename(self) -> str:
        return self.path.name

    def read_bytes(self) -> bytes:
        try:
            return self.path.read_bytes()
        except OSError as exc:
            raise NewsletterError(
                f"unable to read attachment {self.path}: {exc}"
            ) from exc


class NewsletterItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    body: str
    created: str
    modified: str
    url: str


class NewsletterError(UpstreamError):
    """A newsletter request failed.

    ``status`` is the HTTP status when the upstream answered. For a write,
    :attr:`rejected` says whether the upstream definitely did not apply it
    (a 4xx other than 408); anything else (a transport failure, a timeout, a
    5xx, an unreadable 2xx) leaves the effect uncertain.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        safe_summary: str | None = None,
    ) -> None:
        super().__init__(message, safe_summary=safe_summary)
        self.status = status

    @property
    def rejected(self) -> bool:
        return (
            self.status is not None and 400 <= self.status < 500 and self.status != 408
        )


class NewsletterAttachmentRef(BaseModel):
    """An attachment as SharePoint lists it. Size and digest are not listed."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    name: str = Field(alias="FileName")
    server_relative_url: str = Field(default="", alias="ServerRelativeUrl")


class SharePointAttachmentsResponse(BaseModel):
    value: list[NewsletterAttachmentRef] = Field(default_factory=list)


class NewsletterPage(BaseModel):
    """One page of the list. ``next_cursor`` is ``None`` only at the end."""

    model_config = ConfigDict(frozen=True)

    items: list[NewsletterItem]
    next_cursor: str | None = None


class GraphListItemFields(BaseModel):
    id: str = ""
    title: str = Field(default="", alias="Title")
    body: str = Field(default="", alias="Body")
    created: str = Field(default="", alias="Created")
    modified: str = Field(default="", alias="Modified")


class GraphListItem(BaseModel):
    id: str = ""
    web_url: str = Field(default="", alias="webUrl")
    fields: GraphListItemFields = Field(default_factory=GraphListItemFields)

    def to_newsletter_item(self) -> NewsletterItem:
        return NewsletterItem(
            id=self.id or self.fields.id,
            title=self.fields.title,
            body=body_from_html(self.fields.body),
            created=self.fields.created,
            modified=self.fields.modified,
            url=self.web_url,
        )


class GraphListItemsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    value: list[GraphListItem] = Field(default_factory=list)
    next_link: str | None = Field(default=None, alias="@odata.nextLink")


class GraphCreatedListItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    web_url: str = Field(default="", alias="webUrl")


class AzureCliTokenProvider:
    """Fetches a Microsoft Graph access token via the Azure CLI.

    This is the interactive command line's identity: whichever account
    ``az`` is signed in as. Long-running servers must not use it (they use
    :class:`~jake_tools.microsoft_auth.MicrosoftTokenProvider`).

    `az account get-access-token` spawns a subprocess (~1s) and one command
    asks for the same resource's token repeatedly, so each token is reused,
    but only until shortly before the expiry the CLI reports. A token without
    a usable expiry is never reused.
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._cache: dict[str, tuple[str, float]] = {}

    def __call__(self, resource: str = GRAPH_RESOURCE) -> str:
        cached = self._cache.get(resource)
        if cached is not None and self._clock() < cached[1]:
            return cached[0]
        token, expires_at = self._fetch(resource)
        if expires_at is not None:
            self._cache[resource] = (token, expires_at - 300)
        return token

    def _fetch(self, resource: str) -> tuple[str, float | None]:
        try:
            output = subprocess.check_output(
                [
                    "az",
                    "account",
                    "get-access-token",
                    "--resource",
                    resource,
                    "-o",
                    "json",
                ],
                text=True,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            login = (
                "az login --tenant "
                f"{CSU_TENANT_ID} --scope https://graph.microsoft.com/.default --use-device-code"
            )
            raise NewsletterError(
                f"unable to get Microsoft Graph token via Azure CLI. Run: {login}\n{exc.stderr}"
            ) from exc
        try:
            data = json.loads(output)
            token = data["accessToken"]
        except (ValueError, KeyError, TypeError) as exc:
            raise NewsletterError("Azure CLI returned an unreadable token") from exc
        if not isinstance(token, str) or not token:
            raise NewsletterError("Azure CLI returned an unreadable token")
        return token, _az_expiry(data)


def _az_expiry(data: Mapping[str, object]) -> float | None:
    """Epoch seconds the CLI says the token expires, or ``None`` if it does not."""
    expires_on = data.get("expires_on")
    if isinstance(expires_on, int | float) and not isinstance(expires_on, bool):
        return float(expires_on)
    legacy = data.get("expiresOn")
    if isinstance(legacy, str):
        try:
            # Older CLIs print the local wall-clock time, without a zone.
            return datetime.fromisoformat(legacy).timestamp()
        except ValueError:
            return None
    return None


class NewsletterClient:
    def __init__(
        self,
        *,
        token_provider: Callable[[str], str] | None = None,
        graph_root: str = GRAPH_ROOT,
        sharepoint_root: str = SHAREPOINT_ROOT,
        site_id: str = CSU_SITE_ID,
        list_id: str = CSU_WEEKLY_NEWSLETTER_LIST_ID,
        session: HttpSession | None = None,
    ) -> None:
        self._token_provider = token_provider or AzureCliTokenProvider()
        self._graph_root = graph_root.rstrip("/")
        self._sharepoint_root = sharepoint_root.rstrip("/")
        self._site_id = site_id
        self._list_id = list_id
        self._session = session or requests.Session()

    def list_items(self, *, limit: int = 10) -> list[NewsletterItem]:
        return self.list_page(limit=limit).items

    def list_page(
        self, *, limit: int = 10, cursor: str | None = None
    ) -> NewsletterPage:
        """At most ``limit`` newest-first items, and where the next page starts.

        ``cursor`` is the Graph ``$skiptoken`` of an earlier page's
        ``next_cursor``. A next link the client cannot turn back into a cursor
        is an error: never a silently shorter list.
        """
        params = {
            "$expand": "fields($select=Title,Body,Created,Modified)",
            "$top": str(limit),
            "$orderby": "createdDateTime desc",
        }
        if cursor is not None:
            params["$skiptoken"] = cursor
        query = urllib.parse.urlencode(params, safe="(),")
        data = self._graph("GET", self._list_path(f"/items?{query}"))
        response = GraphListItemsResponse.model_validate(data)
        return NewsletterPage(
            items=[item.to_newsletter_item() for item in response.value],
            next_cursor=_next_cursor(response.next_link),
        )

    def get_item(self, item_id: str) -> NewsletterItem:
        data = self._graph("GET", self._list_path(f"/items/{item_id}?$expand=fields"))
        return GraphListItem.model_validate(data).to_newsletter_item()

    def create_item(
        self,
        *,
        title: str,
        body: str,
        attachments: Sequence[NewsletterAttachment] = (),
    ) -> NewsletterItem:
        item_id = self.create_entry(title=title, body=body).id
        for attachment in attachments:
            self.add_attachment(item_id, attachment)
        return self.get_item(item_id)

    def create_entry(self, *, title: str, body: str) -> GraphCreatedListItem:
        """POST the entry and nothing else: the one step that creates it."""
        payload = {"fields": {"Title": title, "Body": body_to_html(body)}}
        data = self._graph("POST", self._list_path("/items"), payload)
        try:
            return GraphCreatedListItem.model_validate(data)
        except ValidationError as exc:
            # The POST was sent; its reply gave no entry ID.
            raise NewsletterError(
                "newsletter create returned no entry id",
                status=200,
                safe_summary="POST items: reply had no entry id",
            ) from exc

    def update_item(
        self,
        item_id: str,
        *,
        title: str | None = None,
        body: str | None = None,
        attachments: Sequence[NewsletterAttachment] = (),
    ) -> NewsletterItem:
        self.update_fields(item_id, title=title, body=body)
        for attachment in attachments:
            self.add_attachment(item_id, attachment)
        return self.get_item(item_id)

    def update_fields(
        self, item_id: str, *, title: str | None = None, body: str | None = None
    ) -> None:
        """PATCH only the supplied fields; untouched fields are not sent."""
        fields: dict[str, str] = {}
        if title is not None:
            fields["Title"] = title
        if body is not None:
            fields["Body"] = body_to_html(body)
        if fields:
            self._graph("PATCH", self._list_path(f"/items/{item_id}/fields"), fields)

    def add_attachment(self, item_id: str, attachment: NewsletterAttachment) -> None:
        self.add_attachment_bytes(item_id, attachment.filename, attachment.read_bytes())

    def add_attachment_bytes(self, item_id: str, filename: str, content: bytes) -> None:
        self._sharepoint(
            "POST",
            f"{self._attachments_path(item_id)}/add(FileName='{_quote_name(filename)}')",
            content,
            content_type="application/octet-stream",
        )

    def list_attachments(self, item_id: str) -> list[NewsletterAttachmentRef]:
        data = self._sharepoint("GET", self._attachments_path(item_id))
        return SharePointAttachmentsResponse.model_validate(data).value

    def download_attachment(self, item_id: str, filename: str) -> bytes:
        response = self._send(
            "GET",
            SHAREPOINT_RESOURCE,
            f"{self._sharepoint_root}{self._attachments_path(item_id)}"
            f"('{_quote_name(filename)}')/$value",
            body=None,
            content_type="application/json",
            accept="*/*",
        )
        return bytes(response.content)

    def absolute_url(self, server_relative_url: str) -> str:
        """An attachment's address, on the SharePoint host this client talks to."""
        parts = urllib.parse.urlsplit(self._sharepoint_root)
        return f"{parts.scheme}://{parts.netloc}{server_relative_url}"

    def _attachments_path(self, item_id: str) -> str:
        return f"/web/lists(guid'{self._list_id}')/items({item_id})/AttachmentFiles"

    def _list_path(self, suffix: str) -> str:
        site_id = urllib.parse.quote(self._site_id, safe="")
        return f"/sites/{site_id}/lists/{self._list_id}{suffix}"

    def _graph(
        self, method: str, path: str, payload: Mapping[str, object] | None = None
    ) -> JsonObject:
        return self._request_json(
            method,
            GRAPH_RESOURCE,
            f"{self._graph_root}{path}",
            body=payload,
            content_type="application/json",
        )

    def _sharepoint(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        *,
        content_type: str = "application/json",
    ) -> JsonObject:
        return self._request_json(
            method,
            SHAREPOINT_RESOURCE,
            f"{self._sharepoint_root}{path}",
            body=body,
            content_type=content_type,
        )

    def _request_json(
        self,
        method: str,
        resource: str,
        url: str,
        *,
        body: Mapping[str, object] | bytes | None,
        content_type: str,
    ) -> JsonObject:
        response = self._send(
            method, resource, url, body=body, content_type=content_type
        )
        if response.status_code == 204 or not response.content:
            return {}

        try:
            data = response.json()
        except ValueError as exc:
            body_text = str(response.text)[:500]
            raise NewsletterError(
                f"newsletter returned invalid JSON for {method} {url}: {exc}; "
                f"body={body_text!r}",
                status=response.status_code,
                safe_summary=f"{method} {_safe_path(url)}: unreadable response",
            ) from exc

        if not isinstance(data, Mapping):
            raise NewsletterError(
                f"newsletter returned unexpected payload for {method} {url}: {data!r}",
                status=response.status_code,
                safe_summary=f"{method} {_safe_path(url)}: unexpected payload",
            )
        return cast(JsonObject, data)

    def _send(
        self,
        method: str,
        resource: str,
        url: str,
        *,
        body: Mapping[str, object] | bytes | None,
        content_type: str,
        accept: str = "application/json",
    ) -> Any:
        try:
            response = self._session.request(
                method,
                url,
                headers={
                    "Authorization": f"Bearer {self._token_provider(resource)}",
                    "Accept": accept,
                    "Content-Type": content_type,
                },
                json=body if isinstance(body, Mapping) else None,
                data=body if isinstance(body, bytes) else None,
                timeout=30,
            )
        except requests.RequestException as exc:
            raise NewsletterError(
                f"newsletter request failed: {exc}",
                safe_summary=f"{method} {_safe_path(url)}: {type(exc).__name__}",
            ) from exc

        if response.status_code >= 400:
            error_body = str(response.text)[:500]
            raise NewsletterError(
                f"newsletter request failed for {method} {url}: "
                f"{response.status_code} {response.reason}\n{error_body}",
                status=response.status_code,
                safe_summary=f"{method} {_safe_path(url)}: HTTP {response.status_code}",
            )
        return response


def _safe_path(url: str) -> str:
    """``url``'s path with no host, query or credentials, for safe summaries."""
    return urllib.parse.urlsplit(url).path


def _quote_name(filename: str) -> str:
    return urllib.parse.quote(filename.replace("'", "''"))


def _next_cursor(next_link: str | None) -> str | None:
    if not next_link:
        return None
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(next_link).query)
    token = query.get("$skiptoken")
    if not token or not token[0]:
        raise NewsletterError(
            "newsletter list has more pages but gave no usable continuation token",
            safe_summary="list continuation token missing",
        )
    return token[0]


def body_to_html(body: str) -> str:
    paragraphs = [
        paragraph.strip()
        for paragraph in body.strip().split("\n\n")
        if paragraph.strip()
    ]
    newline = "\n"
    return "".join(
        f"<p>{html.escape(paragraph).replace(newline, '<br>')}</p>"
        for paragraph in paragraphs
    )


def body_from_html(value: str) -> str:
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
    value = re.sub(r"</div>|</p>|</li>|</h[1-6]>", "\n\n", value, flags=re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()
