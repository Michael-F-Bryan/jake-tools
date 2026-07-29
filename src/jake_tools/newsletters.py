from __future__ import annotations

import html
import re
import subprocess
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import requests
from pydantic import BaseModel, ConfigDict, Field

from .http import HttpSession

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


class NewsletterError(RuntimeError):
    pass


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
    value: list[GraphListItem] = Field(default_factory=list)


class GraphCreatedListItem(BaseModel):
    id: str


class AzureCliTokenProvider:
    """Fetches a Microsoft Graph access token via the Azure CLI.

    `az account get-access-token` spawns a subprocess (~1s). A single HTTP
    call site can request the same resource's token repeatedly within one
    process run (e.g. create_item with attachments calls _graph and
    _sharepoint several times), so cache each resource's token for the
    process lifetime instead of re-shelling out every call.
    """

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    def __call__(self, resource: str = GRAPH_RESOURCE) -> str:
        cached = self._cache.get(resource)
        if cached is not None:
            return cached
        token = self._fetch(resource)
        self._cache[resource] = token
        return token

    def _fetch(self, resource: str) -> str:
        try:
            return subprocess.check_output(
                [
                    "az",
                    "account",
                    "get-access-token",
                    "--resource",
                    resource,
                    "--query",
                    "accessToken",
                    "-o",
                    "tsv",
                ],
                text=True,
                stderr=subprocess.PIPE,
            ).strip()
        except subprocess.CalledProcessError as exc:
            login = (
                "az login --tenant "
                f"{CSU_TENANT_ID} --scope https://graph.microsoft.com/.default --use-device-code"
            )
            raise NewsletterError(
                f"unable to get Microsoft Graph token via Azure CLI. Run: {login}\n{exc.stderr}"
            ) from exc


class NewsletterOperations(Protocol):
    """The subset of :class:`NewsletterClient` the CLI depends on.

    Lets the CLI's typed context inject a fake without subclassing the real
    Graph-backed client.
    """

    def list_items(self, *, limit: int) -> list[NewsletterItem]: ...

    def create_item(
        self,
        *,
        title: str,
        body: str,
        attachments: Sequence[NewsletterAttachment],
    ) -> NewsletterItem: ...

    def update_item(
        self,
        item_id: str,
        *,
        title: str | None,
        body: str | None,
        attachments: Sequence[NewsletterAttachment],
    ) -> NewsletterItem: ...


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
        query = urllib.parse.urlencode(
            {
                "$expand": "fields($select=Title,Body,Created,Modified)",
                "$top": str(limit),
                "$orderby": "createdDateTime desc",
            },
            safe="(),",
        )
        data = self._graph("GET", self._list_path(f"/items?{query}"))
        response = GraphListItemsResponse.model_validate(data)
        return [item.to_newsletter_item() for item in response.value]

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
        payload = {"fields": {"Title": title, "Body": body_to_html(body)}}
        data = self._graph("POST", self._list_path("/items"), payload)
        item_id = GraphCreatedListItem.model_validate(data).id
        for attachment in attachments:
            self.add_attachment(item_id, attachment)
        return self.get_item(item_id)

    def update_item(
        self,
        item_id: str,
        *,
        title: str | None = None,
        body: str | None = None,
        attachments: Sequence[NewsletterAttachment] = (),
    ) -> NewsletterItem:
        fields: dict[str, str] = {}
        if title is not None:
            fields["Title"] = title
        if body is not None:
            fields["Body"] = body_to_html(body)

        if fields:
            self._graph("PATCH", self._list_path(f"/items/{item_id}/fields"), fields)

        for attachment in attachments:
            self.add_attachment(item_id, attachment)

        return self.get_item(item_id)

    def add_attachment(self, item_id: str, attachment: NewsletterAttachment) -> None:
        filename = attachment.filename.replace("'", "''")
        path = (
            f"/web/lists(guid'{self._list_id}')/items({item_id})"
            f"/AttachmentFiles/add(FileName='{urllib.parse.quote(filename)}')"
        )
        self._sharepoint(
            "POST",
            path,
            attachment.read_bytes(),
            content_type="application/octet-stream",
        )

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
        self, method: str, path: str, body: bytes, *, content_type: str
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
        try:
            response = self._session.request(
                method,
                url,
                headers={
                    "Authorization": f"Bearer {self._token_provider(resource)}",
                    "Accept": "application/json",
                    "Content-Type": content_type,
                },
                json=body if isinstance(body, Mapping) else None,
                data=body if isinstance(body, bytes) else None,
                timeout=30,
            )
        except requests.RequestException as exc:
            raise NewsletterError(f"newsletter request failed: {exc}") from exc

        if response.status_code >= 400:
            error_body = str(response.text)[:500]
            raise NewsletterError(
                f"newsletter request failed for {method} {url}: "
                f"{response.status_code} {response.reason}\n{error_body}"
            )

        if response.status_code == 204 or not response.content:
            return {}

        try:
            data = response.json()
        except ValueError as exc:
            body_text = str(response.text)[:500]
            raise NewsletterError(
                f"newsletter returned invalid JSON for {method} {url}: {exc}; "
                f"body={body_text!r}"
            ) from exc

        if not isinstance(data, Mapping):
            raise NewsletterError(
                f"newsletter returned unexpected payload for {method} {url}: {data!r}"
            )
        return cast(JsonObject, data)


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
