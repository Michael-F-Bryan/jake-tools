from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import html
import json
import re
import subprocess
from typing import cast
import urllib.error
import urllib.parse
import urllib.request

from pathlib import Path

from pydantic import BaseModel, Field


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
            raise NewsletterError(f"unable to read attachment {self.path}: {exc}") from exc


@dataclass(frozen=True)
class NewsletterItem:
    id: str
    title: str
    body: str
    created: str
    modified: str
    url: str

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "title": self.title,
            "body": self.body,
            "created": self.created,
            "modified": self.modified,
            "url": self.url,
        }


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
    def __call__(self, resource: str = GRAPH_RESOURCE) -> str:
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


class NewsletterClient:
    def __init__(
        self,
        *,
        token_provider: Callable[[str], str] | None = None,
        graph_root: str = GRAPH_ROOT,
        sharepoint_root: str = SHAREPOINT_ROOT,
        site_id: str = CSU_SITE_ID,
        list_id: str = CSU_WEEKLY_NEWSLETTER_LIST_ID,
    ) -> None:
        self._token_provider = token_provider or AzureCliTokenProvider()
        self._graph_root = graph_root.rstrip("/")
        self._sharepoint_root = sharepoint_root.rstrip("/")
        self._site_id = site_id
        self._list_id = list_id

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
        self._sharepoint("POST", path, attachment.read_bytes(), content_type="application/octet-stream")

    def _list_path(self, suffix: str) -> str:
        site_id = urllib.parse.quote(self._site_id, safe="")
        return f"/sites/{site_id}/lists/{self._list_id}{suffix}"

    def _graph(self, method: str, path: str, payload: Mapping[str, object] | None = None) -> JsonObject:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        return self._request_json(
            method,
            GRAPH_RESOURCE,
            f"{self._graph_root}{path}",
            body=body,
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
        body: bytes | None,
        content_type: str,
    ) -> JsonObject:
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self._token_provider(resource)}",
                "Accept": "application/json",
                "Content-Type": content_type,
            },
        )
        try:
            with urllib.request.urlopen(request) as response:
                if response.status == 204:
                    return {}

                raw = response.read()
                if not raw:
                    return {}

                return cast(JsonObject, json.loads(raw))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode(errors="replace")
            raise NewsletterError(f"newsletter request failed: {exc.code} {exc.reason}\n{details}") from exc
        except urllib.error.URLError as exc:
            raise NewsletterError(f"newsletter request failed: {exc.reason}") from exc

def body_to_html(body: str) -> str:
    paragraphs = [paragraph.strip() for paragraph in body.strip().split("\n\n") if paragraph.strip()]
    return "".join(f"<p>{html.escape(paragraph).replace('\n', '<br>')}</p>" for paragraph in paragraphs)


def body_from_html(value: str) -> str:
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
    value = re.sub(r"</div>|</p>|</li>|</h[1-6]>", "\n\n", value, flags=re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()
