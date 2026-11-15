"""A local stand-in for Microsoft's token endpoint, Graph and SharePoint REST.

One HTTP server speaks the slices the newsletter client uses:

* ``POST /<tenant>/oauth2/v2.0/token`` (the refresh-token grant), issuing
  opaque access tokens bound to the resource named in ``scope`` and to a
  lifetime, plus an unsigned ID token naming tenant and account;
* Graph list items (``GET`` list with ``$top``/``$skiptoken`` paging, ``GET``
  one, ``POST`` create, ``PATCH`` fields) under ``/graph/v1.0``;
* SharePoint REST attachments (list, add, download) under ``/sp/_api``.

Graph and SharePoint each refuse a token issued for the other's audience, an
expired token, or an unknown one (401), so audience separation and refresh are
exercised through the real clients rather than asserted. Scripted failures
(``script``) let a test make one specific call fail, drop the connection, or
reply 500 *after* applying the write. The ID-token claims it emits
(``tid``, ``preferred_username``, ``oid``) are the documented claim names, not
evidence of what Microsoft returns on every refresh. It proves the client's and tools'
mechanics only: it is not Microsoft, and says nothing about live compatibility
(SharePoint's real HTML normalisation, throttling, tenant policy, ...).
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

TENANT = "0a3a5574-cfda-4314-952e-c0b3e1dcac6d"
CLIENT = "11111111-2222-3333-4444-555555555555"
ACCOUNT = "editor@example.test"
REFRESH_TOKEN = "seed-refresh-token-VALUE"
CLIENT_SECRET = "fake-client-secret-VALUE"
LIST_ID = "b07a8e3d-fa7e-4191-8733-549140a576db"
GRAPH_RESOURCE = "https://graph.microsoft.com"
SHAREPOINT_RESOURCE = "https://csuses.sharepoint.com"
ATTACHMENT_DIR = "/sites/leadershipteam/Lists/Newsletter/Attachments"


@dataclass
class Entry:
    id: int
    title: str
    body_html: str
    created: str
    attachments: dict[str, bytes] = field(default_factory=dict)
    modified: str = ""


@dataclass
class Recorded:
    method: str
    path: str
    token: str | None
    body: bytes


def jwt(claims: dict[str, Any]) -> str:
    def part(data: dict[str, Any]) -> str:
        raw = json.dumps(data).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{part({'alg': 'none'})}.{part(claims)}."


class FakeMicrosoft:
    """Start with :meth:`start`; point the roots at :attr:`base_url`."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.entries: dict[int, Entry] = {}
        self.next_id = 1
        self.requests: list[Recorded] = []
        self.token_requests: list[dict[str, str]] = []
        self.issued: dict[str, tuple[str, float]] = {}
        self.refresh_token = REFRESH_TOKEN
        self.rotate = False
        self.expires_in = 3600
        self.token_delay = 0.0
        self.token_error: dict[str, Any] | None = None
        self.token_status = 400
        self.token_redirect = 0
        self.id_claims: dict[str, Any] = {
            "tid": TENANT,
            "preferred_username": ACCOUNT,
        }
        self.omit_id_token = False
        self.script: dict[str, list[str]] = {}
        """``"<METHOD> <route>"`` -> queue of one-shot actions: a status code
        as a string (``"403"``), ``"drop"`` (close without answering) or
        ``"apply-then-500"`` (do the write, then answer 500),
        ``"apply-then-json:<json>"`` (do it, answer 200 with that body), or
        ``"redirect:<status>"`` (answer with a redirect to ``/redirected``)."""
        self.page_size_cap: int | None = None
        self.corrupt_downloads: set[str] = set()
        self.mangle_body = False
        self.forbid_graph_list = False
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle -------------------------------------------------------

    def start(self) -> FakeMicrosoft:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                fake.handle(self)

            def do_POST(self) -> None:
                fake.handle(self)

            def do_PATCH(self) -> None:
                fake.handle(self)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    @property
    def base_url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_port}"

    @property
    def graph_root(self) -> str:
        return f"{self.base_url}/graph/v1.0"

    @property
    def sharepoint_root(self) -> str:
        return f"{self.base_url}/sp/_api"

    def env(self, **extra: str) -> dict[str, str]:
        """The environment that points a real server at this fake."""
        values = {
            "JAKE_TOOLS_MICROSOFT_TENANT_ID": TENANT,
            "JAKE_TOOLS_MICROSOFT_CLIENT_ID": CLIENT,
            "JAKE_TOOLS_MICROSOFT_ACCOUNT": ACCOUNT,
            "JAKE_TOOLS_MICROSOFT_AUTHORITY_URL": self.base_url,
            "JAKE_TOOLS_NEWSLETTERS_GRAPH_ROOT": self.graph_root,
            "JAKE_TOOLS_NEWSLETTERS_SHAREPOINT_ROOT": self.sharepoint_root,
            "MICROSOFT_REFRESH_TOKEN": REFRESH_TOKEN,
        }
        values.update(extra)
        return values

    # -- test helpers ----------------------------------------------------

    def add_entry(
        self, title: str, body: str, *, attachments: dict[str, bytes] | None = None
    ) -> Entry:
        with self.lock:
            entry = Entry(
                self.next_id,
                title,
                body,
                f"2026-11-{min(self.next_id, 28):02d}T00:00:00Z",
                dict(attachments or {}),
            )
            entry.modified = entry.created
            self.entries[entry.id] = entry
            self.next_id += 1
            return entry

    def calls(self, method: str, fragment: str) -> list[Recorded]:
        return [r for r in self.requests if r.method == method and fragment in r.path]

    def creates(self) -> list[Recorded]:
        """Every attempt to create an entry (POST to the list's items)."""
        return [
            r for r in self.requests if r.method == "POST" and r.path.endswith("/items")
        ]

    def refreshes(self, resource: str | None = None) -> list[dict[str, str]]:
        return [
            r
            for r in self.token_requests
            if resource is None or r["scope"].startswith(resource + "/.default")
        ]

    # -- request handling ------------------------------------------------

    def handle(self, request: BaseHTTPRequestHandler) -> None:
        length = int(request.headers.get("Content-Length") or 0)
        body = request.rfile.read(length) if length else b""
        parts = urllib.parse.urlsplit(request.path)
        path = urllib.parse.unquote(parts.path)
        query = urllib.parse.parse_qs(parts.query)
        header = request.headers.get("Authorization") or ""
        token = header.removeprefix("Bearer ") if header.startswith("Bearer ") else None
        with self.lock:
            self.requests.append(Recorded(request.command, path, token, body))
        if path.endswith("/oauth2/v2.0/token"):
            self._token(request, body)
            return
        if path.startswith("/graph/v1.0/"):
            resource = GRAPH_RESOURCE
        elif path.startswith("/sp/_api/"):
            resource = SHAREPOINT_RESOURCE
        else:
            self._reply(request, 404, {"error": "unknown route"})
            return
        if not self._authorised(token, resource):
            self._reply(request, 401, {"error": {"code": "InvalidAuthenticationToken"}})
            return
        route = self._route(request.command, path)
        action = None
        with self.lock:
            queue = self.script.get(route)
            if queue:
                action = queue.pop(0)
        if action == "drop":
            request.connection.close()
            return
        if action is not None and action.isdigit():
            self._reply(request, int(action), {"error": {"code": "scripted"}})
            return
        if action is not None and action.startswith("redirect:"):
            request.send_response(int(action.removeprefix("redirect:")))
            request.send_header("Location", f"{self.base_url}/redirected")
            request.send_header("Content-Length", "0")
            request.end_headers()
            return
        try:
            status, payload, raw = self._dispatch(request.command, path, query, body)
        except KeyError:
            self._reply(request, 404, {"error": {"code": "itemNotFound"}})
            return
        if action == "apply-then-500":
            self._reply(request, 500, {"error": {"code": "generalException"}})
            return
        if action is not None and action.startswith("apply-then-json:"):
            # The write (or read) happened, but the reply has the wrong shape.
            self._reply(
                request, 200, json.loads(action.removeprefix("apply-then-json:"))
            )
            return
        self._reply(request, status, payload, raw=raw)

    @staticmethod
    def _route(method: str, path: str) -> str:
        if "/AttachmentFiles/add(" in path:
            name = re.search(r"FileName='(.*)'\)", path)
            return f"{method} attachment-add:{name.group(1) if name else ''}"
        if path.endswith("/$value"):
            name = re.search(r"AttachmentFiles\('(.*)'\)/\$value", path)
            return f"{method} attachment-get:{name.group(1) if name else ''}"
        if path.endswith("/AttachmentFiles"):
            return f"{method} attachment-list"
        if path.endswith("/fields"):
            return f"{method} fields"
        if re.search(r"/items/\d+$", path):
            return f"{method} item"
        return f"{method} items"

    def _authorised(self, token: str | None, resource: str) -> bool:
        with self.lock:
            issued = self.issued.get(token or "")
        if issued is None:
            return False
        audience, expires_at = issued
        return audience == resource and time.monotonic() < expires_at

    def _token(self, request: BaseHTTPRequestHandler, body: bytes) -> None:
        form = {k: v[0] for k, v in urllib.parse.parse_qs(body.decode()).items()}
        with self.lock:
            self.token_requests.append(form)
        if self.token_delay:
            time.sleep(self.token_delay)
        if self.token_redirect:
            request.send_response(self.token_redirect)
            request.send_header("Location", f"{self.base_url}/redirected")
            request.send_header("Content-Length", "0")
            request.end_headers()
            return
        if self.token_error is not None:
            self._reply(request, self.token_status, self.token_error)
            return
        if form.get("refresh_token") not in {self.refresh_token, REFRESH_TOKEN}:
            self._reply(
                request,
                400,
                {
                    "error": "invalid_grant",
                    "error_description": "AADSTS70008: The provided refresh token is expired.",
                },
            )
            return
        scope = form.get("scope", "")
        resource = scope.split("/.default")[0]
        with self.lock:
            number = len(self.issued) + 1
            access = f"access-{number}-for-{urllib.parse.quote(resource, safe='')}"
            self.issued[access] = (resource, time.monotonic() + self.expires_in)
            payload: dict[str, Any] = {
                "token_type": "Bearer",
                "access_token": access,
                "expires_in": self.expires_in,
                "scope": scope,
            }
            if self.rotate:
                self.refresh_token = f"rotated-refresh-{number}"
                payload["refresh_token"] = self.refresh_token
            if not self.omit_id_token:
                payload["id_token"] = jwt(self.id_claims)
        self._reply(request, 200, payload)

    def _dispatch(
        self,
        method: str,
        path: str,
        query: dict[str, list[str]],
        body: bytes,
    ) -> tuple[int, Any, bytes | None]:
        graph_item = re.search(r"/lists/[^/]+/items(?:/(\d+))?(/fields)?$", path)
        if graph_item:
            item_id, fields = graph_item.groups()
            if item_id is None and method == "GET":
                if self.forbid_graph_list:
                    return 403, {"error": {"code": "accessDenied"}}, None
                return 200, self._list(query), None
            if item_id is None and method == "POST":
                data = json.loads(body)["fields"]
                entry = self.add_entry(data["Title"], data["Body"])
                return 201, self._graph_item(entry, with_fields=False), None
            if item_id is None:
                return 405, {"error": "method not allowed"}, None
            entry = self._entry(int(item_id))
            if fields and method == "PATCH":
                data = json.loads(body)
                with self.lock:
                    entry.title = data.get("Title", entry.title)
                    entry.body_html = data.get("Body", entry.body_html)
                    entry.modified = "2026-11-29T00:00:00Z"
                return 200, data, None
            return 200, self._graph_item(entry), None
        sp = re.search(r"/items\((\d+)\)/AttachmentFiles(.*)$", path)
        if sp:
            entry = self._entry(int(sp.group(1)))
            rest = sp.group(2)
            if rest == "" and method == "GET":
                return 200, self._attachment_list(entry), None
            add = re.match(r"/add\(FileName='(.*)'\)$", rest)
            if add and method == "POST":
                name = add.group(1).replace("''", "'")
                with self.lock:
                    if name in entry.attachments:
                        return 500, {"error": "file already exists"}, None
                    entry.attachments[name] = body
                return (
                    200,
                    {"FileName": name, "ServerRelativeUrl": self._url(entry, name)},
                    None,
                )
            get = re.match(r"\('(.*)'\)/\$value$", rest)
            if get:
                name = get.group(1).replace("''", "'")
                if name not in entry.attachments:
                    return 404, {"error": "no such attachment"}, None
                data = entry.attachments[name]
                if name in self.corrupt_downloads:
                    data = data + b"!"
                return 200, None, data
        return 404, {"error": "unknown route"}, None

    def _entry(self, entry_id: int) -> Entry:
        with self.lock:
            return self.entries[entry_id]

    def _url(self, entry: Entry, name: str) -> str:
        return f"{ATTACHMENT_DIR}/{entry.id}/{urllib.parse.quote(name)}"

    def _attachment_list(self, entry: Entry) -> dict[str, Any]:
        return {
            "value": [
                {"FileName": name, "ServerRelativeUrl": self._url(entry, name)}
                for name in entry.attachments
            ]
        }

    def _graph_item(self, entry: Entry, *, with_fields: bool = True) -> dict[str, Any]:
        item: dict[str, Any] = {
            "id": str(entry.id),
            "webUrl": f"https://csuses.sharepoint.com/Lists/Newsletter/DispForm.aspx?ID={entry.id}",
        }
        if with_fields:
            body = (
                entry.body_html + "<p>mangled</p>"
                if self.mangle_body
                else entry.body_html
            )
            item["fields"] = {
                "id": str(entry.id),
                "Title": entry.title,
                "Body": body,
                "Created": entry.created,
                "Modified": entry.modified,
            }
        return item

    def _list(self, query: dict[str, list[str]]) -> dict[str, Any]:
        top = int(query.get("$top", ["100"])[0])
        if self.page_size_cap is not None:
            top = min(top, self.page_size_cap)
        ordered = sorted(self.entries.values(), key=lambda e: e.id, reverse=True)
        token = query.get("$skiptoken", [None])[0]
        if token is not None:
            ceiling = int(token.removeprefix("tok-"))
            ordered = [e for e in ordered if e.id < ceiling]
        page = ordered[:top]
        payload: dict[str, Any] = {"value": [self._graph_item(e) for e in page]}
        if len(ordered) > top:
            payload["@odata.nextLink"] = (
                f"{self.graph_root}/sites/x/lists/{LIST_ID}/items"
                f"?$top={top}&$skiptoken=tok-{page[-1].id}"
            )
        return payload

    @staticmethod
    def _reply(
        request: BaseHTTPRequestHandler,
        status: int,
        payload: Any,
        *,
        raw: bytes | None = None,
    ) -> None:
        data = raw if raw is not None else json.dumps(payload).encode()
        request.send_response(status)
        request.send_header(
            "Content-Type",
            "application/octet-stream" if raw is not None else "application/json",
        )
        request.send_header("Content-Length", str(len(data)))
        request.end_headers()
        request.wfile.write(data)
