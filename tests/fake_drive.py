"""A local stand-in for the Microsoft Graph drive-item endpoints.

Extends :class:`fake_microsoft.FakeMicrosoft` (its token endpoint, audience and
expiry enforcement) with the slices the file tools use, written from the
documented contracts (Microsoft Learn, Graph v1.0):

* ``GET /shares/u!<base64url>/driveItem`` and
  ``GET /teams/{t}/channels/{c}/filesFolder``;
* ``GET /drives/{d}/items/{id}`` (also ``items/{parent}:/{name}``) and
  ``.../children`` paged by ``$top``/``$skiptoken`` and ``@odata.nextLink``;
* ``GET .../content[?format=pdf]`` answering ``302`` to a pre-authenticated
  ``/download/<token>`` that REFUSES a request carrying ``Authorization``;
* ``POST .../createUploadSession`` (``if-match`` answered ``412`` when stale;
  ``@microsoft.graph.conflictBehavior`` honoured at the final fragment, where
  the documentation says a name conflict surfaces) and ``PUT /upload/<id>``
  with ``Content-Range``, also refusing ``Authorization``.

Failure injection reuses ``script`` (keys below). Hooks let a test make a
concurrent edit land at a chosen moment. It proves the client's and tools'
mechanics only: it is not Microsoft, and whether SharePoint actually enforces
``If-Match`` on a session, resolves a given link form, or renders a given file
is exactly what it cannot say (issue #41).

Script routes: ``GET shares``, ``GET channel``, ``GET item``, ``GET children``,
``GET content``, ``POST session``, ``PUT upload``, ``GET download``.
``PUT upload`` also understands ``apply-then-drop``.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.parse
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from typing import Any

from fake_microsoft import FakeMicrosoft, Recorded

DRIVE_ID = "b!drive-one_ABC"
ROOT_ID = "01ROOT0000"
PDF_EXTENSIONS = {"docx", "doc", "xlsx", "pptx", "rtf", "md"}


@dataclass
class Node:
    id: str
    name: str
    parent: str | None
    folder: bool = False
    content: bytes = b""
    version: int = 1
    modified_by: str = "Original Author"
    guid: str = field(default_factory=lambda: str(uuid.uuid4()).upper())

    @property
    def etag(self) -> str:
        return f'"{{{self.guid}}},{self.version}"'

    @property
    def ctag(self) -> str:
        return f'"c:{{{self.guid}}},{self.version}"'


class FakeDrive(FakeMicrosoft):
    def __init__(self) -> None:
        super().__init__()
        self.nodes: dict[str, Node] = {}
        self.sessions: dict[str, dict[str, Any]] = {}
        self.downloads: dict[str, bytes] = {}
        self.links: dict[str, str] = {}
        self.channels: dict[tuple[str, str], str] = {}
        self.header_log: list[tuple[str, str, dict[str, str]]] = []
        self.hooks: dict[str, Callable[[], None]] = {}
        self.children_cap: int | None = None
        self.corrupt_commit = False
        self.corrupt_download_names: set[str] = set()
        self.not_pdf_names: set[str] = set()
        self.refuse_pdf_names: set[str] = set()
        self.next_node = 1
        self.nodes[ROOT_ID] = Node(ROOT_ID, "root", None, folder=True)

    # -- test helpers ----------------------------------------------------

    def add_folder(self, name: str, parent: str = ROOT_ID) -> Node:
        return self._add(Node(self._new_id(), name, parent, folder=True))

    def add_file(
        self, name: str, content: bytes, parent: str = ROOT_ID, by: str = "Original"
    ) -> Node:
        return self._add(
            Node(self._new_id(), name, parent, content=content, modified_by=by)
        )

    def edit(self, node: Node | str, content: bytes, by: str = "Sam Editor") -> Node:
        """Another person saves a new version."""
        with self.lock:
            target = self.nodes[node if isinstance(node, str) else node.id]
            target.content = content
            target.version += 1
            target.modified_by = by
            return target

    def children_of(self, parent: str) -> list[Node]:
        with self.lock:
            return sorted(
                (n for n in self.nodes.values() if n.parent == parent),
                key=lambda n: n.name,
            )

    def named(self, parent: str, name: str) -> Node | None:
        return next((n for n in self.children_of(parent) if n.name == name), None)

    def graph_calls(self, method: str, fragment: str) -> list[Recorded]:
        return self.calls(method, fragment)

    def env(self, **extra: str) -> dict[str, str]:
        return super().env(JAKE_TOOLS_FILES_GRAPH_ROOT=self.graph_root, **extra)

    def _new_id(self) -> str:
        with self.lock:
            number = self.next_node
            self.next_node += 1
            return f"01ITEM{number:04d}"

    def _add(self, node: Node) -> Node:
        with self.lock:
            self.nodes[node.id] = node
            return node

    # -- request handling ------------------------------------------------

    def handle(self, request: BaseHTTPRequestHandler) -> None:
        path = urllib.parse.unquote(urllib.parse.urlsplit(request.path).path)
        headers = dict(request.headers.items())
        with self.lock:
            self.header_log.append((request.command, path, headers))
        if path.startswith(("/upload/", "/download/")):
            self._handle_pre_authenticated(request, path, headers)
            return
        super().handle(request)

    def _scripted(self, route: str) -> str | None:
        with self.lock:
            queue = self.script.get(route)
            return queue.pop(0) if queue else None

    def _handle_pre_authenticated(
        self, request: BaseHTTPRequestHandler, path: str, headers: dict[str, str]
    ) -> None:
        length = int(headers.get("Content-Length") or 0)
        body = request.rfile.read(length) if length else b""
        with self.lock:
            self.requests.append(Recorded(request.command, path, None, body))
        if any(k.lower() == "authorization" for k in headers):
            # Documented: a bearer token on these URLs can fail with 401.
            self._reply(request, 401, {"error": {"code": "InvalidAuthenticationToken"}})
            return
        kind = "upload" if path.startswith("/upload/") else "download"
        action = self._scripted(f"{request.command} {kind}")
        if action == "drop":
            request.connection.close()
            return
        if action is not None and action.isdigit():
            self._reply(request, int(action), {"error": {"code": "scripted"}})
            return
        key = path.rsplit("/", 1)[-1]
        if kind == "download":
            with self.lock:
                data = self.downloads.get(key)
            if data is None:
                self._reply(request, 404, {"error": {"code": "itemNotFound"}})
                return
            self._reply(request, 200, None, raw=data)
            return
        status, payload = self._commit(key, headers, body)
        if action == "apply-then-500":
            self._reply(request, 500, {"error": {"code": "generalException"}})
            return
        if action == "apply-then-drop":
            request.connection.close()
            return
        if action is not None and action.startswith("apply-then-json:"):
            self._reply(
                request, 200, json.loads(action.removeprefix("apply-then-json:"))
            )
            return
        self._reply(request, status, payload)

    def _commit(
        self, session_id: str, headers: dict[str, str], body: bytes
    ) -> tuple[int, Any]:
        with self.lock:
            session = self.sessions.get(session_id)
        if session is None:
            return 404, {"error": {"code": "itemNotFound"}}
        total = len(body)
        if headers.get("Content-Range") != f"bytes 0-{total - 1}/{total}":
            return 416, {"error": {"code": "invalidRange"}}
        hook = self.hooks.get("before_commit")
        if hook is not None:
            hook()
        stored = body + b"!" if self.corrupt_commit else body
        with self.lock:
            if session["item_id"] is not None:
                node = self.nodes[session["item_id"]]
                node.content = stored
                node.version += 1
                node.modified_by = "Jake's Account"
                self.sessions.pop(session_id)
                return 200, self._driveitem(node)
            parent, name = session["parent"], session["name"]
            exists = self.named(parent, name)
            if exists is not None:
                if session["conflict"] == "rename":
                    stem, dot, extension = name.rpartition(".")
                    name = f"{stem} 1.{extension}" if dot else f"{name} 1"
                else:
                    return 409, {
                        "error": {
                            "code": "nameAlreadyExists",
                            "message": "Another file exists with the same name.",
                        }
                    }
            node = self._add(
                Node(
                    self._new_id(),
                    name,
                    parent,
                    content=stored,
                    modified_by="Jake's Account",
                )
            )
            self.sessions.pop(session_id)
            return 201, self._driveitem(node)

    @staticmethod
    def _route(method: str, path: str) -> str:
        if "/shares/" in path:
            return f"{method} shares"
        if path.endswith("/filesFolder"):
            return f"{method} channel"
        if path.endswith("createUploadSession"):
            return f"{method} session"
        if path.endswith("/children"):
            return f"{method} children"
        if path.endswith("/content"):
            return f"{method} content"
        if "/drives/" in path:
            return f"{method} item"
        return FakeMicrosoft._route(method, path)

    def _dispatch(
        self, method: str, path: str, query: dict[str, list[str]], body: bytes
    ) -> tuple[int, Any, bytes | None]:
        graph = path.removeprefix("/graph/v1.0")
        if shares := re.fullmatch(r"/shares/u!([A-Za-z0-9_-]+)/driveItem", graph):
            token = shares.group(1)
            url = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)).decode()
            node = self.nodes.get(self.links.get(url, ""))
            if node is None:
                return 404, {"error": {"code": "itemNotFound"}}, None
            return 200, self._driveitem(node), None
        if channel := re.fullmatch(
            r"/teams/([^/]+)/channels/([^/]+)/filesFolder", graph
        ):
            node = self.nodes.get(self.channels.get(channel.groups(), ""))  # type: ignore[arg-type]
            if node is None:
                return 404, {"error": {"code": "itemNotFound"}}, None
            return 200, self._driveitem(node), None
        drive = re.match(r"/drives/([^/]+)/items/([^/:]+)(.*)$", graph)
        if drive is None:
            return super()._dispatch(method, path, query, body)
        drive_id, item_id, rest = drive.groups()
        if drive_id != DRIVE_ID:
            return 404, {"error": {"code": "itemNotFound"}}, None
        node = self.nodes.get(item_id)
        if node is None:
            return 404, {"error": {"code": "itemNotFound"}}, None
        if rest == "" and method == "GET":
            before = self.hooks.get("before_get_item")
            if before is not None:
                before()  # an edit that this very reply already shows
            payload = self._driveitem(node)
            after = self.hooks.get("after_get_item")
            if after is not None:
                after()  # an edit that only a LATER read shows
            return 200, payload, None
        if rest == "/children" and method == "GET":
            return 200, self._children(node, query), None
        if rest == "/content" and method == "GET":
            return self._content(node, query)
        if rest.startswith(":/") and method == "GET" and not rest.endswith(":"):
            child = self.named(node.id, rest.removeprefix(":/"))
            if child is None:
                return 404, {"error": {"code": "itemNotFound"}}, None
            return 200, self._driveitem(child), None
        if rest.endswith("createUploadSession") and method == "POST":
            return self._open_session(node, rest, body)
        return 404, {"error": {"code": "unknownRoute"}}, None

    def _content(
        self, node: Node, query: dict[str, list[str]]
    ) -> tuple[int, Any, bytes | None]:
        data = node.content
        if query.get("format") == ["pdf"]:
            extension = node.name.rsplit(".", 1)[-1]
            if extension not in PDF_EXTENSIONS or node.name in self.refuse_pdf_names:
                return 400, {"error": {"code": "invalidRequest"}}, None
            data = (
                b"<html>not a pdf</html>"
                if node.name in self.not_pdf_names
                else b"%PDF-1.7 rendering of " + node.name.encode()
            )
        elif node.name in self.corrupt_download_names:
            data = data + b"!"
        token = uuid.uuid4().hex
        with self.lock:
            self.downloads[token] = data
        return 302, {"__location__": f"{self.base_url}/download/{token}"}, None

    def _open_session(
        self, node: Node, rest: str, body: bytes
    ) -> tuple[int, Any, bytes | None]:
        hook = self.hooks.get("before_session")
        if hook is not None:
            hook()
        info: dict[str, Any] = {"item_id": None, "parent": None, "name": None}
        with self.lock:
            sent = next(
                (
                    h
                    for m, p, h in reversed(self.header_log)
                    if m == "POST" and p.endswith("createUploadSession")
                ),
                {},
            )
            if rest == "/createUploadSession":
                info["item_id"] = node.id
                condition = sent.get("If-Match")
                if condition is not None and condition != node.etag:
                    return 412, {"error": {"code": "preconditionFailed"}}, None
            else:
                spec = json.loads(body or b"{}").get("item", {})
                info.update(
                    parent=node.id,
                    name=rest.removeprefix(":/").removesuffix(":/createUploadSession"),
                    conflict=spec.get("@microsoft.graph.conflictBehavior", "fail"),
                )
            session_id = uuid.uuid4().hex
            self.sessions[session_id] = info
        return (
            200,
            {
                "uploadUrl": f"{self.base_url}/upload/{session_id}",
                "expirationDateTime": "2026-12-01T00:00:00Z",
            },
            None,
        )

    def _children(self, node: Node, query: dict[str, list[str]]) -> dict[str, Any]:
        top = int(query.get("$top", ["200"])[0])
        if self.children_cap is not None:
            top = min(top, self.children_cap)
        children = self.children_of(node.id)
        start = int(query.get("$skiptoken", ["tok-0"])[0].removeprefix("tok-"))
        page = children[start : start + top]
        payload: dict[str, Any] = {"value": [self._driveitem(c) for c in page]}
        if start + top < len(children):
            payload["@odata.nextLink"] = (
                f"{self.graph_root}/drives/{DRIVE_ID}/items/{node.id}/children"
                f"?$top={top}&$skiptoken=tok-{start + top}"
            )
        return payload

    def _driveitem(self, node: Node) -> dict[str, Any]:
        item: dict[str, Any] = {
            "id": node.id,
            "name": node.name,
            "webUrl": "https://csuses.sharepoint.com/sites/team/Shared%20Documents/"
            + urllib.parse.quote(node.name),
            "size": len(node.content),
            "eTag": node.etag,
            "cTag": node.ctag,
            "lastModifiedDateTime": f"2026-11-{min(node.version, 28):02d}T00:00:00Z",
            "lastModifiedBy": {"user": {"displayName": node.modified_by}},
        }
        if node.parent is not None:
            item["parentReference"] = {"driveId": DRIVE_ID, "id": node.parent}
        else:
            item["parentReference"] = {"driveId": DRIVE_ID}
        if node.folder:
            item["folder"] = {"childCount": len(self.children_of(node.id))}
        else:
            item["file"] = {
                "mimeType": "application/octet-stream",
                "hashes": {"quickXorHash": "AAAAAAAAAAAAAAAAAAAAAAAAAAA="},
            }
        return item

    @staticmethod
    def _reply(
        request: BaseHTTPRequestHandler,
        status: int,
        payload: Any,
        *,
        raw: bytes | None = None,
    ) -> None:
        if isinstance(payload, dict) and "__location__" in payload:
            request.send_response(status)
            request.send_header("Location", payload["__location__"])
            request.send_header("Content-Length", "0")
            request.end_headers()
            return
        FakeMicrosoft._reply(request, status, payload, raw=raw)


__all__ = ["DRIVE_ID", "ROOT_ID", "FakeDrive", "Node"]
