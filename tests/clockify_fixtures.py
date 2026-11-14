"""A dataset-driven Clockify workspace for the catalogue/lookup tests.

One :class:`Workspace` answers the same routes the real ``ClockifyClient``
calls, honouring paging and the archived/is-active filters. It is served two
ways so both tests exercise the real client and real operation code: as the
client's ``HttpSession`` seam (:class:`RoutingSession`) and as a local HTTP
peer (:func:`serve`) for the stdio MCP server. Every request is logged, so
tests can count upstream requests and response bytes independently of what a
tool returns to its caller.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

API_KEY = "dummy-clockify-key"


@dataclass
class Workspace:
    id: str = "ws-1"
    clients: list[dict[str, Any]] = field(default_factory=list)
    projects: list[dict[str, Any]] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    # Paths (prefix match on the request path) that answer 503.
    failing: set[str] = field(default_factory=set)
    # Task lists that repeat page 1 forever (a real pagination fault).
    unstable: set[str] = field(default_factory=set)
    requests: list[tuple[str, dict[str, str]]] = field(default_factory=list)
    response_bytes: int = 0

    def add_client(self, id: str, name: str, archived: bool = False) -> None:
        self.clients.append({"id": id, "name": name, "archived": archived})

    def add_project(
        self,
        id: str,
        name: str,
        client: str | None,
        note: str = "",
        archived: bool = False,
    ) -> None:
        self.projects.append(
            {
                "id": id,
                "name": name,
                "note": note,
                "archived": archived,
                "clientId": client or "",
            }
        )

    def add_task(
        self, id: str, name: str, project: str, status: str = "ACTIVE"
    ) -> None:
        self.tasks.append(
            {"id": id, "name": name, "projectId": project, "status": status}
        )

    def reset_counters(self) -> None:
        self.requests.clear()
        self.response_bytes = 0

    def respond(self, path: str, params: dict[str, str]) -> tuple[int, object]:
        self.requests.append((path, params))
        base = f"/workspaces/{self.id}"
        if any(path.startswith(prefix) for prefix in self.failing):
            return 503, {"message": "upstream body that must not leak"}
        archived = params.get("archived")
        if path == base + "/clients":
            return 200, self._page(
                [c for c in self.clients if str(c["archived"]).lower() == archived],
                params,
            )
        if path == base + "/projects":
            return 200, self._page(
                [p for p in self.projects if str(p["archived"]).lower() == archived],
                params,
            )
        if path.startswith(base + "/projects/") and path.endswith("/tasks"):
            project = path.split("/")[-2]
            wanted = "ACTIVE" if params.get("is-active") == "true" else "DONE"
            rows = [
                t
                for t in self.tasks
                if t["projectId"] == project and t["status"] == wanted
            ]
            if path in self.unstable:
                params = {**params, "page": "1"}
            return 200, self._page(rows, params)
        return 404, {"message": "unknown route"}

    @staticmethod
    def _page(rows: list[dict[str, Any]], params: dict[str, str]) -> list[Any]:
        size, page = int(params["page-size"]), int(params["page"])
        return rows[(page - 1) * size : page * size]


class _Response:
    def __init__(self, status: int, payload: object) -> None:
        self.status_code = status
        self.reason = "OK" if status < 400 else "ERR"
        self.text = json.dumps(payload)
        self.content = self.text.encode()
        self._payload = payload

    def json(self) -> object:
        return self._payload


class RoutingSession:
    """``HttpSession`` backed by a :class:`Workspace`: the client seam."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        assert method == "GET"
        assert kwargs["headers"]["X-Api-Key"] == API_KEY
        params = {k: str(v) for k, v in (kwargs.get("params") or {}).items()}
        status, payload = self.workspace.respond(urlsplit(url).path, params)
        response = _Response(status, payload)
        self.workspace.response_bytes += len(response.content)
        return response


@contextmanager
def serve(workspace: Workspace) -> Iterator[str]:
    """Serve ``workspace`` on a local port; yields the base URL."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            url = urlsplit(self.path)
            params = {k: v[0] for k, v in parse_qs(url.query).items()}
            if self.headers.get("X-Api-Key") != API_KEY:
                status, result = 401, {"secret": "never expose upstream body"}
            else:
                status, result = workspace.respond(url.path, params)
            data = json.dumps(result).encode()
            workspace.response_bytes += len(data)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def sunfish_workspace(filler_projects: int = 30) -> Workspace:
    """A workspace shaped like the reported use: a handful of Jira keys to
    resolve in a workspace that is mostly history.

    Keys: SF-101 (one task), SF-102 (completed task), SF-301 (ambiguous: two
    live tasks in different clients), SF-77 (task in an archived project of
    an archived client), SF-100 (a project, via its ``Jira:`` note), SF-999
    (nowhere). SF-1010 and a mid-name mention of SF-555 are decoys: neither
    may satisfy a lookup for SF-10 / SF-101 / SF-555.
    """
    ws = Workspace()
    ws.add_client("c-acme", "Acme Robotics")
    ws.add_client("c-globex", "Globex")
    ws.add_client("c-old", "Legacy Co", archived=True)
    ws.add_project("p-acme", "Acme Phase 1", "c-acme", "Jira: SF-100")
    ws.add_project("p-globex", "Globex Phase 1", "c-globex", "Jira: SF-200")
    ws.add_project("p-old", "Legacy", "c-old", "Jira: SF-50", archived=True)
    ws.add_task("t-101", "SF-101 Wire up auth", "p-acme")
    ws.add_task("t-1010", "SF-1010 Decoy with longer number", "p-acme")
    ws.add_task("t-102", "SF-102 Write docs", "p-acme", "DONE")
    ws.add_task("t-301a", "SF-301 Cutover", "p-acme")
    ws.add_task("t-301b", "SF-301 Cutover (copy)", "p-globex")
    ws.add_task("t-201", "SF-201 Calibrate arm", "p-globex")
    ws.add_task("t-555", "Follow-up to SF-555 later", "p-globex")
    ws.add_task("t-77", "SF-77 Legacy import", "p-old", "DONE")
    for i in range(filler_projects):
        client = ("c-acme", "c-globex", "c-old")[i % 3]
        ws.add_project(
            f"p-fill-{i}",
            f"Historic project {i}",
            client,
            f"Jira: SF-{1000 + i}",
            archived=i % 2 == 0,
        )
        for n in range(4):
            ws.add_task(
                f"t-fill-{i}-{n}",
                f"SF-{2000 + i * 10 + n} Historic work item {n}",
                f"p-fill-{i}",
                "DONE" if n % 2 else "ACTIVE",
            )
    return ws
