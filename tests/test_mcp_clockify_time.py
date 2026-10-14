"""Real stdio MCP server and real ClockifyClient against a local HTTP peer."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from mcp import ClientSession
from pydantic import AnyUrl
from test_clockify_time import END, START, entry

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]
EnvFactory = Callable[..., dict[str, str]]


@dataclass
class Upstream:
    url: str = ""
    entries: list[dict[str, Any]] = field(
        default_factory=lambda: [
            entry("old", "2020-01-01T00:00:00Z", None),
            entry("completed"),
        ]
    )
    requests: list[tuple[str, str, dict[str, list[str]]]] = field(default_factory=list)
    posts: list[dict[str, Any]] = field(default_factory=list)
    outcome: str = "ok"
    task_project: str = "p-1"


@pytest.fixture
def upstream() -> Iterator[Upstream]:
    state = Upstream()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            self.dispatch("GET")

        def do_POST(self) -> None:
            self.dispatch("POST")

        def dispatch(self, method: str) -> None:
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            state.requests.append((method, url.path, query))
            status = 200
            result: object = []
            ws = "/workspaces/ws-1"
            if self.headers.get("X-Api-Key") != "dummy-clockify-key":
                status, result = 401, {"secret": "never expose upstream body"}
            elif url.path == "/user":
                result = {"id": "u-1"}
            elif url.path == ws + "/user/u-1/time-entries":
                assert set(query) == {"page", "page-size"}
                size, page = int(query["page-size"][0]), int(query["page"][0])
                result = state.entries[(page - 1) * size : page * size]
            elif url.path == ws + "/clients":
                archived = query["archived"] == ["true"]
                result = [
                    {
                        "id": "c-old" if archived else "c-1",
                        "name": "Client",
                        "archived": archived,
                    }
                ]
            elif url.path == ws + "/projects":
                archived = query["archived"] == ["true"]
                result = [
                    {
                        "id": "p-old" if archived else "p-1",
                        "name": "Project",
                        "archived": archived,
                        "clientId": "c-1",
                    }
                ]
            elif url.path == ws + "/projects/p-1":
                result = {"id": "p-1", "name": "Project", "clientId": "c-1"}
            elif url.path.endswith("/tasks/t-1"):
                result = {
                    "id": "t-1",
                    "name": "Task",
                    "projectId": state.task_project,
                    "status": "ACTIVE",
                }
            elif url.path.endswith("/tasks"):
                project = url.path.split("/")[-2]
                active = query["is-active"] == ["true"]
                result = [
                    {
                        "id": project + ("-active" if active else "-done"),
                        "name": "Task",
                        "projectId": project,
                        "status": "ACTIVE" if active else "DONE",
                    }
                ]
            elif method == "POST" and url.path == ws + "/time-entries":
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                state.posts.append(body)
                created = entry(
                    "created",
                    body["start"],
                    body["end"],
                    description=body["description"],
                    billable=body["billable"],
                    projectId=body["projectId"],
                    taskId=body["taskId"],
                )
                state.entries.append(created)
                status, result = (
                    (500, {"error": "secret response"})
                    if state.outcome == "post"
                    else (201, {"id": "created"})
                )
            elif url.path == ws + "/time-entries/created":
                result = state.entries[-1]
                if state.outcome == "get":
                    status, result = 503, {"error": "secret response"}
                elif state.outcome == "mismatch":
                    result = {**state.entries[-1], "description": "different"}
            else:
                status, result = 404, {"error": "unknown"}
            data = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def clockify_env(
    upstream: Upstream, mcp_env_factory: EnvFactory, mcp_home: Path
) -> dict[str, str]:
    return mcp_env_factory(
        mcp_home,
        CLOCKIFY_API_KEY="dummy-clockify-key",
        CLOCKIFY_API_BASE_URL=upstream.url,
        JAKE_TOOLS_CLOCKIFY_WORKSPACES="ws-1,ws-2",
    )


async def test_discovery_history_catalogue_and_verified_create(
    mcp_server: ServerFactory, clockify_env: dict[str, str], upstream: Upstream
) -> None:
    async with mcp_server(clockify_env) as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
        assert {"start", "end", "description", "billable"} == set(
            tools["clockify_create_time_entry"].inputSchema["required"]
        )
        assert tools["clockify_time_entries"].outputSchema is None
        skill = await session.read_resource(AnyUrl("skill://clockify-time"))
        assert "post_uncertain" in str(skill)
        assert "clockify_create_time_entry" in str(skill)
        history = await session.call_tool(
            "clockify_time_entries", {"start": START, "end": END, "limit": 1}
        )
        assert not history.isError, history.structuredContent
        report = history.structuredContent
        assert report is not None
        assert report["entries"][0]["entry"]["id"] == "old"
        assert report["entries"][0]["running"]
        second = await session.call_tool(
            "clockify_time_entries",
            {"start": START, "end": END, "limit": 1, "cursor": report["next_cursor"]},
        )
        assert second.structuredContent is not None
        assert second.structuredContent["entries"][0]["entry"]["id"] == "completed"
        catalogue = await session.call_tool("clockify_workspace_catalogue", {})
        assert not catalogue.isError, catalogue.structuredContent
        assert catalogue.structuredContent is not None
        assert [p["archived"] for p in catalogue.structuredContent["projects"]] == [
            False,
            True,
        ]
        assert {t["status"] for t in catalogue.structuredContent["tasks"]} == {
            "ACTIVE",
            "DONE",
        }
        result = await session.call_tool(
            "clockify_create_time_entry",
            {
                "start": START,
                "end": END,
                "description": "work",
                "billable": True,
                "project_id": "p-1",
                "task_id": "t-1",
            },
        )
        assert not result.isError, result.structuredContent
        assert (
            result.structuredContent is not None
            and result.structuredContent["verified"]
        )
        history = await session.call_tool(
            "clockify_time_entries", {"start": START, "end": END}
        )
        assert history.structuredContent is not None
        created = next(
            r
            for r in history.structuredContent["entries"]
            if r["entry"]["id"] == "created"
        )
        assert (created["project_name"], created["task_name"]) == ("Project", "Task")
    assert len(upstream.posts) == 1
    assert all("jira" not in path for _, path, _ in upstream.requests)


async def test_real_upstream_pagination_is_unfiltered(
    mcp_server: ServerFactory, clockify_env: dict[str, str], upstream: Upstream
) -> None:
    upstream.entries = [
        entry(f"old-{i}", "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z")
        for i in range(5000)
    ] + [entry("match")]
    async with mcp_server(clockify_env) as session:
        result = await session.call_tool(
            "clockify_time_entries", {"start": START, "end": END}
        )
    assert not result.isError, result.structuredContent
    assert result.structuredContent is not None
    assert result.structuredContent["total"] == 1
    assert result.structuredContent["coverage"]["reads"][0]["pages"] == 2
    assert [
        q["page"]
        for _, path, q in upstream.requests
        if path.endswith("/user/u-1/time-entries")
    ] == [["1"], ["2"]]


@pytest.mark.parametrize(
    "outcome,expected,known_id",
    [
        ("post", "post_uncertain", None),
        ("get", "verification_unavailable", "created"),
        ("mismatch", "verification_failed", "created"),
    ],
)
async def test_create_errors_include_known_id_never_retry(
    mcp_server: ServerFactory,
    clockify_env: dict[str, str],
    upstream: Upstream,
    outcome: str,
    expected: str,
    known_id: str | None,
) -> None:
    upstream.outcome = outcome
    async with mcp_server(clockify_env) as session:
        result = await session.call_tool(
            "clockify_create_time_entry",
            {"start": START, "end": END, "description": "work", "billable": False},
        )
    assert result.isError and result.structuredContent is not None
    assert result.structuredContent["code"] == "upstream_error"
    assert result.structuredContent["detail"] == {
        "integration": "clockify",
        "outcome": expected,
        "created_entry_id": known_id,
        "retry_safe": False,
    }
    assert "secret response" not in json.dumps(result.structuredContent)
    assert len(upstream.posts) == 1


@pytest.mark.parametrize(
    "tool,args,code",
    [
        (
            "clockify_workspace_catalogue",
            {"workspace_id": "elsewhere"},
            "access_denied",
        ),
        (
            "clockify_time_entries",
            {"start": "2026-01-02", "end": END},
            "invalid_argument",
        ),
        (
            "clockify_time_entries",
            {"start": START, "end": END, "cursor": "garbage"},
            "invalid_argument",
        ),
        (
            "clockify_create_time_entry",
            {
                "start": START,
                "end": END,
                "description": "work",
                "billable": False,
                "project_id": "../evil",
            },
            "invalid_argument",
        ),
    ],
)
async def test_validation_and_allowlist(
    mcp_server: ServerFactory,
    clockify_env: dict[str, str],
    upstream: Upstream,
    tool: str,
    args: dict[str, Any],
    code: str,
) -> None:
    async with mcp_server(clockify_env) as session:
        result = await session.call_tool(tool, args)
    assert result.isError and result.structuredContent is not None
    assert result.structuredContent["code"] == code
    assert upstream.posts == []


async def test_missing_allowlist_and_clockify_credentials(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
) -> None:
    async with mcp_server(mcp_env) as session:
        result = await session.call_tool("clockify_workspace_catalogue", {})
        assert (
            result.structuredContent is not None
            and result.structuredContent["code"] == "access_denied"
        )
    async with mcp_server(
        {**mcp_env, "JAKE_TOOLS_CLOCKIFY_WORKSPACES": "ws-1"}
    ) as session:
        result = await session.call_tool("clockify_workspace_catalogue", {})
        assert (
            result.structuredContent is not None
            and result.structuredContent["code"] == "missing_credentials"
        )


async def test_upstream_error_is_sanitised(
    mcp_server: ServerFactory, clockify_env: dict[str, str]
) -> None:
    async with mcp_server({**clockify_env, "CLOCKIFY_API_KEY": "wrong-key"}) as session:
        result = await session.call_tool("clockify_workspace_catalogue", {})
    assert result.isError and result.structuredContent is not None
    assert result.structuredContent["code"] == "upstream_error"
    assert (
        result.structuredContent["message"]
        == "Clockify error: GET /workspaces/ws-1/clients returned HTTP 401"
    )
    wire = json.dumps(result.structuredContent)
    assert "wrong-key" not in wire and "never expose upstream body" not in wire


async def test_task_mismatch_no_mutation(
    mcp_server: ServerFactory, clockify_env: dict[str, str], upstream: Upstream
) -> None:
    upstream.task_project = "other"
    async with mcp_server(clockify_env) as session:
        result = await session.call_tool(
            "clockify_create_time_entry",
            {
                "start": START,
                "end": END,
                "description": "work",
                "billable": False,
                "project_id": "p-1",
                "task_id": "t-1",
            },
        )
    assert (
        result.structuredContent is not None
        and result.structuredContent["code"] == "invalid_argument"
    )
    assert upstream.posts == []
