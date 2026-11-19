"""``clockify_jira_sync`` over stdio, against a fake Clockify and Jira.

The MCP server is the real ``python -m jake_tools.mcp`` process (see
``conftest.py``), and it talks to Clockify and Jira with the real
``ClockifyClient`` and ``JiraClient`` over real HTTP. The only fake is the
external boundary itself: a local ``http.server`` that speaks just enough of
both REST APIs, reached through ``CLOCKIFY_API_BASE_URL`` and
``JIRA_BASE_URL``, with dummy credential values it checks on every request.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from click.testing import CliRunner
from mcp import ClientSession
from mcp.types import CallToolResult, TextContent

from jake_tools.cli.clockify import clockify as clockify_cli

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]
EnvFactory = Callable[..., dict[str, str]]

CLOCKIFY_KEY = "dummy-clockify-key-8f3a1c"
JIRA_EMAIL = "dummy-jira-email-2b7d@example.test"
JIRA_TOKEN = "dummy-jira-token-c94e05"
WORKSPACE = "ws-1"


# --- The fake upstream ----------------------------------------------------------


@dataclass
class FakeUpstream:
    """Mutable Clockify + Jira state behind one local HTTP server.

    The default inventory needs two task writes: SF-427 is DONE in Clockify
    but in progress and assigned in Jira (reactivate), and SF-304 is ACTIVE
    in Clockify but Done in Jira (mark done).
    """

    base_url: str = ""
    projects: dict[str, dict[str, Any]] = field(default_factory=dict)
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    issues: dict[str, dict[str, Any]] = field(default_factory=dict)
    assigned_active: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    fail_write_number: int | None = None
    # What a single-task read returns instead of the stored field values, by
    # task ID: a write Clockify acknowledged but did not persist as asked.
    readback: dict[str, dict[str, str]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def clockify_url(self) -> str:
        return f"{self.base_url}/clockify/api/v1"

    @property
    def jira_url(self) -> str:
        return f"{self.base_url}/jira"

    def seed_default(self) -> None:
        self.projects["p-131"] = {
            "id": "p-131",
            "name": "Production Vehicle",
            "note": "Jira: SF-131",
            "archived": False,
            "billable": False,
            "color": "#689F38",
            "public": True,
            "clientId": "client-1",
        }
        self.tasks["t-304"] = {
            "id": "t-304",
            "name": "SF-304 Bench test PX4",
            "projectId": "p-131",
            "status": "ACTIVE",
        }
        self.tasks["t-427"] = {
            "id": "t-427",
            "name": "SF-427 Evaluate PX4 external control methods",
            "projectId": "p-131",
            "status": "DONE",
        }
        self.issues["SF-131"] = _jira_issue(
            "SF-131", "Production Vehicle", "In Progress", "Project / Phase"
        )
        self.issues["SF-304"] = _jira_issue("SF-304", "Bench test PX4", "Done", "Task")
        self.issues["SF-427"] = _jira_issue(
            "SF-427", "Evaluate PX4 external control methods", "In Progress", "Task"
        )
        self.assigned_active = ["SF-427"]

    def env(self) -> dict[str, str]:
        return {
            "CLOCKIFY_API_KEY": CLOCKIFY_KEY,
            "CLOCKIFY_API_BASE_URL": self.clockify_url,
            "JIRA_BASE_URL": self.jira_url,
            "JIRA_EMAIL": JIRA_EMAIL,
            "JIRA_API_TOKEN": JIRA_TOKEN,
        }


def _jira_issue(
    key: str,
    summary: str,
    status: str,
    issue_type: str,
    assignee: str = "Michael Bryan",
) -> dict[str, Any]:
    category = {
        "Done": "Done",
        "Won’t Do": "Done",
        "To Do": "To Do",
        "Backlog": "To Do",
        "TRIAGE": "To Do",
        "Investigate": "To Do",
    }.get(status, "In Progress")
    fields: dict[str, Any] = {
        "summary": summary,
        "status": {"name": status, "statusCategory": {"name": category}},
        "assignee": {"displayName": assignee},
        "issuetype": {"name": issue_type},
    }
    if issue_type != "Project / Phase":
        fields["parent"] = {
            "key": "SF-131",
            "fields": {"summary": "Production Vehicle"},
        }
    return {"key": key, "fields": fields}


def _handler_for(upstream: FakeUpstream) -> type[BaseHTTPRequestHandler]:
    expected_basic = "Basic " + base64.b64encode(
        f"{JIRA_EMAIL}:{JIRA_TOKEN}".encode()
    ).decode("ascii")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # quiet
            return

        def _send(self, status: int, body: object) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length)) if length else None

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def do_PUT(self) -> None:
            self._dispatch("PUT")

        def _dispatch(self, method: str) -> None:
            url = urlsplit(self.path)
            query = {key: values[0] for key, values in parse_qs(url.query).items()}
            body = self._body()
            with upstream.lock:
                if url.path.startswith("/clockify/api/v1/"):
                    if self.headers.get("X-Api-Key") != CLOCKIFY_KEY:
                        return self._send(401, {"message": "bad key"})
                    path = url.path.removeprefix("/clockify/api/v1")
                    status, payload = _clockify(upstream, method, path, query, body)
                elif url.path.startswith("/jira/"):
                    if self.headers.get("Authorization") != expected_basic:
                        return self._send(401, {"errorMessages": ["bad auth"]})
                    path = url.path.removeprefix("/jira")
                    status, payload = _jira(upstream, method, path, body)
                else:
                    status, payload = 404, {"message": "unknown"}
            self._send(status, payload)

    return Handler


def _clockify(
    upstream: FakeUpstream,
    method: str,
    path: str,
    query: dict[str, str],
    body: Any,
) -> tuple[int, object]:
    if (method, path) == ("GET", "/user"):
        return 200, {"id": "user-1", "activeWorkspace": WORKSPACE}
    ws = f"/workspaces/{WORKSPACE}"
    if (method, path) == ("GET", f"{ws}/clients"):
        page = [{"id": "client-1", "name": "Sunfish Robotics", "archived": False}]
        return 200, page if query.get("page") == "1" else []
    if (method, path) == ("GET", f"{ws}/projects"):
        archived = query.get("archived") == "true"
        page = [p for p in upstream.projects.values() if p["archived"] is archived]
        return 200, page if query.get("page") == "1" else []
    if match := re.fullmatch(rf"{ws}/projects/([^/]+)/tasks", path):
        project_id = match.group(1)
        if method == "GET":
            want = "ACTIVE" if query.get("is-active") == "true" else "DONE"
            page = [
                t
                for t in upstream.tasks.values()
                if t["projectId"] == project_id and t["status"] == want
            ]
            return 200, page if query.get("page") == "1" else []
        if method == "POST":
            upstream.writes.append(f"POST task {body['name']}")
            if upstream.fail_write_number == len(upstream.writes):
                return 500, {"message": "simulated Clockify outage"}
            created = {
                "id": f"t-new-{len(upstream.writes)}",
                "name": body["name"],
                "projectId": project_id,
                "status": "ACTIVE",
            }
            upstream.tasks[created["id"]] = created
            return 201, created
    if match := re.fullmatch(rf"{ws}/projects/([^/]+)/tasks/([^/]+)", path):
        task = upstream.tasks.get(match.group(2))
        if task is None:
            return 404, {"message": "no such task"}
        if method == "GET":
            return 200, {**task, **upstream.readback.get(task["id"], {})}
        if method == "PUT":
            upstream.writes.append(f"PUT task {task['id']}")
            if upstream.fail_write_number == len(upstream.writes):
                return 500, {"message": "simulated Clockify outage"}
            task.update(name=body["name"], status=body["status"])
            return 200, task
    if match := re.fullmatch(rf"{ws}/projects/([^/]+)", path):
        project = upstream.projects.get(match.group(1))
        if project is None:
            return 404, {"message": "no such project"}
        if method == "GET":
            return 200, project
    return 404, {"message": f"unhandled {method} {path}"}


def _jira(
    upstream: FakeUpstream, method: str, path: str, body: Any
) -> tuple[int, object]:
    if (method, path) == ("POST", "/rest/api/3/search/jql"):
        jql: str = body["jql"]
        if match := re.match(r"key in \(([^)]*)\)", jql):
            keys = [key for key in match.group(1).split(",") if key]
        else:
            match = re.search(r"status in \(([^)]*)\)", jql)
            assert match is not None, jql
            statuses = re.findall(r'"([^"]*)"', match.group(1))
            keys = [
                key
                for key in upstream.assigned_active
                if upstream.issues[key]["fields"]["status"]["name"] in statuses
            ]
        return 200, {
            "issues": [upstream.issues[key] for key in keys if key in upstream.issues]
        }
    if match := re.fullmatch(r"/rest/api/3/issue/([A-Z0-9-]+)", path):
        issue = upstream.issues.get(match.group(1))
        return (200, issue) if issue else (404, {"errorMessages": ["no issue"]})
    return 404, {"errorMessages": [f"unhandled {method} {path}"]}


@pytest.fixture
def upstream() -> Iterator[FakeUpstream]:
    state = FakeUpstream()
    state.seed_default()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(state))
    state.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def sync_env(
    upstream: FakeUpstream, mcp_env_factory: EnvFactory, mcp_home: Path
) -> dict[str, str]:
    return mcp_env_factory(mcp_home, **upstream.env())


async def _call(
    mcp_server: ServerFactory, env: dict[str, str], **arguments: Any
) -> CallToolResult:
    async with mcp_server(env) as session:
        return await session.call_tool("clockify_jira_sync", arguments)


def _ok(result: CallToolResult) -> dict[str, Any]:
    assert result.is_error is False, result.structured_content
    assert result.structured_content is not None
    return result.structured_content


def _err(result: CallToolResult, code: str) -> dict[str, Any]:
    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == code, result.structured_content
    return result.structured_content


# --- Discovery and arguments ----------------------------------------------------


async def test_tool_is_listed_with_its_input_schema(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
) -> None:
    async with mcp_server(mcp_env) as session:
        tools = {tool.name: tool for tool in (await session.list_tools()).tools}

    tool = tools["clockify_jira_sync"]
    properties = tool.input_schema["properties"]
    assert set(properties) == {"issues", "apply", "plan_digest"}
    assert properties["apply"]["default"] is False
    assert tool.input_schema.get("required", []) == []
    assert tool.output_schema is None
    assert tool.description is not None
    assert "planned or underway" in tool.description
    assert "plan_digest" in tool.description
    assert "Never deletes" in tool.description
    # The explicit-versus-automatic distinction the guide repeats.
    assert "regardless of assignee or Jira status" in tool.description
    assert "CREATE_TASK followed by MARK_TASK_DONE" in tool.description


async def test_apply_without_digest_is_invalid_and_touches_nothing(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    error = _err(await _call(mcp_server, sync_env, apply=True), "invalid_argument")

    assert error["detail"] == {"argument": "plan_digest"}
    assert upstream.writes == []


async def test_malformed_issue_key_is_invalid_and_named(
    mcp_server: ServerFactory, sync_env: dict[str, str]
) -> None:
    error = _err(
        await _call(mcp_server, sync_env, issues=["SF-1", "not a key"]),
        "invalid_argument",
    )

    assert "'not a key'" in error["message"]
    assert error["detail"] == {"argument": "issues", "key": "not a key"}


@pytest.mark.parametrize(
    "missing",
    ["CLOCKIFY_API_KEY", "JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN"],
)
async def test_missing_credential_is_named_and_no_secret_is_echoed(
    mcp_server: ServerFactory,
    mcp_env_factory: EnvFactory,
    mcp_home: Path,
    upstream: FakeUpstream,
    missing: str,
) -> None:
    present = {k: v for k, v in upstream.env().items() if k != missing}
    env = mcp_env_factory(mcp_home, **present)

    result = await _call(mcp_server, env)

    error = _err(result, "missing_credentials")
    assert missing in error["message"]
    assert error["detail"] == {"variables": [missing]}
    wire = json.dumps(result.model_dump(mode="json"))
    for secret in (CLOCKIFY_KEY, JIRA_EMAIL, JIRA_TOKEN, upstream.jira_url):
        assert secret not in wire
    assert upstream.writes == []


async def test_upstream_failure_is_upstream_error_without_secrets(
    mcp_server: ServerFactory,
    mcp_env_factory: EnvFactory,
    mcp_home: Path,
    upstream: FakeUpstream,
) -> None:
    env = mcp_env_factory(
        mcp_home, **{**upstream.env(), "JIRA_API_TOKEN": "dummy-wrong-token-77"}
    )

    result = await _call(mcp_server, env)

    error = _err(result, "upstream_error")
    # Operation and status only: no host, no response body.
    assert error["message"] == (
        "Jira error: POST /rest/api/3/search/jql returned HTTP 401"
    )
    assert error["detail"] == {"integration": "jira"}
    wire = json.dumps(result.model_dump(mode="json"))
    jira_host = urlsplit(upstream.jira_url).netloc
    for secret in (
        CLOCKIFY_KEY,
        JIRA_EMAIL,
        "dummy-wrong-token-77",
        upstream.jira_url,
        jira_host,
        "bad auth",
    ):
        assert secret not in wire


async def test_unreachable_jira_host_is_named_by_error_class_only(
    mcp_server: ServerFactory,
    mcp_env_factory: EnvFactory,
    mcp_home: Path,
    upstream: FakeUpstream,
    capfd: pytest.CaptureFixture[str],
) -> None:
    # .invalid never resolves (RFC 2606), so requests raises ConnectionError
    # whose text names the host.
    env = mcp_env_factory(
        mcp_home,
        **{**upstream.env(), "JIRA_BASE_URL": "secret-tenant.invalid/"},
    )

    result = await _call(mcp_server, env)

    error = _err(result, "upstream_error")
    assert error["message"] == (
        "Jira error: POST /rest/api/3/search/jql failed: ConnectionError"
    )
    wire = json.dumps(result.model_dump(mode="json"))
    assert "secret-tenant" not in wire
    # The server's stderr log carries the same sanitised message.
    stderr = capfd.readouterr().err
    assert "Jira error: POST /rest/api/3/search/jql failed" in stderr
    assert "secret-tenant" not in stderr


async def test_short_secret_values_are_not_redacted_into_the_message(
    mcp_server: ServerFactory,
    mcp_env_factory: EnvFactory,
    mcp_home: Path,
    upstream: FakeUpstream,
) -> None:
    env = mcp_env_factory(mcp_home, **{**upstream.env(), "CLOCKIFY_API_KEY": "k"})

    error = _err(await _call(mcp_server, env), "upstream_error")

    assert error["message"] == "Clockify error: GET /user returned HTTP 401"


# --- Preview ------------------------------------------------------------------


async def test_preview_returns_the_sync_report_and_writes_nothing(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    report = _ok(await _call(mcp_server, sync_env))

    assert report["mode"] == "preview"
    assert report["applied"] is False
    assert report["failure"] is None
    assert report["workspace_id"] == WORKSPACE
    assert report["client_id"] == "client-1"
    assert report["jira_project"] == "SF"
    assert report["clockify_client"] == "Sunfish Robotics"
    assert report["scope"] == {
        "kind": "assigned-active",
        "jira_project": "SF",
        "issue_keys": [],
    }
    assert report["inventory"] == {
        "active_issues": 1,
        "jira_issues": 3,
        "projects": 1,
        "tasks": 2,
    }
    assert [
        (a["kind"], a["jira_key"], a["task_id"], a["applied"], a["verified"])
        for a in report["actions"]
    ] == [
        ("REACTIVATE_TASK", "SF-427", "t-427", False, False),
        ("MARK_TASK_DONE", "SF-304", "t-304", False, False),
    ]
    assert report["conflicts"] == []
    assert re.fullmatch(r"[0-9a-f]{64}", report["plan_digest"])
    assert upstream.writes == []


async def test_preview_scoped_to_issues(
    mcp_server: ServerFactory, sync_env: dict[str, str]
) -> None:
    report = _ok(await _call(mcp_server, sync_env, issues=["sf-304"]))

    assert report["scope"]["kind"] == "issues"
    assert report["scope"]["issue_keys"] == ["SF-304"]
    assert [(a["kind"], a["jira_key"]) for a in report["actions"]] == [
        ("MARK_TASK_DONE", "SF-304")
    ]


# --- Apply --------------------------------------------------------------------


@pytest.mark.parametrize("selected_status", ["To Do", "In Progress"])
@pytest.mark.parametrize("finished_status", ["Done", "Won’t Do"])
async def test_apply_with_the_preview_digest_applies_and_verifies(
    mcp_server: ServerFactory,
    sync_env: dict[str, str],
    upstream: FakeUpstream,
    selected_status: str,
    finished_status: str,
) -> None:
    upstream.issues["SF-427"] = _jira_issue(
        "SF-427", "Evaluate PX4 external control methods", selected_status, "Task"
    )
    upstream.issues["SF-304"] = _jira_issue(
        "SF-304", "Bench test PX4", finished_status, "Task"
    )
    async with mcp_server(sync_env) as session:
        preview = _ok(await session.call_tool("clockify_jira_sync", {}))
        result = await session.call_tool(
            "clockify_jira_sync",
            {"apply": True, "plan_digest": preview["plan_digest"]},
        )

    report = _ok(result)
    assert report["mode"] == "apply"
    assert report["applied"] is True
    assert report["failure"] is None
    assert report["plan_digest"] == preview["plan_digest"]
    assert [
        (a["jira_key"], a["applied"], a["verified"]) for a in report["actions"]
    ] == [
        ("SF-427", True, True),
        ("SF-304", True, True),
    ]
    assert upstream.writes == ["PUT task t-427", "PUT task t-304"]
    assert upstream.tasks["t-427"]["status"] == "ACTIVE"
    assert upstream.tasks["t-304"]["status"] == "DONE"


@pytest.mark.parametrize("status", ["Backlog", "TRIAGE", "Investigate"])
@pytest.mark.parametrize("renamed", [False, True])
async def test_backlog_excluded_from_creation_and_deferral_preserves_existing_task(
    mcp_server: ServerFactory,
    sync_env: dict[str, str],
    upstream: FakeUpstream,
    status: str,
    renamed: bool,
) -> None:
    upstream.tasks.pop("t-304")
    upstream.tasks["t-427"]["status"] = "ACTIVE"
    summary = (
        "Deferred investigation" if renamed else "Evaluate PX4 external control methods"
    )
    upstream.issues["SF-427"] = _jira_issue("SF-427", summary, status, "Task")
    upstream.issues["SF-500"] = _jira_issue("SF-500", "Future work", status, "Task")
    upstream.assigned_active = ["SF-427", "SF-500"]

    async with mcp_server(sync_env) as session:
        preview = _ok(await session.call_tool("clockify_jira_sync", {}))
        assert preview["inventory"]["active_issues"] == 0
        assert preview["conflicts"] == []
        assert [(a["kind"], a["jira_key"]) for a in preview["actions"]] == (
            [("RENAME_TASK", "SF-427")] if renamed else []
        )
        report = _ok(
            await session.call_tool(
                "clockify_jira_sync",
                {"apply": True, "plan_digest": preview["plan_digest"]},
            )
        )

    assert report["failure"] is None
    assert all(a["applied"] and a["verified"] for a in report["actions"])
    assert upstream.writes == (["PUT task t-427"] if renamed else [])
    assert upstream.tasks == {
        "t-427": {
            "id": "t-427",
            "name": f"SF-427 {summary}",
            "projectId": "p-131",
            "status": "ACTIVE",
        }
    }


async def test_apply_against_a_changed_upstream_is_plan_stale_and_writes_nothing(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    async with mcp_server(sync_env) as session:
        preview = _ok(await session.call_tool("clockify_jira_sync", {}))
        # Someone renames the Jira issue between preview and apply.
        upstream.issues["SF-427"]["fields"]["summary"] = "Evaluate PX4 offboard mode"
        result = await session.call_tool(
            "clockify_jira_sync",
            {"apply": True, "plan_digest": preview["plan_digest"]},
        )

    error = _err(result, "plan_stale")
    assert error["detail"]["expected"] == preview["plan_digest"]
    assert error["detail"]["actual"] != preview["plan_digest"]
    assert re.fullmatch(r"[0-9a-f]{64}", error["detail"]["actual"])
    assert upstream.writes == []


async def test_partial_failure_returns_a_report_with_the_failure(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    upstream.fail_write_number = 2
    async with mcp_server(sync_env) as session:
        preview = _ok(await session.call_tool("clockify_jira_sync", {}))
        result = await session.call_tool(
            "clockify_jira_sync",
            {"apply": True, "plan_digest": preview["plan_digest"]},
        )

    report = _ok(result)
    assert report["mode"] == "apply"
    assert [
        (a["jira_key"], a["applied"], a["verified"]) for a in report["actions"]
    ] == [
        ("SF-427", True, True),
        ("SF-304", False, False),
    ]
    failure = report["failure"]
    assert failure["jira_key"] == "SF-304"
    assert failure["kind"] == "MARK_TASK_DONE"
    assert failure["message"] == (
        "Clockify error: PUT /workspaces/ws-1/projects/p-131/tasks/t-304 "
        "returned HTTP 500"
    )
    assert "simulated Clockify outage" not in json.dumps(report)
    assert upstream.writes == ["PUT task t-427", "PUT task t-304"]
    assert upstream.tasks["t-427"]["status"] == "ACTIVE"
    assert upstream.tasks["t-304"]["status"] == "ACTIVE"


async def test_conflicts_block_apply(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    upstream.tasks["t-304-dup"] = {**upstream.tasks["t-304"], "id": "t-304-dup"}
    async with mcp_server(sync_env) as session:
        preview = _ok(await session.call_tool("clockify_jira_sync", {}))
        result = await session.call_tool(
            "clockify_jira_sync",
            {"apply": True, "plan_digest": preview["plan_digest"]},
        )

    assert [c["jira_key"] for c in preview["conflicts"]] == ["SF-304"]
    error = _err(result, "invalid_argument")
    assert error["detail"] == {"conflicts": ["SF-304"]}
    assert upstream.writes == []


# --- Explicit issues override discovery filters ----------------------------------

SF691_SUMMARY = "Seaworthiness tab: sensor health report and thruster test"
SF691_TASK = f"SF-691 {SF691_SUMMARY}"


def _add_issue_assigned_elsewhere(upstream: FakeUpstream, status: str) -> None:
    upstream.issues["SF-691"] = _jira_issue(
        "SF-691", SF691_SUMMARY, status, "Task", assignee="David Htet"
    )


def _summary(report: dict[str, Any]) -> list[tuple[str, str]]:
    return [(a["kind"], a["jira_key"]) for a in report["actions"]]


async def _preview_and_apply(
    session: ClientSession, **arguments: Any
) -> tuple[dict[str, Any], CallToolResult]:
    preview = _ok(await session.call_tool("clockify_jira_sync", arguments))
    applied = await session.call_tool(
        "clockify_jira_sync",
        {**arguments, "apply": True, "plan_digest": preview["plan_digest"]},
    )
    return preview, applied


@pytest.mark.parametrize("status", ["Done", "Won’t Do"])
async def test_explicit_finished_issue_assigned_elsewhere_is_created_completed(
    mcp_server: ServerFactory,
    sync_env: dict[str, str],
    upstream: FakeUpstream,
    status: str,
) -> None:
    _add_issue_assigned_elsewhere(upstream, status)
    async with mcp_server(sync_env) as session:
        preview = _ok(
            await session.call_tool("clockify_jira_sync", {"issues": ["SF-691"]})
        )

        # Preview is read-only and names both writes, in order.
        assert upstream.writes == []
        assert preview["scope"]["kind"] == "issues"
        assert preview["inventory"]["active_issues"] == 1
        assert preview["conflicts"] == []
        assert _summary(preview) == [
            ("CREATE_TASK", "SF-691"),
            ("MARK_TASK_DONE", "SF-691"),
        ]
        create, complete = preview["actions"]
        assert create["project_key"] == "SF-131"
        assert create["desired_name"] == SF691_TASK
        assert complete["jira_status"] == status
        assert complete["task_id"] is None

        applied = _ok(
            await session.call_tool(
                "clockify_jira_sync",
                {
                    "issues": ["SF-691"],
                    "apply": True,
                    "plan_digest": preview["plan_digest"],
                },
            )
        )

    assert applied["failure"] is None
    assert [(a["kind"], a["applied"], a["verified"]) for a in applied["actions"]] == [
        ("CREATE_TASK", True, True),
        ("MARK_TASK_DONE", True, True),
    ]
    # The report names the record each write landed on.
    assert [a["task_id"] for a in applied["actions"]] == ["t-new-1", "t-new-1"]
    assert upstream.writes == [f"POST task {SF691_TASK}", "PUT task t-new-1"]
    assert upstream.tasks["t-new-1"] == {
        "id": "t-new-1",
        "name": SF691_TASK,
        "projectId": "p-131",
        "status": "DONE",
    }


async def test_completed_creation_digest_covers_the_final_state(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    _add_issue_assigned_elsewhere(upstream, "Done")
    async with mcp_server(sync_env) as session:
        preview = _ok(
            await session.call_tool("clockify_jira_sync", {"issues": ["SF-691"]})
        )
        # Someone reopens the issue after the plan was reviewed.
        _add_issue_assigned_elsewhere(upstream, "In Progress")
        result = await session.call_tool(
            "clockify_jira_sync",
            {
                "issues": ["SF-691"],
                "apply": True,
                "plan_digest": preview["plan_digest"],
            },
        )

    _err(result, "plan_stale")
    assert upstream.writes == []


@pytest.mark.parametrize("status", ["Backlog", "TRIAGE", "Investigate"])
async def test_explicit_deferred_issue_is_created_and_stays_active(
    mcp_server: ServerFactory,
    sync_env: dict[str, str],
    upstream: FakeUpstream,
    status: str,
) -> None:
    _add_issue_assigned_elsewhere(upstream, status)
    async with mcp_server(sync_env) as session:
        preview, result = await _preview_and_apply(session, issues=["SF-691"])

    assert _summary(preview) == [("CREATE_TASK", "SF-691")]
    assert _ok(result)["failure"] is None
    assert upstream.tasks["t-new-1"]["status"] == "ACTIVE"
    assert upstream.writes == [f"POST task {SF691_TASK}"]


async def test_explicit_issue_already_in_clockify_is_not_duplicated(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    _add_issue_assigned_elsewhere(upstream, "Done")
    upstream.tasks["t-691"] = {
        "id": "t-691",
        "name": "SF-691 Seaworthiness tab",
        "projectId": "p-131",
        "status": "ACTIVE",
    }
    async with mcp_server(sync_env) as session:
        preview, result = await _preview_and_apply(session, issues=["SF-691"])
        # Applied, the same request now has nothing left to do.
        again = _ok(
            await session.call_tool("clockify_jira_sync", {"issues": ["SF-691"]})
        )

    assert _summary(preview) == [("MARK_TASK_DONE", "SF-691")]
    assert preview["actions"][0]["task_id"] == "t-691"
    assert _ok(result)["failure"] is None
    assert again["actions"] == [] and again["conflicts"] == []
    assert upstream.writes == ["PUT task t-691"]
    assert upstream.tasks["t-691"] == {
        "id": "t-691",
        "name": SF691_TASK,
        "projectId": "p-131",
        "status": "DONE",
    }


async def test_automatic_discovery_never_creates_finished_or_unassigned_work(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    # Done and assigned elsewhere: outside discovery, so it stays outside.
    _add_issue_assigned_elsewhere(upstream, "Done")
    # To Do and assigned to the user: discovery still creates it.
    upstream.issues["SF-700"] = _jira_issue("SF-700", "Planned work", "To Do", "Task")
    upstream.assigned_active = ["SF-427", "SF-700"]
    async with mcp_server(sync_env) as session:
        preview = _ok(await session.call_tool("clockify_jira_sync", {}))

    assert preview["scope"]["kind"] == "assigned-active"
    assert ("CREATE_TASK", "SF-700") in _summary(preview)
    assert "SF-691" not in {a["jira_key"] for a in preview["actions"]}


async def test_explicit_finished_issue_with_unresolvable_parent_is_a_keyed_conflict(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    _add_issue_assigned_elsewhere(upstream, "Done")
    upstream.issues["SF-691"]["fields"]["parent"]["key"] = "SF-777"  # not in Jira
    async with mcp_server(sync_env) as session:
        preview, result = await _preview_and_apply(session, issues=["SF-691"])

    assert preview["actions"] == []
    assert [
        (c["jira_key"], "SF-777" in c["message"]) for c in preview["conflicts"]
    ] == [("SF-691", True)]
    assert _err(result, "invalid_argument")["detail"] == {"conflicts": ["SF-691"]}
    assert upstream.writes == []


async def test_explicit_finished_issue_with_duplicate_task_is_a_keyed_conflict(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    _add_issue_assigned_elsewhere(upstream, "Done")
    for task_id in ("t-691-a", "t-691-b"):
        upstream.tasks[task_id] = {
            "id": task_id,
            "name": SF691_TASK,
            "projectId": "p-131",
            "status": "ACTIVE",
        }
    async with mcp_server(sync_env) as session:
        preview, result = await _preview_and_apply(session, issues=["SF-691"])

    assert [c["jira_key"] for c in preview["conflicts"]] == ["SF-691"]
    assert "multiple active Clockify tasks" in preview["conflicts"][0]["message"]
    assert _err(result, "invalid_argument")["detail"] == {"conflicts": ["SF-691"]}
    assert upstream.writes == []


async def test_explicit_issue_missing_from_jira_is_an_error_naming_the_key(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    _add_issue_assigned_elsewhere(upstream, "Done")
    result = await _call(mcp_server, sync_env, issues=["SF-691", "SF-999"])

    error = _err(result, "upstream_error")
    assert error["detail"] == {"integration": "jira"}
    assert "/rest/api/3/issue/SF-999" in error["message"]
    assert upstream.writes == []


# --- Completed creation: partial failures ---------------------------------------


async def test_creation_failure_leaves_nothing_and_skips_the_completion(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    _add_issue_assigned_elsewhere(upstream, "Done")
    upstream.fail_write_number = 1
    async with mcp_server(sync_env) as session:
        _, result = await _preview_and_apply(session, issues=["SF-691"])

    report = _ok(result)
    assert [(a["kind"], a["applied"]) for a in report["actions"]] == [
        ("CREATE_TASK", False),
        ("MARK_TASK_DONE", False),
    ]
    assert report["failure"]["kind"] == "CREATE_TASK"
    assert report["failure"]["jira_key"] == "SF-691"
    assert upstream.writes == [f"POST task {SF691_TASK}"]
    assert "t-new-1" not in upstream.tasks


async def test_creation_readback_mismatch_names_the_created_task_and_stops(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    _add_issue_assigned_elsewhere(upstream, "Done")
    upstream.readback["t-new-1"] = {"name": "SF-691 something else"}
    async with mcp_server(sync_env) as session:
        _, result = await _preview_and_apply(session, issues=["SF-691"])

    report = _ok(result)
    assert [(a["kind"], a["applied"]) for a in report["actions"]] == [
        ("CREATE_TASK", False),
        ("MARK_TASK_DONE", False),
    ]
    assert report["failure"]["kind"] == "CREATE_TASK"
    # The write may have landed; the message says which task to inspect.
    assert "t-new-1" in report["failure"]["message"]
    assert upstream.writes == [f"POST task {SF691_TASK}"]
    assert upstream.tasks["t-new-1"]["status"] == "ACTIVE"


async def test_completion_failure_after_creation_keeps_the_task_for_recovery(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    _add_issue_assigned_elsewhere(upstream, "Done")
    upstream.fail_write_number = 2
    async with mcp_server(sync_env) as session:
        _, result = await _preview_and_apply(session, issues=["SF-691"])
        upstream.fail_write_number = None
        retry_preview, retry = await _preview_and_apply(session, issues=["SF-691"])

    report = _ok(result)
    assert [(a["kind"], a["applied"], a["verified"]) for a in report["actions"]] == [
        ("CREATE_TASK", True, True),
        ("MARK_TASK_DONE", False, False),
    ]
    assert report["actions"][0]["task_id"] == "t-new-1"
    assert report["failure"]["kind"] == "MARK_TASK_DONE"
    assert report["failure"]["jira_key"] == "SF-691"
    # A fresh preview finds the created task: it completes it, creates nothing.
    assert _summary(retry_preview) == [("MARK_TASK_DONE", "SF-691")]
    assert retry_preview["actions"][0]["task_id"] == "t-new-1"
    assert _ok(retry)["failure"] is None
    assert upstream.writes == [
        f"POST task {SF691_TASK}",
        "PUT task t-new-1",
        "PUT task t-new-1",
    ]
    assert upstream.tasks["t-new-1"]["status"] == "DONE"


async def test_completion_readback_mismatch_after_creation_is_reported(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    _add_issue_assigned_elsewhere(upstream, "Done")
    # Clockify acknowledges the completion but reads the task back as active
    # (which is also what the creation readback and the update guard expect).
    upstream.readback["t-new-1"] = {"status": "ACTIVE"}
    async with mcp_server(sync_env) as session:
        _, result = await _preview_and_apply(session, issues=["SF-691"])

    report = _ok(result)
    assert [(a["kind"], a["applied"]) for a in report["actions"]] == [
        ("CREATE_TASK", True),
        ("MARK_TASK_DONE", False),
    ]
    assert report["failure"]["kind"] == "MARK_TASK_DONE"
    assert "t-new-1" in report["failure"]["message"]
    assert upstream.writes == [f"POST task {SF691_TASK}", "PUT task t-new-1"]


# --- CLI parity ---------------------------------------------------------------


async def test_cli_dry_run_json_equals_the_tool_preview(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    tool_result = await _call(mcp_server, sync_env)
    cli = CliRunner().invoke(
        clockify_cli,
        ["jira-sync", "--dry-run", "--json"],
        env=upstream.env(),
    )

    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.output) == _ok(tool_result)
    # The same payload in the tool's text content (compact JSON for a model
    # host) as on the CLI's stdout (indented for a person).
    text = tool_result.content[0]
    assert isinstance(text, TextContent)
    assert json.loads(text.text) == json.loads(cli.output)
    assert upstream.writes == []


def test_cli_text_partial_failure_lists_applied_actions_before_the_error(
    upstream: FakeUpstream,
) -> None:
    upstream.fail_write_number = 2

    cli = CliRunner().invoke(clockify_cli, ["jira-sync", "--apply"], env=upstream.env())

    assert cli.exit_code == 1
    lines = cli.output.splitlines()
    assert lines[:2] == [
        "Applied before failure: 1 change",
        "REACTIVATE_TASK SF-427 \u2014 reactivate as "
        "'SF-427 Evaluate PX4 external control methods'",
    ]
    assert lines[2].startswith(
        "Error: Clockify request failed for PUT "
        "/workspaces/ws-1/projects/p-131/tasks/t-304: 500"
    )


# --- Live: the real account ---------------------------------------------------

_LIVE_VARIABLES = ("CLOCKIFY_API_KEY", "JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN")


@pytest.mark.live
async def test_live_preview_matches_cli_dry_run_json_byte_for_byte(
    mcp_server: ServerFactory, mcp_env_factory: EnvFactory, mcp_home: Path
) -> None:
    """The credentialled acceptance check from the design.

    Reads the real Clockify and Jira accounts (read-only: preview and
    ``--dry-run`` only). Skipped unless every credential is in the
    environment.
    """
    missing = [name for name in _LIVE_VARIABLES if not os.environ.get(name)]
    if missing:
        pytest.skip(f"needs {', '.join(missing)}")
    passthrough = {
        name: os.environ[name]
        for name in (*_LIVE_VARIABLES, "CLOCKIFY_API_BASE_URL")
        if os.environ.get(name)
    }
    env = mcp_env_factory(mcp_home, **passthrough)

    async with mcp_server(env) as session:
        result = await session.call_tool("clockify_jira_sync", {})
    completed = subprocess.run(
        [sys.executable, "-m", "jake_tools", "clockify", "jira-sync"]
        + ["--dry-run", "--json"],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        check=False,
    )

    assert result.is_error is False, result.structured_content
    text = result.content[0]
    assert isinstance(text, TextContent)
    assert completed.stdout == f"{text.text}\n"
