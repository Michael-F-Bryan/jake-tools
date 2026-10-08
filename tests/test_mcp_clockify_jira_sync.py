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


def _jira_issue(key: str, summary: str, status: str, issue_type: str) -> dict[str, Any]:
    category = {"Done": "Done", "To Do": "To Do"}.get(status, "In Progress")
    fields: dict[str, Any] = {
        "summary": summary,
        "status": {"name": status, "statusCategory": {"name": category}},
        "assignee": {"displayName": "Michael Bryan"},
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
    if match := re.fullmatch(rf"{ws}/projects/([^/]+)/tasks/([^/]+)", path):
        task = upstream.tasks.get(match.group(2))
        if task is None:
            return 404, {"message": "no such task"}
        if method == "GET":
            return 200, task
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
            keys = list(upstream.assigned_active)
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
    assert result.isError is False, result.structuredContent
    assert result.structuredContent is not None
    return result.structuredContent


def _err(result: CallToolResult, code: str) -> dict[str, Any]:
    assert result.isError is True
    assert result.structuredContent is not None
    assert result.structuredContent["code"] == code, result.structuredContent
    return result.structuredContent


# --- Discovery and arguments ----------------------------------------------------


async def test_tool_is_listed_with_its_input_schema(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
) -> None:
    async with mcp_server(mcp_env) as session:
        tools = {tool.name: tool for tool in (await session.list_tools()).tools}

    tool = tools["clockify_jira_sync"]
    properties = tool.inputSchema["properties"]
    assert set(properties) == {"issues", "apply", "plan_digest"}
    assert properties["apply"]["default"] is False
    assert tool.inputSchema.get("required", []) == []
    assert tool.outputSchema is None
    assert tool.description is not None
    assert "plan_digest" in tool.description
    assert "Never deletes" in tool.description


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
    assert error["message"].startswith("Jira error:")
    assert "401" in error["message"]
    assert error["detail"] == {"integration": "jira"}
    wire = json.dumps(result.model_dump(mode="json"))
    for secret in (CLOCKIFY_KEY, JIRA_EMAIL, "dummy-wrong-token-77"):
        assert secret not in wire


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


async def test_apply_with_the_preview_digest_applies_and_verifies(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
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
    assert "500" in failure["message"]
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
    # Same serialisation too: the tool's text content is the CLI's stdout.
    text = tool_result.content[0]
    assert isinstance(text, TextContent)
    assert cli.output == f"{text.text}\n"
    assert upstream.writes == []


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

    assert result.isError is False, result.structuredContent
    text = result.content[0]
    assert isinstance(text, TextContent)
    assert completed.stdout == f"{text.text}\n"
