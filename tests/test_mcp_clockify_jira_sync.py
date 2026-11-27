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
    # Workspace members (``status`` ACTIVE or INACTIVE) and their time entries,
    # in Clockify's TimeEntryWithRatesDtoV1 shape, by entry ID.
    users: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"id": "user-1", "name": "Michael", "status": "ACTIVE"}
        ]
    )
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    # A write that lands and then answers HTTP 500 (an uncertain write).
    land_then_fail_write_number: int | None = None
    # Rates Clockify assigns an entry when it is moved into a project, by
    # project ID (simulates the undocumented rate recomputation).
    entry_rates_on_move: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Run after the numbered write lands: Clockify changing a record behind it.
    after_write: dict[int, Callable[[FakeUpstream], None]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def write(self, label: str) -> tuple[bool, bool]:
        """Record a write: (refuse it, apply it but answer 500)."""
        self.writes.append(label)
        number = len(self.writes)
        return (
            self.fail_write_number == number,
            self.land_then_fail_write_number == number,
        )

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
    parent: tuple[str, str] = ("SF-131", "Production Vehicle"),
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
        fields["parent"] = {"key": parent[0], "fields": {"summary": parent[1]}}
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

        def do_DELETE(self) -> None:
            self._dispatch("DELETE")

        def _dispatch(self, method: str) -> None:
            url = urlsplit(self.path)
            query = {key: values[0] for key, values in parse_qs(url.query).items()}
            body = self._body()
            with upstream.lock:
                if url.path.startswith("/clockify/api/v1/"):
                    if self.headers.get("X-Api-Key") != CLOCKIFY_KEY:
                        return self._send(401, {"message": "bad key"})
                    path = url.path.removeprefix("/clockify/api/v1")
                    writes = len(upstream.writes)
                    status, payload = _clockify(upstream, method, path, query, body)
                    if len(upstream.writes) > writes and (
                        hook := upstream.after_write.get(len(upstream.writes))
                    ):
                        hook(upstream)
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
    if (method, path) == ("POST", f"{ws}/projects"):
        refuse, land_then_fail = upstream.write(f"POST project {body['note']}")
        if refuse:
            return 500, {"message": "simulated Clockify outage"}
        project = {
            "id": f"p-new-{len(upstream.writes)}",
            "name": body["name"],
            "note": body["note"],
            "archived": False,
            "billable": body["billable"],
            "color": "#039BE5",
            "public": body["isPublic"],
            "clientId": body["clientId"],
        }
        upstream.projects[project["id"]] = project
        if land_then_fail:
            return 500, {"message": "simulated lost response"}
        return 201, project
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
            refuse, land_then_fail = upstream.write(f"POST task {body['name']}")
            if refuse:
                return 500, {"message": "simulated Clockify outage"}
            created = {
                "id": f"t-new-{len(upstream.writes)}",
                "name": body["name"],
                "projectId": project_id,
                "status": body.get("status", "ACTIVE"),
            }
            # TaskRequestV1's other fields; it has no billable.
            created.update(
                (name, body[name]) for name in _TASK_CREATE_FIELDS if name in body
            )
            upstream.tasks[created["id"]] = created
            if land_then_fail:
                return 500, {"message": "simulated lost response"}
            return 201, created
    if match := re.fullmatch(rf"{ws}/projects/([^/]+)/tasks/([^/]+)", path):
        task = upstream.tasks.get(match.group(2))
        if task is None:
            return 404, {"message": "no such task"}
        if method == "GET":
            return 200, {**task, **upstream.readback.get(task["id"], {})}
        if method == "PUT":
            refuse, land_then_fail = upstream.write(f"PUT task {task['id']}")
            if refuse:
                return 500, {"message": "simulated Clockify outage"}
            task.update(name=body["name"], status=body["status"])
            # UpdateTaskRequest treated as a full replace: a writable field
            # the caller leaves out is cleared, so a lossy payload shows.
            for name in _TASK_UPDATE_FIELDS:
                if name in body:
                    task[name] = body[name]
                elif name in task:
                    del task[name]
            if land_then_fail:
                return 500, {"message": "simulated lost response"}
            return 200, task
    if match := re.fullmatch(
        rf"{ws}/projects/([^/]+)/tasks/([^/]+)/(hourly-rate|cost-rate)", path
    ):
        task = upstream.tasks.get(match.group(2))
        if task is None or method != "PUT":
            return 404, {"message": "no such task"}
        assert set(body) == {"amount"}, body  # never a retroactive "since"
        refuse, land_then_fail = upstream.write(f"PUT {match.group(3)} {task['id']}")
        if refuse:
            return 500, {"message": "simulated Clockify outage"}
        field_name = "hourlyRate" if match.group(3) == "hourly-rate" else "costRate"
        task[field_name] = {"amount": body["amount"], "currency": "AUD"}
        if land_then_fail:
            return 500, {"message": "simulated lost response"}
        return 200, task
    if (method, path) == ("GET", f"{ws}/users"):
        # Without status=ALL, Clockify lists only active members.
        page = [
            user
            for user in upstream.users
            if query.get("status") == "ALL" or user["status"] == "ACTIVE"
        ]
        return 200, page if query.get("page") == "1" else []
    if match := re.fullmatch(rf"{ws}/user/([^/]+)/time-entries", path):
        if method != "GET":
            return 404, {"message": "unhandled"}
        page = [
            entry
            for entry in upstream.entries.values()
            if entry["userId"] == match.group(1)
            and entry["projectId"] == query.get("project")
            and entry["taskId"] == query.get("task")
        ]
        return 200, page if query.get("page") == "1" else []
    if match := re.fullmatch(rf"{ws}/time-entries/([^/]+)", path):
        entry = upstream.entries.get(match.group(1))
        if entry is None:
            return 404, {"message": "no such time entry"}
        if method == "GET":
            return 200, entry
        if method == "PUT":
            refuse, land_then_fail = upstream.write(f"PUT entry {entry['id']}")
            if refuse:
                return 500, {"message": "simulated Clockify outage"}
            # UpdateTimeEntryRequest is a full replace.
            entry.update(
                description=body.get("description", ""),
                billable=body.get("billable", False),
                projectId=body.get("projectId"),
                taskId=body.get("taskId"),
                tagIds=body.get("tagIds") or [],
                type=body.get("type", "REGULAR"),
                customFieldValues=[
                    {
                        "customFieldId": value["customFieldId"],
                        "timeEntryId": entry["id"],
                        "value": value.get("value"),
                    }
                    for value in body.get("customFields") or []
                ],
            )
            entry["timeInterval"] = {
                **entry["timeInterval"],
                "start": body["start"],
                "end": body.get("end"),
            }
            if rates := upstream.entry_rates_on_move.get(entry["projectId"]):
                entry.update(rates)
            if land_then_fail:
                return 500, {"message": "simulated lost response"}
            return 200, entry
    if match := re.fullmatch(rf"{ws}/projects/([^/]+)", path):
        project = upstream.projects.get(match.group(1))
        if project is None:
            return 404, {"message": "no such project"}
        if method == "GET":
            return 200, project
    return 404, {"message": f"unhandled {method} {path}"}


_TASK_CREATE_FIELDS = ("assigneeIds", "userGroupIds", "estimate", "budgetEstimate")
_TASK_UPDATE_FIELDS = ("billable", *_TASK_CREATE_FIELDS)


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
        # Planned, but a finished issue is not counted as planned or underway.
        assert preview["inventory"]["active_issues"] == 0
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


# --- Relocation: Jira decides which project a task lives in ---------------------

# The reported cases: tasks Clockify holds under Production Vehicle (p-131)
# whose Jira Project / Phase is elsewhere. SF-737 exists in Clockify; SF-598
# does not yet. key -> (summary, phase key, phase summary, source task ID).
RELOCATED = {
    "SF-693": ("Alerting runbook", "SF-737", "Site Reliability Engineering", "t-693"),
    "SF-704": ("On-call rota", "SF-737", "Site Reliability Engineering", "t-704"),
    "SF-710": ("Fleet telemetry dashboard", "SF-598", "Fleet Management", "t-710"),
}
_TASK_FIELDS = {
    "assigneeIds": ["user-1"],
    "assigneeId": None,
    "userGroupIds": ["group-sre"],
    "billable": True,
    "estimate": "PT8H",
    "budgetEstimate": 0,
    "hourlyRate": {"amount": 15000, "currency": "AUD"},
    "costRate": None,
}


def _entry(entry_id: str, user_id: str, task_id: str, **fields: Any) -> dict[str, Any]:
    """A time entry as Clockify returns it (TimeEntryWithRatesDtoV1)."""
    return {
        "id": entry_id,
        "userId": user_id,
        "workspaceId": WORKSPACE,
        "projectId": "p-131",
        "taskId": task_id,
        "description": f"work on {task_id}",
        "billable": True,
        "tagIds": ["tag-oncall"],
        "timeInterval": {
            "start": "2026-11-02T01:00:00Z",
            "end": "2026-11-02T02:30:00Z",
            "duration": "PT1H30M",
        },
        "customFieldValues": [
            {"customFieldId": "cf-ticket", "timeEntryId": entry_id, "value": "INC-42"}
        ],
        "type": "REGULAR",
        "kioskId": None,
        "isLocked": False,
        "hourlyRate": {"amount": 15000, "currency": "AUD"},
        "costRate": {"amount": 9000, "currency": "AUD"},
        **fields,
    }


def _seed_relocations(upstream: FakeUpstream, keys: tuple[str, ...]) -> None:
    upstream.projects["p-737"] = {
        **upstream.projects["p-131"],
        "id": "p-737",
        "name": "Site Reliability Engineering",
        "note": "Jira: SF-737",
    }
    for phase, summary in (
        ("SF-737", "Site Reliability Engineering"),
        ("SF-598", "Fleet Management"),
    ):
        upstream.issues[phase] = _jira_issue(
            phase, summary, "In Progress", "Project / Phase"
        )
    for key in keys:
        summary, phase, phase_summary, task_id = RELOCATED[key]
        upstream.issues[key] = _jira_issue(
            key, summary, "In Progress", "Task", parent=(phase, phase_summary)
        )
        upstream.tasks[task_id] = {
            "id": task_id,
            "name": f"{key} {summary}",
            "projectId": "p-131",
            "status": "ACTIVE",
            "duration": "PT0S",
            **_TASK_FIELDS,
        }
        upstream.assigned_active.append(key)
    # A member who has since left: only status=ALL lists them.
    upstream.users.append({"id": "user-2", "name": "Former", "status": "INACTIVE"})
    if "SF-693" in keys:
        upstream.entries["e-693-a"] = _entry("e-693-a", "user-1", "t-693")
        upstream.entries["e-693-b"] = _entry(
            "e-693-b",
            "user-2",
            "t-693",
            description="handover notes",
            billable=False,
            tagIds=[],
            customFieldValues=[],
            timeInterval={
                "start": "2026-10-30T22:15:00Z",
                "end": "2026-10-31T00:00:00Z",
                "duration": "PT1H45M",
            },
        )
    if "SF-710" in keys:
        upstream.entries["e-710-a"] = _entry("e-710-a", "user-1", "t-710")


def _destination(upstream: FakeUpstream, phase: str) -> str:
    [project] = [
        p["id"] for p in upstream.projects.values() if p["note"] == f"Jira: {phase}"
    ]
    return project


def _assert_relocated(
    upstream: FakeUpstream,
    keys: tuple[str, ...],
    original_entries: dict[str, dict[str, Any]],
    original_tasks: dict[str, dict[str, Any]],
) -> None:
    """Each key has exactly one task, in its Jira project, carrying its entries."""
    assert not [
        t for t in upstream.tasks.values() if t["name"].startswith("[relocating")
    ]
    # Nothing was deleted.
    assert set(original_tasks) <= set(upstream.tasks)
    assert set(upstream.entries) == set(original_entries)
    for key in keys:
        summary, phase, _, source_id = RELOCATED[key]
        [task] = [t for t in upstream.tasks.values() if t["name"].startswith(f"{key} ")]
        assert task["projectId"] == _destination(upstream, phase)
        assert task["status"] == "ACTIVE"
        assert task["name"] == f"{key} {summary}"
        assert {name: task.get(name) for name in _TASK_FIELDS} == _TASK_FIELDS
        source = upstream.tasks[source_id]
        assert source == {
            **original_tasks[source_id],
            "name": f"[relocated {key} to {task['id']}] {key} {summary}",
            "status": "DONE",
        }
        for entry_id, before in original_entries.items():
            if before["taskId"] != source_id:
                continue
            # Every field but the placement is exactly as it was.
            assert upstream.entries[entry_id] == {
                **before,
                "projectId": task["projectId"],
                "taskId": task["id"],
            }


def _snapshot(upstream: FakeUpstream) -> tuple[dict[str, Any], dict[str, Any]]:
    return json.loads(json.dumps(upstream.entries)), json.loads(
        json.dumps(upstream.tasks)
    )


async def test_reported_relocations_preview_apply_and_converge(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    keys = ("SF-693", "SF-704", "SF-710")
    _seed_relocations(upstream, keys)
    entries, tasks = _snapshot(upstream)

    async with mcp_server(sync_env) as session:
        preview, result = await _preview_and_apply(session)
        again = _ok(await session.call_tool("clockify_jira_sync", {}))

    # The misplaced tasks no longer block the rest of the sync.
    assert preview["conflicts"] == []
    assert _summary(preview) == [
        ("CREATE_PROJECT", "SF-598"),
        ("RELOCATE_TASK", "SF-693"),
        ("RELOCATE_TASK", "SF-704"),
        ("RELOCATE_TASK", "SF-710"),
        ("REACTIVATE_TASK", "SF-427"),
        ("MARK_TASK_DONE", "SF-304"),
    ]
    moves = {a["jira_key"]: a for a in preview["actions"] if a["relocation"]}
    assert moves["SF-693"]["relocation"] == {
        "source_project_id": "p-131",
        "source_task_id": "t-693",
        "desired_status": "ACTIVE",
        "staged_task_id": None,
        "source_retired": False,
        "time_entry_ids": ["e-693-a", "e-693-b"],
        "moved_entry_ids": [],
    }
    assert (moves["SF-693"]["project_key"], moves["SF-693"]["project_id"]) == (
        "SF-737",
        "p-737",
    )
    assert "2 time entries follow it" in moves["SF-693"]["message"]
    assert moves["SF-704"]["relocation"]["time_entry_ids"] == []
    assert moves["SF-710"]["project_id"] is None
    assert "(created first)" in moves["SF-710"]["message"]

    report = _ok(result)
    assert report["failure"] is None
    assert all(a["applied"] and a["verified"] for a in report["actions"])
    _assert_relocated(upstream, keys, entries, tasks)
    # The report names the task each key now lives in.
    for key in keys:
        [action] = [a for a in report["actions"] if a["jira_key"] == key]
        assert upstream.tasks[action["task_id"]]["name"].startswith(f"{key} ")

    # A rerun converges: nothing left to do, nothing duplicated.
    writes = list(upstream.writes)
    assert again["actions"] == [] and again["conflicts"] == []
    assert upstream.writes == writes
    assert not [w for w in writes if w.startswith("DELETE")]


async def test_relocation_digest_binds_every_entry_field(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    _seed_relocations(upstream, ("SF-693",))
    async with mcp_server(sync_env) as session:
        preview = _ok(await session.call_tool("clockify_jira_sync", {}))
        # Someone edits an entry's description after the preview.
        upstream.entries["e-693-b"]["description"] = "edited later"
        result = await session.call_tool(
            "clockify_jira_sync",
            {"apply": True, "plan_digest": preview["plan_digest"]},
        )

    _err(result, "plan_stale")
    assert upstream.writes == []


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (
            {"timeInterval": {"start": "2026-11-02T01:00:00Z", "end": None}},
            "running timer",
        ),
        ({"isLocked": True}, "locked"),
        ({"type": "TIME_OFF"}, "does not accept"),
    ],
)
async def test_entry_that_cannot_move_blocks_every_write(
    mcp_server: ServerFactory,
    sync_env: dict[str, str],
    upstream: FakeUpstream,
    change: dict[str, Any],
    reason: str,
) -> None:
    _seed_relocations(upstream, ("SF-693",))
    upstream.entries["e-693-a"].update(change)
    async with mcp_server(sync_env) as session:
        preview, result = await _preview_and_apply(session)

    [conflict] = preview["conflicts"]
    assert conflict["jira_key"] == "SF-693"
    assert "e-693-a" in conflict["message"] and reason in conflict["message"]
    assert _err(result, "invalid_argument")["detail"] == {"conflicts": ["SF-693"]}
    assert upstream.writes == []


@pytest.mark.parametrize(
    "stray",
    [
        # A staging task naming a different source task.
        {
            "name": "[relocating SF-693 from t-other] SF-693 Alerting runbook",
            "projectId": "p-737",
        },
        # A valid-looking staging task outside the Jira-derived project.
        {
            "name": "[relocating SF-693 from t-693] SF-693 Alerting runbook",
            "projectId": "p-131",
        },
    ],
)
async def test_unrecognised_staging_task_is_a_conflict_not_a_resume(
    mcp_server: ServerFactory,
    sync_env: dict[str, str],
    upstream: FakeUpstream,
    stray: dict[str, Any],
) -> None:
    _seed_relocations(upstream, ("SF-693",))
    upstream.tasks["t-stray"] = {"id": "t-stray", "status": "ACTIVE", **stray}
    async with mcp_server(sync_env) as session:
        preview, result = await _preview_and_apply(session)

    assert [c["jira_key"] for c in preview["conflicts"]] == ["SF-693"]
    assert "staging task t-stray" in preview["conflicts"][0]["message"]
    _err(result, "invalid_argument")
    assert upstream.writes == []


async def test_rate_changed_by_the_move_fails_without_rolling_back(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    _seed_relocations(upstream, ("SF-693",))
    upstream.entry_rates_on_move["p-737"] = {
        "hourlyRate": {"amount": 20000, "currency": "AUD"}
    }
    async with mcp_server(sync_env) as session:
        _, result = await _preview_and_apply(session)

    report = _ok(result)
    failure = report["failure"]
    assert (failure["jira_key"], failure["kind"]) == ("SF-693", "RELOCATE_TASK")
    assert "e-693-a" in failure["message"] and "hourlyRate" in failure["message"]
    assert '"amount": 15000' in failure["message"]  # the original value
    # The failed entry is not reported as verified; its write is named.
    recovery = failure["recovery"]
    assert "moved and verified" not in recovery
    assert "Unverified write: moving time entry e-693-a" in recovery
    # Stopped at the first mismatch: no rollback, nothing retired or promoted.
    assert upstream.writes[-1] == "PUT entry e-693-a"
    assert upstream.entries["e-693-b"]["taskId"] == "t-693"
    assert upstream.tasks["t-693"]["status"] == "ACTIVE"


@pytest.mark.parametrize("response", ["ok", "lost"])
async def test_preservation_failure_is_not_forgotten_by_the_next_preview(
    mcp_server: ServerFactory,
    sync_env: dict[str, str],
    upstream: FakeUpstream,
    response: str,
) -> None:
    """A move that changed a field Clockify will not let us set back blocks
    every later apply until a person accepts it; it never converges silently.
    With ``lost`` the write lands and answers 500, so the apply never saw the
    read-back: only the staging task's baseline catches it."""
    _seed_relocations(upstream, ("SF-693",))
    entries, tasks = _snapshot(upstream)
    upstream.entry_rates_on_move["p-737"] = {
        "hourlyRate": {"amount": 20000, "currency": "AUD"}
    }
    if response == "lost":
        upstream.land_then_fail_write_number = 5  # the first entry move
    async with mcp_server(sync_env) as session:
        _, first = await _preview_and_apply(session)
        assert _ok(first)["failure"]["kind"] == "RELOCATE_TASK"
        upstream.land_then_fail_write_number = None
        writes = len(upstream.writes)
        for _ in range(2):
            preview, blocked = await _preview_and_apply(session)
            [conflict] = preview["conflicts"]
            assert conflict["jira_key"] == "SF-693"
            assert "baseline" in conflict["message"]
            assert "e-693-a" in conflict["message"]
            _err(blocked, "invalid_argument")
        assert len(upstream.writes) == writes

        # A person checks e-693-a, accepts it by removing the baseline from
        # the staging task's name, and the rest of the relocation finishes.
        upstream.entry_rates_on_move.clear()
        [staging] = [
            t for t in upstream.tasks.values() if t["name"].startswith("[relocating")
        ]
        staging["name"] = re.sub(r" #[0-9a-f]{16}\]", "]", staging["name"])
        accepted, result = await _preview_and_apply(session)
        assert accepted["conflicts"] == []
        [move] = [a for a in accepted["actions"] if a["kind"] == "RELOCATE_TASK"]
        assert "new baseline" in move["message"]
        assert _ok(result)["failure"] is None
        final = _ok(await session.call_tool("clockify_jira_sync", {}))

    assert final["actions"] == [] and final["conflicts"] == []
    # The accepted entry keeps the rate it was given; the other is untouched.
    assert upstream.entries["e-693-a"]["hourlyRate"]["amount"] == 20000
    entries["e-693-a"]["hourlyRate"]["amount"] = 20000
    _assert_relocated(upstream, ("SF-693",), entries, tasks)


async def test_source_changed_by_its_retirement_blocks_the_resume(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    _seed_relocations(upstream, ("SF-710",))

    def drop_estimate(state: FakeUpstream) -> None:
        state.tasks["t-710"]["estimate"] = "PT1H"

    upstream.after_write[6] = drop_estimate  # the retirement
    async with mcp_server(sync_env) as session:
        _, first = await _preview_and_apply(session)
        failure = _ok(first)["failure"]
        assert "estimate" in failure["message"]
        assert "Unverified write: retiring source task t-710" in failure["recovery"]
        preview, blocked = await _preview_and_apply(session)

    assert [c["jira_key"] for c in preview["conflicts"]] == ["SF-710"]
    assert "baseline" in preview["conflicts"][0]["message"]
    _err(blocked, "invalid_argument")
    assert len(upstream.writes) == 6


async def test_interrupted_relocation_is_resumed_after_its_issue_leaves_discovery(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    keys = ("SF-710",)
    _seed_relocations(upstream, keys)
    entries, tasks = _snapshot(upstream)
    upstream.fail_write_number = 7  # the promotion, after the source retired
    async with mcp_server(sync_env) as session:
        await _preview_and_apply(session)
        upstream.fail_write_number = None
        # The issue is finished and handed to someone else.
        upstream.issues["SF-710"]["fields"]["status"] = {
            "name": "Done",
            "statusCategory": {"name": "Done"},
        }
        upstream.issues["SF-710"]["fields"]["assignee"] = {"displayName": "David"}
        upstream.assigned_active.remove("SF-710")
        preview, result = await _preview_and_apply(session)
        final = _ok(await session.call_tool("clockify_jira_sync", {}))

    [move] = [a for a in preview["actions"] if a["jira_key"] == "SF-710"]
    assert move["kind"] == "RELOCATE_TASK"
    assert move["relocation"]["source_retired"] is True
    assert move["relocation"]["desired_status"] == "DONE"
    assert _ok(result)["failure"] is None
    assert final["actions"] == [] and final["conflicts"] == []
    [task] = [t for t in upstream.tasks.values() if t["name"].startswith("SF-710 ")]
    assert task["status"] == "DONE"
    assert task["projectId"] == _destination(upstream, "SF-598")
    assert upstream.entries["e-710-a"]["taskId"] == task["id"]


def test_cli_text_failure_prints_the_relocation_recovery(
    upstream: FakeUpstream,
) -> None:
    _seed_relocations(upstream, ("SF-710",))
    upstream.land_then_fail_write_number = 5  # the entry move

    cli = CliRunner().invoke(clockify_cli, ["jira-sync", "--apply"], env=upstream.env())

    assert cli.exit_code == 1
    assert "Recovery: Relocation of SF-710" in cli.output
    assert "Unverified write: moving time entry e-710-a" in cli.output


# One relocation into a missing project, then the default inventory's two
# task updates. Clockify's create-task body has no billable, and rates have
# their own endpoint, so both are copied onto the staging task before any
# entry moves; the source is retired before the replacement is promoted.
SF710_CLEAN_WRITES = [
    "POST project Jira: SF-598",
    "POST task [relocating SF-710 from t-710 #BASELINE] SF-710 Fleet telemetry dashboard",
    "PUT task t-new-2",
    "PUT hourly-rate t-new-2",
    "PUT entry e-710-a",
    "PUT task t-710",
    "PUT task t-new-2",
    "PUT task t-427",
    "PUT task t-304",
]


async def test_relocation_write_order_is_stage_move_retire_promote(
    mcp_server: ServerFactory, sync_env: dict[str, str], upstream: FakeUpstream
) -> None:
    _seed_relocations(upstream, ("SF-710",))
    async with mcp_server(sync_env) as session:
        _, result = await _preview_and_apply(session)

    assert _ok(result)["failure"] is None
    assert _writes(upstream) == SF710_CLEAN_WRITES


def _writes(upstream: FakeUpstream) -> list[str]:
    return [re.sub(r"#[0-9a-f]{16}\]", "#BASELINE]", w) for w in upstream.writes]


@pytest.mark.parametrize("mode", ["refused", "landed"])
@pytest.mark.parametrize("write_number", range(1, len(SF710_CLEAN_WRITES) + 1))
async def test_interrupted_relocation_resumes_and_converges(
    mcp_server: ServerFactory,
    sync_env: dict[str, str],
    upstream: FakeUpstream,
    mode: str,
    write_number: int,
) -> None:
    keys = ("SF-710",)
    _seed_relocations(upstream, keys)
    entries, tasks = _snapshot(upstream)
    if mode == "refused":
        upstream.fail_write_number = write_number
    else:
        upstream.land_then_fail_write_number = write_number

    async with mcp_server(sync_env) as session:
        _, first = await _preview_and_apply(session)
        failure = _ok(first)["failure"]
        assert failure is not None
        # The upstream's response body never reaches the report.
        assert "simulated" not in json.dumps(_ok(first))
        if failure["kind"] == "RELOCATE_TASK":
            assert "Preview again before retrying" in failure["recovery"]
            assert "Unverified write:" in failure["recovery"]
        upstream.fail_write_number = upstream.land_then_fail_write_number = None

        # Each fresh preview reads what landed and plans only what remains.
        resumed, second = await _preview_and_apply(session)
        assert resumed["conflicts"] == []
        assert _ok(second)["failure"] is None
        final = _ok(await session.call_tool("clockify_jira_sync", {}))

    assert final["actions"] == [] and final["conflicts"] == []
    _assert_relocated(upstream, keys, entries, tasks)
    assert upstream.tasks["t-427"]["status"] == "ACTIVE"
    assert upstream.tasks["t-304"]["status"] == "DONE"
    # At most the interrupted write and one repeat of it on top of the clean
    # run's writes (a refused write is retried; a landed one is not redone).
    assert len(upstream.writes) <= len(SF710_CLEAN_WRITES) + 1


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
