"""Exact work-item lookup through the real stdio MCP server and a local HTTP peer."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

import pytest
from clockify_fixtures import API_KEY, Workspace, serve, sunfish_workspace
from mcp import ClientSession

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]
EnvFactory = Callable[..., dict[str, str]]

KEYS = ["SF-101", "SF-301", "SF-999", "SF-77", "SF-102"]


@pytest.fixture
def workspace() -> Iterator[tuple[Workspace, dict[str, str]]]:
    ws = sunfish_workspace(60)
    with serve(ws) as url:
        yield ws, {
            "CLOCKIFY_API_KEY": API_KEY,
            "CLOCKIFY_API_BASE_URL": url,
            "JAKE_TOOLS_CLOCKIFY_WORKSPACES": "ws-1",
        }


@pytest.fixture
def env(
    workspace: tuple[Workspace, dict[str, str]],
    mcp_env_factory: EnvFactory,
    mcp_home: Path,
) -> dict[str, str]:
    return mcp_env_factory(mcp_home, **workspace[1])


async def call(session: ClientSession, args: dict[str, Any]) -> dict[str, Any]:
    result = await session.call_tool("clockify_workspace_catalogue", args)
    assert result.structured_content is not None
    return result.structured_content


async def test_lookup_answers_several_keys_and_measures_both_costs(
    mcp_server: ServerFactory,
    workspace: tuple[Workspace, dict[str, str]],
    env: dict[str, str],
) -> None:
    ws = workspace[0]
    async with mcp_server(env) as session:
        ws.reset_counters()
        full = await call(session, {})
        full_requests, full_chars = len(ws.requests), len(json.dumps(full))

        ws.reset_counters()
        lookup = await call(session, {"keys": KEYS})
        unscoped_requests, unscoped_chars = len(ws.requests), len(json.dumps(lookup))

        ws.reset_counters()
        scoped = await call(
            session,
            {"keys": ["SF-101", "SF-301"], "project_ids": ["p-acme", "p-globex"]},
        )
        scoped_requests = len(ws.requests)

    # Full mode keeps its contract.
    assert set(full) == {"workspace_id", "clients", "projects", "tasks", "coverage"}
    outcomes = {r["key"]: r["outcome"] for r in lookup["results"]}
    assert outcomes == {
        "SF-101": "resolved",
        "SF-301": "ambiguous",
        "SF-999": "not_found",
        "SF-77": "resolved",
        "SF-102": "resolved",
    }
    by_key = {r["key"]: r for r in lookup["results"]}
    (task,) = by_key["SF-101"]["candidates"]
    assert task["task"]["id"] == "t-101" and task["client"]["id"] == "c-acme"
    assert len(by_key["SF-301"]["candidates"]) == 2
    assert by_key["SF-77"]["candidates"][0]["project"]["archived"] is True
    assert by_key["SF-102"]["candidates"][0]["task"]["completed"] is True
    assert lookup["coverage"]["complete"] and lookup["coverage"]["scope"][
        "workspace_wide"
    ]
    # Volume and upstream cost are separate facts.
    assert unscoped_chars * 10 < full_chars
    assert unscoped_requests == full_requests
    assert lookup["coverage"]["upstream_requests"] == unscoped_requests
    assert scoped["coverage"]["upstream_requests"] == scoped_requests
    assert scoped_requests < full_requests // 4
    assert not scoped["coverage"]["scope"]["workspace_wide"]
    print(
        f"\nfull: {full_requests} requests, {full_chars} chars;"
        f" lookup: {unscoped_requests} requests, {unscoped_chars} chars;"
        f" scoped lookup: {scoped_requests} requests,"
        f" {len(json.dumps(scoped))} chars"
    )


async def test_unreadable_project_is_incomplete_not_absent(
    mcp_server: ServerFactory,
    workspace: tuple[Workspace, dict[str, str]],
    env: dict[str, str],
) -> None:
    ws = workspace[0]
    ws.failing.add("/workspaces/ws-1/projects/p-old/tasks")
    async with mcp_server(env) as session:
        result = await call(session, {"keys": ["SF-77", "SF-999"]})
    assert {r["key"]: r["outcome"] for r in result["results"]} == {
        "SF-77": "incomplete",
        "SF-999": "incomplete",
    }
    coverage = result["coverage"]
    assert coverage["complete"] is False
    assert [u["project_id"] for u in coverage["unread"]] == ["p-old"]
    assert "must not leak" not in json.dumps(result)


@pytest.mark.parametrize(
    "args,message",
    [
        ({"client_ids": ["c-acme"]}, "pass keys"),
        ({"keys": []}, "1..50"),
        ({"keys": ["not a key"]}, "Invalid Jira issue key"),
        ({"keys": ["SF-1"], "client_ids": ["c-nope"]}, "Unknown client"),
        ({"keys": ["SF-1"], "project_ids": ["../x"]}, "letters, digits"),
    ],
)
async def test_bad_arguments_are_invalid_argument(
    mcp_server: ServerFactory,
    workspace: tuple[Workspace, dict[str, str]],
    env: dict[str, str],
    args: dict[str, Any],
    message: str,
) -> None:
    async with mcp_server(env) as session:
        result = await call(session, args)
    assert result["code"] == "invalid_argument" and message in result["message"]


async def test_input_schema_advertises_the_lookup_arguments(
    mcp_server: ServerFactory, env: dict[str, str]
) -> None:
    async with mcp_server(env) as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
    schema = tools["clockify_workspace_catalogue"].input_schema
    assert set(schema["properties"]) == {
        "workspace_id",
        "keys",
        "client_ids",
        "project_ids",
    }
    assert not schema.get("required")


async def test_skill_teaches_lookup_scope_and_catalogue_reuse(
    mcp_server: ServerFactory, env: dict[str, str]
) -> None:
    async with mcp_server(env) as session:
        text = str((await session.read_resource("skill://clockify-time")).contents[0])
    for needle in (
        "`keys`",
        "ambiguous",
        "incomplete",
        "upstream_requests",
        "Reusing a catalogue",
        "response size is not upstream work",
    ):
        assert needle in text, needle
