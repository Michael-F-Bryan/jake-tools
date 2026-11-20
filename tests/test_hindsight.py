"""Shared Hindsight memory for delegated workers: contract and mechanics.

Real pieces throughout: the real config loader, the real stdio server, worker
and SDK. Only the ``claude`` executable is the fake CLI, which records its
argument vector; ``test_hindsight_live.py`` (slow) runs the real CLI.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession
from test_claude_tasks import (
    FAST,
    argv_files,
    detail,
    flag,
    poll_until,
    runs_dir,
    start,
    terminal,
    workdir,  # noqa: F401 - fixture
)

from jake_tools.claude_runs.models import RunSpec
from jake_tools.claude_runs.worker import agent_spec
from jake_tools.config import ConfigError, load_config
from jake_tools.hindsight import HindsightMemory, check_api_url

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]

URL = "http://127.0.0.1:8888"
BANK = "michael-personal"
ENV = {
    "JAKE_TOOLS_HINDSIGHT_API_URL": URL,
    "JAKE_TOOLS_HINDSIGHT_BANK_ID": BANK,
}


# --- the connection ------------------------------------------------------------


def test_the_route_is_the_single_bank_path_and_never_the_multi_bank_one() -> None:
    memory = HindsightMemory(api_url=URL + "/", bank_id=BANK)

    assert memory.api_url == URL
    assert memory.mcp_url == "http://127.0.0.1:8888/mcp/michael-personal/"
    # No header, token or other credential travels with the definition.
    from jake_tools.claude import http_mcp_server

    assert http_mcp_server(memory.mcp_url) == {"type": "http", "url": memory.mcp_url}


def test_access_decides_whether_retain_is_approved() -> None:
    read = HindsightMemory(api_url=URL, bank_id=BANK, access="read")
    write = HindsightMemory(api_url=URL, bank_id=BANK)

    assert read.tool_names == ("mcp__hindsight__recall", "mcp__hindsight__reflect")
    assert write.tool_names == (*read.tool_names, "mcp__hindsight__retain")
    # Only the read and retain tools: nothing destructive or administrative.
    for memory in (read, write):
        assert not [
            n
            for n in memory.tool_names
            if n.rsplit("__", 1)[1].startswith(
                ("delete", "clear", "update", "create", "invalidate", "cancel")
            )
        ]


@pytest.mark.parametrize(
    "url",
    [
        "http://hindsight.example.com",  # cleartext to a remote host
        "ftp://127.0.0.1",
        "http://user:pw@127.0.0.1:8888",
        "http://127.0.0.1:8888/mcp/other/",  # the endpoint must not pick a bank
        "http://127.0.0.1:8888?bank=x",
        "http://127.0.0.1:99999",
        "127.0.0.1:8888",
    ],
)
def test_an_endpoint_that_could_redirect_or_expose_memory_is_refused(url: str) -> None:
    with pytest.raises(ValueError):
        check_api_url(url)


@pytest.mark.parametrize(
    "bank", ["", "a/b", "../x", "a b", "bank?x=1", "-lead", ".dot", "x" * 129, "é"]
)
def test_a_bank_that_is_not_one_plain_path_segment_is_refused(bank: str) -> None:
    with pytest.raises(ValueError):
        HindsightMemory(api_url=URL, bank_id=bank)


# --- configuration -------------------------------------------------------------


def test_nothing_is_configured_by_default_and_no_bank_is_invented() -> None:
    config = load_config({})

    assert config.hindsight is None
    assert config.hindsight_access.value == "readwrite"
    assert config.hindsight_api_url.source == "default"


def test_flags_beat_the_environment_and_report_their_source() -> None:
    config = load_config(
        {**ENV, "JAKE_TOOLS_HINDSIGHT_ACCESS": "read"},
        overrides={
            "hindsight.bank_id": "infin-michael",
            "hindsight.access": "readwrite",
        },
    )

    assert config.hindsight == HindsightMemory(api_url=URL, bank_id="infin-michael")
    assert config.hindsight_bank_id.source == "flag"
    assert config.hindsight_api_url.source == "env"
    assert config.hindsight_api_url.origin == "JAKE_TOOLS_HINDSIGHT_API_URL"


def test_blank_values_mean_unset() -> None:
    config = load_config(
        {"JAKE_TOOLS_HINDSIGHT_API_URL": " ", "JAKE_TOOLS_HINDSIGHT_BANK_ID": ""}
    )

    assert config.hindsight is None


@pytest.mark.parametrize("name", ENV)
def test_a_server_without_a_bank_or_a_bank_without_a_server_gives_no_memory(
    name: str,
) -> None:
    config = load_config({name: ENV[name]})

    assert config.hindsight is None  # never a default or fallback bank


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("JAKE_TOOLS_HINDSIGHT_API_URL", "http://example.com"),
        ("JAKE_TOOLS_HINDSIGHT_BANK_ID", "a/b"),
        ("JAKE_TOOLS_HINDSIGHT_ACCESS", "admin"),
    ],
)
def test_an_unusable_value_names_its_setting(variable: str, value: str) -> None:
    with pytest.raises(ConfigError) as caught:
        load_config({**ENV, variable: value})

    assert variable in str(caught.value)


# --- the worker's launch -------------------------------------------------------


def _spec(**extra: Any) -> RunSpec:
    return RunSpec.model_validate(
        {
            "task_id": "20260101T000000Z-" + "0" * 32,
            "created_at": "2026-01-01T00:00:00Z",
            "working_directory": "/",
            "model": "fake-model",
            "max_turns": 3,
            "timeout_seconds": 60,
            "max_budget_usd": 1.0,
            **extra,
        }
    )


def test_a_run_without_memory_is_launched_exactly_as_before() -> None:
    options = agent_spec(_spec()).to_options()

    assert options.mcp_servers == {}
    assert options.allowed_tools == []
    assert options.tools == []
    assert options.system_prompt is None
    assert options.strict_mcp_config is True


def test_memory_tools_are_exposed_and_approved_even_with_no_builtin_tools() -> None:
    memory = HindsightMemory(api_url=URL, bank_id=BANK)
    options = agent_spec(_spec(hindsight=memory.model_dump())).to_options()

    assert options.tools == []  # no built-in tool was requested
    assert options.mcp_servers == {"hindsight": {"type": "http", "url": memory.mcp_url}}
    assert options.allowed_tools == list(memory.tool_names)
    assert options.setting_sources == []  # no settings file, hooks or plugins
    assert options.strict_mcp_config is True
    assert options.permission_mode == "dontAsk"
    assert options.system_prompt is None  # the worker's prompt is untouched


def test_memory_approvals_add_to_the_callers_rules_without_widening_them() -> None:
    memory = HindsightMemory(api_url=URL, bank_id=BANK, access="read")
    spec = _spec(
        hindsight=memory.model_dump(), tools=["Read", "Bash"], allowed_tools=["Read"]
    )
    options = agent_spec(spec).to_options()

    assert options.tools == ["Read", "Bash"]
    assert options.allowed_tools == ["Read", *memory.tool_names]


async def test_the_server_launches_workers_with_the_parents_bank_and_nothing_else(
    mcp_server: ServerFactory,
    fake_env: dict[str, str],
    fake_claude_state: Path,
    workdir: Path,  # noqa: F811
    mcp_home: Path,
) -> None:
    env = {**fake_env, **ENV, "JAKE_TOOLS_HINDSIGHT_ACCESS": "read"}
    # A secret the server holds must still never reach the worker or its files.
    env["JIRA_API_TOKEN"] = "SENTINEL-SECRET-VALUE"
    async with mcp_server(env) as session:
        started = await start(session, workdir, "[[scenario:echo]] hi", tools=[])
        await poll_until(session, started["task_id"], terminal)
        info = await detail(session, started["task_id"])

    (argv,) = argv_files(fake_claude_state)
    config = json.loads(flag(argv, "--mcp-config") or "{}")
    assert config == {
        "mcpServers": {
            "hindsight": {"type": "http", "url": f"{URL}/mcp/{BANK}/"},
        }
    }
    assert "--strict-mcp-config" in argv
    assert flag(argv, "--tools") == ""
    assert flag(argv, "--allowedTools") == (
        "mcp__hindsight__recall,mcp__hindsight__reflect"
    )
    assert flag(argv, "--system-prompt") == ""  # unchanged: nothing added
    assert flag(argv, "--setting-sources") == ""

    spec = json.loads(next(runs_dir(mcp_home).glob("*/spec.json")).read_text())
    assert spec["hindsight"] == {"api_url": URL, "bank_id": BANK, "access": "read"}
    assert info["status"] == "completed"
    everything = json.dumps(spec) + json.dumps(argv)
    assert "SENTINEL-SECRET-VALUE" not in everything


async def test_a_worker_without_configured_memory_gets_no_memory_server(
    mcp_server: ServerFactory,
    fake_env: dict[str, str],
    fake_claude_state: Path,
    workdir: Path,  # noqa: F811
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(session, workdir, "[[scenario:echo]] hi", tools=[])
        await poll_until(session, started["task_id"], terminal)

    (argv,) = argv_files(fake_claude_state)
    assert "hindsight" not in json.dumps(argv)
    assert flag(argv, "--allowedTools") in (None, "")


async def test_the_max_turns_handoff_stays_tool_free_when_memory_is_configured(
    mcp_server: ServerFactory,
    fake_env: dict[str, str],
    fake_claude_state: Path,
    workdir: Path,  # noqa: F811
) -> None:
    """The handoff layers its own spec over the worker's defaults, and only
    fields it sets win: it must explicitly drop the memory server and its
    grants, or the "tool-free" resume would inherit them."""
    async with mcp_server({**fake_env, **ENV}) as session:
        started = await start(
            session, workdir, "[[scenario:max_turns]] go", tools=["Bash"]
        )
        await poll_until(session, started["task_id"], terminal)
        info = await detail(session, started["task_id"])

    assert info["result"]["handoff"]["status"] == "written"
    first, second = argv_files(fake_claude_state)
    assert "hindsight" in (flag(first, "--mcp-config") or "")
    assert "mcp__hindsight__recall" in (flag(first, "--allowedTools") or "")
    assert flag(second, "--resume") == started["session_id"]
    assert flag(second, "--tools") == ""
    assert flag(second, "--allowedTools") in (None, "")
    assert "hindsight" not in json.dumps(second)
    assert "mcp__" not in json.dumps(second)
    assert FAST["max_turns"]  # the shared fast limits were used
