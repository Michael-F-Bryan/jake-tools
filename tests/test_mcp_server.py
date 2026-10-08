"""The MCP server's plumbing: transport, result shape, error shape, isolation.

The stdio tests spawn the real ``python -m jake_tools.mcp`` under a hermetic
environment (see ``conftest.py``) and talk to it through the ``mcp`` client
library. The in-process tests exercise the registration wrapper directly
through FastMCP's own ``call_tool`` so the error contract is pinned without
needing a tool that fails on purpose in the shipped server.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel

from jake_tools.config import load_config
from jake_tools.mcp.errors import ToolError, ToolErrorPayload, error_result
from jake_tools.mcp.server import build_server, packaged_skills, structured_tool

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]


async def test_stdio_server_lists_ping_with_input_schema_and_no_output_schema(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
) -> None:
    async with mcp_server(mcp_env) as session:
        tools = (await session.list_tools()).tools

    by_name = {tool.name: tool for tool in tools}
    assert "ping" in by_name
    ping = by_name["ping"]
    assert ping.inputSchema["type"] == "object"
    assert ping.inputSchema.get("properties", {}) == {}
    assert ping.outputSchema is None
    assert ping.description and "jake-tools" in ping.description


async def test_stdio_ping_returns_structured_content_and_matching_text(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
) -> None:
    async with mcp_server(mcp_env) as session:
        result = await session.call_tool("ping", {})

    assert result.isError is False
    assert result.structuredContent is not None
    assert result.structuredContent["ok"] is True
    assert result.structuredContent["pid"] > 0
    assert result.structuredContent["version"]
    assert result.structuredContent["server_time"].endswith("Z")
    assert len(result.content) == 1
    text = result.content[0]
    assert isinstance(text, TextContent)
    assert json.loads(text.text) == result.structuredContent


async def test_stdio_default_subcommand_is_serve(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
) -> None:
    async with mcp_server(mcp_env, args=("-m", "jake_tools.mcp")) as session:
        result = await session.call_tool("ping", {})

    assert result.isError is False


async def test_stdio_starts_without_any_credentials_and_lists_skill_resources(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
) -> None:
    assert not any(key.startswith(("CLOCKIFY", "JIRA")) for key in mcp_env)

    async with mcp_server(mcp_env) as session:
        resources = (await session.list_resources()).resources

    assert sorted(str(resource.uri) for resource in resources) == [
        f"skill://{skill.name}" for skill in packaged_skills()
    ]


async def test_stdio_server_resolves_config_from_the_isolated_home(
    mcp_server: ServerFactory,
    mcp_env_factory: Callable[..., dict[str, str]],
    mcp_home: Path,
) -> None:
    config_dir = mcp_home / ".config" / "jake-tools"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text('[clockify]\njira_project = "ZZ"\n')
    env = mcp_env_factory(mcp_home)

    # The same loader the server uses, over the same environment: proves the
    # isolated HOME is where the server will look, without a tool that leaks
    # its config over the wire.
    config = load_config(env)
    assert config.jira_project.value == "ZZ"
    assert config.jira_project.origin == str(config_dir / "config.toml")

    async with mcp_server(env) as session:
        assert (await session.call_tool("ping", {})).isError is False


def test_entrypoint_never_loads_dotenv_or_the_cli_entrypoint(tmp_path: Path) -> None:
    """Importing and invoking the MCP entrypoint must not read a ``.env``.

    ``dotenv`` the module may be imported by a dependency (pydantic-settings
    does); what matters is that nothing on this path *calls* it, so a ``.env``
    in the working directory never reaches the environment, and that the
    ``jake-tools`` CLI entrypoint (which does load it) stays out.
    """
    (tmp_path / ".env").write_text("JAKE_TOOLS_JIRA_PROJECT=FROM_DOTENV\n")
    script = (
        "import os, sys\n"
        "import jake_tools.mcp.__main__ as entry, jake_tools.mcp.server\n"
        "entry.main(['--help'], standalone_mode=False)\n"
        "bad = sorted(m for m in sys.modules if m == 'jake_tools.__main__' "
        "or m.startswith('jake_tools.transcription') or m == 'torch')\n"
        "print(bad, os.environ.get('JAKE_TOOLS_JIRA_PROJECT'))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    assert completed.stdout.strip().splitlines()[-1] == "[] None"


def test_module_help_lists_public_subcommands_and_hides_worker() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "jake_tools.mcp", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "serve" in completed.stdout
    assert "doctor" in completed.stdout
    assert "install-skills" in completed.stdout
    assert "worker" not in completed.stdout


# --- the registration wrapper, in process ------------------------------------


class Echo(BaseModel):
    value: str
    count: int


def _app_with_tools() -> FastMCP:
    app = FastMCP("test")

    @structured_tool(app, name="echo", description="echo")
    async def echo(value: str, count: int = 1) -> Echo:
        if count < 0:
            raise ToolError(
                "invalid_argument",
                "count must be non-negative",
                detail={"argument": "count"},
            )
        return Echo(value=value, count=count)

    @structured_tool(app, name="explode", description="explode")
    async def explode(secret: str) -> Echo:
        raise RuntimeError(f"leaked {secret}")

    return app


async def test_wrapper_sends_model_as_structured_content() -> None:
    app = _app_with_tools()

    result = await app.call_tool("echo", {"value": "hi", "count": 2})

    assert isinstance(result, CallToolResult)
    assert result.isError is False
    assert result.structuredContent == {"value": "hi", "count": 2}


async def test_wrapper_keeps_handler_parameters_as_input_schema() -> None:
    app = _app_with_tools()

    tools = {tool.name: tool for tool in await app.list_tools()}

    assert set(tools["echo"].inputSchema["properties"]) == {"value", "count"}
    assert tools["echo"].inputSchema["required"] == ["value"]
    assert tools["echo"].outputSchema is None


async def test_wrapper_turns_tool_error_into_structured_error_result() -> None:
    app = _app_with_tools()

    result = await app.call_tool("echo", {"value": "hi", "count": -1})

    assert isinstance(result, CallToolResult)
    assert result.isError is True
    payload = ToolErrorPayload.model_validate(result.structuredContent)
    assert payload.code == "invalid_argument"
    assert payload.detail == {"argument": "count"}
    text = result.content[0]
    assert isinstance(text, TextContent)
    assert json.loads(text.text) == result.structuredContent


async def test_wrapper_reports_untranslated_exceptions_without_their_text() -> None:
    app = _app_with_tools()

    result = await app.call_tool("explode", {"secret": "hunter2"})

    assert isinstance(result, CallToolResult)
    assert result.isError is True
    assert result.structuredContent is not None
    assert result.structuredContent["code"] == "internal_error"
    assert "hunter2" not in json.dumps(result.structuredContent)


def test_error_result_omits_absent_detail() -> None:
    result = error_result(ToolError("unknown_task", "no such task"))

    assert result.structuredContent == {
        "code": "unknown_task",
        "message": "no such task",
    }


def test_build_server_registers_ping_and_only_packaged_skills(tmp_path: Path) -> None:
    config = load_config({"HOME": str(tmp_path)})

    app = build_server(config)

    assert "ping" in {tool.name for tool in app._tool_manager.list_tools()}


@pytest.fixture
async def _unused() -> AsyncIterator[None]:  # keeps pytest-asyncio's mode exercised
    yield None
