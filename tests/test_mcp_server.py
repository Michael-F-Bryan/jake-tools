"""The MCP server's plumbing: transport, result shape, error shape, isolation.

The stdio tests spawn the real ``python -m jake_tools.mcp`` under a hermetic
environment (see ``conftest.py``) and talk to it through the ``mcp`` client
library. The in-process tests exercise the registration wrapper directly
through MCPServer's own ``call_tool`` so the error contract is pinned without
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
from mcp.server.mcpserver import MCPServer
from mcp.types import (
    CallToolResult,
    InputRequiredResult,
    TextContent,
    TextResourceContents,
)
from pydantic import BaseModel

from jake_tools.config import load_config
from jake_tools.mcp.errors import ToolError, ToolErrorPayload, error_result
from jake_tools.mcp.server import (
    build_server,
    packaged_skills,
    skill_description,
    structured_tool,
)

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]


async def test_stdio_server_lists_ping_with_input_schema_and_no_output_schema(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
) -> None:
    async with mcp_server(mcp_env) as session:
        tools = (await session.list_tools()).tools

    by_name = {tool.name: tool for tool in tools}
    assert "ping" in by_name
    ping = by_name["ping"]
    assert ping.input_schema["type"] == "object"
    assert ping.input_schema.get("properties", {}) == {}
    assert ping.output_schema is None
    assert ping.description and "jake-tools" in ping.description


async def test_stdio_ping_returns_structured_content_and_matching_text(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
) -> None:
    async with mcp_server(mcp_env) as session:
        result = await session.call_tool("ping", {})

    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["ok"] is True
    assert result.structured_content["pid"] > 0
    assert result.structured_content["version"]
    assert result.structured_content["server_time"].endswith("Z")
    assert len(result.content) == 1
    text = result.content[0]
    assert isinstance(text, TextContent)
    assert json.loads(text.text) == result.structured_content


async def test_stdio_default_subcommand_is_serve(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
) -> None:
    async with mcp_server(mcp_env, args=("-m", "jake_tools.mcp")) as session:
        result = await session.call_tool("ping", {})

    assert result.is_error is False


async def test_stdio_starts_without_any_credentials_and_lists_skill_resources(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
) -> None:
    assert not any(key.startswith(("CLOCKIFY", "JIRA")) for key in mcp_env)

    async with mcp_server(mcp_env) as session:
        resources = (await session.list_resources()).resources

    expected = {f"skill://{skill.name}" for skill in packaged_skills()}
    expected.update(
        f"skill://{skill.name}/{path.relative_to(skill).as_posix()}"
        for skill in packaged_skills()
        for path in (skill / "references").rglob("*.md")
    )
    assert {str(resource.uri) for resource in resources} == expected
    by_uri = {str(resource.uri): resource for resource in resources}
    for skill in packaged_skills():
        description = next(
            line.removeprefix("description: ")
            for line in (skill / "SKILL.md").read_text().splitlines()
            if line.startswith("description: ")
        )
        assert by_uri[f"skill://{skill.name}"].description == description


async def test_stdio_reads_optional_clockify_reference_without_inlining_it(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
) -> None:
    uri = "skill://clockify-time/references/direct-api-workflows.md"
    async with mcp_server(mcp_env) as session:
        resources = (await session.list_resources()).resources
        assert uri in {str(resource.uri) for resource in resources}
        reference = await session.read_resource(uri)
        skill = await session.read_resource("skill://clockify-time")

    assert isinstance(reference.contents[0], TextResourceContents)
    assert isinstance(skill.contents[0], TextResourceContents)
    source = next(p for p in packaged_skills() if p.name == "clockify-time")
    assert (
        reference.contents[0].text
        == (source / "references/direct-api-workflows.md").read_text()
    )
    assert skill.contents[0].text == (source / "SKILL.md").read_text()
    assert "## Merging duplicate Jira-backed projects" not in skill.contents[0].text


async def test_nested_markdown_references_are_individually_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jake_tools.mcp import server

    root = tmp_path / "skills"
    skill = root / "example"
    nested = skill / "references" / "cleanup"
    nested.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: example\ndescription: Exact-ID cleanup guidance\n---\n"
        "Read references/cleanup/exact-ids.md only when needed."
    )
    (nested / "exact-ids.md").write_text("Exact-ID cleanup procedure")
    (nested / "openapi.json").write_text('{"not": "a default resource"}')
    monkeypatch.setattr(server, "SKILLS_DIR", root)
    app = build_server(load_config({"HOME": str(tmp_path)}))

    assert {str(r.uri) for r in await app.list_resources()} == {
        "skill://example",
        "skill://example/references/cleanup/exact-ids.md",
    }
    result = await app.read_resource("skill://example/references/cleanup/exact-ids.md")
    assert not isinstance(result, InputRequiredResult)
    assert [part.content for part in result] == ["Exact-ID cleanup procedure"]


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("Read Clockify history.", "Read Clockify history."),
        ('"Read: Clockify # history"', "Read: Clockify # history"),
        (">-\n  Read Clockify\n  history safely.", "Read Clockify history safely."),
        ("|\n  Read Clockify.\n  Verify entries.", "Read Clockify.\nVerify entries.\n"),
    ],
)
def test_skill_description_reads_yaml_strings(
    tmp_path: Path, field: str, expected: str
) -> None:
    path = tmp_path / "SKILL.md"
    path.write_text(f"---\nname: example\ndescription: {field}\n---\n# Body\n")
    assert skill_description(path) == expected


@pytest.mark.parametrize(
    "text",
    [
        "# No frontmatter",
        "---\nname: example\n",
        "---\nname: example\n---\n",
        "---\ndescription: 123\n---\n",
        "---\ndescription: ''\n---\n",
        "---\n- not a mapping\n---\n",
    ],
)
def test_skill_description_rejects_invalid_packaged_metadata(
    tmp_path: Path, text: str
) -> None:
    path = tmp_path / "SKILL.md"
    path.write_text(text)
    with pytest.raises(ValueError, match="frontmatter"):
        skill_description(path)


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
        assert (await session.call_tool("ping", {})).is_error is False


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
    from jake_tools.mcp.__main__ import main

    assert set(main.commands) == {"serve", "doctor", "worker"}
    assert "serve" in completed.stdout
    assert "doctor" in completed.stdout
    assert "worker" not in completed.stdout


# --- the registration wrapper, in process ------------------------------------


class Echo(BaseModel):
    value: str
    count: int


def _app_with_tools() -> MCPServer:
    app = MCPServer("test")

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
    assert result.is_error is False
    assert result.structured_content == {"value": "hi", "count": 2}


async def test_wrapper_keeps_handler_parameters_as_input_schema() -> None:
    app = _app_with_tools()

    tools = {tool.name: tool for tool in await app.list_tools()}

    assert set(tools["echo"].input_schema["properties"]) == {"value", "count"}
    assert tools["echo"].input_schema["required"] == ["value"]
    assert tools["echo"].output_schema is None


async def test_wrapper_turns_tool_error_into_structured_error_result() -> None:
    app = _app_with_tools()

    result = await app.call_tool("echo", {"value": "hi", "count": -1})

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    payload = ToolErrorPayload.model_validate(result.structured_content)
    assert payload.code == "invalid_argument"
    assert payload.detail == {"argument": "count"}
    text = result.content[0]
    assert isinstance(text, TextContent)
    assert json.loads(text.text) == result.structured_content


async def test_wrapper_reports_untranslated_exceptions_without_their_text() -> None:
    app = _app_with_tools()

    result = await app.call_tool("explode", {"secret": "hunter2"})

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "internal_error"
    assert "hunter2" not in json.dumps(result.structured_content)


def test_error_result_omits_absent_detail() -> None:
    result = error_result(ToolError("unknown_task", "no such task"))

    assert result.structured_content == {
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
