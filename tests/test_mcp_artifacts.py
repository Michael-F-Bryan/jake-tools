"""``artifact_*`` through the real MCP server with no opt-in setting.

The server is spawned over stdio under an isolated HOME with a stand-in
``claude`` CLI first on PATH (``fake_claude.py``). The stand-in exposes no
``Artifact`` tool, which is the genuine prerequisite failure: it proves the
tools are discoverable by default and that a missing prerequisite is reported
actionably and before anything is published. It does not prove a publish
works; ``test_artifacts_live.py`` (``--slow``, ``live``) does.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path

from mcp import ClientSession
from mcp.types import CallToolResult

from jake_tools.mcp.errors import ToolErrorPayload

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]
URL = "https://claude.ai/artifact/V4iHumJLsDrW9EBEtZfqZj"
ARTIFACT_TOOLS = {"artifact_publish", "artifact_read", "artifact_update"}


def _error(result: CallToolResult) -> ToolErrorPayload:
    assert result.is_error is True, result
    return ToolErrorPayload.model_validate(result.structured_content)


async def test_artifact_tools_are_discoverable_with_no_opt_in(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
) -> None:
    assert not [name for name in mcp_env if "ARTIFACT" in name]
    async with mcp_server(mcp_env) as session:
        tools = {tool.name: tool for tool in (await session.list_tools()).tools}
        resources = {str(r.uri) for r in (await session.list_resources()).resources}
    assert set(tools) >= ARTIFACT_TOOLS
    assert "skill://artifact-publishing" in resources
    for name in ARTIFACT_TOOLS:
        description = tools[name].description or ""
        assert "`skill://artifact-publishing`" in description, name


async def test_a_missing_artifact_tool_is_an_actionable_error_before_any_write(
    mcp_server: ServerFactory, fake_env: dict[str, str], tmp_path: Path
) -> None:
    page = tmp_path / "page.html"
    page.write_text("<!doctype html><title>t</title><p>synthetic</p>")
    env = {**fake_env, "FAKE_CLAUDE_UNAVAILABLE_TOOLS": "Artifact"}
    async with mcp_server(env) as session:
        publish = await session.call_tool("artifact_publish", {"file_path": str(page)})
        read = await session.call_tool("artifact_read", {"url": URL})
        update = await session.call_tool(
            "artifact_update",
            {"url": URL, "file_path": str(page), "expected_version_id": "v1"},
        )
    for result in (publish, read, update):
        payload = _error(result)
        assert payload.code == "upstream_error"
        assert payload.detail is not None
        assert payload.detail["reason"] == "tool_not_exposed"
        assert payload.detail["publish_attempted"] is False
        assert "enableArtifact" in payload.message


async def test_argument_errors_come_from_the_handlers_not_from_a_missing_tool(
    mcp_server: ServerFactory, mcp_env: dict[str, str], tmp_path: Path
) -> None:
    async with mcp_server(mcp_env) as session:
        missing = await session.call_tool(
            "artifact_publish", {"file_path": str(tmp_path / "absent.html")}
        )
        bad_url = await session.call_tool(
            "artifact_read", {"url": "https://example.com/not-an-artifact"}
        )
    assert _error(missing).code == "invalid_argument"
    assert _error(bad_url).code == "invalid_argument"
