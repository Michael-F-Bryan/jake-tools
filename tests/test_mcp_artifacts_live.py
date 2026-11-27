"""Real publish → read → update through the real MCP server, no enable setting.

``slow`` and ``live``; CREATES ONE REAL PRIVATE ARTIFACT of a synthetic file and
then publishes a second version of that same artifact. It runs only with the
same explicit opt-in as ``test_artifacts_live.py``:
``JAKE_TOOLS_LIVE_ARTIFACT_PUBLISH=create-a-new-artifact``. It never touches any
other artifact. The server keeps the caller's real ``HOME`` (the signed-in
Claude account lives there) but isolated XDG state. Set
``JAKE_TOOLS_LIVE_ARTIFACT_RECEIPT_DIR`` to keep the three results as JSON.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

import pytest
from conftest import isolated_mcp_env
from mcp import ClientSession

pytestmark = [
    pytest.mark.slow,
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("JAKE_TOOLS_LIVE_ARTIFACT_PUBLISH") != "create-a-new-artifact",
        reason="creates a real artifact; needs JAKE_TOOLS_LIVE_ARTIFACT_PUBLISH="
        "create-a-new-artifact and explicit authority for the synthetic file",
    ),
]

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]


def _page(marker: str, body: str) -> str:
    return (
        "<!doctype html>\n<html><head><meta charset=utf-8>"
        f"<title>jake-tools live probe</title></head>\n<body><p>{marker}</p>"
        f"<p>{body}</p></body></html>\n"
    )


def _ok(result: Any) -> dict[str, Any]:
    assert result.is_error is False, result.structured_content or result.content
    assert isinstance(result.structured_content, dict)
    return result.structured_content


async def test_publish_read_update_through_mcp_with_no_opt_in_setting(
    mcp_server: ServerFactory, tmp_path: Path
) -> None:
    marker = os.environ.get("JAKE_TOOLS_LIVE_ARTIFACT_MARKER", "jake-tools-live-mcp")
    first = tmp_path / "v1.html"
    second = tmp_path / "v2.html"
    first.write_text(_page(marker, "version one"), encoding="utf-8")
    second.write_text(_page(marker, "version two"), encoding="utf-8")

    env = isolated_mcp_env(tmp_path / "state-home")
    env["HOME"] = os.environ["HOME"]
    for name in ("CLAUDE_CONFIG_DIR", "ANTHROPIC_API_KEY"):
        if name in os.environ:
            env[name] = os.environ[name]
    assert not [name for name in env if "ARTIFACT" in name]

    async with mcp_server(env) as session:
        names = {tool.name for tool in (await session.list_tools()).tools}
        assert {"artifact_publish", "artifact_read", "artifact_update"} <= names
        published = _ok(
            await session.call_tool("artifact_publish", {"file_path": str(first)})
        )
        assert published["verified"] is True, published["problems"]
        assert published["private_confirmed"] is True
        url, version_id = published["url"], published["version_id"]

        read = _ok(
            await session.call_tool(
                "artifact_read",
                {
                    "url": url,
                    "file_path": str(first),
                    "expected_version_id": version_id,
                },
            )
        )
        assert read["private"] is True
        assert read["matches_file"] is True
        assert read["version_matches"] is True

        updated = _ok(
            await session.call_tool(
                "artifact_update",
                {
                    "url": url,
                    "file_path": str(second),
                    "expected_version_id": version_id,
                },
            )
        )
        assert updated["verified"] is True, updated["problems"]
        assert updated["url"] == url
        assert updated["previous_version_id"] == version_id
        assert updated["version"] == published["version"] + 1

    receipts = os.environ.get("JAKE_TOOLS_LIVE_ARTIFACT_RECEIPT_DIR")
    if receipts:
        out = Path(receipts)
        out.mkdir(parents=True, exist_ok=True)
        for name, payload in (
            ("publish", published),
            ("read", read),
            ("update", updated),
        ):
            (out / f"{name}-receipt.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
    print(json.dumps({"url": url, "published": published, "updated": updated}))
