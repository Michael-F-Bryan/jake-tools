"""``polish_transcript`` through the real MCP server over stdio.

The server, the operation, the pipeline and the Claude Agent SDK run for
real; the ``claude`` CLI is ``tests/fake_claude.py``, which returns the
structured replies this test chooses. That proves discovery, argument
validation, error mapping, the tool-less CLI invocation and the result's
shape, never polishing quality (``test_polish_transcript_live.py``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

from mcp import ClientSession
from polish_fixtures import (
    EMPTY_EDITS,
    EMPTY_REVIEW,
    write_transcript,
)

from jake_tools.mcp.errors import ToolErrorPayload

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]

TURNS: list[tuple[str | None, str]] = [
    ("speaker_1", "Um, so we we need the room."),
    ("speaker_2", "Yeah. Not on Tuesday, though?"),
]


def _error(result: Any) -> ToolErrorPayload:
    assert result.is_error is True, result
    return ToolErrorPayload.model_validate(result.structured_content)


async def test_the_tool_is_listed_with_its_cloud_boundary(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
) -> None:
    async with mcp_server(mcp_env) as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
    tool = tools["polish_transcript"]
    assert tool.input_schema["required"] == ["transcript_path", "output_directory"]
    assert set(tool.input_schema["properties"]) == {
        "transcript_path",
        "output_directory",
        "glossary",
        "start_seconds",
        "end_seconds",
        "max_budget_usd",
    }
    assert tool.output_schema is None
    description = " ".join((tool.description or "").split())
    assert "sent to Anthropic's Claude models; the audio is not" in description
    assert "skill://transcription" in description


async def test_bad_arguments_are_invalid_argument_and_write_nothing(
    mcp_server: ServerFactory, mcp_env: dict[str, str], tmp_path: Path
) -> None:
    source = write_transcript(tmp_path / "raw.json", TURNS)
    existing = tmp_path / "existing"
    existing.mkdir()
    cases = {
        "transcript_path": {
            "transcript_path": "raw.json",
            "output_directory": str(tmp_path / "a"),
        },
        "output_directory": {
            "transcript_path": str(source),
            "output_directory": str(existing),
        },
        "glossary": {
            "transcript_path": str(source),
            "output_directory": str(tmp_path / "b"),
            "glossary": ["X", "x"],
        },
        "max_budget_usd": {
            "transcript_path": str(source),
            "output_directory": str(tmp_path / "c"),
            "max_budget_usd": 1000,
        },
    }
    async with mcp_server(mcp_env) as session:
        for argument, arguments in cases.items():
            error = _error(await session.call_tool("polish_transcript", arguments))
            assert error.code == "invalid_argument"
            assert error.detail == {"argument": argument}
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "existing",
        "home",
        "raw.json",
    ]
    assert list(existing.iterdir()) == []


async def test_a_polish_runs_tool_less_through_the_real_sdk(
    mcp_server: ServerFactory,
    fake_env: dict[str, str],
    fake_claude_state: Path,
    tmp_path: Path,
) -> None:
    replies = {
        "EditDecisions": {
            **EMPTY_EDITS,
            "deletions": [
                {
                    "first_word": 2,
                    "last_word": 2,
                    "text": "we",
                    "kind": "repetition",
                    "reason": "stutter",
                }
            ],
        },
        "ReviewFindings": {
            **EMPTY_REVIEW,
            "issues": [
                {
                    "first_word": 9,
                    "last_word": 10,
                    "text": "on Tuesday,",
                    "category": "unclear_meaning",
                    "note": "Which Tuesday?",
                }
            ],
        },
    }
    env = {**fake_env, "FAKE_CLAUDE_STRUCTURED_OUTPUT": json.dumps(replies)}
    source = write_transcript(tmp_path / "raw.json", TURNS)
    raw = source.read_bytes()
    out = tmp_path / "polished"
    async with mcp_server(env) as session:
        result = await session.call_tool(
            "polish_transcript",
            {"transcript_path": str(source), "output_directory": str(out)},
        )
    assert result.is_error is not True, result
    payload = result.structured_content
    assert payload is not None
    assert payload["status"] == "complete"
    assert payload["source_unchanged"] is True and source.read_bytes() == raw
    assert payload["counts"]["removed_by_kind"] == {"filler": 1, "repetition": 1}
    assert payload["counts"]["review_items"] == 1
    assert "model" not in payload["chunks"][0]["edit"]  # nulls are omitted
    markdown = (out / "transcript.md").read_text()
    assert "So we need the room." in markdown
    assert "Not on [R1] Tuesday, though?" in markdown
    argvs = [
        json.loads(p.read_text()) for p in sorted(fake_claude_state.glob("argv-*.json"))
    ]
    assert len(argvs) == 2
    for argv in argvs:
        assert argv[argv.index("--tools") + 1] == ""
        assert "--json-schema" in argv and "--strict-mcp-config" in argv
        assert "--setting-sources=" in argv  # no settings files at all
        assert not any(a.startswith("--allowedTools") for a in argv)
    assert {argv[argv.index("--model") + 1] for argv in argvs} == {"sonnet", "opus"}
