"""The SDK stdout message limit, through the real SDK transport and parser.

``tests/fake_claude.py`` stands in for the CLI binary (the true external
boundary). Everything after it is production code: the SDK's subprocess
transport, its line framing and size guard, JSON parsing and message
parsing, driven by ``ClaudeAgent.converse``. Nothing here fakes the parser.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from claude_agent_sdk import CLIJSONDecodeError, ToolResultBlock, UserMessage
from conftest import install_fake_claude

from jake_tools import claude
from jake_tools.claude import AgentSpec, ClaudeAgent

MIB = 1024 * 1024


@pytest.fixture
def fake_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir, state = install_fake_claude(tmp_path / "fake")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_CLAUDE_STATE", str(state))
    workdir = tmp_path / "work"
    workdir.mkdir()
    return workdir


async def read_image(workdir: Path, size: int, spec: AgentSpec | None = None) -> int:
    """Have the CLI answer a ``Read`` with a ``size``-byte base64 image block.

    Returns the image data length the parsed message carries.
    """
    agent = ClaudeAgent()
    spec = (spec or AgentSpec()).model_copy(
        update={"model": "fake-model", "cwd": workdir}
    )
    seen: list[int] = []
    async with agent.converse(spec) as conversation:
        await conversation.send(f"[[scenario:image_result:{size}]] look", uuid="u-1")
        async for message in conversation.messages():
            if isinstance(message, UserMessage) and isinstance(message.content, list):
                for block in message.content:
                    if isinstance(block, ToolResultBlock) and isinstance(
                        block.content, list
                    ):
                        seen.extend(
                            len(part["source"]["data"]) for part in block.content
                        )
            if type(message).__name__ == "ResultMessage":
                break
    assert len(seen) == 1
    return seen[0]


def test_the_default_limit_is_finite_and_above_the_sdk_default() -> None:
    options = AgentSpec().to_options()
    assert options.max_buffer_size == claude.DEFAULT_MAX_BUFFER_SIZE
    assert MIB < claude.DEFAULT_MAX_BUFFER_SIZE <= 64 * MIB


def test_a_limit_must_be_positive() -> None:
    with pytest.raises(ValueError):
        AgentSpec(max_buffer_size=0)


async def test_an_image_result_over_the_sdks_one_mib_default_is_parsed_whole(
    fake_cli: Path,
) -> None:
    size = 2 * MIB
    assert size > MIB  # what the SDK rejects when no limit is set
    assert await read_image(fake_cli, size) == size


async def test_a_message_over_the_configured_limit_fails_without_truncation(
    fake_cli: Path,
) -> None:
    limit = 2 * MIB
    spec = AgentSpec(max_buffer_size=limit)
    # The envelope around the data adds a few hundred characters, so this
    # one is just under the limit and is delivered intact.
    assert await read_image(fake_cli, limit - 4096, spec) == limit - 4096
    # One just over it is refused, never cut short and parsed.
    with pytest.raises(CLIJSONDecodeError) as raised:
        await read_image(fake_cli, limit + 1, spec)
    assert f"maximum buffer size of {limit} bytes" in str(raised.value)
