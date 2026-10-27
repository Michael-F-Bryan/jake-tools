"""The ``transcribe`` tool through the real MCP server over stdio.

The server is spawned under a hermetic HOME; the only fake is the transcription
service at its HTTP boundary (``fake_transcription.py``). These establish
discovery gating, upload mechanics, result shape and error mapping, not
transcription quality.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

import pytest
from fake_transcription import TOKEN, FakeTranscriptionService
from mcp import ClientSession
from mcp.types import TextResourceContents

from jake_tools.mcp.errors import ToolErrorPayload
from jake_tools.transcription import TranscriptionResult

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]


@pytest.fixture
def service() -> Iterator[FakeTranscriptionService]:
    fake = FakeTranscriptionService().start()
    try:
        yield fake
    finally:
        fake.stop()


@pytest.fixture
def recordings(tmp_path: Path) -> list[Path]:
    first = tmp_path / "meeting-1.m4a"
    second = tmp_path / "meeting-2.m4a"
    first.write_bytes(b"\x00\x01" * 500)
    second.write_bytes(b"\x02\x03" * 700)
    return [first, second]


async def _tool_names(session: ClientSession) -> set[str]:
    return {tool.name for tool in (await session.list_tools()).tools}


def _payload(result: Any) -> ToolErrorPayload:
    assert result.is_error is True, result
    return ToolErrorPayload.model_validate(result.structured_content)


async def test_no_endpoint_means_no_transcribe_tool(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
) -> None:
    async with mcp_server(mcp_env) as session:
        names = await _tool_names(session)
        resources = {str(r.uri) for r in (await session.list_resources()).resources}
    assert "ping" in names
    assert "transcribe" not in names
    # The guidance ships regardless; only the tool is gated.
    assert "skill://transcription" in resources


async def test_endpoint_without_token_exposes_the_tool_and_reports_missing_credentials(
    mcp_server: ServerFactory,
    mcp_env_factory: Callable[..., dict[str, str]],
    mcp_home: Path,
    recordings: list[Path],
) -> None:
    env = mcp_env_factory(mcp_home, TRANSCRIPTION_API_BASE_URL="http://127.0.0.1:9")
    async with mcp_server(env) as session:
        tools = {tool.name: tool for tool in (await session.list_tools()).tools}
        assert "transcribe" in tools
        assert tools["transcribe"].input_schema["required"] == ["files"]
        assert tools["transcribe"].output_schema is None
        description = " ".join((tools["transcribe"].description or "").split())
        assert "Before using this tool, read `skill://transcription`" in description
        resource = await session.read_resource("skill://transcription")
        result = await session.call_tool(
            "transcribe", {"files": [str(p) for p in recordings]}
        )
    content = resource.contents[0]
    assert isinstance(content, TextResourceContents)
    assert "speaker_1" in content.text
    payload = _payload(result)
    assert payload.code == "missing_credentials"
    assert payload.detail == {"variables": ["TRANSCRIPTION_API_TOKEN"]}


async def test_endpoint_from_config_file_also_exposes_the_tool(
    mcp_server: ServerFactory,
    mcp_env_factory: Callable[..., dict[str, str]],
    mcp_home: Path,
) -> None:
    config_dir = mcp_home / ".config" / "jake-tools"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        '[transcription]\napi_base_url = "http://127.0.0.1:9"\ntimeout_seconds = 12\n'
    )
    async with mcp_server(mcp_env_factory(mcp_home)) as session:
        assert "transcribe" in await _tool_names(session)


async def test_transcribe_uploads_ordered_files_and_returns_the_structured_result(
    mcp_server: ServerFactory,
    mcp_env_factory: Callable[..., dict[str, str]],
    mcp_home: Path,
    service: FakeTranscriptionService,
    recordings: list[Path],
) -> None:
    env = mcp_env_factory(
        mcp_home,
        TRANSCRIPTION_API_BASE_URL=service.base_url + "/",
        TRANSCRIPTION_API_TOKEN=TOKEN,
    )
    async with mcp_server(env) as session:
        result = await session.call_tool(
            "transcribe", {"files": [str(recordings[1]), str(recordings[0])]}
        )

    assert result.is_error is False, result.structured_content
    parsed = TranscriptionResult.model_validate(result.structured_content)
    assert [s.path for s in parsed.inputs] == [str(recordings[1]), str(recordings[0])]
    assert [s.bytes for s in parsed.inputs] == [1400, 1000]
    assert [f.name for f in parsed.transcript.files] == [
        "meeting-2.m4a",
        "meeting-1.m4a",
    ]
    assert parsed.transcript.timestamp_unit == "seconds"
    assert parsed.transcript.segments[0].words[0].speaker == "speaker_1"

    [post] = service.posts()
    assert post.authorization == f"Bearer {TOKEN}"
    assert [(n, f) for n, f, _ in post.parts] == [
        ("file", "meeting-2.m4a"),
        ("file", "meeting-1.m4a"),
    ]
    assert [data for _, _, data in post.parts] == [
        recordings[1].read_bytes(),
        recordings[0].read_bytes(),
    ]
    # Nothing but the upload was sent, and the token never appears in the result.
    assert TOKEN not in str(result.structured_content)


async def test_file_problems_are_invalid_argument_without_contacting_the_service(
    mcp_server: ServerFactory,
    mcp_env_factory: Callable[..., dict[str, str]],
    mcp_home: Path,
    service: FakeTranscriptionService,
    recordings: list[Path],
    tmp_path: Path,
) -> None:
    env = mcp_env_factory(
        mcp_home,
        TRANSCRIPTION_API_BASE_URL=service.base_url,
        TRANSCRIPTION_API_TOKEN=TOKEN,
    )
    missing = tmp_path / "nope.wav"
    async with mcp_server(env) as session:
        empty = _payload(await session.call_tool("transcribe", {"files": []}))
        blank = _payload(await session.call_tool("transcribe", {"files": [""]}))
        absent = _payload(
            await session.call_tool(
                "transcribe", {"files": [str(recordings[0]), str(missing)]}
            )
        )
        too_many = _payload(
            await session.call_tool("transcribe", {"files": [str(recordings[0])] * 65})
        )

    assert empty.code == "invalid_argument" and empty.detail == {"argument": "files"}
    assert blank.code == "invalid_argument"
    assert absent.code == "invalid_argument"
    assert absent.detail == {"argument": "files", "file": str(missing)}
    assert "does not exist" in absent.message
    assert too_many.code == "invalid_argument"
    assert service.posts() == []


@pytest.mark.parametrize(
    ("status", "service_code", "expected_code", "reason"),
    [
        (401, "unauthorized", "access_denied", "unauthorized"),
        (429, "overloaded", "capacity_exceeded", "overloaded"),
        (503, "not_ready", "upstream_error", "not_ready"),
        (400, "invalid_input", "invalid_argument", "invalid_input"),
        (413, "payload_too_large", "invalid_argument", "payload_too_large"),
        (500, "inference_failed", "upstream_error", "inference_failed"),
    ],
)
async def test_service_errors_map_to_stable_codes_with_actionable_detail(
    mcp_server: ServerFactory,
    mcp_env_factory: Callable[..., dict[str, str]],
    mcp_home: Path,
    service: FakeTranscriptionService,
    recordings: list[Path],
    status: int,
    service_code: str,
    expected_code: str,
    reason: str,
) -> None:
    token = "wrong-token" if status == 401 else TOKEN
    if status != 401:
        service.script.status = status
        service.script.error = (service_code, f"service message for {service_code}")
        service.script.retry_after = 20 if status in (429, 503) else None
    env = mcp_env_factory(
        mcp_home,
        TRANSCRIPTION_API_BASE_URL=service.base_url,
        TRANSCRIPTION_API_TOKEN=token,
    )
    async with mcp_server(env) as session:
        payload = _payload(
            await session.call_tool("transcribe", {"files": [str(recordings[0])]})
        )

    assert payload.code == expected_code
    assert payload.detail is not None
    assert payload.detail["integration"] == "transcription"
    assert payload.detail["reason"] == reason
    assert payload.detail["http_status"] == status
    assert payload.detail["service_code"] == service_code
    if status in (429, 503):
        assert payload.detail["retry_after_seconds"] == 20
    if status == 401:
        assert "TRANSCRIPTION_API_TOKEN" in payload.message
        assert "wrong-token" not in payload.message
    else:
        assert payload.message == f"service message for {service_code}"


async def test_unreachable_and_slow_services_are_upstream_errors_never_retried(
    mcp_server: ServerFactory,
    mcp_env_factory: Callable[..., dict[str, str]],
    mcp_home: Path,
    service: FakeTranscriptionService,
    recordings: list[Path],
) -> None:
    service.script.delay_seconds = 0.8
    env = mcp_env_factory(
        mcp_home,
        TRANSCRIPTION_API_BASE_URL=service.base_url,
        TRANSCRIPTION_API_TOKEN=TOKEN,
        JAKE_TOOLS_TRANSCRIPTION_TIMEOUT_SECONDS="0.3",
    )
    async with mcp_server(env) as session:
        timeout = _payload(
            await session.call_tool("transcribe", {"files": [str(recordings[0])]})
        )
    assert timeout.code == "upstream_error"
    assert timeout.detail is not None
    assert timeout.detail["reason"] == "timeout"
    assert timeout.detail["retry_safe"] is False
    assert "0.3" in timeout.message
    assert len(service.posts()) == 1

    service.stop()
    async with mcp_server(env) as session:
        gone = _payload(
            await session.call_tool("transcribe", {"files": [str(recordings[0])]})
        )
    assert gone.code == "upstream_error"
    assert gone.detail is not None
    assert gone.detail["reason"] == "unavailable"
    assert "transcription.api_base_url" in gone.message
    assert service.base_url not in gone.message


def test_the_client_path_imports_no_native_inference_dependency() -> None:
    """A proxy-only (Linux) install must not drag in Apple or ML frameworks."""
    script = (
        "import sys\n"
        "import jake_tools.transcription, jake_tools.mcp.server, jake_tools.mcp.tools.transcribe\n"
        "native = {'torch', 'coremltools', 'mlx', 'numpy', 'onnxruntime', 'ctranslate2',"
        " 'faster_whisper', 'whisper', 'mlx_whisper', 'objc', 'Foundation', 'AVFoundation'}\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in native))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "[]"


@pytest.fixture
async def _unused() -> AsyncIterator[None]:  # keeps pytest-asyncio's mode exercised
    yield None
