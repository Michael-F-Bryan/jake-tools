"""``transcribe``: upload client-local audio to the native transcription service.

Registered only when ``transcription.api_base_url`` is configured; an agent
that cannot see the tool has no transcription capability, by design. The
handler is a thin adapter: it checks the files exist here, hands them to
:class:`~jake_tools.transcription.TranscriptionClient`, and maps that
client's errors onto the server's stable codes.
"""

from __future__ import annotations

import functools
from pathlib import Path

import anyio.to_thread
from mcp.server.mcpserver import MCPServer

from ...config import Config
from ...transcription import (
    TranscriptionClient,
    TranscriptionError,
    TranscriptionInputError,
    TranscriptionResult,
)
from ..errors import ToolError
from ..server import structured_tool

MAX_FILES_PER_CALL = 64


def transcription_client(config: Config) -> TranscriptionClient:
    base_url = config.transcription_api_base_url.value
    if base_url is None:
        raise ToolError(
            "invalid_argument",
            "transcription.api_base_url is not configured",
        )
    token = config.transcription_api_token.value
    if not token:
        raise ToolError(
            "missing_credentials",
            "Set TRANSCRIPTION_API_TOKEN in the server environment",
            detail={"variables": ["TRANSCRIPTION_API_TOKEN"]},
        )
    try:
        return TranscriptionClient(
            base_url,
            token,
            timeout_seconds=config.transcription_timeout_seconds.value,
        )
    except ValueError as exc:
        raise ToolError(
            "invalid_argument", f"Invalid transcription configuration: {exc}"
        ) from exc


def tool_error(exc: TranscriptionError) -> ToolError:
    """The client's classification onto the server's codes, with what a caller
    can act on in ``detail``."""
    detail: dict[str, object] = {"integration": "transcription", "reason": exc.kind}
    if exc.code is not None:
        detail["service_code"] = exc.code
    if exc.status is not None:
        detail["http_status"] = exc.status
    if exc.retry_after_seconds is not None:
        detail["retry_after_seconds"] = exc.retry_after_seconds
    message = exc.safe_summary or str(exc)
    match exc.kind:
        case "unauthorized":
            return ToolError(
                "access_denied",
                "TRANSCRIPTION_API_TOKEN was rejected by the transcription service",
                detail=detail,
            )
        case "overloaded":
            return ToolError("capacity_exceeded", message, detail=detail)
        case "invalid_input" | "payload_too_large":
            return ToolError("invalid_argument", message, detail=detail)
        case "timeout":
            detail["retry_safe"] = False
            return ToolError("upstream_error", message, detail=detail)
        case _:
            return ToolError("upstream_error", message, detail=detail)


def register(app: MCPServer, config: Config) -> None:
    if not config.has_transcription_endpoint:
        return

    @structured_tool(
        app,
        name="transcribe",
        description=(
            "Transcribe and diarise audio files that exist on THIS server's filesystem "
            "(paths the MCP server can open; not the agent's or a container's view "
            "unless they share it). Pass files in recording order; they are uploaded "
            "and treated as ONE continuous recording with request-wide anonymous speaker "
            "labels (speaker_1, speaker_2, …). Returns the complete structured transcript: "
            "speaker segments with word timings in seconds from the start of the "
            "concatenated recording, per-file time spans, speaker summaries, the engine's "
            "full text, limitations, and processing times. Fully local inference on the "
            "native host service; one job at a time with a bounded queue, so a long "
            "recording can take minutes. A timeout or disconnect is reported, never "
            "retried automatically: the service may still be busy with the request."
            "\n\nBefore using this tool, read `skill://transcription` via this server's "
            "`read_resource` capability, unless its contents are already loaded in the "
            "current context. The guide explains the transcript schema, what speaker "
            "and overlap fields do and do not promise, and how to handle each error. "
            "Reuse the loaded guide for subsequent calls."
        ),
    )
    async def transcribe(files: list[str]) -> TranscriptionResult:
        if not files:
            raise ToolError(
                "invalid_argument",
                "files must list at least one audio file path",
                detail={"argument": "files"},
            )
        if len(files) > MAX_FILES_PER_CALL:
            raise ToolError(
                "invalid_argument",
                f"files may list at most {MAX_FILES_PER_CALL} paths",
                detail={"argument": "files"},
            )
        if any(not isinstance(item, str) or not item for item in files):
            raise ToolError(
                "invalid_argument",
                "every entry in files must be a non-empty path",
                detail={"argument": "files"},
            )
        client = transcription_client(config)
        paths = [Path(item).expanduser() for item in files]
        try:
            return await anyio.to_thread.run_sync(
                functools.partial(client.transcribe_result, paths)
            )
        except TranscriptionInputError as exc:
            raise ToolError(
                "invalid_argument",
                f"{exc.path}: {exc.reason}",
                detail={"argument": "files", "file": str(exc.path)},
            ) from exc
        except TranscriptionError as exc:
            raise tool_error(exc) from exc
