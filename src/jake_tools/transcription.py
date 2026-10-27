"""Client for the native transcription service and its transcript contract.

The service (``transcription-service/`` in this repository) is a long-running
macOS process that keeps FluidAudio's speech-recognition and diarisation
models warm. This module is the only thing that talks to it: it uploads
client-local audio files over HTTP and validates the structured transcript
that comes back. It never decodes audio, never loads a model, and depends on
nothing beyond ``requests`` and ``pydantic``, so a Linux container can run it.

Wire contract (``transcription-service/README.md`` is the full description):

- ``GET /health`` (no auth) reports readiness.
- ``POST /v1/transcriptions`` takes a ``multipart/form-data`` body whose parts
  are all named ``file``, each with a ``filename``; part order is recording
  order. It answers with one complete :class:`Transcript`, or an error body
  ``{"error": {"code", "message"}}`` whose HTTP status says what went wrong.

Timestamps are seconds from the first sample of the concatenated recording.
"""

from __future__ import annotations

import json
import mimetypes
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import requests
from pydantic import BaseModel, ConfigDict, ValidationError

from .http import HttpSession, UpstreamError

TRANSCRIPT_SCHEMA_VERSION = 1
CONNECT_TIMEOUT_SECONDS = 10.0

# --- the contract ---------------------------------------------------------------


class TranscriptEngine(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    asr_model: str
    diarization_model: str


class TranscriptFile(BaseModel):
    """Where one uploaded file sits on the concatenated timeline."""

    model_config = ConfigDict(frozen=True)

    index: int
    name: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float


class TranscriptWord(BaseModel):
    """A recognised word. ``speaker`` is ``None`` when no diarised turn covers
    or is near it; ``overlap`` means several speakers' turns covered it and
    the attribution is the largest overlap, which may be wrong."""

    model_config = ConfigDict(frozen=True)

    text: str
    start_seconds: float
    end_seconds: float
    confidence: float
    speaker: str | None = None
    overlap: bool = False


class TranscriptSegment(BaseModel):
    """A run of consecutive words attributed to one speaker."""

    model_config = ConfigDict(frozen=True)

    speaker: str | None = None
    start_seconds: float
    end_seconds: float
    text: str
    words: tuple[TranscriptWord, ...]


class TranscriptSpeaker(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    speaking_seconds: float
    segment_count: int


class TranscriptProcessing(BaseModel):
    model_config = ConfigDict(frozen=True)

    decode_seconds: float
    asr_seconds: float
    diarization_seconds: float
    total_seconds: float


class Transcript(BaseModel):
    """The service's complete result for one request. Never partial."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1]
    engine: TranscriptEngine
    timestamp_unit: Literal["seconds"]
    timestamp_origin: str
    timing_resolution_seconds: float
    duration_seconds: float
    files: tuple[TranscriptFile, ...]
    speakers: tuple[TranscriptSpeaker, ...]
    segments: tuple[TranscriptSegment, ...]
    text: str
    unassigned_word_count: int
    limitations: tuple[str, ...]
    processing: TranscriptProcessing


class ServiceQueue(BaseModel):
    model_config = ConfigDict(frozen=True)

    running: int
    queued: int
    max_queued: int


class ServiceHealth(BaseModel):
    """``GET /health``."""

    model_config = ConfigDict(frozen=True)

    status: str
    ready: bool
    version: str
    engine: TranscriptEngine | None = None
    detail: str | None = None
    queue: ServiceQueue


class SourceFile(BaseModel):
    """What the MCP tool uploaded, as the caller named it."""

    model_config = ConfigDict(frozen=True)

    index: int
    path: str
    bytes: int


class TranscriptionResult(BaseModel):
    """What the ``transcribe`` tool returns."""

    model_config = ConfigDict(frozen=True)

    inputs: tuple[SourceFile, ...]
    transcript: Transcript


# --- errors ---------------------------------------------------------------------

TranscriptionErrorKind = Literal[
    "unavailable",
    "timeout",
    "unauthorized",
    "not_ready",
    "overloaded",
    "invalid_input",
    "payload_too_large",
    "inference_failed",
    "bad_response",
]

_STATUS_KINDS: dict[int, TranscriptionErrorKind] = {
    400: "invalid_input",
    401: "unauthorized",
    413: "payload_too_large",
    415: "invalid_input",
    429: "overloaded",
    500: "inference_failed",
    503: "not_ready",
}


class TranscriptionInputError(ValueError):
    """A file the caller named cannot be uploaded. Raised before any request."""

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


class TranscriptionError(UpstreamError):
    """The service could not be reached, refused the request, or failed.

    ``kind`` is the client's classification; ``code`` and ``message`` are the
    service's own when it answered with an error body. Messages never carry
    the token, audio or transcript content, only file names and limits.
    """

    def __init__(
        self,
        kind: TranscriptionErrorKind,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message, safe_summary=message)
        self.kind: TranscriptionErrorKind = kind
        self.status = status
        self.code = code
        self.retry_after_seconds = retry_after_seconds


# --- the client -----------------------------------------------------------------


def validate_base_url(value: str) -> str:
    """An ``http``/``https`` URL with a host and no query or fragment."""
    parsed = urlsplit(value.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            "transcription.api_base_url must be an http:// or https:// URL with a host"
        )
    if parsed.query or parsed.fragment:
        raise ValueError(
            "transcription.api_base_url must not carry a query or fragment"
        )
    return value.strip().rstrip("/")


class TranscriptionClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_seconds: float,
        session: HttpSession | None = None,
    ) -> None:
        self.base_url = validate_base_url(base_url)
        if not token:
            raise ValueError("a transcription service token is required")
        self._token = token
        self.timeout_seconds = timeout_seconds
        self._session = session or requests.Session()

    def health(self) -> ServiceHealth:
        """``GET /health``; the report whether or not the service is ready."""
        try:
            response = self._session.request(
                "GET",
                f"{self.base_url}/health",
                timeout=(CONNECT_TIMEOUT_SECONDS, CONNECT_TIMEOUT_SECONDS),
            )
        except requests.RequestException as exc:
            raise _transport_error(exc, self.timeout_seconds) from exc
        try:
            return ServiceHealth.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise TranscriptionError(
                "bad_response",
                f"GET /health returned HTTP {response.status_code} without a valid health report",
                status=response.status_code,
            ) from exc

    def check_token(self) -> None:
        """Prove the token without inference: an empty POST is ``invalid_input``
        for a good token and ``unauthorized`` for a bad one."""
        try:
            response = self._session.request(
                "POST",
                f"{self.base_url}/v1/transcriptions",
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=(CONNECT_TIMEOUT_SECONDS, CONNECT_TIMEOUT_SECONDS),
            )
        except requests.RequestException as exc:
            raise _transport_error(exc, self.timeout_seconds) from exc
        if response.status_code == 400:
            return
        raise _response_error(response)

    def transcribe(self, files: Sequence[Path]) -> Transcript:
        """Upload ``files`` in order and return the validated transcript.

        Raises :class:`TranscriptionInputError` for a file that cannot be read
        (nothing is sent) and :class:`TranscriptionError` for everything the
        service or the network does.
        """
        if not files:
            raise TranscriptionInputError(Path(""), "no files were given")
        body = MultipartUpload([Path(file) for file in files])
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": f"multipart/form-data; boundary={body.boundary}",
            "Content-Length": str(len(body)),
        }
        try:
            response = self._session.request(
                "POST",
                f"{self.base_url}/v1/transcriptions",
                data=body,
                headers=headers,
                timeout=(CONNECT_TIMEOUT_SECONDS, self.timeout_seconds),
            )
        except TranscriptionInputError:
            raise
        except requests.RequestException as exc:
            raise _transport_error(exc, self.timeout_seconds) from exc
        if response.status_code != 200:
            raise _response_error(response)
        try:
            transcript = Transcript.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise TranscriptionError(
                "bad_response",
                "the transcription service returned a transcript that does not match the contract",
                status=200,
            ) from exc
        if len(transcript.files) != len(files):
            raise TranscriptionError(
                "bad_response",
                f"the transcript covers {len(transcript.files)} files but {len(files)} were sent",
                status=200,
            )
        return transcript


def _transport_error(
    exc: requests.RequestException, timeout_seconds: float
) -> TranscriptionError:
    if isinstance(exc, requests.ConnectTimeout | requests.ConnectionError):
        return TranscriptionError(
            "unavailable",
            "the transcription service is unreachable at transcription.api_base_url; "
            "check that it is running and reachable from the MCP server",
        )
    if isinstance(exc, requests.Timeout):
        return TranscriptionError(
            "timeout",
            f"no transcript within transcription.timeout_seconds ({timeout_seconds:g}s); "
            "the service may still be processing this request, so do not resubmit it blindly",
        )
    return TranscriptionError(
        "unavailable",
        f"the request to the transcription service failed ({type(exc).__name__})",
    )


def _response_error(response: Any) -> TranscriptionError:
    status = int(response.status_code)
    kind = _STATUS_KINDS.get(status, "bad_response")
    code: str | None = None
    message: str | None = None
    try:
        payload = response.json()
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            code = error.get("code") if isinstance(error.get("code"), str) else None
            message = (
                error.get("message") if isinstance(error.get("message"), str) else None
            )
    except ValueError:
        pass
    retry_after: int | None = None
    raw_retry = response.headers.get("Retry-After") if response.headers else None
    if raw_retry and str(raw_retry).isdigit():
        retry_after = int(raw_retry)
    if message is None:
        message = (
            f"the transcription service answered HTTP {status} without an error body"
        )
    return TranscriptionError(
        kind, message, status=status, code=code, retry_after_seconds=retry_after
    )


class MultipartUpload:
    """A ``multipart/form-data`` body that streams files from disk.

    ``requests`` sends any object with ``read`` and ``__len__`` with a
    ``Content-Length`` and without buffering it, so even a long recording is
    never held in memory. Every file is opened and sized up front; one that
    cannot be read raises :class:`TranscriptionInputError` before any byte is
    sent, and one that shrinks mid-upload raises it mid-stream, which aborts
    the request.
    """

    def __init__(self, paths: Sequence[Path]) -> None:
        self.boundary = f"jake-tools-{uuid.uuid4().hex}"
        self.sources: list[SourceFile] = []
        self._segments: list[bytes | tuple[Path, int]] = []
        for index, path in enumerate(paths):
            size = _readable_size(path)
            name = path.name or f"recording-{index + 1}"
            content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
            header = (
                f"--{self.boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{_quote(name)}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode()
            self._segments.append(header)
            self._segments.append((path, size))
            self._segments.append(b"\r\n")
            self.sources.append(SourceFile(index=index, path=str(path), bytes=size))
        self._segments.append(f"--{self.boundary}--\r\n".encode())
        self._length = sum(
            len(segment) if isinstance(segment, bytes) else segment[1]
            for segment in self._segments
        )
        self._index = 0
        self._offset = 0
        self._handle: Any = None

    def __len__(self) -> int:
        return self._length

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._length
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0 and self._index < len(self._segments):
            segment = self._segments[self._index]
            if isinstance(segment, bytes):
                piece = segment[self._offset : self._offset + remaining]
                self._offset += len(piece)
                if self._offset >= len(segment):
                    self._advance()
            else:
                path, expected = segment
                if self._handle is None:
                    self._handle = path.open("rb")
                piece = self._handle.read(min(remaining, expected - self._offset))
                if not piece:
                    self._handle.close()
                    self._handle = None
                    raise TranscriptionInputError(
                        path, "file shrank while it was being uploaded"
                    )
                self._offset += len(piece)
                if self._offset >= expected:
                    self._handle.close()
                    self._handle = None
                    self._advance()
            chunks.append(piece)
            remaining -= len(piece)
        return b"".join(chunks)

    def _advance(self) -> None:
        self._index += 1
        self._offset = 0


def _readable_size(path: Path) -> int:
    try:
        if path.is_dir():
            raise TranscriptionInputError(path, "is a directory, not an audio file")
        if not path.is_file():
            raise TranscriptionInputError(
                path, "does not exist or is not a regular file"
            )
        size = path.stat().st_size
        with path.open("rb"):
            pass
    except TranscriptionInputError:
        raise
    except OSError as exc:
        raise TranscriptionInputError(
            path, exc.strerror or "cannot be read by the MCP server"
        ) from exc
    if size == 0:
        raise TranscriptionInputError(path, "is empty")
    return size


def _quote(name: str) -> str:
    return (
        name.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "")
        .replace("\n", "")
    )


def transcript_summary(transcript: Transcript) -> dict[str, Any]:
    """A compact view for logs and tests: counts only, no content."""
    return {
        "duration_seconds": transcript.duration_seconds,
        "files": len(transcript.files),
        "speakers": len(transcript.speakers),
        "segments": len(transcript.segments),
        "words": sum(len(segment.words) for segment in transcript.segments),
        "unassigned_word_count": transcript.unassigned_word_count,
    }


__all__ = [
    "CONNECT_TIMEOUT_SECONDS",
    "TRANSCRIPT_SCHEMA_VERSION",
    "MultipartUpload",
    "ServiceHealth",
    "ServiceQueue",
    "SourceFile",
    "Transcript",
    "TranscriptEngine",
    "TranscriptFile",
    "TranscriptProcessing",
    "TranscriptSegment",
    "TranscriptSpeaker",
    "TranscriptWord",
    "TranscriptionClient",
    "TranscriptionError",
    "TranscriptionErrorKind",
    "TranscriptionInputError",
    "TranscriptionResult",
    "transcript_summary",
    "validate_base_url",
]

# ``json`` and ``os`` are used by callers' type checkers only through re-export
# of this module's names; keep the imports explicit for the error helpers.
_ = (json, os)
