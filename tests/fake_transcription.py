"""A stand-in for the native transcription service, at the HTTP boundary.

Speaks the wire contract in ``transcription-service/README.md``: ``GET
/health`` and ``POST /v1/transcriptions`` with a multipart body. Records what
it was sent (token, part order, file names and bytes) and answers with a
canned transcript or a scripted error. It proves the Python client's and MCP
tool's mechanics only; it transcribes nothing.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

TOKEN = "fake-service-token"


def canned_transcript(
    files: list[tuple[str, int]], *, duration_per_file: float = 1.5
) -> dict[str, Any]:
    spans = []
    offset = 0.0
    for index, (name, _size) in enumerate(files):
        spans.append(
            {
                "index": index,
                "name": name,
                "start_seconds": round(offset, 3),
                "end_seconds": round(offset + duration_per_file, 3),
                "duration_seconds": duration_per_file,
            }
        )
        offset += duration_per_file
    words = [
        {
            "text": "hello",
            "start_seconds": 0.08,
            "end_seconds": 0.4,
            "confidence": 0.93,
            "speaker": "speaker_1",
            "overlap": False,
        },
        {
            "text": "there",
            "start_seconds": 0.48,
            "end_seconds": 0.8,
            "confidence": 0.9,
            "speaker": "speaker_1",
            "overlap": False,
        },
        {
            "text": "hi",
            "start_seconds": 1.6,
            "end_seconds": 1.84,
            "confidence": 0.7,
            "speaker": None,
            "overlap": True,
        },
    ]
    return {
        "schema_version": 1,
        "engine": {
            "name": "fake",
            "version": "0",
            "asr_model": "fake-asr@0",
            "diarization_model": "fake-diar@0",
        },
        "timestamp_unit": "seconds",
        "timestamp_origin": "first sample of the concatenated recording",
        "timing_resolution_seconds": 0.08,
        "duration_seconds": round(offset, 3),
        "files": spans,
        "speakers": [{"id": "speaker_1", "speaking_seconds": 1.0, "segment_count": 1}],
        "segments": [
            {
                "speaker": "speaker_1",
                "start_seconds": 0.08,
                "end_seconds": 0.8,
                "text": "hello there",
                "words": words[:2],
            },
            {
                "speaker": None,
                "start_seconds": 1.6,
                "end_seconds": 1.84,
                "text": "hi",
                "words": words[2:],
            },
        ],
        "text": "hello there hi",
        "unassigned_word_count": 1,
        "limitations": ["fake engine"],
        "processing": {
            "decode_seconds": 0.01,
            "asr_seconds": 0.02,
            "diarization_seconds": 0.03,
            "total_seconds": 0.06,
        },
    }


@dataclass
class RecordedRequest:
    method: str
    path: str
    authorization: str | None
    content_type: str | None
    content_length: int | None
    transfer_encoding: str | None
    parts: list[tuple[str | None, str | None, bytes]] = field(default_factory=list)
    raw_body: bytes = b""


@dataclass
class Script:
    """What the next ``POST`` should do."""

    status: int = 200
    error: tuple[str, str] | None = None
    retry_after: int | None = None
    delay_seconds: float = 0.0
    body_override: bytes | None = None
    ready: bool = True
    health_status: str = "ready"


class FakeTranscriptionService:
    def __init__(self, token: str = TOKEN) -> None:
        self.token = token
        self.requests: list[RecordedRequest] = []
        self.script = Script()
        self._lock = threading.Lock()
        handler = self._handler()
        # Threaded: the client keeps HTTP/1.1 connections alive, and a
        # single-threaded server would block every second connection.
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._stopped = False

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}"

    def start(self) -> FakeTranscriptionService:
        self._thread.start()
        return self

    def stop(self) -> None:
        """Idempotent, so a test can take the service down mid-way."""
        if self._stopped:
            return
        self._stopped = True
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def posts(self) -> list[RecordedRequest]:
        return [r for r in self.requests if r.method == "POST"]

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        service = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass

            def _send(
                self, status: int, payload: bytes, headers: dict[str, str] | None = None
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(payload)

            def _error(
                self, status: int, code: str, message: str, retry: int | None
            ) -> None:
                headers = {"Retry-After": str(retry)} if retry is not None else None
                self._send(
                    status,
                    json.dumps({"error": {"code": code, "message": message}}).encode(),
                    headers,
                )

            def do_GET(self) -> None:
                record = RecordedRequest(
                    "GET",
                    self.path,
                    self.headers.get("Authorization"),
                    None,
                    None,
                    None,
                )
                with service._lock:
                    service.requests.append(record)
                if self.path != "/health":
                    self._error(404, "not_found", "no such route", None)
                    return
                script = service.script
                report = {
                    "status": script.health_status,
                    "ready": script.ready,
                    "version": "fake",
                    "engine": {
                        "name": "fake",
                        "version": "0",
                        "asr_model": "fake-asr@0",
                        "diarization_model": "fake-diar@0",
                    }
                    if script.ready
                    else None,
                    "detail": None if script.ready else "fake not ready",
                    "queue": {"running": 0, "queued": 0, "max_queued": 4},
                }
                self._send(200 if script.ready else 503, json.dumps(report).encode())

            def do_POST(self) -> None:
                length = self.headers.get("Content-Length")
                body = self.rfile.read(int(length)) if length else b""
                record = RecordedRequest(
                    "POST",
                    self.path,
                    self.headers.get("Authorization"),
                    self.headers.get("Content-Type"),
                    int(length) if length else None,
                    self.headers.get("Transfer-Encoding"),
                    raw_body=body,
                )
                content_type = self.headers.get("Content-Type") or ""
                if content_type.startswith("multipart/form-data") and body:
                    message = BytesParser(policy=HTTP).parsebytes(
                        f"Content-Type: {content_type}\r\n\r\n".encode() + body
                    )
                    for part in message.iter_parts():
                        name = part.get_param("name", header="content-disposition")
                        payload = part.get_payload(decode=True)
                        record.parts.append(
                            (
                                str(name) if name is not None else None,
                                part.get_filename(),
                                payload if isinstance(payload, bytes) else b"",
                            )
                        )
                with service._lock:
                    service.requests.append(record)
                script = service.script
                if self.path != "/v1/transcriptions":
                    self._error(404, "not_found", "no such route", None)
                    return
                if record.authorization != f"Bearer {service.token}":
                    self._error(
                        401, "unauthorized", "the bearer token was rejected", None
                    )
                    return
                if script.delay_seconds:
                    time.sleep(script.delay_seconds)
                if script.body_override is not None:
                    self._send(script.status, script.body_override)
                    return
                if script.error is not None:
                    self._error(script.status, *script.error, script.retry_after)
                    return
                if not record.parts:
                    self._error(
                        400, "invalid_input", "the request carried no files", None
                    )
                    return
                files = [
                    (filename or "", len(data)) for _n, filename, data in record.parts
                ]
                self._send(200, json.dumps(canned_transcript(files)).encode())

        return Handler
