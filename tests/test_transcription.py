"""The transcription HTTP client against a fake service at the HTTP boundary.

These prove upload mechanics (streaming multipart, order, names, sizes,
Content-Length), contract validation and error classification. They say
nothing about transcription quality; ``test_transcription_live.py`` does.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fake_transcription import TOKEN, FakeTranscriptionService

from jake_tools.transcription import (
    MultipartUpload,
    TranscriptionClient,
    TranscriptionError,
    TranscriptionInputError,
    validate_base_url,
)


@pytest.fixture
def service() -> Iterator[FakeTranscriptionService]:
    fake = FakeTranscriptionService().start()
    try:
        yield fake
    finally:
        fake.stop()


@pytest.fixture
def client(service: FakeTranscriptionService) -> TranscriptionClient:
    return TranscriptionClient(service.base_url, TOKEN, timeout_seconds=5)


@pytest.fixture
def recordings(tmp_path: Path) -> list[Path]:
    first = tmp_path / "part one.m4a"
    second = tmp_path / "nested" / "part-two.wav"
    second.parent.mkdir()
    first.write_bytes(b"first-bytes" * 100)
    second.write_bytes(bytes(range(256)) * 20)
    return [first, second]


def test_base_url_validation() -> None:
    assert (
        validate_base_url(" https://host.example:8787/ ") == "https://host.example:8787"
    )
    assert validate_base_url("http://10.0.0.2:8787/api/") == "http://10.0.0.2:8787/api"
    for bad in ["host:8787", "ftp://host", "http://", "http://h/?x=1", "http://h/#f"]:
        with pytest.raises(ValueError, match="transcription.api_base_url"):
            validate_base_url(bad)
    with pytest.raises(ValueError, match="token"):
        TranscriptionClient("http://h", "", timeout_seconds=1)


def test_multipart_upload_streams_files_in_order_with_a_known_length(
    recordings: list[Path],
) -> None:
    body = MultipartUpload(recordings)

    streamed = b""
    while chunk := body.read(7):
        streamed += chunk
    assert len(streamed) == len(body)
    first_at = streamed.index(b'filename="part one.m4a"')
    second_at = streamed.index(b'filename="part-two.wav"')
    assert first_at < second_at
    assert recordings[0].read_bytes() in streamed
    assert recordings[1].read_bytes() in streamed
    assert streamed.endswith(f"--{body.boundary}--\r\n".encode())
    assert streamed.count(b'name="file"') == 2
    assert (
        b"Content-Type: audio/mp4" in streamed
        or b"Content-Type: audio/x-m4a" in streamed
    )
    assert [s.path for s in body.sources] == [str(p) for p in recordings]
    assert [s.bytes for s in body.sources] == [p.stat().st_size for p in recordings]


def test_unreadable_inputs_fail_before_any_request(
    client: TranscriptionClient, service: FakeTranscriptionService, tmp_path: Path
) -> None:
    missing = tmp_path / "missing.wav"
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    for path, reason in [
        (missing, "does not exist"),
        (empty, "is empty"),
        (tmp_path, "is a directory"),
    ]:
        with pytest.raises(TranscriptionInputError, match=reason) as excinfo:
            client.transcribe([path])
        assert excinfo.value.path == path
    with pytest.raises(TranscriptionInputError, match="no files"):
        client.transcribe([])
    assert service.posts() == []


def test_a_file_that_shrinks_mid_upload_aborts_the_request(
    client: TranscriptionClient, recordings: list[Path]
) -> None:
    body = MultipartUpload(recordings)
    recordings[0].write_bytes(b"short")
    with pytest.raises(TranscriptionInputError, match="shrank"):
        while body.read(64):
            pass


def test_transcribe_uploads_in_order_and_returns_the_validated_transcript(
    client: TranscriptionClient,
    service: FakeTranscriptionService,
    recordings: list[Path],
) -> None:
    transcript = client.transcribe(recordings)

    [post] = service.posts()
    assert post.authorization == f"Bearer {TOKEN}"
    assert post.content_type is not None
    assert post.content_type.startswith("multipart/form-data; boundary=")
    assert post.transfer_encoding is None
    assert post.content_length == len(post.raw_body)
    assert [(name, filename) for name, filename, _ in post.parts] == [
        ("file", "part one.m4a"),
        ("file", "part-two.wav"),
    ]
    assert [data for _, _, data in post.parts] == [p.read_bytes() for p in recordings]

    assert transcript.schema_version == 1
    assert [f.name for f in transcript.files] == ["part one.m4a", "part-two.wav"]
    assert [f.start_seconds for f in transcript.files] == [0.0, 1.5]
    assert transcript.duration_seconds == 3.0
    assert transcript.speakers[0].id == "speaker_1"
    assert transcript.segments[1].speaker is None
    assert transcript.segments[1].words[0].overlap is True
    assert transcript.unassigned_word_count == 1


@pytest.mark.parametrize(
    ("status", "code", "kind"),
    [
        (400, "invalid_input", "invalid_input"),
        (413, "payload_too_large", "payload_too_large"),
        (429, "overloaded", "overloaded"),
        (500, "inference_failed", "inference_failed"),
        (503, "not_ready", "not_ready"),
        (418, "teapot", "bad_response"),
    ],
)
def test_error_bodies_are_classified_by_status_and_carry_the_service_message(
    client: TranscriptionClient,
    service: FakeTranscriptionService,
    recordings: list[Path],
    status: int,
    code: str,
    kind: str,
) -> None:
    service.script.status = status
    service.script.error = (code, f"service says {code}")
    service.script.retry_after = 30 if status in (429, 503) else None

    with pytest.raises(TranscriptionError) as excinfo:
        client.transcribe(recordings)

    error = excinfo.value
    assert error.kind == kind
    assert error.status == status
    assert error.code == code
    assert error.safe_summary == f"service says {code}"
    assert error.retry_after_seconds == (30 if status in (429, 503) else None)


def test_wrong_token_is_unauthorized(
    service: FakeTranscriptionService, recordings: list[Path]
) -> None:
    client = TranscriptionClient(service.base_url, "nope", timeout_seconds=5)
    with pytest.raises(TranscriptionError) as excinfo:
        client.transcribe(recordings)
    assert excinfo.value.kind == "unauthorized"
    assert "nope" not in str(excinfo.value)


def test_off_contract_success_bodies_are_bad_response(
    client: TranscriptionClient,
    service: FakeTranscriptionService,
    recordings: list[Path],
) -> None:
    service.script.body_override = b"not json"
    with pytest.raises(TranscriptionError) as excinfo:
        client.transcribe(recordings)
    assert excinfo.value.kind == "bad_response"

    service.script.body_override = json.dumps({"schema_version": 2}).encode()
    with pytest.raises(TranscriptionError) as excinfo:
        client.transcribe(recordings)
    assert excinfo.value.kind == "bad_response"

    # A transcript covering a different number of files than were sent.
    service.script.body_override = None
    with pytest.raises(TranscriptionError, match="covers 1 files but 2"):
        client.transcribe = client.transcribe  # keep pyright quiet about reassignment
        _one_file_echo(client, service, recordings)


def _one_file_echo(
    client: TranscriptionClient,
    service: FakeTranscriptionService,
    recordings: list[Path],
) -> None:
    from fake_transcription import canned_transcript

    service.script.body_override = json.dumps(
        canned_transcript([("only.wav", 1)])
    ).encode()
    client.transcribe(recordings)


def test_timeouts_and_unreachable_services_are_distinguished(
    service: FakeTranscriptionService, recordings: list[Path]
) -> None:
    service.script.delay_seconds = 0.6
    slow = TranscriptionClient(service.base_url, TOKEN, timeout_seconds=0.2)
    with pytest.raises(TranscriptionError) as excinfo:
        slow.transcribe(recordings)
    assert excinfo.value.kind == "timeout"
    assert "may still be processing" in str(excinfo.value)

    service.stop()
    gone = TranscriptionClient(service.base_url, TOKEN, timeout_seconds=1)
    with pytest.raises(TranscriptionError) as excinfo:
        gone.transcribe(recordings)
    assert excinfo.value.kind == "unavailable"
    with pytest.raises(TranscriptionError) as excinfo:
        gone.health()
    assert excinfo.value.kind == "unavailable"


def test_a_tls_failure_is_not_reported_as_an_unreachable_service(
    service: FakeTranscriptionService, recordings: list[Path]
) -> None:
    # TLS against the plain-HTTP fake fails the handshake, as an untrusted
    # certificate does: the service is up, the client cannot talk to it.
    https = TranscriptionClient(
        service.base_url.replace("http://", "https://"), TOKEN, timeout_seconds=5
    )
    with pytest.raises(TranscriptionError) as excinfo:
        https.transcribe(recordings)
    assert excinfo.value.kind == "tls_failed"
    assert "REQUESTS_CA_BUNDLE" in str(excinfo.value)
    assert "unreachable" not in str(excinfo.value)
    with pytest.raises(TranscriptionError) as excinfo:
        https.health()
    assert excinfo.value.kind == "tls_failed"


def test_health_and_token_check(
    client: TranscriptionClient, service: FakeTranscriptionService
) -> None:
    health = client.health()
    assert health.ready is True
    assert health.engine is not None and health.engine.name == "fake"
    assert health.queue.max_queued == 4
    client.check_token()  # 400 for an empty authenticated POST

    bad = TranscriptionClient(service.base_url, "wrong", timeout_seconds=5)
    with pytest.raises(TranscriptionError) as excinfo:
        bad.check_token()
    assert excinfo.value.kind == "unauthorized"

    service.script.ready = False
    service.script.health_status = "starting"
    not_ready = client.health()
    assert not_ready.ready is False
    assert not_ready.status == "starting"
    assert not_ready.engine is None
