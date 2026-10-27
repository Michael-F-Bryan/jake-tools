"""The complete path against the real native service and real models.

``uv run pytest --slow -k transcription_live``

Spawns ``transcription-service serve`` from this repository's Swift build,
waits until it is warm, synthesises a scripted two-voice, two-file recording
with macOS ``say``, and drives it through the real MCP server's ``transcribe``
tool. This is the only test that says anything about transcription itself:
that speech comes back as the words spoken, that two voices become two
request-wide speakers whose identity survives the file boundary, that
timestamps land in the right files, and how long a warm request takes.

Text-to-speech audio is clean and never overlaps, so a pass here is evidence
of mechanics and of basic recognition quality, not of accuracy on real
meetings. Skipped (with the reason) when the binary, the models or the TTS
voices are not available; never faked.

Requirements: ``transcription-service/.build/{debug,release}/transcription-service``
(or ``JAKE_TRANSCRIPTION_SERVICE_BIN``) and provisioned models
(``TRANSCRIPTION_MODELS_DIR`` or FluidAudio's default cache).
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager
from pathlib import Path

import pytest
import requests
from mcp import ClientSession
from transcription_fixtures import (
    Conversation,
    normalise,
    synthesise_conversation,
    tools_available,
)

from jake_tools.transcription import (
    Transcript,
    TranscriptionClient,
    TranscriptionResult,
    transcript_summary,
)

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parent.parent
SERVICE_DIR = REPO / "transcription-service"
READY_TIMEOUT_SECONDS = 600.0

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]


def _service_binary() -> Path | None:
    override = os.environ.get("JAKE_TRANSCRIPTION_SERVICE_BIN")
    candidates = [Path(override)] if override else []
    candidates += [
        SERVICE_DIR / ".build" / "release" / "transcription-service",
        SERVICE_DIR / ".build" / "debug" / "transcription-service",
    ]
    return next((c for c in candidates if c.is_file() and os.access(c, os.X_OK)), None)


def _models_root() -> Path:
    override = os.environ.get("TRANSCRIPTION_MODELS_DIR")
    if override:
        return Path(override)
    return Path.home() / "Library" / "Application Support" / "FluidAudio" / "Models"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LiveService:
    def __init__(self, binary: Path, models_root: Path, work_dir: Path) -> None:
        self.token = secrets.token_urlsafe(24)
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        token_file = work_dir / "token"
        token_file.write_text(self.token)
        self.log_path = work_dir / "service.log"
        self._log = self.log_path.open("w")
        self.process = subprocess.Popen(
            [
                str(binary),
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--token-file",
                str(token_file),
                "--models-dir",
                str(models_root),
                "--work-dir",
                str(work_dir / "work"),
                "--max-queued",
                "2",
            ],
            stdout=self._log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        self.started = time.monotonic()
        self.ready_after: float | None = None

    def wait_ready(self) -> None:
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"service exited with {self.process.returncode}:\n{self.log()}"
                )
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
            except requests.RequestException:
                time.sleep(0.2)
                continue
            report = response.json()
            if report.get("ready"):
                self.ready_after = time.monotonic() - self.started
                return
            if report.get("status") == "failed":
                raise RuntimeError(f"models failed to load: {report.get('detail')}")
            time.sleep(0.5)
        raise RuntimeError(
            f"service not ready after {READY_TIMEOUT_SECONDS}s:\n{self.log()}"
        )

    def log(self) -> str:
        self._log.flush()
        return self.log_path.read_text()

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self._log.close()


@pytest.fixture(scope="module")
def live_service(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LiveService]:
    if sys.platform != "darwin":
        pytest.skip("the native service runs on macOS only")
    binary = _service_binary()
    if binary is None:
        pytest.skip(
            "transcription-service is not built; run `swift build` in transcription-service/"
        )
    models = _models_root()
    if not (models / "parakeet-tdt-0.6b-v3").is_dir():
        pytest.skip(
            f"models are not provisioned under {models}; run `transcription-service provision`"
        )
    reason = tools_available()
    if reason is not None:
        pytest.skip(f"cannot synthesise fixtures: {reason}")
    service = LiveService(binary, models, tmp_path_factory.mktemp("service"))
    try:
        service.wait_ready()
        yield service
    finally:
        service.stop()


@pytest.fixture(scope="module")
def conversation(tmp_path_factory: pytest.TempPathFactory) -> Conversation:
    return synthesise_conversation(tmp_path_factory.mktemp("fixtures"))


def _words(transcript: Transcript) -> list[str]:
    return [normalise(w.text) for s in transcript.segments for w in s.words]


def _recall(expected: set[str], found: set[str]) -> float:
    return len(expected & found) / len(expected)


def _assert_transcript_matches_conversation(
    transcript: Transcript, conversation: Conversation
) -> None:
    summary = transcript_summary(transcript)
    assert transcript.schema_version == 1
    assert transcript.engine.name == "FluidAudio"
    assert transcript.timing_resolution_seconds == 0.08

    # One continuous timeline whose file boundaries are where we built them.
    total = sum(conversation.file_durations)
    assert abs(transcript.duration_seconds - total) < 0.05
    assert [f.index for f in transcript.files] == [0, 1]
    assert (
        abs(transcript.files[1].start_seconds - conversation.file_durations[0]) < 0.05
    )

    # Words: almost everything scripted was recognised, in order.
    expected = [w for u in conversation.utterances for w in u.words]
    found = _words(transcript)
    recall = _recall(set(expected), set(found))
    assert recall >= 0.9, f"word recall {recall:.2f}: {summary}\n{found}"
    assert transcript.text
    assert all(
        w.end_seconds >= w.start_seconds for s in transcript.segments for w in s.words
    )
    starts = [w.start_seconds for s in transcript.segments for w in s.words]
    assert starts == sorted(starts)

    # Two voices became two request-wide speakers, each holding most of the
    # words that voice spoke, and the first speaker is the first voice.
    assert len(transcript.speakers) == 2, summary
    assert [s.id for s in transcript.speakers] == ["speaker_1", "speaker_2"]
    by_speaker: dict[str | None, set[str]] = {}
    for segment in transcript.segments:
        for word in segment.words:
            by_speaker.setdefault(word.speaker, set()).add(normalise(word.text))
    for label, voice in zip(
        ("speaker_1", "speaker_2"), conversation.voices, strict=True
    ):
        own = conversation.words_for(voice) - conversation.words_for(
            next(v for v in conversation.voices if v != voice)
        )
        attributed = by_speaker.get(label, set())
        share = _recall(own, attributed)
        assert share >= 0.8, f"{label} holds {share:.2f} of {voice}'s distinctive words"
    assert transcript.unassigned_word_count <= 0.1 * len(found)

    # Speaker identity survives the file boundary: the voice that closes
    # file one opens file two under the same label.
    boundary = conversation.file_durations[0]
    before = [
        w for s in transcript.segments for w in s.words if w.end_seconds <= boundary
    ]
    after = [
        w for s in transcript.segments for w in s.words if w.start_seconds >= boundary
    ]
    assert before and after
    assert before[-1].speaker == after[0].speaker
    closing_voice = conversation.utterances[5].voice
    opening_voice = conversation.utterances[6].voice
    assert closing_voice == opening_voice  # the fixture's design
    # Each utterance's words were attributed to the speaker that spoke them.
    for utterance in conversation.utterances:
        words_in = [
            w
            for s in transcript.segments
            for w in s.words
            if utterance.start_seconds - 0.3
            <= w.start_seconds
            <= utterance.end_seconds + 0.3
        ]
        if not words_in:
            continue
        label = f"speaker_{conversation.voices.index(utterance.voice) + 1}"
        agreeing = sum(1 for w in words_in if w.speaker == label)
        assert agreeing / len(words_in) >= 0.8, (
            utterance.text,
            [(w.text, w.speaker) for w in words_in],
        )


def test_warm_service_transcribes_two_files_through_the_real_mcp_server(
    live_service: LiveService,
    conversation: Conversation,
    mcp_server: ServerFactory,
    mcp_env_factory: Callable[..., dict[str, str]],
    mcp_home: Path,
) -> None:
    import anyio

    env = mcp_env_factory(
        mcp_home,
        TRANSCRIPTION_API_BASE_URL=live_service.base_url,
        TRANSCRIPTION_API_TOKEN=live_service.token,
    )
    files = [str(p) for p in conversation.m4a_files]

    async def run() -> tuple[TranscriptionResult, float, TranscriptionResult, float]:
        async with mcp_server(env) as session:
            assert "transcribe" in {t.name for t in (await session.list_tools()).tools}
            start = time.monotonic()
            first = await session.call_tool("transcribe", {"files": files})
            first_seconds = time.monotonic() - start
            assert first.is_error is False, first.structured_content
            start = time.monotonic()
            second = await session.call_tool("transcribe", {"files": files})
            second_seconds = time.monotonic() - start
            assert second.is_error is False, second.structured_content
        return (
            TranscriptionResult.model_validate(first.structured_content),
            first_seconds,
            TranscriptionResult.model_validate(second.structured_content),
            second_seconds,
        )

    first, first_seconds, second, second_seconds = anyio.run(run)
    assert [i.path for i in first.inputs] == files
    _assert_transcript_matches_conversation(first.transcript, conversation)
    _assert_transcript_matches_conversation(second.transcript, conversation)
    # Warm reuse: no model reload between requests, and the second request is
    # not slower than the first by more than noise.
    log = live_service.log()
    assert log.count("ASR models loaded") == 1
    assert log.count("diarization models loaded") == 1
    assert second_seconds < first_seconds * 1.5 + 1.0
    # Visible with ``-s``; the numbers are the point of this test.
    print(
        "\n[transcription_live] "
        + json.dumps(
            {
                "cold_start_to_ready_seconds": round(live_service.ready_after or 0, 2),
                "recording_seconds": round(first.transcript.duration_seconds, 2),
                "first_request_seconds": round(first_seconds, 2),
                "second_request_seconds": round(second_seconds, 2),
                "service_processing": first.transcript.processing.model_dump(),
                "summary": transcript_summary(first.transcript),
                "text": first.transcript.text,
            }
        )
    )


def test_wav_and_m4a_inputs_agree_and_single_file_timeline_starts_at_zero(
    live_service: LiveService, conversation: Conversation
) -> None:
    client = TranscriptionClient(
        live_service.base_url, live_service.token, timeout_seconds=300
    )
    from_wav = client.transcribe(conversation.wav_files)
    from_m4a = client.transcribe(conversation.m4a_files)
    assert _recall(set(_words(from_wav)), set(_words(from_m4a))) >= 0.9
    single = client.transcribe([conversation.wav_files[1]])
    assert len(single.files) == 1
    assert single.files[0].start_seconds == 0.0
    assert abs(single.duration_seconds - conversation.file_durations[1]) < 0.05
    expected = {
        w for u in conversation.utterances if u.file_index == 1 for w in u.words
    }
    assert _recall(expected, set(_words(single))) >= 0.9


def test_queue_bound_and_disconnect_against_the_real_service(
    live_service: LiveService, conversation: Conversation
) -> None:
    """Four concurrent uploads against max-queued 2: one runs, two wait, one
    is refused as overloaded; a client that drops mid-inference frees the slot."""
    import concurrent.futures

    def post(timeout: float) -> int:
        with open(conversation.wav_files[0], "rb") as handle:
            response = requests.post(
                f"{live_service.base_url}/v1/transcriptions",
                headers={"Authorization": f"Bearer {live_service.token}"},
                files=[("file", ("a.wav", handle, "audio/wav"))],
                timeout=timeout,
            )
        return response.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(post, 300) for _ in range(4)]
        statuses = sorted(f.result() for f in futures)
    assert statuses.count(200) == 3, statuses
    assert statuses.count(429) == 1, statuses

    with pytest.raises(requests.ReadTimeout):
        post(0.5)
    deadline = time.monotonic() + 60
    queue: dict[str, int] = {}
    while time.monotonic() < deadline:
        queue = requests.get(f"{live_service.base_url}/health", timeout=5).json()[
            "queue"
        ]
        if queue["running"] == 0 and queue["queued"] == 0:
            break
        time.sleep(0.5)
    assert queue == {"running": 0, "queued": 0, "max_queued": 2}
    assert "client disconnected; cancelling request" in live_service.log()
    work = Path(live_service.log_path).parent / "work"
    assert not any(work.iterdir())


@pytest.fixture
async def _unused() -> AsyncIterator[None]:  # keeps pytest-asyncio's mode exercised
    yield None
