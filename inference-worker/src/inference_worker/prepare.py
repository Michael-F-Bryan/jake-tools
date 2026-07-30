"""Stage 1: normalise the source media to 16 kHz mono PCM s16le via ffmpeg.

Runs once per request. This is the only place ffmpeg is invoked — no
lossy intermediate encode, straight from the source artefact to the wav
the ASR/diarisation stages consume.
"""

from __future__ import annotations

import subprocess
import time
import wave
from pathlib import Path

from inference_worker.models import (
    AudioPreparationConfig,
    PreparedAudio,
    PrepareStageResult,
    StageError,
)
from inference_worker.provenance import config_hash, sha256_file, stage_observations

PREPARED_FILENAME = "prepared.wav"


def ffmpeg_version() -> str:
    """First line of ``ffmpeg -version`` (folded into the config hash,
    M2). Best-effort: "unknown" if ffmpeg can't be queried — the actual
    conversion attempt below still reports a proper ``ffmpeg-unavailable``
    stage failure in that case, this is only for provenance."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, timeout=5, check=False
        )
    except OSError:
        return "unknown"
    if result.returncode != 0 or not result.stdout:
        return "unknown"
    return result.stdout.decode("utf-8", errors="replace").splitlines()[0]


def prepare_stage_config_hash(config: AudioPreparationConfig) -> str:
    return config_hash({**config.model_dump(), "ffmpeg_version": ffmpeg_version()})


def _ffmpeg_file_arg(path: Path) -> str:
    """Force ffmpeg to treat `path` as a literal filename, never a
    protocol URL. ffmpeg's protocol detection treats anything shaped like
    `scheme:...` as a URL — a source filename containing a colon (e.g. a
    meeting title like "14:30 sync.wav") would otherwise misparse as an
    unknown protocol and fail (m1)."""
    return f"file:{path.resolve()}"


def prepare_audio(
    source_path: Path,
    out_dir: Path,
    config: AudioPreparationConfig,
    source_sha256: str,
    timeout_s: float,
) -> PrepareStageResult:
    stage_config_hash = prepare_stage_config_hash(config)
    output_path = out_dir / PREPARED_FILENAME
    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i",
        _ffmpeg_file_arg(source_path),
        "-vn",
        "-sn",
        "-ac",
        str(config.channels),
        "-ar",
        str(config.sample_rate_hz),
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
        _ffmpeg_file_arg(output_path),
    ]
    started = time.monotonic()
    try:
        result = subprocess.run(
            command, capture_output=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired:
        return _failed(
            stage_config_hash,
            started,
            source_sha256,
            error_class="timeout",
            message=f"ffmpeg exceeded its {timeout_s}s timeout",
            retryable=True,
        )
    except OSError as exc:
        return _failed(
            stage_config_hash,
            started,
            source_sha256,
            error_class="ffmpeg-unavailable",
            # A missing/unlaunchable ffmpeg binary is an environment
            # problem the caller can plausibly fix (install/PATH) and
            # retry, unlike ffmpeg rejecting the input itself (m4).
            message=f"could not launch ffmpeg: {exc}",
            retryable=True,
        )

    if result.returncode != 0:
        stderr_tail = result.stderr.decode("utf-8", errors="replace")[-2000:]
        return PrepareStageResult(
            status="failed",
            config_hash=stage_config_hash,
            source_sha256=source_sha256,
            observations=stage_observations(started),
            error=StageError(
                error_class="ffmpeg-failed",
                message=f"ffmpeg exited {result.returncode}: {stderr_tail}",
                retryable=False,
            ),
        )

    return PrepareStageResult(
        status="completed",
        config_hash=stage_config_hash,
        source_sha256=source_sha256,
        observations=stage_observations(started),
        output=PreparedAudio(
            path=str(output_path),
            sha256=sha256_file(output_path),
            duration_ms=_wav_duration_ms(output_path),
        ),
    )


def _wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
    return round(frames / rate * 1000)


def _failed(
    stage_config_hash: str,
    started: float,
    source_sha256: str,
    *,
    error_class: str,
    message: str,
    retryable: bool,
) -> PrepareStageResult:
    return PrepareStageResult(
        status="failed",
        config_hash=stage_config_hash,
        source_sha256=source_sha256,
        observations=stage_observations(started),
        error=StageError(error_class=error_class, message=message, retryable=retryable),
    )
