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


def prepare_stage_config_hash(config: AudioPreparationConfig) -> str:
    return config_hash(config.model_dump())


def prepare_audio(
    source_path: Path,
    out_dir: Path,
    config: AudioPreparationConfig,
    timeout_s: float,
) -> PrepareStageResult:
    stage_config_hash = prepare_stage_config_hash(config)
    output_path = out_dir / PREPARED_FILENAME
    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i",
        str(source_path),
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
        str(output_path),
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
            error_class="timeout",
            message=f"ffmpeg exceeded its {timeout_s}s timeout",
            retryable=True,
        )
    except OSError as exc:
        return _failed(
            stage_config_hash,
            started,
            error_class="ffmpeg-unavailable",
            message=f"could not launch ffmpeg: {exc}",
            retryable=False,
        )

    if result.returncode != 0:
        stderr_tail = result.stderr.decode("utf-8", errors="replace")[-2000:]
        return PrepareStageResult(
            status="failed",
            config_hash=stage_config_hash,
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
    *,
    error_class: str,
    message: str,
    retryable: bool,
) -> PrepareStageResult:
    return PrepareStageResult(
        status="failed",
        config_hash=stage_config_hash,
        observations=stage_observations(started),
        error=StageError(error_class=error_class, message=message, retryable=retryable),
    )
