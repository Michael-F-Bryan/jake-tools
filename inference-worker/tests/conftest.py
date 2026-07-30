"""Shared test fixtures: synthetic audio generation and request building.

Synthetic sine-wave audio only — stdlib `wave`/`struct`/`math`, no numpy
or soundfile dependency. Tests never read any file under the Obsidian
vault or the evaluation corpus.
"""

from __future__ import annotations

import math
import struct
import wave
from collections.abc import Callable
from pathlib import Path

import pytest

from inference_worker.models import (
    AudioArtefact,
    InferenceRequest,
    ModelIdentity,
    RuntimeProvenance,
    SpeakerConstraints,
    StageTimeouts,
)
from inference_worker.provenance import sha256_file

# A structurally valid M1 request ID: "<prefix>_<uuid7>". The uuid7 half
# has version nibble 7 and variant nibble in {8,9,a,b}; it doesn't need to
# encode a real timestamp for these tests, only the right shape.
VALID_REQUEST_ID = "test_018f4c3e-1c1a-7f00-8b1a-2f6b6c1b0a11"


def write_sine_wav(
    path: Path,
    *,
    seconds: float = 1.0,
    freq_hz: float = 440.0,
    sample_rate: int = 44100,
    channels: int = 1,
    sample_width: int = 2,
    amplitude: int = 3000,
) -> Path:
    """Write a small PCM sine-wave wav file for tests. No numpy required."""
    n_samples = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            value = int(amplitude * math.sin(2 * math.pi * freq_hz * i / sample_rate))
            frames += struct.pack("<h", value) * channels
        wf.writeframes(bytes(frames))
    return path


def make_request(
    *,
    audio_path: Path,
    audio_sha256: str | None = None,
    request_id: str = VALID_REQUEST_ID,
    speaker_constraints: SpeakerConstraints | None = None,
    prepare_s: float = 30.0,
    asr_s: float = 60.0,
    diarise_s: float = 60.0,
) -> InferenceRequest:
    """A minimal, valid InferenceRequest pointed at a real audio file."""
    return InferenceRequest(
        request_id=request_id,
        audio=AudioArtefact(
            path=str(audio_path),
            sha256=audio_sha256 or sha256_file(audio_path),
        ),
        asr_model=ModelIdentity(
            name="mlx-community/parakeet-tdt-0.6b-v2", version="expected"
        ),
        diarisation_model=ModelIdentity(
            name="pyannote/speaker-diarization-community-1", version="expected"
        ),
        runtime_provenance=RuntimeProvenance(
            python_version="3.12.11",
            ml_framework_versions={"parakeet-mlx": "0.5.2"},
            worker_package_version="0.1.0",
            dependency_lockfile_sha256="0" * 64,
        ),
        speaker_constraints=speaker_constraints or SpeakerConstraints(),
        stage_timeouts=StageTimeouts(
            prepare_s=prepare_s, asr_s=asr_s, diarise_s=diarise_s
        ),
    )


@pytest.fixture
def sine_wav_factory() -> Callable[..., Path]:
    return write_sine_wav


@pytest.fixture
def request_factory() -> Callable[..., InferenceRequest]:
    return make_request
