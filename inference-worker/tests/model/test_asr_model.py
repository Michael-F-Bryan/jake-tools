"""Real Parakeet ASR smoke test. Heavyweight (model download/load) —
opt-in via `-m model`, excluded from the default `pytest -q` run.
"""

from __future__ import annotations

import pytest

from inference_worker.asr import run_asr
from inference_worker.provenance import sha256_file

pytestmark = pytest.mark.model


def test_run_asr_completes_on_synthetic_audio(sine_wav_factory, tmp_path):
    # Real Parakeet ASR doesn't care about the source sample rate/channel
    # layout the way ffmpeg's prepare stage does, but the M11 pipeline
    # always hands it an already-prepared 16 kHz mono wav, so match that.
    wav = sine_wav_factory(
        tmp_path / "prepared.wav", seconds=3.0, sample_rate=16000, channels=1
    )
    wav_sha256 = sha256_file(wav)

    result = run_asr(wav, wav_sha256, timeout_s=180.0)

    assert result.status == "completed", result.error
    assert result.input_audio_sha256 == wav_sha256
    assert result.model_provenance is not None
    assert result.model_provenance.identity.name == "mlx-community/parakeet-tdt-0.6b-v2"
    assert result.model_provenance.package_versions["parakeet-mlx"]
    assert result.output is not None
    # Sine tones aren't speech: empty text/tokens is the expected,
    # non-crashing outcome — the point is the typed pipeline runs for real.
    assert isinstance(result.output.text, str)
    assert isinstance(result.output.tokens, list)
