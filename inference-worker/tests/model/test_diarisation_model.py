"""Real Pyannote Community-1 diarisation test. Heavyweight (gated model
download/load) — opt-in via `-m model`, excluded from the default
`pytest -q` run.

This currently exercises whichever of the two honest M11 outcomes the
worker's Hugging Face credentials produce: `completed` once
`pyannote/speaker-diarization-community-1`'s gated terms are accepted, or
a clean `model-access-denied` failure while they aren't. Both are correct
per the KNOWN CONSTRAINT in this worker's brief — this test must never
assert one branch only, and must never fall back to a hosted API (M15).
"""

from __future__ import annotations

import pytest

from inference_worker.diarise import run_diarisation
from inference_worker.models import SpeakerConstraints
from inference_worker.provenance import sha256_file

pytestmark = pytest.mark.model


def test_run_diarisation_completes_or_cleanly_denies_access(sine_wav_factory, tmp_path):
    wav = sine_wav_factory(
        tmp_path / "prepared.wav", seconds=4.0, sample_rate=16000, channels=1
    )
    wav_sha256 = sha256_file(wav)

    result = run_diarisation(
        wav,
        wav_sha256,
        SpeakerConstraints(min_speakers=1, max_speakers=2),
        timeout_s=180.0,
    )

    assert result.status in ("completed", "failed")
    if result.status == "completed":
        assert result.input_audio_sha256 == wav_sha256
        assert result.output is not None
        assert isinstance(result.output.segments, list)
    else:
        assert result.error is not None
        assert result.error.error_class == "model-access-denied"
        assert result.error.retryable is True

    # Either way, public repo metadata (the pinned revision) must have
    # resolved — it doesn't require the gated weights.
    assert result.model_provenance is not None
    assert (
        result.model_provenance.identity.name
        == "pyannote/speaker-diarization-community-1"
    )
