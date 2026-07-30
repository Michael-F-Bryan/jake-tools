"""Real Pyannote Community-1 diarisation test. Heavyweight (gated model
download/load) — opt-in via `-m model`, excluded from the default
`pytest -q` run.

m7: access is granted on this machine, so this asserts `completed`
outright rather than accepting either outcome — an either/or assertion
would pass silently even with a genuinely broken diariser. If access is
ever revoked (a `model-access-denied` failure, per the KNOWN CONSTRAINT
in this worker's brief), the test skips with the reason instead of
failing, so a real regression in the diariser itself is never confused
with an access/credentials problem. Never falls back to a hosted API
(M15) regardless of which branch this hits.
"""

from __future__ import annotations

import pytest

from inference_worker.diarise import run_diarisation
from inference_worker.models import SpeakerConstraints
from inference_worker.provenance import sha256_file

pytestmark = pytest.mark.model


def test_run_diarisation_completes_on_synthetic_audio(sine_wav_factory, tmp_path):
    wav = sine_wav_factory(
        tmp_path / "prepared.wav", seconds=4.0, sample_rate=16000, channels=1
    )
    wav_sha256 = sha256_file(wav)

    result = run_diarisation(
        wav,
        wav_sha256,
        SpeakerConstraints(min_speakers=1, max_speakers=2),
        duration_ms=4000,
        timeout_s=180.0,
    )

    if (
        result.status == "failed"
        and result.error is not None
        and result.error.error_class == "model-access-denied"
    ):
        pytest.skip(
            f"Community-1 gated access denied on this machine: {result.error.message}"
        )

    assert result.status == "completed", result.error
    assert result.input_audio_sha256 == wav_sha256
    assert result.output is not None
    assert isinstance(result.output.segments, list)
    # Public repo metadata (the pinned revision) must have resolved from
    # the local cache — it's what we just loaded the weights from.
    assert result.model_provenance is not None
    assert (
        result.model_provenance.identity.name
        == "pyannote/speaker-diarization-community-1"
    )
