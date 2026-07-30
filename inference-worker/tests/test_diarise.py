"""Diarisation stage behaviour that doesn't require downloading gated
weights.

Real `run_diarisation` call throughout — a near-zero timeout reliably
interrupts it before it gets anywhere near real inference, which is what
makes this fast enough for the default (non-model) suite while still
exercising the real timeout/provenance code path rather than a mock of it.
The model-access-denied / completed split is covered honestly by the
`-m model` test in tests/model/test_diarisation_model.py.
"""

from __future__ import annotations

from pathlib import Path

from inference_worker.diarise import _speaker_kwargs, run_diarisation
from inference_worker.models import SpeakerConstraints


def test_run_diarisation_reports_timeout_and_still_carries_model_provenance():
    result = run_diarisation(
        Path("does-not-need-to-exist.wav"),
        "a" * 64,
        SpeakerConstraints(),
        timeout_s=0.0001,
    )

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.error_class == "timeout"
    assert result.error.retryable is True
    # Public repo metadata resolves without needing the gated weights, so
    # provenance should be available even though inference never ran.
    assert result.model_provenance is not None
    assert (
        result.model_provenance.identity.name
        == "pyannote/speaker-diarization-community-1"
    )


def test_speaker_kwargs_maps_exact_to_num_speakers():
    assert _speaker_kwargs(SpeakerConstraints(exact_speakers=3)) == {"num_speakers": 3}
    assert _speaker_kwargs(SpeakerConstraints(min_speakers=1, max_speakers=4)) == {
        "min_speakers": 1,
        "max_speakers": 4,
    }
    assert _speaker_kwargs(SpeakerConstraints()) == {}
