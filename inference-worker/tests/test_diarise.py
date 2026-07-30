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

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from inference_worker.diarise import (
    PIPELINE_ID,
    _build_segments,
    _speaker_kwargs,
    diarisation_stage_config_hash,
    run_diarisation,
)
from inference_worker.models import SpeakerConstraints
from inference_worker.provenance import config_hash


def test_diarisation_stage_config_hash_folds_in_the_resolved_model_revision():
    """M2: two runs against different cached model revisions must hash
    differently — config_hash must not cover only the compile-time
    PIPELINE_ID constant. Compares against a hash built from PIPELINE_ID
    (+ constraints) alone to prove the revision and package versions are
    actually mixed in, without needing to fake a second real revision."""
    constraints = SpeakerConstraints()

    hash_with_revision = diarisation_stage_config_hash(constraints)
    hash_without_revision = config_hash(
        {"pipeline_id": PIPELINE_ID, **constraints.model_dump()}
    )

    assert hash_with_revision != hash_without_revision


def test_diarisation_stage_config_hash_survives_a_missing_package(monkeypatch):
    """N3: diarisation_stage_config_hash() runs before run_diarisation's
    own try block even starts — a strict package_version() call here
    would raise PackageNotFoundError and degrade a per-stage failure into
    a whole-run worker-internal-error. Simulates a genuinely missing
    package by making the real importlib.metadata.version lookup fail,
    the actual boundary a missing package would hit."""
    import importlib.metadata

    def _always_raise(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _always_raise)

    result = diarisation_stage_config_hash(SpeakerConstraints())

    assert len(result) == 64  # a real sha256 hex digest — it didn't raise


def test_run_diarisation_reports_timeout_and_still_carries_model_provenance():
    result = run_diarisation(
        Path("does-not-need-to-exist.wav"),
        "a" * 64,
        SpeakerConstraints(),
        duration_ms=1000,
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


def test_run_diarisation_retains_the_wav_on_failure():
    """M5: retained_artefacts is the flagship partial-failure record — a
    failed diarisation stage must name the prepared wav that still
    survives it, not leave the field dead/empty. N5: as a bare filename
    relative to the out-dir, consistent with ArtefactRef's filenames —
    not the absolute path the stage was actually called with."""
    wav_path = Path("some/out-dir/prepared.wav")

    result = run_diarisation(
        wav_path, "a" * 64, SpeakerConstraints(), duration_ms=1000, timeout_s=0.0001
    )

    assert result.error is not None
    assert result.error.retained_artefacts == ["prepared.wav"]


def test_speaker_kwargs_maps_exact_to_num_speakers():
    assert _speaker_kwargs(SpeakerConstraints(exact_speakers=3)) == {"num_speakers": 3}
    assert _speaker_kwargs(SpeakerConstraints(min_speakers=1, max_speakers=4)) == {
        "min_speakers": 1,
        "max_speakers": 4,
    }
    assert _speaker_kwargs(SpeakerConstraints()) == {}


# --- M6: overshoot clamp + span validation on raw diarisation output -----


@dataclass
class _FakeTurn:
    start: float
    end: float


class _FakeAnnotation:
    """Stands in for pyannote's Annotation — a real external-library
    boundary type, not the code under test — exposing just the
    itertracks() shape run_diarisation actually consumes."""

    def __init__(self, tracks: list[tuple[_FakeTurn, None, str]]) -> None:
        self._tracks = tracks

    def itertracks(self, yield_label: bool = True):
        return iter(self._tracks)


def test_build_segments_clamps_end_ms_overshoot_past_duration():
    """M6: pyannote's frame-boundary rounding can put a segment's end a
    few ms past the audio's actual duration (verified: 38ms overshoot on
    a real run) — clamp to what's real."""
    annotation = _FakeAnnotation(
        [(_FakeTurn(start=1.0, end=4.038), None, "SPEAKER_00")]
    )

    segments = _build_segments(annotation, duration_ms=4000)

    assert segments[0].start_ms == 1000
    assert segments[0].end_ms == 4000


def test_build_segments_preserves_a_segment_within_duration_unclamped():
    annotation = _FakeAnnotation([(_FakeTurn(start=1.0, end=2.0), None, "SPEAKER_00")])

    segments = _build_segments(annotation, duration_ms=10_000)

    assert segments[0].start_ms == 1000
    assert segments[0].end_ms == 2000


def test_build_segments_raises_when_clamping_cannot_fix_a_reversed_span():
    """A segment starting after the file's own duration can't be clamped
    into a valid span — that's a modelling error, not evidence worth
    keeping, so run_diarisation converts this into a stage failure."""
    annotation = _FakeAnnotation([(_FakeTurn(start=5.0, end=5.5), None, "SPEAKER_00")])

    with pytest.raises(ValidationError):
        _build_segments(annotation, duration_ms=4000)
