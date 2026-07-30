"""Request/response round-trip and validation-error behaviour (M11, M1, M6)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inference_worker.models import (
    ArtefactRef,
    AsrOutput,
    AsrStageResult,
    AsrToken,
    AudioArtefact,
    DiarisationSegment,
    InferenceRequest,
    InferenceResponse,
    PreparedAudio,
    PrepareStageResult,
    RuntimeProvenance,
    SpeakerConstraints,
    StageObservations,
)

_VALID_ID = "attempt_018f4c3e-1c1a-7f00-8b1a-2f6b6c1b0a11"


def test_request_round_trips_through_json(request_factory, sine_wav_factory, tmp_path):
    wav = sine_wav_factory(tmp_path / "source.wav")
    request = request_factory(audio_path=wav)

    restored = InferenceRequest.model_validate_json(request.model_dump_json())

    assert restored == request


def test_request_id_must_match_prefix_uuid7_format(
    request_factory, sine_wav_factory, tmp_path
):
    wav = sine_wav_factory(tmp_path / "source.wav")

    with pytest.raises(ValidationError, match="M1"):
        request_factory(audio_path=wav, request_id="not-a-valid-id")


def test_request_id_rejects_uuid4_shape(request_factory, sine_wav_factory, tmp_path):
    wav = sine_wav_factory(tmp_path / "source.wav")
    # Valid UUID shape, but version nibble '4' (uuid4), not '7'.
    uuid4_id = "attempt_018f4c3e-1c1a-4f00-8b1a-2f6b6c1b0a11"

    with pytest.raises(ValidationError, match="M1"):
        request_factory(audio_path=wav, request_id=uuid4_id)


@pytest.mark.parametrize("prefix", ["banana", "test", "req", "infreq"])
def test_request_id_rejects_prefixes_outside_the_m1_set(
    request_factory, sine_wav_factory, tmp_path, prefix
):
    """m3: the prefix is drawn from M1's fixed set ("attempt" for the
    normal per-operation-attempt invocation, "run" for manual diagnostic
    runs) — an arbitrary prefix like "banana", the old fixture prefix
    "test", or the interim "infreq" reservation must be rejected."""
    wav = sine_wav_factory(tmp_path / "source.wav")
    other_prefix_id = f"{prefix}_018f4c3e-1c1a-7f00-8b1a-2f6b6c1b0a11"

    with pytest.raises(ValidationError, match="prefix set"):
        request_factory(audio_path=wav, request_id=other_prefix_id)


def test_request_id_accepts_the_reserved_attempt_prefix(
    request_factory, sine_wav_factory, tmp_path
):
    wav = sine_wav_factory(tmp_path / "source.wav")

    request = request_factory(audio_path=wav, request_id=_VALID_ID)

    assert request.request_id == _VALID_ID


def test_audio_sha256_must_be_hex_digest(request_factory, sine_wav_factory, tmp_path):
    wav = sine_wav_factory(tmp_path / "source.wav")

    with pytest.raises(ValidationError):
        request_factory(audio_path=wav, audio_sha256="not-hex")


@pytest.mark.parametrize(
    "constraints_kwargs",
    [
        {"exact_speakers": 2, "min_speakers": 1},
        {"exact_speakers": 2, "max_speakers": 3},
        {"min_speakers": 3, "max_speakers": 2},
    ],
)
def test_inconsistent_speaker_constraints_are_rejected(constraints_kwargs):
    with pytest.raises(ValidationError):
        SpeakerConstraints(**constraints_kwargs)


def test_consistent_speaker_constraints_are_accepted():
    exact = SpeakerConstraints(exact_speakers=2)
    bounded = SpeakerConstraints(min_speakers=1, max_speakers=3)
    assert exact.exact_speakers == 2
    assert bounded.min_speakers == 1 and bounded.max_speakers == 3


def test_completed_stage_result_requires_output():
    with pytest.raises(ValidationError, match="requires output"):
        AsrStageResult(
            status="completed",
            config_hash="abc",
            observations=StageObservations(wall_time_ms=1),
        )


def test_completed_stage_result_rejects_error():
    with pytest.raises(ValidationError, match="must not carry an error"):
        AsrStageResult(
            status="completed",
            config_hash="abc",
            observations=StageObservations(wall_time_ms=1),
            output=AsrOutput(text="", tokens=[]),
            error={"error_class": "x", "message": "x", "retryable": False},
        )


def test_failed_stage_result_requires_error():
    with pytest.raises(ValidationError, match="requires an error"):
        PrepareStageResult(
            status="failed",
            config_hash="abc",
            source_sha256="a" * 64,
            observations=StageObservations(wall_time_ms=1),
        )


# --- M6: span invariants ----------------------------------------------


@pytest.mark.parametrize("span_cls", [AsrToken, DiarisationSegment])
def test_reversed_span_is_rejected(span_cls):
    """M6: end_ms < start_ms is a modelling error, not raw evidence — it
    must never silently construct."""
    kwargs = {"start_ms": 100, "end_ms": 50}
    if span_cls is AsrToken:
        kwargs |= {"text": "x", "confidence": 0.5}
    else:
        kwargs |= {"speaker_label": "SPEAKER_00"}

    with pytest.raises(ValidationError, match="end_ms"):
        span_cls(**kwargs)


@pytest.mark.parametrize("span_cls", [AsrToken, DiarisationSegment])
def test_negative_span_bound_is_rejected(span_cls):
    kwargs = {"start_ms": -1, "end_ms": 10}
    if span_cls is AsrToken:
        kwargs |= {"text": "x", "confidence": 0.5}
    else:
        kwargs |= {"speaker_label": "SPEAKER_00"}

    with pytest.raises(ValidationError):
        span_cls(**kwargs)


def test_zero_length_asr_token_is_preserved_as_evidence():
    """M6: parakeet's duration head structurally emits some zero-length
    tokens (observed 1 in 82 on a real run) — start_ms == end_ms is valid
    and must be preserved as-is, not rejected or silently dropped."""
    token = AsrToken(text="", start_ms=1234, end_ms=1234, confidence=0.9)

    assert token.start_ms == token.end_ms == 1234


def test_zero_length_diarisation_segment_is_preserved_as_evidence():
    segment = DiarisationSegment(start_ms=500, end_ms=500, speaker_label="SPEAKER_00")

    assert segment.start_ms == segment.end_ms == 500


# --- response shape (M3, M4, m6) ---------------------------------------


def test_inference_response_round_trips_with_artefact_refs_and_delta():
    """InferenceResponse (m6) only carries ArtefactRef pointers to the
    separately-written asr.json/diarisation.json files — the actual
    partial-failure content lives in those files and is exercised
    end-to-end in test_orchestrator.py. This test covers the response's
    own shape: the echoed source audio ref (M3) and the declared-vs-
    observed runtime delta (M4)."""
    response = InferenceResponse(
        request_id=_VALID_ID,
        audio=AudioArtefact(path="source.wav", sha256="a" * 64),
        prepare=PrepareStageResult(
            status="completed",
            config_hash="prep-hash",
            source_sha256="a" * 64,
            observations=StageObservations(wall_time_ms=10),
            output=PreparedAudio(
                path="prepared.wav", sha256="b" * 64, duration_ms=1000
            ),
        ),
        asr=ArtefactRef(filename="asr.json", sha256="c" * 64, status="completed"),
        diarisation=ArtefactRef(
            filename="diarisation.json", sha256="d" * 64, status="failed"
        ),
        runtime=RuntimeProvenance(
            python_version="3.12.11",
            ml_framework_versions={},
            worker_package_version="0.1.0",
            dependency_lockfile_sha256="0" * 64,
        ),
        runtime_provenance_delta={"python_version": "declared=3.12.0 observed=3.12.11"},
    )

    restored = InferenceResponse.model_validate_json(response.model_dump_json())
    assert restored.audio.sha256 == "a" * 64
    assert restored.asr.status == "completed"
    assert restored.asr.filename == "asr.json"
    assert restored.diarisation.status == "failed"
    assert restored.runtime_provenance_delta == {
        "python_version": "declared=3.12.0 observed=3.12.11"
    }


def test_prepare_stage_result_always_carries_source_sha256():
    """M3: source_sha256 is known before ffmpeg ever runs (it's already
    verified against the request), so it's present even on a failed
    prepare stage — resume-by-hash needs the source hash, not just the
    (possibly never-produced) prepared wav's."""
    result = PrepareStageResult(
        status="failed",
        config_hash="prep-hash",
        source_sha256="a" * 64,
        observations=StageObservations(wall_time_ms=1),
        error={"error_class": "ffmpeg-failed", "message": "boom", "retryable": False},
    )

    assert result.source_sha256 == "a" * 64
    assert result.output is None
