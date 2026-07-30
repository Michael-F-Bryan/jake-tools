"""Request/response round-trip and validation-error behaviour (M11, M1)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inference_worker.models import (
    AsrOutput,
    AsrStageResult,
    AsrToken,
    InferenceRequest,
    InferenceResponse,
    PreparedAudio,
    PrepareStageResult,
    RuntimeProvenance,
    SpeakerConstraints,
    StageObservations,
)


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
    uuid4_id = "test_018f4c3e-1c1a-4f00-8b1a-2f6b6c1b0a11"

    with pytest.raises(ValidationError, match="M1"):
        request_factory(audio_path=wav, request_id=uuid4_id)


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
            observations=StageObservations(wall_time_ms=1),
        )


def test_inference_response_round_trips_with_partial_failure():
    """The exact M11 shape this contract cares about most: a completed ASR
    artefact surviving alongside a failed-retryable diarisation artefact."""
    response = InferenceResponse(
        request_id="test_018f4c3e-1c1a-7f00-8b1a-2f6b6c1b0a11",
        prepare=PrepareStageResult(
            status="completed",
            config_hash="prep-hash",
            observations=StageObservations(wall_time_ms=10),
            output=PreparedAudio(
                path="prepared.wav", sha256="a" * 64, duration_ms=1000
            ),
        ),
        asr=AsrStageResult(
            status="completed",
            config_hash="asr-hash",
            observations=StageObservations(wall_time_ms=20),
            output=AsrOutput(
                text="hi",
                tokens=[AsrToken(text="hi", start_ms=0, end_ms=100, confidence=0.9)],
            ),
        ),
        diarisation={
            "status": "failed",
            "config_hash": "diar-hash",
            "observations": {"wall_time_ms": 5},
            "error": {
                "error_class": "model-access-denied",
                "message": "gated",
                "retryable": True,
            },
        },
        runtime=RuntimeProvenance(
            python_version="3.12.11",
            ml_framework_versions={},
            worker_package_version="0.1.0",
            dependency_lockfile_sha256="0" * 64,
        ),
    )

    restored = InferenceResponse.model_validate_json(response.model_dump_json())
    assert restored.asr.status == "completed"
    assert restored.diarisation.status == "failed"
    assert restored.diarisation.error.retryable is True
    assert restored.diarisation.error.error_class == "model-access-denied"
