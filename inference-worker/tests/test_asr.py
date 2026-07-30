"""ASR stage behaviour that doesn't require downloading model weights.

Real `run_asr` call throughout — a near-zero timeout reliably interrupts
it before it gets anywhere near real inference, which is what makes this
fast enough for the default (non-model) suite while still exercising the
real timeout/provenance code path rather than a mock of it.
"""

from __future__ import annotations

from pathlib import Path

from inference_worker.asr import run_asr


def test_run_asr_reports_timeout_and_still_carries_model_provenance():
    result = run_asr(Path("does-not-need-to-exist.wav"), "a" * 64, timeout_s=0.0001)

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.error_class == "timeout"
    assert result.error.retryable is True
    # Public repo metadata resolves without needing the model weights, so
    # provenance should be available even though inference never ran.
    assert result.model_provenance is not None
    assert result.model_provenance.identity.name == "mlx-community/parakeet-tdt-0.6b-v2"
