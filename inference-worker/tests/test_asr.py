"""ASR stage behaviour that doesn't require downloading model weights.

Real `run_asr` call throughout — a near-zero timeout reliably interrupts
it before it gets anywhere near real inference, which is what makes this
fast enough for the default (non-model) suite while still exercising the
real timeout/provenance code path rather than a mock of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from inference_worker.asr import MODEL_ID, _build_tokens, asr_stage_config_hash, run_asr
from inference_worker.provenance import config_hash


def test_asr_stage_config_hash_folds_in_the_resolved_model_revision():
    """M2: two runs against different cached model revisions must hash
    differently — config_hash must not cover only the compile-time
    MODEL_ID constant. Compares against a hash built from MODEL_ID alone
    to prove the revision (and package versions) are actually mixed in,
    without needing to fake a second real revision on disk."""
    hash_with_revision = asr_stage_config_hash()
    hash_without_revision = config_hash({"model_id": MODEL_ID})

    assert hash_with_revision != hash_without_revision


def test_asr_stage_config_hash_survives_a_missing_package(monkeypatch):
    """N3: asr_stage_config_hash() runs before run_asr's own try block
    even starts — on a platform where a package is genuinely absent (e.g.
    mlx on non-Apple-Silicon Linux), a strict package_version() call here
    would raise PackageNotFoundError and degrade a per-stage failure into
    a whole-run worker-internal-error. Simulates that by making the real
    importlib.metadata.version lookup fail, the actual boundary a missing
    package would hit — not a patch of this module's own code."""
    import importlib.metadata

    def _always_raise(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _always_raise)

    result = asr_stage_config_hash()

    assert len(result) == 64  # a real sha256 hex digest — it didn't raise


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


def test_run_asr_retains_the_wav_on_failure():
    """M5: retained_artefacts is the flagship partial-failure record — a
    failed ASR stage must name the prepared wav that still survives it,
    not leave the field dead/empty. N5: as a bare filename relative to
    the out-dir, consistent with ArtefactRef's filenames — not the
    absolute path the stage was actually called with."""
    wav_path = Path("some/out-dir/prepared.wav")

    result = run_asr(wav_path, "a" * 64, timeout_s=0.0001)

    assert result.error is not None
    assert result.error.retained_artefacts == ["prepared.wav"]


# --- M6: span validation on raw ASR output -------------------------------


@dataclass
class _FakeToken:
    text: str
    start: float
    end: float
    confidence: float = 0.9


def test_build_tokens_preserves_a_zero_length_token():
    """Parakeet's duration head structurally emits some zero-length
    tokens (observed 1 in 82 on a real run) — must be preserved as
    evidence, not dropped or rejected."""
    tokens = _build_tokens([_FakeToken(text="", start=1.234, end=1.234)])

    assert tokens[0].start_ms == tokens[0].end_ms == 1234


def test_build_tokens_raises_on_a_reversed_span():
    """A token with end before start is a modelling error, not evidence
    worth keeping — run_asr converts this into a clean asr-invalid-output
    stage failure rather than letting it propagate raw."""
    with pytest.raises(ValidationError):
        _build_tokens([_FakeToken(text="x", start=2.0, end=1.0)])
