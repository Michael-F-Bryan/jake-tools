"""Provenance and hashing helpers: real hashing, real importlib.metadata
lookups, real (local-only) cache scans — no mocking, no network."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pytest

from inference_worker.models import RuntimeProvenance
from inference_worker.provenance import (
    compute_runtime_delta,
    config_hash,
    local_model_revision,
    observed_runtime_provenance,
    package_version,
    peak_rss_bytes,
    sha256_file,
)


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"some bytes to hash" * 1000)

    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_config_hash_is_stable_regardless_of_key_order():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_config_hash_changes_with_content():
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_package_version_resolves_a_real_installed_package():
    # pydantic is a direct dependency of inference-worker itself.
    version = package_version("pydantic")
    assert version and version[0].isdigit()


def test_package_version_raises_for_unknown_distribution():
    from importlib.metadata import PackageNotFoundError

    with pytest.raises(PackageNotFoundError):
        package_version("definitely-not-a-real-package-xyz")


# --- B4 / M1: local-cache-only model revision resolution ----------------


def test_local_model_revision_resolves_a_cached_repo_from_disk():
    # Cached on this machine by this suite's own prior -m model runs.
    # scan_cache_dir() is a pure local filesystem scan, so this never
    # touches the network (B4) and reports the exact snapshot that would
    # actually load (M1), not the remote repo's current HEAD.
    revision = local_model_revision("mlx-community/parakeet-tdt-0.6b-v2")

    assert revision is not None
    assert len(revision) == 40  # git commit sha
    assert all(c in "0123456789abcdef" for c in revision)


def test_local_model_revision_returns_none_for_an_uncached_repo():
    # M1: an unresolvable revision is None, never a fabricated "unknown".
    revision = local_model_revision(
        "this-org-does-not-exist/this-repo-does-not-exist-either"
    )

    assert revision is None


def test_local_model_revision_is_fast_because_it_never_touches_the_network():
    """B4: hf_repo_revision() used to call the remote Hub API with no
    timeout before a stage's own enforce_timeout window even started —
    a stalled HF endpoint turned a 0.5s stage budget into 75.5s wall
    while the response recorded a falsified ~500ms observation.
    local_model_revision() must be a pure disk scan: bounded and fast
    regardless of network reachability, for a repo id that would
    definitely require a network round-trip to resolve remotely."""
    started = time.monotonic()
    local_model_revision("this-org-does-not-exist/this-repo-does-not-exist-either")
    elapsed_s = time.monotonic() - started

    assert elapsed_s < 2.0


# --- B3: graceful degradation, never raise away a completed run --------


def test_observed_runtime_provenance_degrades_on_missing_lockfile(tmp_path):
    missing_lockfile = tmp_path / "does-not-exist" / "uv.lock"

    runtime = observed_runtime_provenance(missing_lockfile)

    assert runtime.dependency_lockfile_sha256.startswith("unresolved:")
    # Every other field still resolves normally — one bad field must not
    # take the rest down with it.
    assert runtime.python_version.startswith("3.12")
    assert runtime.worker_package_version
    assert all(v for v in runtime.ml_framework_versions.values())


def test_observed_runtime_provenance_populates_all_fields():
    lockfile = Path(__file__).resolve().parents[1] / "uv.lock"

    runtime = observed_runtime_provenance(lockfile)

    assert runtime.python_version.startswith("3.12")
    assert set(runtime.ml_framework_versions) == {
        "parakeet-mlx",
        "mlx",
        "pyannote-audio",
        "torch",
    }
    assert all(v for v in runtime.ml_framework_versions.values())
    assert runtime.worker_package_version
    assert len(runtime.dependency_lockfile_sha256) == 64


def test_peak_rss_bytes_is_a_positive_int_or_none():
    value = peak_rss_bytes()
    assert value is None or value > 0


# --- M4: declared vs observed runtime delta -----------------------------


def _runtime(**overrides) -> RuntimeProvenance:
    base = {
        "python_version": "3.12.11",
        "ml_framework_versions": {"torch": "2.13.0"},
        "worker_package_version": "0.1.0",
        "dependency_lockfile_sha256": "a" * 64,
    }
    return RuntimeProvenance(**{**base, **overrides})


def test_compute_runtime_delta_is_empty_when_everything_matches():
    provenance = _runtime()

    assert compute_runtime_delta(provenance, provenance) == {}


def test_compute_runtime_delta_reports_mismatched_fields_only():
    declared = _runtime(
        python_version="3.12.0", ml_framework_versions={"torch": "2.12.0"}
    )
    observed = _runtime(
        python_version="3.12.11", ml_framework_versions={"torch": "2.13.0"}
    )

    delta = compute_runtime_delta(declared, observed)

    assert delta["python_version"] == "declared=3.12.0 observed=3.12.11"
    assert delta["ml_framework_versions.torch"] == "declared=2.12.0 observed=2.13.0"
    assert "worker_package_version" not in delta
    assert "dependency_lockfile_sha256" not in delta
