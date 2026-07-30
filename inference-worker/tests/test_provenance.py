"""Provenance and hashing helpers: real hashing, real importlib.metadata
lookups, real (if unreachable) HTTP fallback — no mocking."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_worker.provenance import (
    config_hash,
    hf_repo_revision,
    observed_runtime_provenance,
    package_version,
    peak_rss_bytes,
    sha256_file,
)


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

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


def test_hf_repo_revision_falls_back_to_unknown_for_nonexistent_repo():
    revision = hf_repo_revision(
        "this-org-does-not-exist/this-repo-does-not-exist-either"
    )
    assert revision == "unknown"


def test_peak_rss_bytes_is_a_positive_int_or_none():
    value = peak_rss_bytes()
    assert value is None or value > 0


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
