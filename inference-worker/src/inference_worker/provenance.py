"""Hashing, package/runtime version lookup, and best-effort peak memory.

Kept separate from the stage modules because every stage needs at least
one of these (input hashing, config hashing, or provenance lookup) — this
is the "one obvious home" for that cross-cutting concern.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

from huggingface_hub import model_info

from inference_worker.models import RuntimeProvenance, StageObservations

_HASH_CHUNK_SIZE = 1 << 20  # 1 MiB
_ML_DISTRIBUTIONS = ("parakeet-mlx", "mlx", "pyannote-audio", "torch")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(payload: dict[str, Any]) -> str:
    """Stable hash of a stage's effective configuration."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def package_version(distribution_name: str) -> str:
    return importlib.metadata.version(distribution_name)


def hf_repo_revision(repo_id: str) -> str:
    """Best-effort exact model version: the HF repo's current commit sha.

    Public repo metadata is reachable via the Hub API even when a gated
    repo's weights are not, so this resolves independently of
    ``model-access-denied`` failures.
    """
    try:
        return model_info(repo_id).sha or "unknown"
    except Exception:
        return "unknown"


def peak_rss_bytes() -> int | None:
    """Best-effort peak resident set size for this process.

    macOS reports ``ru_maxrss`` in bytes; Linux reports it in KiB.
    Returns None if the platform doesn't expose ``resource.getrusage``.
    """
    try:
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError):
        return None
    return raw if sys.platform == "darwin" else raw * 1024


def stage_observations(started: float) -> StageObservations:
    """A stage's wall-time + best-effort peak-RSS observation.

    Every stage builds this the same way (elapsed time since it started,
    plus current peak RSS), so it lives here rather than being
    reimplemented per stage module.
    """
    elapsed_ms = int((time.monotonic() - started) * 1000)
    return StageObservations(wall_time_ms=elapsed_ms, peak_rss_bytes=peak_rss_bytes())


def observed_runtime_provenance(lockfile_path: Path) -> RuntimeProvenance:
    """The worker's own actual runtime, as installed right now."""
    return RuntimeProvenance(
        python_version=platform.python_version(),
        ml_framework_versions={
            name: package_version(name) for name in _ML_DISTRIBUTIONS
        },
        worker_package_version=package_version("inference-worker"),
        dependency_lockfile_sha256=sha256_file(lockfile_path),
    )
