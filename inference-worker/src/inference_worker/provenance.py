"""Hashing, package/runtime version lookup, local model-cache resolution,
and best-effort peak memory.

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

from inference_worker.models import (
    ModelProvenance,
    RuntimeProvenance,
    StageObservations,
)

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


def local_model_revision(repo_id: str) -> str | None:
    """The commit hash of the snapshot ``repo_id``'s "main" ref currently
    resolves to in huggingface_hub's on-disk cache, or None if nothing is
    cached yet.

    This is a pure local filesystem scan (``scan_cache_dir`` reads
    ``.../refs/main`` and ``.../snapshots/<sha>/`` under the cache root) —
    it never touches the network (B4: a stalled HF endpoint must never be
    able to blow a stage's timeout budget) and it reports the exact
    snapshot that will actually load (M1: the remote repo's current HEAD,
    via ``model_info().sha``, is not necessarily what's on disk — pinning
    to "main" plus a slow release cadence made this hard to notice, but
    they're different values by construction).
    """
    from huggingface_hub import scan_cache_dir

    try:
        cache_info = scan_cache_dir()
    except Exception:
        return None
    for repo in cache_info.repos:
        if repo.repo_id == repo_id and repo.repo_type == "model":
            for revision in repo.revisions:
                if "main" in revision.refs:
                    return revision.commit_hash
    return None


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
    plus current *process-wide, cumulative* peak RSS — see
    ``StageObservations.process_peak_rss_bytes``), so it lives here rather
    than being reimplemented per stage module.
    """
    elapsed_ms = int((time.monotonic() - started) * 1000)
    return StageObservations(
        wall_time_ms=elapsed_ms, process_peak_rss_bytes=peak_rss_bytes()
    )


def best_effort_package_version(distribution_name: str) -> str:
    try:
        return package_version(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        return f"unresolved: {exc}"


def _best_effort_lockfile_sha256(lockfile_path: Path) -> str:
    try:
        return sha256_file(lockfile_path)
    except OSError as exc:
        return f"unresolved: {exc}"


def observed_runtime_provenance(lockfile_path: Path) -> RuntimeProvenance:
    """The worker's own actual runtime, as installed right now.

    B3: a missing lockfile or an unresolvable package version (e.g. an
    ML distribution that isn't installed on this platform) must never
    raise here and throw away an otherwise-completed run — every field
    degrades independently, recording what it can and marking what it
    can't rather than aborting the whole response.
    """
    return RuntimeProvenance(
        python_version=platform.python_version(),
        ml_framework_versions={
            name: best_effort_package_version(name) for name in _ML_DISTRIBUTIONS
        },
        worker_package_version=best_effort_package_version("inference-worker"),
        dependency_lockfile_sha256=_best_effort_lockfile_sha256(lockfile_path),
    )


def compute_runtime_delta(
    declared: RuntimeProvenance, observed: RuntimeProvenance
) -> dict[str, str]:
    """M4: the fields where the request's declared runtime provenance and
    the worker's observed one disagree, described as "declared=X
    observed=Y". Empty when everything matches. Not enforced (only the
    audio hash gets hard contract-failure semantics) — recorded so a
    caller can detect drift instead of it being silently dropped."""
    delta: dict[str, str] = {}
    for field in (
        "python_version",
        "worker_package_version",
        "dependency_lockfile_sha256",
    ):
        declared_value, observed_value = (
            getattr(declared, field),
            getattr(observed, field),
        )
        if declared_value != observed_value:
            delta[field] = f"declared={declared_value} observed={observed_value}"
    for name, declared_version in declared.ml_framework_versions.items():
        observed_version = observed.ml_framework_versions.get(name, "<not observed>")
        if declared_version != observed_version:
            delta[f"ml_framework_versions.{name}"] = (
                f"declared={declared_version} observed={observed_version}"
            )
    return delta


def compute_model_revision_delta(
    field_name: str, declared_version: str, observed: ModelProvenance | None
) -> dict[str, str]:
    """N4: a declared model *version* is deliberately not enforced (unlike
    the model *name* — see __main__._refuse_unpinned_models; callers may
    not pin an exact revision) but silently accepting a mismatch without
    recording it would hide real drift. Folds into the same delta shape
    as compute_runtime_delta. No entry when nothing was actually observed
    (the stage never ran, or never resolved a revision)."""
    if observed is None or declared_version == observed.identity.version:
        return {}
    return {
        field_name: f"declared={declared_version} observed={observed.identity.version}"
    }
