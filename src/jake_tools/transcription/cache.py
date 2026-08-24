"""Content-addressed run cache for the transcription pipeline.

Every pipeline stage writes its output into a run directory keyed by a run id
(note-stem plus audio-hash prefix — computed by callers, not here). A rerun
that hits an already-cached stage skips straight to the next one, and the
integrate step reads its tier-b baseline back out of the same cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..ai_usage import AICallTelemetry, AITelemetry, AITotals, TelemetrySink
from ..cache_models import CacheEnvelope
from .models import StageTiming, StageTimingLog, StaleState

_DEFAULT_ROOT = Path.home() / "Library" / "Caches" / "jake-tools" / "transcription"
_CHUNK_SIZE = 1024 * 1024
_RESUMABLE_NAMES = {
    "asr_checkpoint",
    "diarisation_checkpoint",
    "raw_transcript",
    "timings",
    "ai_telemetry",
}


class StageManifest(CacheEnvelope):
    """Content/config provenance required before reusing a model artefact."""

    stage: str
    input_hash: str
    config_hash: str


class CacheTelemetrySink:
    """Bind one run cache to the SDK seam without global mutable state."""

    def __init__(self, cache: RunCache, run_id: str) -> None:
        self.cache = cache
        self.run_id = run_id

    def record_call(self, record: AICallTelemetry) -> None:
        self.cache.record_ai_call(self.run_id, record)

    def record_cache_hit(self, *, stage: str, model: str | None = None) -> None:
        self.cache.record_ai_call(
            self.run_id,
            AICallTelemetry.cache_hit(stage=stage, model=model),
        )


class RunCache:
    """JSON and text artefacts for one or more pipeline runs, keyed by run id.

    ``root`` defaults to a fixed macOS cache directory — this is a personal
    tool, not a cross-platform one — and is a constructor argument only so
    tests can point it at ``tmp_path``.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root if root is not None else _DEFAULT_ROOT

    def run_dir(self, run_id: str) -> Path:
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load[TModel: BaseModel](
        self, run_id: str, name: str, model_type: type[TModel]
    ) -> TModel | None:
        """Load ``name`` as a ``model_type``, or ``None`` if it was never stored.

        A file that exists but fails to parse raises: a lying cache is worse
        than a cold one.
        """
        path = self._model_path(run_id, name)
        if not path.exists():
            return None
        return model_type.model_validate_json(path.read_text())

    def store(self, run_id: str, name: str, value: BaseModel) -> Path:
        path = self._model_path(run_id, name)
        _atomic_write(path, value.model_dump_json(indent=2))
        return path

    def store_text(self, run_id: str, name: str, text: str) -> Path:
        path = self._text_path(run_id, name)
        _atomic_write(path, text)
        return path

    def load_resumable[TModel: BaseModel](
        self, run_id: str, name: str, model_type: type[TModel]
    ) -> TModel | None:
        """Load a local resumable artefact, invalidating partial/invalid state.

        Ordinary cache reads remain fail-closed: a corrupt non-resumable
        product still raises. These recomputable checkpoints and telemetry
        documents are different because a partial write must become a cache
        miss rather than block recovery.
        """
        if name not in _RESUMABLE_NAMES:
            return self.load(run_id, name, model_type)
        path = self._model_path(run_id, name)
        if not path.exists():
            return None
        try:
            return model_type.model_validate_json(path.read_text())
        except ValueError:
            path.unlink(missing_ok=True)
            return None

    def load_text(self, run_id: str, name: str) -> str | None:
        path = self._text_path(run_id, name)
        if not path.exists():
            return None
        return path.read_text()

    def telemetry_sink(self, run_id: str) -> TelemetrySink:
        return CacheTelemetrySink(self, run_id)

    def record_ai_call(self, run_id: str, record: AICallTelemetry) -> None:
        telemetry = self.load_resumable(run_id, "ai_telemetry", AITelemetry)
        if telemetry is None:
            telemetry = AITelemetry(calls=[], stages=[], totals=AITotals())
        next_attempt = (
            max(
                (
                    call.attempt
                    for call in telemetry.calls
                    if call.stage == record.stage
                ),
                default=0,
            )
            + 1
        )
        telemetry.append(record.model_copy(update={"attempt": next_attempt}))
        self.store(run_id, "ai_telemetry", telemetry)

    def load_manifest(self, run_id: str, stage: str) -> StageManifest | None:
        return self.load(run_id, f"{stage}.manifest", StageManifest)

    def store_manifest(self, run_id: str, manifest: StageManifest) -> Path:
        return self.store(run_id, f"{manifest.stage}.manifest", manifest)

    def stage_manifest(
        self,
        stage: str,
        *,
        inputs: dict[str, Any],
        config: dict[str, Any],
    ) -> StageManifest:
        return StageManifest(
            stage=stage,
            input_hash=stable_hash(inputs),
            config_hash=stable_hash(config),
        )

    def invalidate_artefacts(self, run_id: str, names: set[str]) -> list[str]:
        """Remove only named products and their manifests, preserving audit telemetry."""
        run_dir = self.run_dir(run_id)
        removed: list[str] = []
        for name in names:
            for suffix in (".json", ".txt"):
                path = run_dir / f"{name}{suffix}"
                if path.exists():
                    path.unlink()
                    removed.append(path.name)
        return sorted(removed)

    def record_timing(self, run_id: str, timing: StageTiming) -> None:
        """Append one durable timing record to the run's local-stage log."""
        log = self.load_resumable(run_id, "timings", StageTimingLog)
        if log is None:
            log = StageTimingLog(stages=[])
        log.stages.append(timing)
        self.store(run_id, "timings", log)

    def load_timings(self, run_id: str) -> list[StageTiming]:
        log = self.load_resumable(run_id, "timings", StageTimingLog)
        return [] if log is None else log.stages

    def invalidate_downstream(self, run_id: str, *, reason: str) -> StaleState:
        """Remove products derived from speaker evidence, preserving human state."""
        run_dir = self.run_dir(run_id)
        names = {
            "resolved_transcript",
            "chapters",
            "polished",
            "polish_issues",
            "minutes",
            "review",
            "candidate",
            "decision",
        }
        manifest_names = {
            "chapterise.manifest",
            "polish.manifest",
            "minutes.manifest",
        }
        removed: list[str] = []
        for path in run_dir.glob("*.json"):
            if (
                path.stem in names
                or path.stem in manifest_names
                or any(
                    token in path.stem for token in ("review", "candidate", "decision")
                )
            ):
                path.unlink()
                removed.append(path.name)
        for path in run_dir.glob("*.txt"):
            if any(token in path.stem for token in ("review", "candidate", "decision")):
                path.unlink()
                removed.append(path.name)
        stale = StaleState(reason=reason, artefacts=sorted(removed))
        self.store(run_id, "stale", stale)
        return stale

    def _model_path(self, run_id: str, name: str) -> Path:
        return self.run_dir(run_id) / f"{name}.json"

    def _text_path(self, run_id: str, name: str) -> Path:
        return self.run_dir(run_id) / f"{name}.txt"


def _atomic_write(path: Path, content: str) -> None:
    """Replace a cache artefact only after its complete contents are durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def sha256_of(path: Path) -> str:
    """Hash ``path``'s contents, streaming so large audio files stay off the heap."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    """Hash canonical JSON so manifests bind exact typed inputs/configuration."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
