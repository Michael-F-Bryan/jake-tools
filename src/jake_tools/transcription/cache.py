"""Content-addressed run cache for the transcription pipeline.

Every pipeline stage writes its output into a run directory keyed by a run id
(note-stem plus audio-hash prefix — computed by callers, not here). A rerun
that hits an already-cached stage skips straight to the next one, and the
integrate step reads its tier-b baseline back out of the same cache.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel

from .models import StageTiming, StageTimingLog, StaleState

_DEFAULT_ROOT = Path.home() / "Library" / "Caches" / "jake-tools" / "transcription"
_CHUNK_SIZE = 1024 * 1024
_RESUMABLE_NAMES = {
    "asr_checkpoint",
    "diarisation_checkpoint",
    "raw_transcript",
    "timings",
}


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
        product still raises. These four local artefacts are different because
        they are recomputable checkpoints or telemetry, so a partial write
        must become a cache miss rather than block recovery.
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

    def record_timing(self, run_id: str, timing: StageTiming) -> None:
        """Append one durable timing record to the run's local-stage log."""
        log = self.load_resumable(run_id, "timings", StageTimingLog)
        if log is None:
            log = StageTimingLog()
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
        removed: list[str] = []
        for path in run_dir.glob("*.json"):
            if path.stem in names or any(
                token in path.stem for token in ("review", "candidate", "decision")
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
