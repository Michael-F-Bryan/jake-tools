"""Content-addressed run cache for the transcription pipeline.

Every pipeline stage writes its output into a run directory keyed by a run id
(note-stem plus audio-hash prefix — computed by callers, not here). A rerun
that hits an already-cached stage skips straight to the next one, and the
integrate step reads its tier-b baseline back out of the same cache.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel

_DEFAULT_ROOT = Path.home() / "Library" / "Caches" / "jake-tools" / "transcription"
_CHUNK_SIZE = 1024 * 1024


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
        path.write_text(value.model_dump_json(indent=2))
        return path

    def store_text(self, run_id: str, name: str, text: str) -> Path:
        path = self._text_path(run_id, name)
        path.write_text(text)
        return path

    def load_text(self, run_id: str, name: str) -> str | None:
        path = self._text_path(run_id, name)
        if not path.exists():
            return None
        return path.read_text()

    def _model_path(self, run_id: str, name: str) -> Path:
        return self.run_dir(run_id) / f"{name}.json"

    def _text_path(self, run_id: str, name: str) -> Path:
        return self.run_dir(run_id) / f"{name}.txt"


def sha256_of(path: Path) -> str:
    """Hash ``path``'s contents, streaming so large audio files stay off the heap."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
