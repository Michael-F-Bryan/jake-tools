from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from .audit_models import DiscoveredRecord
from .models import AuditStage


def utc_now() -> datetime:
    return datetime.now(UTC)


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temp file + ``os.replace``.

    Readers (``tune``, ``audit``) never observe a partially written file:
    either the old content is still there, or the new content is there
    complete. A crash mid-write leaves only a stray ``.tmp`` file behind,
    never a corrupt target.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_path = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)
        raise


def append_model(path: Path, record: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def truncate_records(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def read_models[TModel: BaseModel](path: Path, model: type[TModel]) -> list[TModel]:
    if not path.exists():
        return []
    records: list[TModel] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(model.model_validate_json(line))
    return records


def write_model(path: Path, payload: BaseModel) -> None:
    atomic_write_text(
        path,
        json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True),
    )


def load_model[TModel: BaseModel](path: Path, model: type[TModel]) -> TModel:
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def read_discovered_candidates(path: Path) -> list[DiscoveredRecord]:
    if not path.exists():
        return []
    records: list[DiscoveredRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("stage") != AuditStage.DISCOVERED:
            continue
        records.append(DiscoveredRecord.model_validate(payload))
    return records
