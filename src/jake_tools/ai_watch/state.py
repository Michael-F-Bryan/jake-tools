from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path

from .audit import append_model, read_models, utc_now
from .audit_models import SeenCandidateRecord
from .models import CuratorDecisionType, candidate_id_for, content_hash_for


class SeenIndex:
    def __init__(self, state_root: Path) -> None:
        self.state_root = state_root
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.seen_path = state_root / "seen-candidates.jsonl"
        self.url_index_path = state_root / "url-index.json"
        self.hash_index_path = state_root / "content-hash-index.json"
        self.surfaced_index_path = state_root / "surfaced-index.json"
        self._records = self._load_seen()
        # Indexes are loaded once and mutated in memory; flush() is the only
        # thing that touches disk for them, so a run that updates the same
        # index many times does one write instead of one per record.
        self._url_index = self._read_index(self.url_index_path)
        self._hash_index = self._read_index(self.hash_index_path)
        self._surfaced_index = self._read_index(self.surfaced_index_path)

    def _load_seen(self) -> dict[str, SeenCandidateRecord]:
        records: dict[str, SeenCandidateRecord] = {}
        for row in read_models(self.seen_path, SeenCandidateRecord):
            records[row.candidate_id] = row
        return records

    def check_seen(self, *, url: str) -> SeenCandidateRecord | None:
        candidate_id = candidate_id_for(url=url)
        canonical = url.strip().rstrip("/").lower()
        if candidate_id in self._records:
            return self._records[candidate_id]
        if canonical in self._url_index:
            other_id = self._url_index[canonical]
            return self._records.get(other_id)
        return None

    def record_seen(
        self,
        *,
        candidate_id: str,
        url: str,
        title: str,
        source: str,
        decision: CuratorDecisionType | None = None,
        content_hash: str | None = None,
        content_path: str | None = None,
        obsidian_path: str | None = None,
        duplicate_of: str | None = None,
    ) -> None:
        now = utc_now()
        existing = self._records.get(candidate_id)
        record = SeenCandidateRecord(
            candidate_id=candidate_id,
            canonical_url=url.strip().rstrip("/").lower(),
            url=url,
            title=title,
            sources=sorted(set(existing.sources if existing else []) | {source}),
            first_seen_at=existing.first_seen_at if existing else now,
            last_seen_at=now,
            latest_decision=decision
            or (existing.latest_decision if existing else None),
            content_hash=content_hash or (existing.content_hash if existing else None),
            latest_content_path=content_path
            or (existing.latest_content_path if existing else None),
            obsidian_note_path=obsidian_path
            or (existing.obsidian_note_path if existing else None),
            duplicate_of=duplicate_of or (existing.duplicate_of if existing else None),
        )
        self._records[candidate_id] = record
        append_model(self.seen_path, record)
        self._url_index[record.canonical_url] = candidate_id
        if content_hash:
            self._hash_index[content_hash] = candidate_id
        if obsidian_path:
            self._surfaced_index[candidate_id] = obsidian_path

    def record_content_hash(self, text: str, candidate_id: str) -> str:
        digest = content_hash_for(text)
        self._hash_index[digest] = candidate_id
        return digest

    def flush(self) -> None:
        """Persist the in-memory indexes to disk, each in one atomic write."""
        self._write_index(self.url_index_path, self._url_index)
        self._write_index(self.hash_index_path, self._hash_index)
        self._write_index(self.surfaced_index_path, self._surfaced_index)

    def _write_index(self, path: Path, index: dict[str, str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(index, indent=2, sort_keys=True)
        descriptor, tmp_path = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_path)
            raise

    def _read_index(self, path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return {str(key): str(value) for key, value in loaded.items()}
        return {}
