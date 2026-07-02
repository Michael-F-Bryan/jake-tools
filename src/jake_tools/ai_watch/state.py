from __future__ import annotations

import json
from pathlib import Path

from .audit import append_model, read_models, utc_now_iso
from .audit_models import SeenCandidateRecord
from .models import candidate_id_for, content_hash_for


class SeenIndex:
    def __init__(self, state_root: Path) -> None:
        self.state_root = state_root
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.seen_path = state_root / "seen-candidates.jsonl"
        self.url_index_path = state_root / "url-index.json"
        self.hash_index_path = state_root / "content-hash-index.json"
        self.surfaced_index_path = state_root / "surfaced-index.json"
        self._records = self._load_seen()

    def _load_seen(self) -> dict[str, SeenCandidateRecord]:
        records: dict[str, SeenCandidateRecord] = {}
        for row in read_models(self.seen_path, SeenCandidateRecord):
            records[row.candidate_id] = row
        return records

    def check_seen(self, *, url: str, title: str = "") -> SeenCandidateRecord | None:
        candidate_id = candidate_id_for(url=url, title=title)
        canonical = url.strip().rstrip("/").lower()
        if candidate_id in self._records:
            return self._records[candidate_id]
        url_index = self._read_index(self.url_index_path)
        if canonical in url_index:
            other_id = url_index[canonical]
            return self._records.get(other_id)
        return None

    def record_seen(
        self,
        *,
        candidate_id: str,
        url: str,
        title: str,
        source: str,
        decision: str | None = None,
        content_hash: str | None = None,
        content_path: str | None = None,
        obsidian_path: str | None = None,
        duplicate_of: str | None = None,
    ) -> None:
        now = utc_now_iso()
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
        self._update_index(self.url_index_path, record.canonical_url, candidate_id)
        if content_hash:
            self._update_index(self.hash_index_path, content_hash, candidate_id)
        if obsidian_path:
            self._update_index(self.surfaced_index_path, candidate_id, obsidian_path)

    def record_content_hash(self, text: str, candidate_id: str) -> str:
        digest = content_hash_for(text)
        self._update_index(self.hash_index_path, digest, candidate_id)
        return digest

    def _update_index(self, path: Path, key: str, value: str) -> None:
        index = self._read_index(path)
        index[key] = value
        path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")

    def _read_index(self, path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return {str(key): str(value) for key, value in loaded.items()}
        return {}
