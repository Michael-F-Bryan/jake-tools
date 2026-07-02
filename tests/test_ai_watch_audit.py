from __future__ import annotations

from pathlib import Path

from jake_tools.ai_watch.audit import append_model, read_models, truncate_records
from jake_tools.ai_watch.audit_models import DiscoveredRecord


def test_audit_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    append_model(
        path,
        DiscoveredRecord(
            run_id="2026-07-02",
            candidate_id="sha256:test",
            timestamp="2026-07-02T00:00:00+00:00",
            source="test",
            url="https://example.com",
            title="Example",
        ),
    )
    rows = read_models(path, DiscoveredRecord)
    assert rows[0].candidate_id == "sha256:test"
    truncate_records(path)
    assert read_models(path, DiscoveredRecord) == []
