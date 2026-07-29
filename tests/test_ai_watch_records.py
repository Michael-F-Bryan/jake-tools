from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jake_tools.ai_watch.audit_models import (
    DiscoveredRecord,
    RunManifest,
    RunManifestCounts,
    RunManifestPaths,
)
from jake_tools.ai_watch.models import RunStatus
from jake_tools.ai_watch.records import (
    append_model,
    atomic_write_text,
    load_model,
    read_models,
    truncate_records,
    write_model,
)

TIMESTAMP = "2026-07-02T00:00:00+00:00"


def test_audit_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    append_model(
        path,
        DiscoveredRecord(
            run_id="2026-07-02",
            candidate_id="sha256:test",
            timestamp=datetime.fromisoformat(TIMESTAMP),
            source="test",
            url="https://example.com",
            title="Example",
        ),
    )
    rows = read_models(path, DiscoveredRecord)
    assert rows[0].candidate_id == "sha256:test"
    truncate_records(path)
    assert read_models(path, DiscoveredRecord) == []


def test_timestamp_round_trips_through_jsonl_as_a_datetime(tmp_path: Path) -> None:
    """timestamp is typed as datetime, not str. Pydantic writes it as ISO
    text (with a 'Z' UTC suffix rather than '+00:00', a cosmetic difference
    from the old plain-string field) and reads it back as the same instant,
    and it can still read JSONL written by the old str-typed field."""
    path = tmp_path / "events.jsonl"
    append_model(
        path,
        DiscoveredRecord(
            run_id="2026-07-02",
            candidate_id="sha256:test",
            timestamp=datetime.fromisoformat(TIMESTAMP),
            source="test",
            url="https://example.com",
            title="Example",
        ),
    )
    on_disk = path.read_text(encoding="utf-8")
    assert '"timestamp": "2026-07-02T00:00:00Z"' in on_disk
    rows = read_models(path, DiscoveredRecord)
    assert rows[0].timestamp == datetime.fromisoformat(TIMESTAMP)


def test_atomic_write_text_leaves_no_tmp_file_and_writes_full_content(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    atomic_write_text(path, "hello world")

    assert path.read_text(encoding="utf-8") == "hello world"
    # No stray temp file from the write should survive.
    assert list(tmp_path.glob(".*")) == []


def test_atomic_write_text_replaces_existing_file_wholesale(tmp_path: Path) -> None:
    """A second write must fully replace the first, never merge or leave a
    truncated remainder from the previous (longer) content."""
    path = tmp_path / "manifest.json"
    atomic_write_text(path, "a much longer first payload")
    atomic_write_text(path, "short")

    assert path.read_text(encoding="utf-8") == "short"
    assert list(tmp_path.glob(".*")) == []


def _sample_manifest() -> RunManifest:
    return RunManifest(
        run_id="2026-07-02",
        status=RunStatus.OK,
        paths=RunManifestPaths(
            root="/tmp/run",
            digest="/tmp/run/digest.md",
            summary="/tmp/run/summary.json",
        ),
        counts=RunManifestCounts(
            candidates=1, fetched=1, scouted=1, curated=1, surfaced=1, speculative=0
        ),
    )


def test_write_model_writes_atomically_and_load_model_reads_it_back(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    manifest = _sample_manifest()

    write_model(path, manifest)

    assert load_model(path, RunManifest) == manifest
    assert list(tmp_path.glob(".*")) == []


def test_reads_pre_existing_plus_offset_timestamp_format(tmp_path: Path) -> None:
    """JSONL written before the str -> datetime migration used a '+00:00'
    offset suffix instead of 'Z'; existing archives must stay readable."""
    path = tmp_path / "events.jsonl"
    legacy_line = (
        '{"candidate_id": "sha256:legacy", "description": "", "run_id": "2026-07-02", '
        '"source": "test", "stage": "discovered", '
        f'"timestamp": "{TIMESTAMP}", "title": "Legacy", "url": "https://example.com"}}\n'
    )
    path.write_text(legacy_line, encoding="utf-8")
    rows = read_models(path, DiscoveredRecord)
    assert rows[0].timestamp == datetime.fromisoformat(TIMESTAMP)
