from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from jake_tools.ai_usage import AITelemetry
from jake_tools.transcription.cache import RunCache, sha256_of
from jake_tools.transcription.models import SourceClip, StageTimingLog


def test_store_then_load_returns_equal_model(tmp_path: Path) -> None:
    cache = RunCache(tmp_path)
    clip = SourceClip(path="a.wav", offset_seconds=0.0, duration_seconds=10.0)

    cache.store("run-1", "clip", clip)

    assert cache.load("run-1", "clip", SourceClip) == clip


def test_load_of_missing_name_returns_none(tmp_path: Path) -> None:
    cache = RunCache(tmp_path)

    assert cache.load("run-1", "missing", SourceClip) is None


def test_load_of_corrupt_json_raises(tmp_path: Path) -> None:
    cache = RunCache(tmp_path)
    path = cache.run_dir("run-1") / "clip.json"
    path.write_text("{not valid json")

    with pytest.raises(ValueError):
        cache.load("run-1", "clip", SourceClip)


def test_load_of_valid_wrong_schema_raises(tmp_path: Path) -> None:
    cache = RunCache(tmp_path)
    path = cache.run_dir("run-1") / "timings.json"
    path.write_text('{"wrong": 1}')

    with pytest.raises(ValueError):
        cache.load("run-1", "timings", StageTimingLog)


def test_load_resumable_valid_wrong_schema_quarantines_it_as_a_cache_miss(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path)
    path = cache.run_dir("run-1") / "timings.json"
    path.write_text('{"wrong": 1}')

    assert cache.load_resumable("run-1", "timings", StageTimingLog) is None
    assert not path.exists()


def test_load_resumable_accepts_supported_legacy_timing_schema(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path)
    path = cache.run_dir("run-1") / "timings.json"
    path.write_text('{"stages": []}')

    assert cache.load_resumable("run-1", "timings", StageTimingLog) == StageTimingLog(
        stages=[]
    )
    assert path.exists()


def test_load_resumable_unsupported_schema_version_quarantines_it(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path)
    path = cache.run_dir("run-1") / "ai_telemetry.json"
    path.write_text('{"schema_version": 2, "calls": [], "stages": [], "totals": {}}')

    assert cache.load_resumable("run-1", "ai_telemetry", AITelemetry) is None
    assert not path.exists()


def test_load_resumable_corrupt_json_quarantines_it_as_a_cache_miss(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path)
    path = cache.run_dir("run-1") / "timings.json"
    path.write_text("{")

    assert cache.load_resumable("run-1", "timings", StageTimingLog) is None
    assert not path.exists()


def test_store_text_then_load_text_round_trips(tmp_path: Path) -> None:
    cache = RunCache(tmp_path)

    cache.store_text("run-1", "baseline", "tier-b baseline text")

    assert cache.load_text("run-1", "baseline") == "tier-b baseline text"


def test_load_text_of_missing_name_returns_none(tmp_path: Path) -> None:
    cache = RunCache(tmp_path)

    assert cache.load_text("run-1", "missing") is None


def test_store_creates_run_dir_on_demand(tmp_path: Path) -> None:
    cache = RunCache(tmp_path)

    run_dir = cache.run_dir("run-1")

    assert run_dir.is_dir()
    assert run_dir == tmp_path / "run-1"


def test_sha256_of_matches_hashlib_for_small_file(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.bin"
    fixture.write_bytes(b"some fixture bytes for hashing" * 100)

    expected = hashlib.sha256(fixture.read_bytes()).hexdigest()

    assert sha256_of(fixture) == expected
