from __future__ import annotations

from pathlib import Path

from jake_tools.ai_watch.models import CuratorDecisionType
from jake_tools.ai_watch.state import SeenIndex


def test_seen_index_round_trip(tmp_path: Path) -> None:
    state = SeenIndex(tmp_path / "state")
    state.record_seen(
        candidate_id="sha256:abc",
        url="https://example.com/post",
        title="Post",
        source="test",
    )
    seen = state.check_seen(url="https://example.com/post")
    assert seen is not None
    assert seen.candidate_id == "sha256:abc"


def test_check_seen_skips_duplicate_url(tmp_path: Path) -> None:
    state = SeenIndex(tmp_path / "state")
    state.record_seen(
        candidate_id="sha256:first",
        url="https://example.com/post/",
        title="Post",
        source="test",
    )
    seen = state.check_seen(url="https://example.com/post")
    assert seen is not None
    assert seen.candidate_id == "sha256:first"


def test_indexes_stay_in_memory_until_flush(tmp_path: Path) -> None:
    """record_seen() must not write the index files itself; only flush()
    touches disk for them, so a run doing many updates writes each index
    file once instead of once per record."""
    state_root = tmp_path / "state"
    state = SeenIndex(state_root)
    state.record_seen(
        candidate_id="sha256:abc",
        url="https://example.com/post",
        title="Post",
        source="test",
        content_hash="sha256:content",
        obsidian_path="3 Resources/AI/Post.md",
    )

    assert not state.url_index_path.exists()
    assert not state.hash_index_path.exists()
    assert not state.surfaced_index_path.exists()

    state.flush()

    assert state.url_index_path.exists()
    assert state.hash_index_path.exists()
    assert state.surfaced_index_path.exists()
    # The atomic write must not leave its temp file behind.
    assert list(state_root.glob("*.tmp")) == []


def test_flush_persists_indexes_readable_by_a_fresh_instance(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state = SeenIndex(state_root)
    state.record_seen(
        candidate_id="sha256:abc",
        url="https://example.com/post",
        title="Post",
        source="test",
    )
    state.flush()

    reloaded = SeenIndex(state_root)
    seen = reloaded.check_seen(url="https://example.com/post")
    assert seen is not None
    assert seen.candidate_id == "sha256:abc"


def test_record_seen_decision_round_trips_as_enum_through_flush(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    state = SeenIndex(state_root)
    state.record_seen(
        candidate_id="sha256:abc",
        url="https://example.com/post",
        title="Post",
        source="test",
        decision=CuratorDecisionType.SURFACE,
    )
    state.flush()

    reloaded = SeenIndex(state_root)
    seen = reloaded.check_seen(url="https://example.com/post")
    assert seen is not None
    assert seen.latest_decision == CuratorDecisionType.SURFACE
    assert isinstance(seen.latest_decision, CuratorDecisionType)
