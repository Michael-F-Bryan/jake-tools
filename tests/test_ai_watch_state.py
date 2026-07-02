from __future__ import annotations

from pathlib import Path

from jake_tools.ai_watch.state import SeenIndex


def test_seen_index_round_trip(tmp_path: Path) -> None:
    state = SeenIndex(tmp_path / "state")
    state.record_seen(
        candidate_id="sha256:abc",
        url="https://example.com/post",
        title="Post",
        source="test",
    )
    seen = state.check_seen(url="https://example.com/post", title="Post")
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
    seen = state.check_seen(url="https://example.com/post", title="Post")
    assert seen is not None
    assert seen.candidate_id == "sha256:first"
