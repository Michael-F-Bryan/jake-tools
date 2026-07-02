from __future__ import annotations

from datetime import date
from pathlib import Path

from jake_tools.ai_watch.models import candidate_id_for
from jake_tools.ai_watch.paths import AiWatchPaths


def test_paths_for_date_layout(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    assert paths.root == tmp_path / "ai-watch" / "2026-07-02"
    assert paths.state_root == tmp_path / "ai-watch" / "state"
    assert paths.candidates.name == "candidates.jsonl"
    assert paths.root.exists()


def test_candidate_id_stable() -> None:
    first = candidate_id_for(url="https://example.com/a/")
    second = candidate_id_for(url="https://example.com/a")
    assert first == second
