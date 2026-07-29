from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from jake_tools.ai_watch.audit_models import (
    CuratorDecisionRecord,
    DiscoveredRecord,
)
from jake_tools.ai_watch.digest import run_digest
from jake_tools.ai_watch.models import (
    CuratorDecisionType,
    DigestLane,
    ObsidianRecommendation,
)
from jake_tools.ai_watch.paths import AiWatchPaths
from jake_tools.ai_watch.records import append_model


def test_digest_renders_surfaced_items(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    append_model(
        paths.candidates,
        DiscoveredRecord(
            run_id="2026-07-02",
            candidate_id="sha256:test",
            timestamp=datetime.fromisoformat("2026-07-02T00:00:00+00:00"),
            source="test",
            url="https://example.com/harness",
            title="Harness design",
        ),
    )
    append_model(
        paths.curator_decisions,
        CuratorDecisionRecord(
            run_id="2026-07-02",
            candidate_id="sha256:test",
            timestamp=datetime.fromisoformat("2026-07-02T00:00:00+00:00"),
            model="gpt-5.5",
            decision=CuratorDecisionType.SURFACE,
            lane=DigestLane.MAIN_DIGEST,
            reason="Transferable harness evaluator pattern.",
            digest_summary="Harness write-up with evaluator loops.",
            obsidian_recommendation=ObsidianRecommendation(
                should_create_note=True,
                path="3 Resources/AI/Harness design.md",
                placement_reason="AI engineering",
            ),
        ),
    )
    surfaced, speculative = run_digest(paths=paths)
    assert surfaced == 1
    assert speculative == 0
    digest = paths.digest.read_text(encoding="utf-8")
    assert "## Harness design" in digest
    assert "Why it matters:" in digest
    assert "https://example.com/harness" in digest


def test_empty_digest_renders_without_error(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    surfaced, speculative = run_digest(paths=paths)
    assert surfaced == 0
    assert speculative == 0
    assert "_No items crossed the bar._" in paths.digest.read_text(encoding="utf-8")
