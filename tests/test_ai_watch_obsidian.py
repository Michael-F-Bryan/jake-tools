from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from jake_tools.ai_watch.audit import append_model, write_model
from jake_tools.ai_watch.audit_models import ArticleMetadata, CuratorDecisionRecord
from jake_tools.ai_watch.models import (
    AiWatchCommandOptions,
    CuratorDecisionType,
    DigestLane,
    ObsidianRecommendation,
    VaultPathEscapeError,
)
from jake_tools.ai_watch.obsidian import render_obsidian_note, run_obsidian_sync
from jake_tools.ai_watch.paths import AiWatchPaths


def test_render_obsidian_note_has_callout() -> None:
    note = render_obsidian_note(
        title="Harness design",
        url="https://example.com",
        digest_summary="Concrete harness pattern.",
        body="Article body",
        curator_reason="Transferable harness design.",
        candidate_id="sha256:test",
        run_id="2026-07-02",
        tags=["agent-harnesses"],
        target_date=date(2026, 7, 2),
    )
    assert "> [!Summary] TL;DR:" in note
    assert "note/capture" in note
    assert "source/article/clipping" in note
    assert "Concrete harness pattern." in note
    assert 'Link: "https://example.com"' in note


def _surfaced_decision(candidate_id: str, *, path: str) -> CuratorDecisionRecord:
    return CuratorDecisionRecord(
        run_id="2026-07-02",
        candidate_id=candidate_id,
        timestamp="2026-07-02T00:00:00+00:00",
        model="gpt-5.5",
        decision=CuratorDecisionType.SURFACE,
        lane=DigestLane.MAIN_DIGEST,
        reason="Transferable harness evaluator pattern.",
        digest_summary="Harness write-up.",
        obsidian_recommendation=ObsidianRecommendation(
            should_create_note=True,
            path=path,
            placement_reason="bad",
        ),
    )


def test_run_obsidian_sync_rejects_absolute_path_escape(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    vault = tmp_path / "vault"
    append_model(
        paths.curator_decisions,
        _surfaced_decision("sha256:escape", path="/etc/passwd"),
    )
    options = AiWatchCommandOptions(
        target_date=date(2026, 7, 2), base_dir=tmp_path, vault_path=vault
    )
    with pytest.raises(VaultPathEscapeError):
        run_obsidian_sync(options=options, paths=paths)


def test_run_obsidian_sync_rejects_dot_dot_traversal(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    vault = tmp_path / "vault"
    append_model(
        paths.curator_decisions,
        _surfaced_decision("sha256:traversal", path="3 Resources/../../../etc/x"),
    )
    options = AiWatchCommandOptions(
        target_date=date(2026, 7, 2), base_dir=tmp_path, vault_path=vault
    )
    with pytest.raises(VaultPathEscapeError):
        run_obsidian_sync(options=options, paths=paths)


def test_run_obsidian_sync_sanitizes_slash_in_fallback_title(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    vault = tmp_path / "vault"
    candidate_id = "sha256:slash-title"
    append_model(
        paths.curator_decisions,
        _surfaced_decision(candidate_id, path=""),
    )
    write_model(
        paths.article_metadata(candidate_id),
        ArticleMetadata(
            candidate_id=candidate_id,
            url="https://example.com/weird",
            title="Weird/Title",
            source="test",
            content_hash="sha256:abc",
        ),
    )
    options = AiWatchCommandOptions(
        target_date=date(2026, 7, 2), base_dir=tmp_path, vault_path=vault
    )
    run_obsidian_sync(options=options, paths=paths)

    assert not (vault / "3 Resources/AI/Weird").exists()
    assert (vault / "3 Resources/AI/Weird-Title.md").exists()
