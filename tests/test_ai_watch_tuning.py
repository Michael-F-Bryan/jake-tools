from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from jake_tools.ai_usage import AITotals
from jake_tools.ai_watch.audit_models import (
    CuratorDecisionRecord,
    DiscoveredRecord,
    RunManifest,
    ScoutEvaluationRecord,
)
from jake_tools.ai_watch.digest import run_digest
from jake_tools.ai_watch.manifest import write_manifest
from jake_tools.ai_watch.models import (
    AiWatchCommandOptions,
    CuratorDecisionType,
    DigestLane,
    ObsidianRecommendation,
    RunStatus,
    ScoutRecommendation,
    StageFailure,
)
from jake_tools.ai_watch.paths import AiWatchPaths
from jake_tools.ai_watch.records import append_model, load_model
from jake_tools.ai_watch.tuning import apply_surface_policy, run_tune


def _append_candidate_run(
    paths: AiWatchPaths,
    *,
    candidate_id: str,
    title: str,
    fit: int,
    practicality: int,
    novelty: int,
    noise: int,
) -> None:
    append_model(
        paths.candidates,
        DiscoveredRecord(
            run_id="2026-07-07",
            candidate_id=candidate_id,
            timestamp=datetime.fromisoformat("2026-07-07T00:00:00+00:00"),
            source="test",
            url=f"https://example.com/{candidate_id}",
            title=title,
        ),
    )
    append_model(
        paths.scout_evaluations,
        ScoutEvaluationRecord(
            run_id="2026-07-07",
            candidate_id=candidate_id,
            timestamp=datetime.fromisoformat("2026-07-07T00:00:00+00:00"),
            model="fake-scout",
            tags=["agent-harnesses"],
            fit_score=fit,
            novelty_score=novelty,
            practicality_score=practicality,
            noise_risk=noise,
            evidence_quotes=["agent harness"],
            recommendation=ScoutRecommendation.PROMOTE_TO_CURATOR,
            reason=f"{title} scout reason",
            summary=f"{title} scout summary",
        ),
    )
    append_model(
        paths.curator_decisions,
        CuratorDecisionRecord(
            run_id="2026-07-07",
            candidate_id=candidate_id,
            timestamp=datetime.fromisoformat("2026-07-07T00:00:00+00:00"),
            model="fake-curator",
            decision=CuratorDecisionType.SURFACE,
            lane=DigestLane.MAIN_DIGEST,
            reason=f"{title} curator reason",
            digest_summary=f"{title} digest summary",
            obsidian_recommendation=ObsidianRecommendation(
                should_create_note=True,
                path=f"3 Resources/AI/{title}.md",
                placement_reason="Agent workflow evidence.",
            ),
        ),
    )
    paths.article_markdown(candidate_id).write_text(
        f"# {title}\n\nPublished Jul 1, 2026\n\nAgent harness details.",
        encoding="utf-8",
    )


def test_surface_policy_keeps_only_highest_ranked_items(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 7)).create()
    _append_candidate_run(
        paths,
        candidate_id="sha256:low",
        title="Lower signal",
        fit=4,
        practicality=4,
        novelty=3,
        noise=2,
    )
    _append_candidate_run(
        paths,
        candidate_id="sha256:best",
        title="Best signal",
        fit=5,
        practicality=5,
        novelty=4,
        noise=1,
    )
    _append_candidate_run(
        paths,
        candidate_id="sha256:second",
        title="Second signal",
        fit=5,
        practicality=5,
        novelty=3,
        noise=1,
    )
    options = AiWatchCommandOptions(
        target_date=date(2026, 7, 7),
        base_dir=tmp_path,
        surface_limit=2,
    )

    result = apply_surface_policy(options=options, paths=paths)

    assert result.before_surface_count == 3
    assert result.after_surface_count == 2
    assert result.demoted_count == 1

    rows = [
        CuratorDecisionRecord.model_validate_json(line)
        for line in paths.curator_decisions.read_text(encoding="utf-8").splitlines()
    ]
    decisions = {row.candidate_id: row for row in rows}
    assert decisions["sha256:best"].decision == CuratorDecisionType.SURFACE
    assert decisions["sha256:second"].decision == CuratorDecisionType.SURFACE
    assert decisions["sha256:low"].decision == CuratorDecisionType.SPECULATIVE_WATCH
    assert decisions["sha256:low"].lane == DigestLane.SPECULATIVE_WATCH
    assert decisions["sha256:low"].obsidian_recommendation.should_create_note is False
    assert "surface limit" in decisions["sha256:low"].reason

    surfaced, speculative = run_digest(paths=paths)
    assert surfaced == 2
    assert speculative == 1
    digest = paths.digest.read_text(encoding="utf-8")
    assert "## Best signal" in digest
    assert "## Second signal" in digest
    assert "## Lower signal" not in digest


def test_surface_policy_can_remove_safe_unsurfaced_notes(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 7)).create()
    vault = tmp_path / "vault"
    note = vault / "3 Resources/AI/Lower signal.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        "# Lower signal\n\nCandidate ID: sha256:low\nRun ID: 2026-07-07\n",
        encoding="utf-8",
    )
    _append_candidate_run(
        paths,
        candidate_id="sha256:low",
        title="Lower signal",
        fit=4,
        practicality=4,
        novelty=3,
        noise=2,
    )
    _append_candidate_run(
        paths,
        candidate_id="sha256:best",
        title="Best signal",
        fit=5,
        practicality=5,
        novelty=4,
        noise=1,
    )
    options = AiWatchCommandOptions(
        target_date=date(2026, 7, 7),
        base_dir=tmp_path,
        vault_path=vault,
        surface_limit=1,
    )

    result = apply_surface_policy(options=options, paths=paths, remove_notes=True)

    assert result.demoted_count == 1
    assert result.removed_note_count == 1
    assert not note.exists()


def test_surface_policy_does_not_remove_unmarked_notes(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 7)).create()
    vault = tmp_path / "vault"
    note = vault / "3 Resources/AI/Lower signal.md"
    note.parent.mkdir(parents=True)
    note.write_text("# Handwritten note\n", encoding="utf-8")
    _append_candidate_run(
        paths,
        candidate_id="sha256:low",
        title="Lower signal",
        fit=4,
        practicality=4,
        novelty=3,
        noise=2,
    )
    _append_candidate_run(
        paths,
        candidate_id="sha256:best",
        title="Best signal",
        fit=5,
        practicality=5,
        novelty=4,
        noise=1,
    )
    options = AiWatchCommandOptions(
        target_date=date(2026, 7, 7),
        base_dir=tmp_path,
        vault_path=vault,
        surface_limit=1,
    )

    result = apply_surface_policy(options=options, paths=paths, remove_notes=True)

    assert result.demoted_count == 1
    assert result.removed_note_count == 0
    assert note.exists()
    skipped = json.loads(result.skipped_note_paths[0])
    assert skipped["reason"] == "marker_mismatch"


def test_surface_policy_demotes_stale_articles_even_under_limit(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 7)).create()
    _append_candidate_run(
        paths,
        candidate_id="sha256:old",
        title="Old agent advice",
        fit=5,
        practicality=5,
        novelty=5,
        noise=1,
    )
    paths.article_markdown("sha256:old").write_text(
        "# Old agent advice\n\nPublished Dec 19, 2024\n\nUseful but old agent patterns.",
        encoding="utf-8",
    )
    options = AiWatchCommandOptions(
        target_date=date(2026, 7, 7),
        base_dir=tmp_path,
        surface_limit=2,
        max_article_age_days=90,
    )

    result = apply_surface_policy(options=options, paths=paths)

    assert result.before_surface_count == 1
    assert result.after_surface_count == 0
    assert result.demoted_count == 1
    decision = CuratorDecisionRecord.model_validate_json(
        paths.curator_decisions.read_text(encoding="utf-8").strip()
    )
    assert decision.decision == CuratorDecisionType.SPECULATIVE_WATCH
    assert "older than 90 days" in decision.reason


def test_surface_policy_demotes_xcode_only_articles(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 7)).create()
    _append_candidate_run(
        paths,
        candidate_id="sha256:xcode",
        title="Xcode Claude Agent SDK",
        fit=5,
        practicality=5,
        novelty=5,
        noise=1,
    )
    options = AiWatchCommandOptions(
        target_date=date(2026, 7, 7),
        base_dir=tmp_path,
        surface_limit=2,
    )

    result = apply_surface_policy(options=options, paths=paths)

    assert result.before_surface_count == 1
    assert result.after_surface_count == 0
    assert result.demoted_count == 1
    decision = CuratorDecisionRecord.model_validate_json(
        paths.curator_decisions.read_text(encoding="utf-8").strip()
    )
    assert decision.decision == CuratorDecisionType.SPECULATIVE_WATCH
    assert "Xcode-specific" in decision.reason


def test_surface_policy_excludes_hard_demoted_items_from_ranking(
    tmp_path: Path,
) -> None:
    """A condemned top-ranked article must not consume a limit slot that an
    eligible article needs; both eligible items should surface."""
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 7)).create()
    _append_candidate_run(
        paths,
        candidate_id="sha256:xcode-top",
        title="Xcode agent tips",
        fit=5,
        practicality=5,
        novelty=5,
        noise=1,
    )
    _append_candidate_run(
        paths,
        candidate_id="sha256:eligible-a",
        title="Eligible A",
        fit=4,
        practicality=4,
        novelty=3,
        noise=2,
    )
    _append_candidate_run(
        paths,
        candidate_id="sha256:eligible-b",
        title="Eligible B",
        fit=3,
        practicality=3,
        novelty=3,
        noise=2,
    )
    options = AiWatchCommandOptions(
        target_date=date(2026, 7, 7),
        base_dir=tmp_path,
        surface_limit=2,
    )

    result = apply_surface_policy(options=options, paths=paths)

    assert result.before_surface_count == 3
    assert result.after_surface_count == 2
    assert result.demoted_count == 1

    rows = [
        CuratorDecisionRecord.model_validate_json(line)
        for line in paths.curator_decisions.read_text(encoding="utf-8").splitlines()
    ]
    decisions = {row.candidate_id: row for row in rows}
    assert (
        decisions["sha256:xcode-top"].decision == CuratorDecisionType.SPECULATIVE_WATCH
    )
    assert decisions["sha256:eligible-a"].decision == CuratorDecisionType.SURFACE
    assert decisions["sha256:eligible-b"].decision == CuratorDecisionType.SURFACE


def test_run_tune_preserves_failed_manifest_status(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 7)).create()
    _append_candidate_run(
        paths,
        candidate_id="sha256:best",
        title="Best signal",
        fit=5,
        practicality=5,
        novelty=4,
        noise=1,
    )
    options = AiWatchCommandOptions(
        target_date=date(2026, 7, 7),
        base_dir=tmp_path,
        surface_limit=2,
    )
    write_manifest(
        paths=paths,
        options=options,
        status=RunStatus.FAIL,
        surfaced_count=0,
        speculative_count=0,
        failed_stages=[StageFailure(stage="scout", error="boom")],
        summary=AITotals(),
    )

    result = run_tune(options=options, paths=paths)

    assert result.surfaced == 1
    manifest = load_model(paths.manifest, RunManifest)
    assert manifest.status == RunStatus.FAIL
    assert manifest.failed_stages == [StageFailure(stage="scout", error="boom")]
    assert manifest.counts.surfaced == 1
