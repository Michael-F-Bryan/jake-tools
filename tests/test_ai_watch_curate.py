from __future__ import annotations

from pathlib import Path

from jake_tools.ai_watch.audit_models import CuratorDecisionRecord
from jake_tools.ai_watch.models import (
    CuratorDecision,
    CuratorDecisionType,
    DigestLane,
    ObsidianRecommendation,
)
from jake_tools.ai_watch.validation import sanitize_digest_summary, validate_curated

VAULT_ROOT = Path("/Users/work/Documents/Vault")


def test_validate_curated_rejects_generic_reason() -> None:
    record = CuratorDecisionRecord(
        run_id="2026-07-02",
        candidate_id="sha256:generic",
        timestamp="2026-07-02T00:00:00+00:00",
        model="gpt-5.5",
        decision=CuratorDecisionType.REJECT,
        reason="interesting article",
        digest_summary="",
    )
    errors = validate_curated([record], vault_root=VAULT_ROOT)
    assert any("generic reason" in error for error in errors)


def test_validate_curated_requires_vault_relative_placement() -> None:
    record = CuratorDecisionRecord(
        run_id="2026-07-02",
        candidate_id="sha256:bad-path",
        timestamp="2026-07-02T00:00:00+00:00",
        model="gpt-5.5",
        decision=CuratorDecisionType.SURFACE,
        lane=DigestLane.MAIN_DIGEST,
        reason="Transferable harness evaluator pattern for long-running agents.",
        digest_summary="Harness write-up.",
        obsidian_recommendation=ObsidianRecommendation(
            should_create_note=True,
            path="/etc/passwd",
            placement_reason="bad",
        ),
    )
    errors = validate_curated([record], vault_root=VAULT_ROOT)
    assert any("outside vault" in error for error in errors)


def test_validate_curated_rejects_dot_dot_traversal_placement() -> None:
    record = CuratorDecisionRecord(
        run_id="2026-07-02",
        candidate_id="sha256:traversal",
        timestamp="2026-07-02T00:00:00+00:00",
        model="gpt-5.5",
        decision=CuratorDecisionType.SURFACE,
        lane=DigestLane.MAIN_DIGEST,
        reason="Transferable harness evaluator pattern for long-running agents.",
        digest_summary="Harness write-up.",
        obsidian_recommendation=ObsidianRecommendation(
            should_create_note=True,
            path="3 Resources/../../../etc/x",
            placement_reason="bad",
        ),
    )
    errors = validate_curated([record], vault_root=VAULT_ROOT)
    assert any("outside vault" in error for error in errors)


def test_sanitize_digest_summary_truncates_overlong_text() -> None:
    overlong = "Harness pattern " * 80
    truncated = sanitize_digest_summary(overlong)
    assert len(truncated) <= 500
    assert truncated.endswith("...")
    assert not truncated.endswith(" ...")


def test_validate_curated_accepts_sanitized_digest_summary() -> None:
    overlong = "Transferable evaluator loop pattern for long-running agents. " * 20
    record = CuratorDecisionRecord(
        run_id="2026-07-02",
        candidate_id="sha256:long-summary",
        timestamp="2026-07-02T00:00:00+00:00",
        model="gpt-5.5",
        decision=CuratorDecisionType.SURFACE,
        lane=DigestLane.MAIN_DIGEST,
        reason="Transferable harness evaluator pattern for long-running agents.",
        digest_summary=sanitize_digest_summary(overlong),
        obsidian_recommendation=ObsidianRecommendation(
            should_create_note=True,
            path="3 Resources/AI Watch/long-summary.md",
            placement_reason="Concrete harness pattern.",
        ),
    )
    errors = validate_curated([record], vault_root=VAULT_ROOT)
    assert errors == []


def test_curator_decision_surface_requires_obsidian_note() -> None:
    decision = CuratorDecision(
        decision=CuratorDecisionType.SURFACE,
        lane=DigestLane.MAIN_DIGEST,
        reason="Transferable harness evaluator pattern for long-running agents.",
        digest_summary="Harness write-up.",
        obsidian_recommendation=ObsidianRecommendation(should_create_note=False),
    )
    record = CuratorDecisionRecord.from_decision(
        run_id="2026-07-02",
        candidate_id="sha256:test",
        timestamp="2026-07-02T00:00:00+00:00",
        model="gpt-5.5",
        decision=decision,
    )
    errors = validate_curated([record], vault_root=VAULT_ROOT)
    assert any("without obsidian note" in error for error in errors)
