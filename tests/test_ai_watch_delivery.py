from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from jake_tools.ai_watch.audit import append_model
from jake_tools.ai_watch.audit_models import CuratorDecisionRecord, DeliveryStatus
from jake_tools.ai_watch.delivery import (
    DISCORD_PAYLOAD_MAX_CHARS,
    FakeSender,
    build_discord_payload,
    run_delivery,
)
from jake_tools.ai_watch.models import (
    AiWatchCommandOptions,
    CuratorDecisionType,
    DigestLane,
    ObsidianRecommendation,
)
from jake_tools.ai_watch.paths import AiWatchPaths


def _long_item(title: str, url: str, obsidian: str) -> str:
    summary = (
        "A long summary about agent tooling, MCP servers, orchestration patterns, "
        "and how large codebases should expose structure for autonomous navigation. "
        * 4
    )
    why = (
        "Strong fit because it maps directly to Hermes-style harness design, tool "
        "selection, sandboxing, and practical trade-offs for agent-operable repos. " * 4
    )
    return (
        f"## {title}\n\n"
        f"{summary.strip()}\n\n"
        f"Why it matters: {why.strip()}\n\n"
        f"URL: {url}\n"
        f"Obsidian: {obsidian}\n"
    )


def test_delivery_skips_empty_digest(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    paths.digest.write_text("_No items crossed the bar._\n", encoding="utf-8")
    options = AiWatchCommandOptions(
        target_date=date(2026, 7, 2),
        base_dir=tmp_path,
        discord_target="discord:user",
    )
    sender = FakeSender()
    result = run_delivery(options=options, paths=paths, sender=sender)
    assert result.status == DeliveryStatus.SKIPPED
    assert sender.calls == []


def _append_surfaced_decision(paths: AiWatchPaths, candidate_id: str) -> None:
    append_model(
        paths.curator_decisions,
        CuratorDecisionRecord(
            run_id="2026-07-02",
            candidate_id=candidate_id,
            timestamp=datetime.fromisoformat("2026-07-02T00:00:00+00:00"),
            model="gpt-5.5",
            decision=CuratorDecisionType.SURFACE,
            lane=DigestLane.MAIN_DIGEST,
            reason="Transferable harness evaluator pattern.",
            digest_summary="Harness write-up.",
            obsidian_recommendation=ObsidianRecommendation(
                should_create_note=True,
                path=f"3 Resources/AI/{candidate_id}.md",
                placement_reason="AI engineering",
            ),
        ),
    )


def test_delivery_payload_under_discord_limit(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    digest = "\n\n".join(
        [
            _long_item(
                "How Claude Code works in large codebases - Hacker News",
                "https://news.ycombinator.com/item?id=48144494",
                "3 Resources/AI/How Claude Code works in large codebases.md",
            ),
            _long_item(
                "Show HN: Mcp-Agent – Build effective agents with Model Context Protocol",
                "https://news.ycombinator.com/item?id=42867050",
                "3 Resources/AI/Mcp-Agent - Build effective agents with Model Context Protocol.md",
            ),
            _long_item(
                "MCP: An in-depth introduction - Hacker News",
                "https://news.ycombinator.com/item?id=43972334",
                "3 Resources/AI/MCP as JSON-RPC tool infrastructure.md",
            ),
            _long_item(
                "Releases · anthropics/claude-code - GitHub",
                "https://github.com/anthropics/claude-code/releases",
                "3 Resources/AI/Claude Code background agents and draft PR workflow.md",
            ),
        ]
    )
    assert len(digest) > 2000
    paths.digest.write_text(digest, encoding="utf-8")
    for candidate_id in ("sha256:one", "sha256:two", "sha256:three", "sha256:four"):
        _append_surfaced_decision(paths, candidate_id)
    options = AiWatchCommandOptions(
        target_date=date(2026, 7, 2),
        base_dir=tmp_path,
        dry_run=True,
        discord_target="discord:user",
    )
    result = run_delivery(options=options, paths=paths)
    payload = (paths.root / "delivery-payload.txt").read_text(encoding="utf-8")
    assert result.status == DeliveryStatus.DRY_RUN
    assert len(payload) <= DISCORD_PAYLOAD_MAX_CHARS
    assert payload.count("## ") == 4
    assert "URL: https://news.ycombinator.com/item?id=48144494" in payload
    assert (
        "Obsidian: 3 Resources/AI/How Claude Code works in large codebases.md"
        in payload
    )


def test_delivery_surfaced_count_ignores_literal_heading_in_summary(
    tmp_path: Path,
) -> None:
    """A curator summary containing a literal '## ' line must not inflate the
    surfaced count derived from the rendered digest markdown."""
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    digest_summary_with_fake_heading = (
        "Intro text.\n## Fake heading inside summary\nMore text."
    )
    paths.digest.write_text(
        f"## Real item\n\n{digest_summary_with_fake_heading}\n\n"
        "Why it matters: reasons\n\nURL: https://example.com/real\n",
        encoding="utf-8",
    )
    append_model(
        paths.curator_decisions,
        CuratorDecisionRecord(
            run_id="2026-07-02",
            candidate_id="sha256:real",
            timestamp=datetime.fromisoformat("2026-07-02T00:00:00+00:00"),
            model="gpt-5.5",
            decision=CuratorDecisionType.SURFACE,
            lane=DigestLane.MAIN_DIGEST,
            reason="Transferable harness evaluator pattern.",
            digest_summary=digest_summary_with_fake_heading,
            obsidian_recommendation=ObsidianRecommendation(
                should_create_note=True,
                path="3 Resources/AI/Real item.md",
                placement_reason="AI engineering",
            ),
        ),
    )
    options = AiWatchCommandOptions(
        target_date=date(2026, 7, 2),
        base_dir=tmp_path,
        dry_run=True,
        discord_target="discord:user",
    )
    result = run_delivery(options=options, paths=paths)
    assert result.surfaced_count == 1


def test_build_discord_payload_keeps_priority_fields() -> None:
    digest = _long_item(
        "Priority fields stay visible",
        "https://example.com/article",
        "3 Resources/AI/Priority fields.md",
    )
    payload = build_discord_payload(digest)
    assert len(payload) <= DISCORD_PAYLOAD_MAX_CHARS
    assert "## Priority fields stay visible" in payload
    assert "URL: https://example.com/article" in payload
    assert "Obsidian: 3 Resources/AI/Priority fields.md" in payload
