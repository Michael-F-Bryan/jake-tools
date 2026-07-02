from __future__ import annotations

from datetime import date
from pathlib import Path

from jake_tools.ai_watch.audit_models import DeliveryStatus
from jake_tools.ai_watch.delivery import (
    DISCORD_PAYLOAD_MAX_CHARS,
    FakeSender,
    build_discord_payload,
    run_delivery,
)
from jake_tools.ai_watch.models import AiWatchCommandOptions
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
