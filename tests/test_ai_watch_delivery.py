from __future__ import annotations

from datetime import date
from pathlib import Path

from jake_tools.ai_watch.audit_models import DeliveryStatus
from jake_tools.ai_watch.delivery import FakeSender, run_delivery
from jake_tools.ai_watch.models import AiWatchCommandOptions
from jake_tools.ai_watch.paths import AiWatchPaths


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
    paths.digest.write_text(
        "Daily digest\n\n## Item\n\nSummary\n\nWhy it matters: reason\n\n",
        encoding="utf-8",
    )
    options = AiWatchCommandOptions(
        target_date=date(2026, 7, 2),
        base_dir=tmp_path,
        dry_run=True,
        discord_target="discord:user",
    )
    result = run_delivery(options=options, paths=paths)
    payload = (paths.root / "delivery-payload.txt").read_text(encoding="utf-8")
    assert result.status == DeliveryStatus.DRY_RUN
    assert len(payload) < 2000
