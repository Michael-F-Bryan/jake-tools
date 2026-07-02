from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .audit import append_model, read_models, utc_now_iso
from .audit_models import (
    CuratorDecisionRecord,
    CuratorDecisionType,
    DeliveryRecord,
    DeliveryStatus,
)
from .models import AiWatchCommandOptions
from .paths import AiWatchPaths


class MessageSender(Protocol):
    def send(self, *, target: str, message_path: Path) -> None: ...


@dataclass
class DeliveryResult:
    status: DeliveryStatus
    surfaced_count: int
    speculative_count: int


class HermesSendSender:
    def send(self, *, target: str, message_path: Path) -> None:
        subprocess.run(
            ["hermes", "send", "--to", target, "--file", str(message_path)],
            check=True,
            capture_output=True,
            text=True,
        )


class FakeSender:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def send(self, *, target: str, message_path: Path) -> None:
        self.calls.append((target, message_path))


def run_delivery(
    *,
    options: AiWatchCommandOptions,
    paths: AiWatchPaths,
    sender: MessageSender | None = None,
) -> DeliveryResult:
    paths.create()
    run_id = options.target_date.isoformat()
    digest_text = (
        paths.digest.read_text(encoding="utf-8") if paths.digest.exists() else ""
    )
    if digest_text.startswith("_No items"):
        surfaced_count = 0
    else:
        surfaced_count = sum(
            1 for line in digest_text.splitlines() if line.startswith("## ")
        )
    speculative_count = len(
        [
            record
            for record in read_models(paths.curator_decisions, CuratorDecisionRecord)
            if record.decision == CuratorDecisionType.SPECULATIVE_WATCH
        ]
    )
    payload_path = paths.root / "delivery-payload.txt"
    payload_path.write_text(digest_text, encoding="utf-8")

    target = options.discord_target
    if surfaced_count == 0:
        append_model(
            paths.delivery,
            DeliveryRecord(
                run_id=run_id,
                timestamp=utc_now_iso(),
                target="discord",
                status=DeliveryStatus.SKIPPED,
                reason="empty_digest",
                digest_path=str(paths.digest),
                surfaced_count=0,
                speculative_count=speculative_count,
            ),
        )
        return DeliveryResult(
            status=DeliveryStatus.SKIPPED,
            surfaced_count=0,
            speculative_count=speculative_count,
        )

    if options.dry_run or not target:
        append_model(
            paths.delivery,
            DeliveryRecord(
                run_id=run_id,
                timestamp=utc_now_iso(),
                target="discord",
                status=DeliveryStatus.DRY_RUN,
                digest_path=str(paths.digest),
                payload_path=str(payload_path),
                surfaced_count=surfaced_count,
                speculative_count=speculative_count,
            ),
        )
        return DeliveryResult(
            status=DeliveryStatus.DRY_RUN,
            surfaced_count=surfaced_count,
            speculative_count=speculative_count,
        )

    active_sender = sender or HermesSendSender()
    active_sender.send(target=target, message_path=payload_path)
    append_model(
        paths.delivery,
        DeliveryRecord(
            run_id=run_id,
            timestamp=utc_now_iso(),
            target="discord",
            status=DeliveryStatus.SENT,
            digest_path=str(paths.digest),
            surfaced_count=surfaced_count,
            speculative_count=speculative_count,
        ),
    )
    return DeliveryResult(
        status=DeliveryStatus.SENT,
        surfaced_count=surfaced_count,
        speculative_count=speculative_count,
    )
