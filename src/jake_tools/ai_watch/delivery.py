from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .audit import append_model, read_models, utc_now
from .audit_models import (
    CuratorDecisionRecord,
    CuratorDecisionType,
    DeliveryRecord,
    DeliveryStatus,
)
from .models import AiWatchCommandOptions
from .paths import AiWatchPaths

DISCORD_PAYLOAD_MAX_CHARS = 1900


class MessageSender(Protocol):
    def send(self, *, target: str, message_path: Path) -> None: ...


@dataclass
class DeliveryResult:
    status: DeliveryStatus
    surfaced_count: int
    speculative_count: int


@dataclass(frozen=True)
class DigestItem:
    title: str
    summary: str
    why_it_matters: str
    url: str
    obsidian_path: str


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


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3].rstrip() + "..."


def parse_digest_items(digest_text: str) -> list[DigestItem]:
    items: list[DigestItem] = []
    for block in re.split(r"(?m)^## ", digest_text.strip()):
        if not block.strip():
            continue
        lines = block.splitlines()
        title = lines[0].strip()
        summary_lines: list[str] = []
        why_it_matters = ""
        url = ""
        obsidian_path = ""
        section = "summary"
        for line in lines[1:]:
            stripped = line.strip()
            if stripped.startswith("Why it matters:"):
                section = "why"
                why_it_matters = stripped.removeprefix("Why it matters:").strip()
            elif stripped.startswith("URL:"):
                section = "fields"
                url = stripped.removeprefix("URL:").strip()
            elif stripped.startswith("Obsidian:"):
                section = "fields"
                obsidian_path = stripped.removeprefix("Obsidian:").strip()
            elif section == "summary" and stripped:
                summary_lines.append(stripped)
            elif section == "why" and stripped:
                why_it_matters = f"{why_it_matters} {stripped}".strip()
        items.append(
            DigestItem(
                title=title,
                summary=" ".join(summary_lines),
                why_it_matters=why_it_matters,
                url=url,
                obsidian_path=obsidian_path,
            )
        )
    return items


def render_digest_item(
    item: DigestItem,
    *,
    summary_limit: int | None = None,
    why_limit: int | None = None,
    include_summary: bool = True,
    include_why: bool = True,
) -> str:
    lines = [f"## {item.title}"]
    if include_summary and item.summary:
        summary = (
            _truncate(item.summary, summary_limit)
            if summary_limit is not None
            else item.summary
        )
        lines.extend(["", summary])
    if include_why and item.why_it_matters:
        why = (
            _truncate(item.why_it_matters, why_limit)
            if why_limit is not None
            else item.why_it_matters
        )
        lines.extend(["", f"Why it matters: {why}"])
    if item.url:
        lines.extend(["", f"URL: {item.url}"])
    if item.obsidian_path:
        lines.append(f"Obsidian: {item.obsidian_path}")
    return "\n".join(lines)


def render_digest_payload(
    items: list[DigestItem],
    *,
    summary_limit: int | None = None,
    why_limit: int | None = None,
    include_summary: bool = True,
    include_why: bool = True,
) -> str:
    return "\n\n".join(
        render_digest_item(
            item,
            summary_limit=summary_limit,
            why_limit=why_limit,
            include_summary=include_summary,
            include_why=include_why,
        )
        for item in items
    )


def build_discord_payload(
    digest_text: str, *, max_chars: int = DISCORD_PAYLOAD_MAX_CHARS
) -> str:
    items = parse_digest_items(digest_text)
    if not items:
        return digest_text

    compression_steps = (
        (None, None, True, True),
        (220, 180, True, True),
        (140, 100, True, True),
        (90, None, True, False),
        (None, None, False, False),
    )
    for summary_limit, why_limit, include_summary, include_why in compression_steps:
        payload = render_digest_payload(
            items,
            summary_limit=summary_limit,
            why_limit=why_limit,
            include_summary=include_summary,
            include_why=include_why,
        )
        if len(payload) <= max_chars:
            return payload

    minimal_items = [
        DigestItem(
            title=_truncate(item.title, 120),
            summary="",
            why_it_matters="",
            url=item.url,
            obsidian_path=item.obsidian_path,
        )
        for item in items
    ]
    payload = render_digest_payload(
        minimal_items, include_summary=False, include_why=False
    )
    if len(payload) <= max_chars:
        return payload
    return _truncate(payload, max_chars)


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
    curator_decisions = read_models(paths.curator_decisions, CuratorDecisionRecord)
    # Counts come from the typed curator decisions, not from sniffing the
    # rendered digest markdown: an item summary containing a literal "## "
    # line would otherwise inflate the surfaced count. The digest file is
    # rendering output only.
    surfaced_count = len(
        [
            record
            for record in curator_decisions
            if record.decision == CuratorDecisionType.SURFACE
        ]
    )
    speculative_count = len(
        [
            record
            for record in curator_decisions
            if record.decision == CuratorDecisionType.SPECULATIVE_WATCH
        ]
    )
    payload_path = paths.root / "delivery-payload.txt"
    payload_text = build_discord_payload(digest_text) if surfaced_count else digest_text
    payload_path.write_text(payload_text, encoding="utf-8")

    target = options.discord_target
    if surfaced_count == 0:
        append_model(
            paths.delivery,
            DeliveryRecord(
                run_id=run_id,
                timestamp=utc_now(),
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
                timestamp=utc_now(),
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
            timestamp=utc_now(),
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
