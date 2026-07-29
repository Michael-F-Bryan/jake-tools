from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .archive import load_article_metadata
from .audit import append_model, read_models, utc_now_iso, write_model
from .audit_models import (
    CuratorDecisionRecord,
    CuratorDecisionType,
    ObsidianPreview,
    ObsidianSyncRecord,
    ObsidianSyncStatus,
    ScoutEvaluationRecord,
)
from .cleanup import strip_page_chrome
from .models import AiWatchCommandOptions, resolve_vault_path
from .paths import AiWatchPaths

_FALLBACK_TITLE_MAX_CHARS = 150


@dataclass
class ObsidianSyncResult:
    created: int
    skipped: int


def _placement_fallback(tags: list[str], title: str) -> str:
    joined = " ".join(tags + [title]).lower()
    if any(token in joined for token in ("pkm", "obsidian", "knowledge")):
        return "3 Resources/Personal Knowledge Management"
    if any(token in joined for token in ("ui", "interface", "design")):
        return "3 Resources/Software Design & Architecture"
    return "3 Resources/AI"


def _sanitize_fallback_title(title: str) -> str:
    """Strip path separators and cap length before using a title as a filename.

    Curator-provided titles are free text; without this a `/` creates
    unintended subdirectories and an unbounded title can blow past filesystem
    filename limits.
    """
    cleaned = title.replace("/", "-").replace("\\", "-").strip()
    return (cleaned or "untitled")[:_FALLBACK_TITLE_MAX_CHARS].rstrip()


def render_obsidian_note(
    *,
    title: str,
    url: str,
    digest_summary: str,
    body: str,
    curator_reason: str,
    candidate_id: str,
    run_id: str,
    tags: list[str],
    target_date: date,
) -> str:
    tag_lines = "\n".join(
        [
            '  - "note/capture"',
            '  - "source/article/clipping"',
            *[f'  - "{tag}"' for tag in tags],
        ]
    )
    summary_lines = "\n".join(
        f"> {line}" for line in digest_summary.splitlines() if line
    )
    return f"""---
Link: "{url}"
Author:
Published:
Created: "[[{target_date.strftime("%B")} {target_date.day}, {target_date.year}]]"
tags:
{tag_lines}
---
> [!Summary] TL;DR:
{summary_lines}

{body}

---
Curator reason: {curator_reason}
Candidate ID: {candidate_id}
Run ID: {run_id}
"""


def run_obsidian_sync(
    *,
    options: AiWatchCommandOptions,
    paths: AiWatchPaths,
) -> ObsidianSyncResult:
    created = 0
    skipped = 0
    run_id = options.target_date.isoformat()
    vault = options.vault_path
    scout_by_id = {
        record.candidate_id: record
        for record in read_models(paths.scout_evaluations, ScoutEvaluationRecord)
    }

    for decision in read_models(paths.curator_decisions, CuratorDecisionRecord):
        if decision.decision != CuratorDecisionType.SURFACE:
            continue
        candidate_id = decision.candidate_id
        rel_path = decision.obsidian_recommendation.path.strip()
        markdown_path = paths.article_markdown(candidate_id)
        metadata_path = paths.article_metadata(candidate_id)
        metadata = (
            load_article_metadata(metadata_path) if metadata_path.exists() else None
        )
        scout_record = scout_by_id.get(candidate_id)
        if not rel_path:
            fallback_title = _sanitize_fallback_title(
                metadata.title if metadata else candidate_id
            )
            rel_path = (
                f"{_placement_fallback(scout_record.tags if scout_record else [], fallback_title)}/"
                f"{fallback_title}.md"
            )
        note_path = resolve_vault_path(vault=vault, rel_path=rel_path)
        raw_body = (
            markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else ""
        )
        body = strip_page_chrome(raw_body)
        note = render_obsidian_note(
            title=metadata.title if metadata else candidate_id,
            url=metadata.url if metadata else "",
            digest_summary=decision.digest_summary,
            body=body,
            curator_reason=decision.reason,
            candidate_id=candidate_id,
            run_id=run_id,
            tags=scout_record.tags if scout_record else [],
            target_date=options.target_date,
        )
        if options.dry_run:
            write_model(
                paths.evidence / f"{candidate_id}.obsidian-preview.json",
                ObsidianPreview(note_path=str(note_path), preview=note[:500]),
            )
            skipped += 1
            append_model(
                paths.obsidian_sync,
                ObsidianSyncRecord(
                    run_id=run_id,
                    candidate_id=candidate_id,
                    timestamp=utc_now_iso(),
                    status=ObsidianSyncStatus.DRY_RUN,
                    note_path=str(note_path),
                ),
            )
            continue
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(note, encoding="utf-8")
        created += 1
        append_model(
            paths.obsidian_sync,
            ObsidianSyncRecord(
                run_id=run_id,
                candidate_id=candidate_id,
                timestamp=utc_now_iso(),
                status=ObsidianSyncStatus.CREATED,
                note_path=str(note_path),
                placement_reason=decision.obsidian_recommendation.placement_reason,
            ),
        )
    return ObsidianSyncResult(created=created, skipped=skipped)
