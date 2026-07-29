from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from ..ai_usage import AITotals
from .audit_models import (
    CuratorDecisionRecord,
    CuratorDecisionType,
    ObsidianSyncRecord,
    RunManifest,
    ScoutEvaluationRecord,
)
from .delivery import build_discord_payload
from .digest import run_digest
from .manifest import write_manifest
from .models import AiWatchCommandOptions, DigestLane, RunStatus
from .paths import AiWatchPaths
from .records import load_model, read_models


@dataclass(frozen=True)
class SurfacePolicyResult:
    before_surface_count: int
    after_surface_count: int
    demoted_count: int
    removed_note_count: int = 0
    skipped_note_paths: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TuneResult:
    surface_policy: SurfacePolicyResult
    surfaced: int
    speculative: int


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

DATE_PATTERNS = [
    re.compile(
        r"(?:published\s+)?(?P<month>[A-Z][a-z]+)\s+(?P<day>\d{1,2}),\s+(?P<year>20\d{2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:published\s+)?(?P<year>20\d{2})-(?P<month>\d{1,2})-(?P<day>\d{1,2})",
        re.IGNORECASE,
    ),
]


def _surface_rank(
    record: CuratorDecisionRecord, scouts: dict[str, ScoutEvaluationRecord]
) -> tuple[int, int, int, int, int]:
    scout = scouts.get(record.candidate_id)
    if scout is None:
        return (0, 0, 0, 0, 0)
    weighted_score = (
        scout.fit_score * 3
        + scout.practicality_score * 2
        + scout.novelty_score
        - scout.noise_risk * 2
    )
    return (
        weighted_score,
        scout.fit_score,
        scout.practicality_score,
        scout.novelty_score,
        -scout.noise_risk,
    )


def _read_article_text(paths: AiWatchPaths, candidate_id: str) -> str:
    path = paths.article_markdown(candidate_id)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _extract_published_date(article_text: str) -> date | None:
    for pattern in DATE_PATTERNS:
        match = pattern.search(article_text[:2000])
        if match is None:
            continue
        month = match.group("month")
        if not month.isdigit():
            month_number = MONTHS.get(month.lower())
            if month_number is None:
                continue
            month = str(month_number)
        try:
            return datetime(
                int(match.group("year")), int(month), int(match.group("day"))
            ).date()
        except ValueError:
            continue
    return None


def _demotion_reason_for_article(
    *,
    record: CuratorDecisionRecord,
    article_text: str,
    options: AiWatchCommandOptions,
) -> str | None:
    combined = (
        f"{record.digest_summary}\n{record.reason}\n{article_text[:2000]}".lower()
    )
    if "xcode" in combined and "cursor" not in combined:
        return "Xcode-specific developer tooling is low value for Michael's Cursor-based workflow."

    published = _extract_published_date(article_text)
    if published is None:
        return None
    age_days = (options.target_date - published).days
    if age_days > options.max_article_age_days:
        return (
            f"Article is {age_days} days old, older than "
            f"{options.max_article_age_days} days; AI Watch should prioritise recent developments."
        )
    return None


def _demote_surface_record(
    record: CuratorDecisionRecord, reason: str
) -> CuratorDecisionRecord:
    recommendation = record.obsidian_recommendation.model_copy(
        update={"should_create_note": False}
    )
    return record.model_copy(
        update={
            "decision": CuratorDecisionType.SPECULATIVE_WATCH,
            "lane": DigestLane.SPECULATIVE_WATCH,
            "reason": f"{record.reason} Demoted from the main digest: {reason}",
            "obsidian_recommendation": recommendation,
        }
    )


def _write_decisions(path: Path, records: list[CuratorDecisionRecord]) -> None:
    # write_text() already overwrites the whole file, so a truncate_records()
    # call first would just be discarded work; only the mkdir is needed here.
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(
        json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n"
        for record in records
    )
    path.write_text(encoded, encoding="utf-8")


def _remove_safe_note(
    *,
    vault_path: Path,
    record: CuratorDecisionRecord,
    run_id: str,
    synced_note_path: str | None = None,
) -> tuple[bool, str | None]:
    rel_path = record.obsidian_recommendation.path.strip()
    if rel_path:
        note_path = vault_path / rel_path
    elif synced_note_path:
        note_path = Path(synced_note_path)
    else:
        return False, None
    if not note_path.exists():
        return False, None
    try:
        body = note_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False, json.dumps({"path": str(note_path), "reason": "decode_error"})
    if (
        f"Candidate ID: {record.candidate_id}" not in body
        or f"Run ID: {run_id}" not in body
    ):
        return False, json.dumps({"path": str(note_path), "reason": "marker_mismatch"})
    note_path.unlink()
    return True, None


def apply_surface_policy(
    *,
    options: AiWatchCommandOptions,
    paths: AiWatchPaths,
    remove_notes: bool = False,
) -> SurfacePolicyResult:
    records = read_models(paths.curator_decisions, CuratorDecisionRecord)
    surface_records = [
        record for record in records if record.decision == CuratorDecisionType.SURFACE
    ]
    before_surface_count = len(surface_records)
    hard_demotions = {
        record.candidate_id: reason
        for record in surface_records
        if (
            reason := _demotion_reason_for_article(
                record=record,
                article_text=_read_article_text(paths, record.candidate_id),
                options=options,
            )
        )
    }
    if (
        not remove_notes
        and not hard_demotions
        and (
            options.surface_limit is None
            or before_surface_count <= options.surface_limit
        )
    ):
        return SurfacePolicyResult(
            before_surface_count=before_surface_count,
            after_surface_count=before_surface_count,
            demoted_count=0,
        )

    if options.surface_limit is None:
        surface_limit = before_surface_count
    else:
        surface_limit = options.surface_limit

    scouts = {
        record.candidate_id: record
        for record in read_models(paths.scout_evaluations, ScoutEvaluationRecord)
    }
    # Hard-demoted items never compete for a surface slot: including them here
    # would let a condemned top-ranked article consume a limit slot instead of
    # an eligible one, shipping the digest under the configured limit.
    eligible_records = [
        record
        for record in surface_records
        if record.candidate_id not in hard_demotions
    ]
    keep_ids = {
        record.candidate_id
        for record in sorted(
            eligible_records,
            key=lambda record: _surface_rank(record, scouts),
            reverse=True,
        )[:surface_limit]
    }

    demoted_originals: list[CuratorDecisionRecord] = []
    rewritten: list[CuratorDecisionRecord] = []
    for record in records:
        if record.decision != CuratorDecisionType.SURFACE:
            rewritten.append(record)
            continue
        if reason := hard_demotions.get(record.candidate_id):
            demoted_originals.append(record)
            rewritten.append(_demote_surface_record(record, reason))
            continue
        if record.candidate_id not in keep_ids:
            demoted_originals.append(record)
            rewritten.append(
                _demote_surface_record(
                    record, f"per-run surface limit ({surface_limit})."
                )
            )
            continue
        rewritten.append(record)

    _write_decisions(paths.curator_decisions, rewritten)

    removed_note_count = 0
    skipped_note_paths: list[str] = []
    if remove_notes:
        synced_note_paths = {
            record.candidate_id: record.note_path
            for record in read_models(paths.obsidian_sync, ObsidianSyncRecord)
        }
        for record in rewritten:
            if record.decision == CuratorDecisionType.SURFACE:
                continue
            removed, skipped = _remove_safe_note(
                vault_path=options.vault_path,
                record=record,
                run_id=options.target_date.isoformat(),
                synced_note_path=synced_note_paths.get(record.candidate_id),
            )
            if removed:
                removed_note_count += 1
            if skipped:
                skipped_note_paths.append(skipped)

    return SurfacePolicyResult(
        before_surface_count=before_surface_count,
        after_surface_count=len(keep_ids),
        demoted_count=len(demoted_originals),
        removed_note_count=removed_note_count,
        skipped_note_paths=skipped_note_paths,
    )


def run_tune(
    *,
    options: AiWatchCommandOptions,
    paths: AiWatchPaths,
    remove_notes: bool = False,
) -> TuneResult:
    """Re-apply the surface policy, re-render the digest, and refresh counts.

    Tuning only revises which items surface; it never re-runs the paid
    stages, so it must not silently turn a FAILED run into an OK one. The
    prior manifest's status and failed_stages are preserved verbatim.
    """
    surface_policy = apply_surface_policy(
        options=options, paths=paths, remove_notes=remove_notes
    )
    surfaced, speculative = run_digest(paths=paths)
    digest_text = paths.digest.read_text(encoding="utf-8")
    (paths.root / "delivery-payload.txt").write_text(
        build_discord_payload(digest_text) if surfaced else digest_text,
        encoding="utf-8",
    )
    summary = (
        AITotals.model_validate_json(paths.summary.read_text(encoding="utf-8"))
        if paths.summary.exists()
        else AITotals()
    )
    previous_manifest = (
        load_model(paths.manifest, RunManifest) if paths.manifest.exists() else None
    )
    status = previous_manifest.status if previous_manifest else RunStatus.OK
    failed_stages = previous_manifest.failed_stages if previous_manifest else []
    write_manifest(
        paths=paths,
        options=options,
        status=status,
        surfaced_count=surfaced,
        speculative_count=speculative,
        failed_stages=failed_stages,
        summary=summary,
    )
    return TuneResult(
        surface_policy=surface_policy, surfaced=surfaced, speculative=speculative
    )
