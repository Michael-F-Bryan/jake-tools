from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .audit import read_models, truncate_records
from .audit_models import (
    CuratorDecisionRecord,
    CuratorDecisionType,
    ObsidianSyncRecord,
    ScoutEvaluationRecord,
)
from .models import AiWatchCommandOptions, DigestLane
from .paths import AiWatchPaths


@dataclass(frozen=True)
class SurfacePolicyResult:
    before_surface_count: int
    after_surface_count: int
    demoted_count: int
    removed_note_count: int = 0
    skipped_note_paths: list[str] = field(default_factory=list)


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


def _demote_surface_record(
    record: CuratorDecisionRecord, surface_limit: int
) -> CuratorDecisionRecord:
    recommendation = record.obsidian_recommendation.model_copy(
        update={"should_create_note": False}
    )
    return record.model_copy(
        update={
            "decision": CuratorDecisionType.SPECULATIVE_WATCH,
            "lane": DigestLane.SPECULATIVE_WATCH,
            "reason": (
                f"{record.reason} Demoted from the main digest by the "
                f"per-run surface limit ({surface_limit})."
            ),
            "obsidian_recommendation": recommendation,
        }
    )


def _write_decisions(path: Path, records: list[CuratorDecisionRecord]) -> None:
    truncate_records(path)
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
    if not remove_notes and (
        options.surface_limit is None or before_surface_count <= options.surface_limit
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
    keep_ids = {
        record.candidate_id
        for record in sorted(
            surface_records,
            key=lambda record: _surface_rank(record, scouts),
            reverse=True,
        )[:surface_limit]
    }

    demoted_originals: list[CuratorDecisionRecord] = []
    rewritten: list[CuratorDecisionRecord] = []
    for record in records:
        if (
            record.decision == CuratorDecisionType.SURFACE
            and record.candidate_id not in keep_ids
        ):
            demoted_originals.append(record)
            rewritten.append(_demote_surface_record(record, surface_limit))
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
