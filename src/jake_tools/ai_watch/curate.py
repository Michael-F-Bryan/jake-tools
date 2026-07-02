from __future__ import annotations

from dataclasses import dataclass

from ..ai_usage import Usage
from .archive import load_article_metadata
from .audit import append_model, read_models, truncate_records, utc_now_iso
from .audit_models import CuratorDecisionRecord, ScoutEvaluationRecord
from .models import AiWatchCommandOptions, ScoutRecommendation
from .paths import AiWatchPaths
from .stages import AiWatchStages, load_interest_profile
from .validation import validate_curated


@dataclass
class CurateRunResult:
    evaluated: int
    surfaced: int
    speculative: int
    usage: Usage


def run_curate(
    *,
    options: AiWatchCommandOptions,
    paths: AiWatchPaths,
    stages: AiWatchStages,
) -> CurateRunResult:
    paths.create()
    truncate_records(paths.curator_decisions)
    run_id = options.target_date.isoformat()
    interest_profile = load_interest_profile()
    evaluated = 0
    surfaced = 0
    speculative = 0
    usage = Usage()

    for scout_record in read_models(paths.scout_evaluations, ScoutEvaluationRecord):
        if (
            options.force_candidate
            and scout_record.candidate_id != options.force_candidate
        ):
            continue
        if (
            not options.force_candidate
            and scout_record.recommendation != ScoutRecommendation.PROMOTE_TO_CURATOR
        ):
            continue
        markdown_path = paths.article_markdown(scout_record.candidate_id)
        if not markdown_path.exists():
            continue
        article_text = markdown_path.read_text(encoding="utf-8")
        metadata_path = paths.article_metadata(scout_record.candidate_id)
        metadata = (
            load_article_metadata(metadata_path) if metadata_path.exists() else None
        )

        decision, reply = stages.run_curate(
            scout_record=scout_record,
            article_text=article_text,
            metadata=metadata,
            options=options,
            interest_profile=interest_profile,
        )
        usage = usage + reply.usage
        append_model(
            paths.curator_decisions,
            CuratorDecisionRecord.from_decision(
                run_id=run_id,
                candidate_id=scout_record.candidate_id,
                timestamp=utc_now_iso(),
                model=options.curator_model,
                decision=decision,
            ),
        )
        evaluated += 1
        if decision.decision.value == "surface":
            surfaced += 1
        elif decision.decision.value == "speculative_watch":
            speculative += 1
        if evaluated >= options.max_candidates:
            break

    errors = validate_curated(
        read_models(paths.curator_decisions, CuratorDecisionRecord),
        vault_root=str(options.vault_path),
    )
    if errors:
        raise RuntimeError("; ".join(errors))
    return CurateRunResult(
        evaluated=evaluated, surfaced=surfaced, speculative=speculative, usage=usage
    )
