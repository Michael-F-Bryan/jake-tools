from __future__ import annotations

from dataclasses import dataclass

from ..ai_usage import Usage
from .archive import load_article_metadata
from .audit import (
    append_model,
    read_discovered_candidates,
    read_models,
    truncate_records,
    utc_now_iso,
)
from .audit_models import (
    FetchRecord,
    FetchStatus,
    ScoutEvaluationRecord,
)
from .models import AiWatchCommandOptions, ScoutRecommendation
from .paths import AiWatchPaths
from .stages import AiWatchStages, load_interest_profile
from .validation import sanitize_evidence_quotes, validate_scouted


@dataclass
class ScoutRunResult:
    evaluated: int
    promoted: int
    usage: Usage


def _fetched_candidates(paths: AiWatchPaths):
    fetched_ids = {
        record.candidate_id
        for record in read_models(paths.fetch_results, FetchRecord)
        if record.status == FetchStatus.OK
    }
    return [
        candidate
        for candidate in read_discovered_candidates(paths.candidates)
        if candidate.candidate_id in fetched_ids
    ]


def run_scout(
    *,
    options: AiWatchCommandOptions,
    paths: AiWatchPaths,
    stages: AiWatchStages,
) -> ScoutRunResult:
    paths.create()
    truncate_records(paths.scout_evaluations)
    run_id = options.target_date.isoformat()
    interest_profile = load_interest_profile()
    evaluated = 0
    promoted = 0
    usage = Usage()

    for candidate in _fetched_candidates(paths):
        markdown_path = paths.article_markdown(candidate.candidate_id)
        if not markdown_path.exists():
            continue
        article_text = markdown_path.read_text(encoding="utf-8")
        metadata_path = paths.article_metadata(candidate.candidate_id)
        metadata = (
            load_article_metadata(metadata_path) if metadata_path.exists() else None
        )

        output, reply = stages.run_scout(
            article_text=article_text,
            metadata=metadata,
            options=options,
            interest_profile=interest_profile,
        )
        output = output.model_copy(
            update={
                "evidence_quotes": sanitize_evidence_quotes(
                    quotes=output.evidence_quotes,
                    article=article_text,
                )
            }
        )
        usage = usage + reply.usage
        append_model(
            paths.scout_evaluations,
            ScoutEvaluationRecord.from_output(
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                timestamp=utc_now_iso(),
                model=options.scout_model,
                output=output,
            ),
        )
        evaluated += 1
        if output.recommendation == ScoutRecommendation.PROMOTE_TO_CURATOR:
            promoted += 1
        if evaluated >= options.max_candidates:
            break

    article_texts = {
        candidate.candidate_id: paths.article_markdown(
            candidate.candidate_id
        ).read_text(encoding="utf-8")
        for candidate in _fetched_candidates(paths)
        if paths.article_markdown(candidate.candidate_id).exists()
    }
    errors = validate_scouted(
        read_models(paths.scout_evaluations, ScoutEvaluationRecord),
        article_texts=article_texts,
    )
    if errors:
        raise RuntimeError("; ".join(errors))
    return ScoutRunResult(evaluated=evaluated, promoted=promoted, usage=usage)
