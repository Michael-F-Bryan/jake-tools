from __future__ import annotations

from dataclasses import dataclass

from ..ai_usage import Usage
from .archive import load_article_metadata
from .audit import (
    append_model,
    read_discovered_candidates,
    read_models,
    truncate_records,
    utc_now,
)
from .audit_models import (
    FetchRecord,
    FetchStatus,
    ScoutEvaluationRecord,
    ScoutFailureRecord,
)
from .models import AiWatchCommandOptions, AiWatchStageError, ScoutRecommendation
from .paths import AiWatchPaths
from .stages import AiWatchStages, load_interest_profile
from .validation import sanitize_evidence_quotes, validate_scouted


@dataclass
class ScoutRunResult:
    evaluated: int
    promoted: int
    usage: Usage
    failed: int = 0


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


async def run_scout(
    *,
    options: AiWatchCommandOptions,
    paths: AiWatchPaths,
    stages: AiWatchStages,
) -> ScoutRunResult:
    paths.create()
    truncate_records(paths.scout_evaluations)
    truncate_records(paths.scout_failures)
    run_id = options.target_date.isoformat()
    interest_profile = load_interest_profile()
    attempted = 0
    evaluated = 0
    promoted = 0
    failed = 0
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

        attempted += 1
        try:
            output, reply = await stages.run_scout(
                article_text=article_text,
                metadata=metadata,
                options=options,
                interest_profile=interest_profile,
            )
        except Exception as error:  # noqa: BLE001 - isolate one bad candidate
            # A single malformed LLM reply must not discard the tokens
            # already spent evaluating every other candidate in this batch.
            failed += 1
            append_model(
                paths.scout_failures,
                ScoutFailureRecord(
                    run_id=run_id,
                    candidate_id=candidate.candidate_id,
                    timestamp=utc_now(),
                    errors=[f"scout call failed: {error}"],
                ),
            )
            if attempted >= options.max_candidates:
                break
            continue

        usage = usage + reply.usage
        output = output.model_copy(
            update={
                "evidence_quotes": sanitize_evidence_quotes(
                    quotes=output.evidence_quotes,
                    article=article_text,
                )
            }
        )
        record = ScoutEvaluationRecord.from_output(
            run_id=run_id,
            candidate_id=candidate.candidate_id,
            timestamp=utc_now(),
            model=options.scout_model,
            output=output,
        )
        # Validate grounding here, against the article text already in hand,
        # instead of re-reading every article from disk after the loop and
        # failing the whole (already paid-for) batch over one bad candidate.
        record_errors = validate_scouted(
            [record], article_texts={candidate.candidate_id: article_text}
        )
        if record_errors:
            failed += 1
            append_model(
                paths.scout_failures,
                ScoutFailureRecord(
                    run_id=run_id,
                    candidate_id=candidate.candidate_id,
                    timestamp=utc_now(),
                    errors=record_errors,
                ),
            )
        else:
            append_model(paths.scout_evaluations, record)
            evaluated += 1
            if output.recommendation == ScoutRecommendation.PROMOTE_TO_CURATOR:
                promoted += 1
        if attempted >= options.max_candidates:
            break

    if attempted and evaluated == 0:
        failures = read_models(paths.scout_failures, ScoutFailureRecord)
        raise AiWatchStageError(
            stage="scout",
            errors=[
                f"{failure.candidate_id}: {'; '.join(failure.errors)}"
                for failure in failures
            ],
        )
    return ScoutRunResult(
        evaluated=evaluated, promoted=promoted, usage=usage, failed=failed
    )
