from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .audit import append_model, truncate_records, utc_now_iso
from .audit_models import DiscoveredRecord, SeenCheckRecord
from .calibration import DEFAULT_CALIBRATION_CASES_PATH, load_calibration_cases
from .models import AiWatchCommandOptions, SearchResult, candidate_id_for
from .paths import AiWatchPaths
from .sources import DEFAULT_SOURCE_QUERIES, SourceQuery
from .state import SeenIndex
from .web_tools import WebTools


@dataclass
class CollectResult:
    discovered: int
    skipped_seen: int


def run_collect(
    *,
    options: AiWatchCommandOptions,
    paths: AiWatchPaths,
    state: SeenIndex,
    web_tools: WebTools,
    source_queries: tuple[SourceQuery, ...] = DEFAULT_SOURCE_QUERIES,
    fixtures_path: Path | None = None,
) -> CollectResult:
    paths.create()
    truncate_records(paths.candidates)
    run_id = options.target_date.isoformat()
    discovered = 0
    skipped_seen = 0
    seen_urls: set[str] = set()

    if options.calibration_only:
        calibration_cases = load_calibration_cases(
            fixtures_path or DEFAULT_CALIBRATION_CASES_PATH
        )
        rows = [
            SearchResult(
                url=case.url,
                title=case.title_hint,
                description="calibration fixture",
                source="calibration",
            )
            for case in calibration_cases
        ]
        source_queries = ()
    else:
        rows = []
        for query in source_queries:
            rows.extend(
                [
                    result.model_copy(update={"source": query.source_id})
                    for result in web_tools.search(query.query, limit=query.limit)
                ]
            )

    for result in rows:
        canonical = result.url.strip().rstrip("/").lower()
        if canonical in seen_urls:
            continue
        seen_urls.add(canonical)
        candidate_id = candidate_id_for(url=result.url)
        seen = state.check_seen(url=result.url)
        if seen is not None and not options.calibration_only:
            skipped_seen += 1
            append_model(
                paths.candidates,
                SeenCheckRecord(
                    run_id=run_id,
                    candidate_id=candidate_id,
                    timestamp=utc_now_iso(),
                    first_seen_at=seen.first_seen_at,
                    latest_decision=seen.latest_decision,
                ),
            )
            continue

        append_model(
            paths.candidates,
            DiscoveredRecord(
                run_id=run_id,
                candidate_id=candidate_id,
                timestamp=utc_now_iso(),
                source=result.source,
                url=result.url,
                title=result.title,
                description=result.description,
            ),
        )
        state.record_seen(
            candidate_id=candidate_id,
            url=result.url,
            title=result.title,
            source=result.source,
        )
        discovered += 1
        if discovered >= options.max_candidates:
            break

    return CollectResult(discovered=discovered, skipped_seen=skipped_seen)
