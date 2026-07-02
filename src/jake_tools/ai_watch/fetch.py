from __future__ import annotations

from dataclasses import dataclass

from .archive import archive_extract
from .audit import (
    append_model,
    read_discovered_candidates,
    read_models,
    truncate_records,
    utc_now_iso,
)
from .audit_models import FetchRecord, FetchStatus
from .calibration import load_calibration_cases, resolve_calibration_extract
from .models import AiWatchCommandOptions
from .paths import AiWatchPaths
from .state import SeenIndex
from .validation import validate_fetched
from .web_tools import WebTools


@dataclass
class FetchResult:
    fetched: int
    failed: int
    skipped: int


def _discovered_candidates(paths: AiWatchPaths) -> list:
    return read_discovered_candidates(paths.candidates)


def run_fetch(
    *,
    options: AiWatchCommandOptions,
    paths: AiWatchPaths,
    state: SeenIndex,
    web_tools: WebTools,
) -> FetchResult:
    paths.create()
    truncate_records(paths.fetch_results)
    run_id = options.target_date.isoformat()
    fetched = 0
    failed = 0
    skipped = 0
    calibration_cases = load_calibration_cases() if options.calibration_only else []

    for candidate in _discovered_candidates(paths):
        seen = state.check_seen(url=candidate.url, title=candidate.title)
        if seen and seen.latest_content_path and not options.calibration_only:
            skipped += 1
            append_model(
                paths.fetch_results,
                FetchRecord(
                    run_id=run_id,
                    candidate_id=candidate.candidate_id,
                    timestamp=utc_now_iso(),
                    status=FetchStatus.SKIPPED,
                    reason="already_archived",
                    content_path=seen.latest_content_path,
                ),
            )
            continue

        extracted = web_tools.extract([candidate.url])[0]
        if calibration_cases:
            extracted = resolve_calibration_extract(
                url=candidate.url,
                title=candidate.title or "",
                extracted=extracted,
                calibration_cases=calibration_cases,
            )
        if extracted.status != "ok" or not extracted.content.strip():
            failed += 1
            append_model(
                paths.fetch_results,
                FetchRecord(
                    run_id=run_id,
                    candidate_id=candidate.candidate_id,
                    timestamp=utc_now_iso(),
                    status=FetchStatus.FAIL,
                    error=extracted.error or "empty content",
                ),
            )
            continue

        markdown_path, metadata_path, content_hash = archive_extract(
            paths_articles=paths.articles,
            paths_raw=paths.raw,
            candidate_id=candidate.candidate_id,
            extracted=extracted,
            source=candidate.source,
            save_raw=options.save_raw,
        )
        state.record_content_hash(extracted.content, candidate.candidate_id)
        state.record_seen(
            candidate_id=candidate.candidate_id,
            url=candidate.url,
            title=candidate.title or extracted.title,
            source=candidate.source,
            content_hash=content_hash,
            content_path=str(markdown_path),
        )
        append_model(
            paths.fetch_results,
            FetchRecord(
                run_id=run_id,
                candidate_id=candidate.candidate_id,
                timestamp=utc_now_iso(),
                status=FetchStatus.OK,
                content_path=str(markdown_path.relative_to(paths.root)),
                metadata_path=str(metadata_path.relative_to(paths.root)),
                content_hash=content_hash,
            ),
        )
        fetched += 1

    errors = validate_fetched(read_models(paths.fetch_results, FetchRecord))
    if errors:
        raise RuntimeError("; ".join(errors))
    return FetchResult(fetched=fetched, failed=failed, skipped=skipped)
