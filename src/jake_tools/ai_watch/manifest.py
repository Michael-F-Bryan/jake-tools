from __future__ import annotations

from ..ai_usage import AITotals
from .audit import read_discovered_candidates, read_models, write_model
from .audit_models import (
    CuratorDecisionRecord,
    FetchRecord,
    FetchStatus,
    RunManifest,
    RunManifestCounts,
    RunManifestPaths,
    ScoutEvaluationRecord,
)
from .models import AiWatchCommandOptions, RunStatus
from .paths import AiWatchPaths


def build_manifest(
    *,
    paths: AiWatchPaths,
    options: AiWatchCommandOptions,
    status: RunStatus,
    surfaced_count: int,
    speculative_count: int,
    failed_stages: list[str],
) -> RunManifest:
    return RunManifest(
        run_id=options.target_date.isoformat(),
        status=status,
        paths=RunManifestPaths(
            root=str(paths.root),
            digest=str(paths.digest),
            summary=str(paths.summary),
        ),
        counts=RunManifestCounts(
            candidates=len(read_discovered_candidates(paths.candidates)),
            fetched=len(
                [
                    record
                    for record in read_models(paths.fetch_results, FetchRecord)
                    if record.status == FetchStatus.OK
                ]
            ),
            scouted=len(read_models(paths.scout_evaluations, ScoutEvaluationRecord)),
            curated=len(read_models(paths.curator_decisions, CuratorDecisionRecord)),
            surfaced=surfaced_count,
            speculative=speculative_count,
        ),
        failed_stages=failed_stages,
        dry_run=options.dry_run,
    )


def write_manifest(
    *,
    paths: AiWatchPaths,
    options: AiWatchCommandOptions,
    status: RunStatus,
    surfaced_count: int,
    speculative_count: int,
    failed_stages: list[str],
    summary: AITotals,
) -> None:
    manifest = build_manifest(
        paths=paths,
        options=options,
        status=status,
        surfaced_count=surfaced_count,
        speculative_count=speculative_count,
        failed_stages=failed_stages,
    )
    write_model(paths.manifest, manifest)
    write_model(paths.summary, summary)
