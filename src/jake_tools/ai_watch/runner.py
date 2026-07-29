from __future__ import annotations

from dataclasses import dataclass, field

from ..ai_usage import AIStageStats, build_ai_totals
from ..claude import ClaudeAgent
from .collect import run_collect
from .curate import run_curate
from .delivery import HermesSendSender, MessageSender, run_delivery
from .digest import run_digest
from .fetch import run_fetch
from .manifest import write_manifest
from .models import AiWatchCommandOptions, AiWatchCommandResult, RunStatus, StageFailure
from .obsidian import run_obsidian_sync
from .paths import AiWatchPaths
from .records import truncate_records
from .scout import run_scout
from .stages import AiWatchStages, ClaudeAiWatchStages
from .state import SeenIndex
from .web_tools import HermesWebTools, WebTools


@dataclass
class RunnerDeps:
    """Everything the run reaches the outside world through."""

    web_tools: WebTools = field(default_factory=HermesWebTools)
    sender: MessageSender = field(default_factory=HermesSendSender)
    agent: ClaudeAgent | None = None
    stages: AiWatchStages | None = None


async def run_ai_watch_command(
    *,
    options: AiWatchCommandOptions,
    deps: RunnerDeps | None = None,
) -> AiWatchCommandResult:
    resolved = deps or RunnerDeps()
    paths = AiWatchPaths.for_date(options.base_dir, options.target_date).create()
    state = SeenIndex(paths.state_root)
    failed_stages: list[StageFailure] = []
    stage_stats: list[AIStageStats] = []
    run_id = options.target_date.isoformat()
    surfaced = 0
    speculative = 0

    stages = resolved.stages or ClaudeAiWatchStages(resolved.agent or ClaudeAgent())

    try:
        run_collect(
            options=options, paths=paths, state=state, web_tools=resolved.web_tools
        )
    except Exception as error:  # noqa: BLE001
        failed_stages.append(StageFailure(stage="collect", error=str(error)))

    try:
        run_fetch(
            options=options, paths=paths, state=state, web_tools=resolved.web_tools
        )
    except Exception as error:  # noqa: BLE001
        failed_stages.append(StageFailure(stage="fetch", error=str(error)))

    try:
        scout_result = await run_scout(options=options, paths=paths, stages=stages)
        stage_stats.append(AIStageStats(stage="scout", usage=scout_result.usage))
    except Exception as error:  # noqa: BLE001
        failed_stages.append(StageFailure(stage="scout", error=str(error)))

    # The cost cap only has teeth if it is checked *before* the next paid
    # stage spends money, not after every stage (including delivery) has
    # already run. Scout and curate are the only two paid stages, so this is
    # the one gate the cap needs.
    cost_so_far = build_ai_totals(stage_stats).estimated_cost_usd
    cost_cap_exceeded = (
        options.cost_cap_usd is not None and cost_so_far > options.cost_cap_usd
    )
    if cost_cap_exceeded:
        truncate_records(paths.curator_decisions)
        failed_stages.append(
            StageFailure(
                stage="curate",
                error=(
                    f"skipped: cost cap ${options.cost_cap_usd:.2f} exceeded after "
                    f"scout (spent ${cost_so_far:.2f})"
                ),
            )
        )
    else:
        try:
            curate_result = await run_curate(
                options=options, paths=paths, stages=stages
            )
            stage_stats.append(AIStageStats(stage="curate", usage=curate_result.usage))
            surfaced = curate_result.surfaced
            speculative = curate_result.speculative
        except Exception as error:  # noqa: BLE001
            failed_stages.append(StageFailure(stage="curate", error=str(error)))

    try:
        run_obsidian_sync(options=options, paths=paths)
    except Exception as error:  # noqa: BLE001
        failed_stages.append(StageFailure(stage="obsidian_sync", error=str(error)))

    try:
        surfaced, speculative = run_digest(paths=paths)
    except Exception as error:  # noqa: BLE001
        failed_stages.append(StageFailure(stage="digest", error=str(error)))

    try:
        run_delivery(options=options, paths=paths, sender=resolved.sender)
    except Exception as error:  # noqa: BLE001
        failed_stages.append(StageFailure(stage="deliver", error=str(error)))

    totals = build_ai_totals(stage_stats)
    status = RunStatus.FAIL if failed_stages else RunStatus.OK
    write_manifest(
        paths=paths,
        options=options,
        status=status,
        surfaced_count=surfaced,
        speculative_count=speculative,
        failed_stages=failed_stages,
        summary=totals,
    )

    return AiWatchCommandResult(
        status=status,
        run_id=run_id,
        root=paths.root,
        digest_path=paths.digest,
        summary_path=paths.summary,
        surfaced_count=surfaced,
        speculative_count=speculative,
        failed_stages=failed_stages,
    )
