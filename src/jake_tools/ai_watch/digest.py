from __future__ import annotations

from .audit import read_discovered_candidates, read_models
from .audit_models import CuratorDecisionRecord, CuratorDecisionType
from .models import AiWatchCommandOptions
from .paths import AiWatchPaths


def render_digest(
    paths: AiWatchPaths, options: AiWatchCommandOptions
) -> tuple[int, int]:
    main_items: list[str] = []
    speculative_items: list[str] = []
    rejected_items: list[str] = []

    curator_rows = read_models(paths.curator_decisions, CuratorDecisionRecord)
    discovered_by_id = {
        candidate.candidate_id: candidate
        for candidate in read_discovered_candidates(paths.candidates)
    }

    for row in curator_rows:
        discovered = discovered_by_id.get(row.candidate_id)
        title = discovered.title if discovered else row.candidate_id
        url = discovered.url if discovered else ""
        block = (
            f"## {title}\n\n"
            f"{row.digest_summary}\n\n"
            f"Why it matters: {row.reason}\n\n"
            f"URL: {url}\n"
        )
        if row.decision == CuratorDecisionType.SURFACE:
            block += f"Obsidian: {row.obsidian_recommendation.path}\n"
            main_items.append(block)
        elif row.decision == CuratorDecisionType.SPECULATIVE_WATCH:
            speculative_items.append(block)
        elif row.decision == CuratorDecisionType.REJECT:
            rejected_items.append(block)

    paths.digest.write_text(
        "\n\n".join(main_items) if main_items else "_No items crossed the bar._\n",
        encoding="utf-8",
    )
    paths.speculative.write_text(
        "\n\n".join(speculative_items)
        if speculative_items
        else "_No speculative items._\n",
        encoding="utf-8",
    )
    paths.rejected.write_text(
        "\n\n".join(rejected_items) if rejected_items else "_No rejects recorded._\n",
        encoding="utf-8",
    )
    return len(main_items), len(speculative_items)


def run_digest(
    *, options: AiWatchCommandOptions, paths: AiWatchPaths
) -> tuple[int, int]:
    paths.create()
    return render_digest(paths, options)
