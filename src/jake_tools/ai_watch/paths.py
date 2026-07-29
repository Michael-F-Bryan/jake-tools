from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict


class AiWatchPaths(BaseModel):
    model_config = ConfigDict(frozen=True)

    root: Path
    state_root: Path
    articles: Path
    prompts: Path
    evidence: Path
    manifest: Path
    summary: Path
    digest: Path
    speculative: Path
    rejected: Path
    candidates: Path
    fetch_results: Path
    scout_evaluations: Path
    scout_failures: Path
    curator_decisions: Path
    obsidian_sync: Path
    delivery: Path

    @classmethod
    def for_date(cls, base_dir: Path, target_date: date) -> Self:
        watch_root = base_dir / "ai-watch"
        root = watch_root / target_date.isoformat()
        state_root = watch_root / "state"
        return cls(
            root=root,
            state_root=state_root,
            articles=root / "articles",
            prompts=root / "prompts",
            evidence=root / "evidence",
            manifest=root / "manifest.json",
            summary=root / "summary.json",
            digest=root / "digest.md",
            speculative=root / "speculative.md",
            rejected=root / "rejected.md",
            candidates=root / "candidates.jsonl",
            fetch_results=root / "fetch-results.jsonl",
            scout_evaluations=root / "scout-evaluations.jsonl",
            scout_failures=root / "scout-failures.jsonl",
            curator_decisions=root / "curator-decisions.jsonl",
            obsidian_sync=root / "obsidian-sync.jsonl",
            delivery=root / "delivery.jsonl",
        )

    def create(self) -> Self:
        for directory in [
            self.root,
            self.articles,
            self.prompts,
            self.evidence,
            self.state_root,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def article_markdown(self, candidate_id: str) -> Path:
        return self.articles / f"{candidate_id}.md"

    def article_metadata(self, candidate_id: str) -> Path:
        return self.articles / f"{candidate_id}.metadata.json"
