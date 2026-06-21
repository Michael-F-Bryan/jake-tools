from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class DailyReportPaths:
    root: Path
    subtasks: Path
    evidence: Path
    prompts: Path
    logs: Path
    drafts: Path
    report: Path
    summary: Path
    manifest: Path
    lane_events: Path

    @classmethod
    def for_date(cls, base_dir: Path, target_date: date) -> "DailyReportPaths":
        root = base_dir / f"daily-report-{target_date.isoformat()}"
        return cls(
            root=root,
            subtasks=root / "subtasks",
            evidence=root / "evidence",
            prompts=root / "prompts",
            logs=root / "logs",
            drafts=root / "drafts",
            report=root / "report.md",
            summary=root / "summary.json",
            manifest=root / "manifest.json",
            lane_events=root / "lane-events.jsonl",
        )

    def create(self) -> "DailyReportPaths":
        for directory in [
            self.root,
            self.subtasks,
            self.evidence,
            self.prompts,
            self.logs,
            self.drafts,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
        return self
