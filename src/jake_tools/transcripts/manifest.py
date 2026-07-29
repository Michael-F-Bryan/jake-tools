from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..ai_usage import AIStageStats, build_ai_stage_stats, build_ai_totals
from ..claude import Reply
from .models import RunManifest, RunStageStatus


def write_json(path: Path, payload: Any) -> None:
    """Write `payload` as pretty, deterministically ordered JSON."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class ManifestBuilder:
    """Accumulates a recipe run's stage statuses and artefact paths.

    Use `step()` around each unit of work so the stage is recorded "pass" on
    success or "fail" on exception; call `skip()` for stages that were
    deliberately not run (e.g. a dry-run vault write). Call `write()` from a
    `finally` block so a run that fails partway still leaves a manifest on
    disk with the failing stage marked, instead of no manifest at all.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.stages: list[RunStageStatus] = []
        self.artefact_paths: dict[str, Path] = {}
        self.warnings: list[str] = []
        self.ai_stage_stats: list[AIStageStats] = []

    def artefact(self, key: str, path: Path) -> None:
        self.artefact_paths[key] = path

    def add_warnings(self, warnings: list[str]) -> None:
        self.warnings.extend(warnings)

    def record_reply(self, stage: str, reply: Reply | None) -> None:
        stats = build_ai_stage_stats(stage, reply)
        if stats is not None:
            self.ai_stage_stats.append(stats)

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        try:
            yield
        except BaseException:
            self.stages.append(RunStageStatus(stage=name, status="fail"))
            raise
        else:
            self.stages.append(RunStageStatus(stage=name, status="pass"))

    def skip(self, name: str) -> None:
        self.stages.append(RunStageStatus(stage=name, status="skipped"))

    def build(self) -> RunManifest:
        return RunManifest(
            run_id=self.run_id,
            stages=self.stages,
            artefact_paths=self.artefact_paths,
            ai_totals=build_ai_totals(self.ai_stage_stats)
            if self.ai_stage_stats
            else None,
            warnings=self.warnings,
        )

    def write(self, manifest_path: Path) -> None:
        manifest_path.write_text(
            json.dumps(self.build().model_dump(mode="json"), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
