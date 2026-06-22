from __future__ import annotations

from typing import Protocol

from .models import DailyReportLaneOptions, LaneOutput, LaneShape, LaneSpec
from ..hermes import AgentSpec, Hermes, HermesResult


class DailyReportStages(Protocol):
    def run_lane(
        self,
        spec: LaneSpec,
        options: DailyReportLaneOptions,
    ) -> tuple[LaneOutput, HermesResult]: ...


class HermesDailyReportStages:
    def __init__(self, hermes: Hermes) -> None:
        self.hermes = hermes

    def run_lane(
        self,
        spec: LaneSpec,
        options: DailyReportLaneOptions,
    ) -> tuple[LaneOutput, HermesResult]:
        if spec.shape is LaneShape.PRE_FED:
            return self.hermes.run_structured_with_result(
                spec.prompt,
                model=spec.model,
                provider=spec.provider,
            )

        agent_spec = AgentSpec(
            model=spec.model,
            provider=spec.provider,
            enabled_toolsets=list(spec.enabled_toolsets),
            parent_session_id=options.parent_session_id,
            session_db=options.session_db,
            max_iterations=options.max_iterations,
        )
        return self.hermes.run_agent_structured(agent_spec, spec.prompt)
