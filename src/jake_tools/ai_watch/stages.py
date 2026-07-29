from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Protocol

from pydantic import BaseModel

from ..claude import AgentSpec, ClaudeAgent, Reply
from ..prompting import StructuredPrompt
from .audit_models import ArticleMetadata, ScoutEvaluationRecord
from .models import AiWatchCommandOptions, CuratorDecision, ScoutOutput


class AiWatchStages(Protocol):
    async def run_scout(
        self,
        *,
        article_text: str,
        metadata: ArticleMetadata | None,
        options: AiWatchCommandOptions,
        interest_profile: str,
    ) -> tuple[ScoutOutput, Reply]: ...

    async def run_curate(
        self,
        *,
        scout_record: ScoutEvaluationRecord,
        article_text: str,
        metadata: ArticleMetadata | None,
        options: AiWatchCommandOptions,
        interest_profile: str,
    ) -> tuple[CuratorDecision, Reply]: ...


class ScoutPrompt(StructuredPrompt[ScoutOutput]):
    response_model: ClassVar[type[BaseModel]] = ScoutOutput

    template = """You are the cheap scout lane for AI Watch.

Score this article against Michael's interest profile. Be inclusive: your job is recall, not final judgement.
Treat old evergreen agent advice and Xcode-only developer tooling as low priority unless the article contains a genuinely new, transferable technique.

Interest profile:
{{ interest_profile }}

Article metadata:
{{ metadata | json }}

Article text (may be truncated):
{{ article_text }}

Return structured JSON only.

evidence_quotes MUST be copied verbatim from the Article text above — contiguous substrings,
not paraphrases or summaries. Each quote must appear exactly in the article (whitespace and
curly-quote variants are tolerated). Paraphrased quotes fail validation. Include 2-5 short
quotes that support your scores.

Recommend promote_to_curator when fit_score >= 3 and noise_risk <= 3.
"""

    interest_profile: str
    metadata: ArticleMetadata | None
    article_text: str


class CuratorPrompt(StructuredPrompt[CuratorDecision]):
    response_model: ClassVar[type[BaseModel]] = CuratorDecision

    template = """You are the smart curator lane for AI Watch.

Apply Michael's interest profile with high precision. Most items should be reject.
Prefer recent AI techniques, tools, releases, and workflow changes. Reject old evergreen agent advice unless it adds genuinely new capability or unusually concrete evidence. Reject Xcode-specific tooling unless the non-Xcode transfer value is explicit.

Interest profile:
{{ interest_profile }}

Scout evaluation:
{{ scout_record | json }}

Article metadata:
{{ metadata | json }}

Article text (may be truncated):
{{ article_text }}

Decide surface only for concrete, transferable patterns relevant to agents, harnesses, MCP, generative UI, or workflow automation.
For surface decisions, digest_summary must be at most 500 characters — a concise digest blurb for the newsletter.
For surface decisions, include obsidian_recommendation with a vault-relative path under 3 Resources/.

{% if calibration_replay -%}
Calibration replay mode: evaluate decision quality as if this article were new. Do NOT return duplicate because a vault note exists. Return the decision you would make for a first-time discovery.
{% endif -%}
"""

    interest_profile: str
    scout_record: ScoutEvaluationRecord
    metadata: ArticleMetadata | None
    article_text: str
    calibration_replay: bool = False


class ClaudeAiWatchStages:
    def __init__(self, agent: ClaudeAgent) -> None:
        self.agent = agent

    async def run_scout(
        self,
        *,
        article_text: str,
        metadata: ArticleMetadata | None,
        options: AiWatchCommandOptions,
        interest_profile: str,
    ) -> tuple[ScoutOutput, Reply]:
        prompt = ScoutPrompt(
            interest_profile=interest_profile,
            metadata=metadata,
            article_text=article_text[:12000],
        )
        return await self.agent.run_structured(
            prompt, AgentSpec(model=options.scout_model)
        )

    async def run_curate(
        self,
        *,
        scout_record: ScoutEvaluationRecord,
        article_text: str,
        metadata: ArticleMetadata | None,
        options: AiWatchCommandOptions,
        interest_profile: str,
    ) -> tuple[CuratorDecision, Reply]:
        prompt = CuratorPrompt(
            interest_profile=interest_profile,
            scout_record=scout_record,
            metadata=metadata,
            article_text=article_text[:16000],
            calibration_replay=options.calibration_only,
        )
        return await self.agent.run_structured(
            prompt, AgentSpec(model=options.curator_model)
        )


def load_interest_profile() -> str:
    path = Path(__file__).resolve().parent / "assets" / "interest-profile.md"
    return path.read_text(encoding="utf-8")
