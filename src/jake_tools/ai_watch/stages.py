from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Protocol

from pydantic import BaseModel

from ..hermes import AgentSpec, Reply
from ..prompting import StructuredPrompt
from .audit_models import ArticleMetadata, ScoutEvaluationRecord
from .models import AiWatchCommandOptions, CuratorDecision, ScoutOutput


class AiWatchStages(Protocol):
    def run_scout(
        self,
        *,
        article_text: str,
        metadata: ArticleMetadata | None,
        options: AiWatchCommandOptions,
        interest_profile: str,
    ) -> tuple[ScoutOutput, Reply]: ...

    def run_curate(
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


class HermesAiWatchStages:
    def __init__(self, hermes) -> None:
        self.hermes = hermes

    def run_scout(
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
        spec = AgentSpec(model=options.scout_model, provider=options.scout_provider)
        return self.hermes.run_structured(prompt, spec)

    def run_curate(
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
        spec = AgentSpec(model=options.curator_model, provider=options.curator_provider)
        return self.hermes.run_structured(prompt, spec)


def load_interest_profile() -> str:
    path = Path(__file__).resolve().parent / "assets" / "interest-profile.md"
    return path.read_text(encoding="utf-8")
