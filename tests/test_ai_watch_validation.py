from __future__ import annotations

from datetime import datetime

from jake_tools.ai_watch.audit_models import ScoutEvaluationRecord
from jake_tools.ai_watch.models import AuditStage, ScoutRecommendation
from jake_tools.ai_watch.validation import (
    normalize_quote_text,
    quote_found_in_article,
    resolve_evidence_quote,
    sanitize_evidence_quotes,
    validate_scouted,
    validate_stage,
)


def test_validate_stage_rejects_empty_records() -> None:
    errors = validate_stage(AuditStage.DISCOVERED, [])
    assert errors == ["discovered: no records"]


def test_quote_found_exact_match() -> None:
    article = "Harness design with evaluator loops for long-running agents."
    assert quote_found_in_article(quote="evaluator loops", article=article)


def test_quote_found_rejects_paraphrase() -> None:
    article = "Harness design with evaluator loops for long-running agents."
    assert not quote_found_in_article(
        quote="The article explains evaluator loop patterns",
        article=article,
    )


def test_quote_found_normalizes_whitespace_and_smart_quotes() -> None:
    article = "Instead, I've started preferring HTML as an output format."
    quote = "Instead, I\u2019ve started preferring HTML as an output format."
    assert quote_found_in_article(quote=quote, article=article)


def test_quote_found_normalizes_non_breaking_space() -> None:
    article = "Use HTML\u00a0to understand code that the agent has written."
    quote = "Use HTML to understand code that the agent has written."
    assert quote_found_in_article(quote=quote, article=article)


def test_normalize_quote_text_collapses_whitespace() -> None:
    assert normalize_quote_text("hello   world\n\ttoday") == "hello world today"


def test_validate_scouted_passes_with_normalized_quotes() -> None:
    article = (
        "Use HTML\u00a0to understand code that the agent has written, to review code."
    )
    record = ScoutEvaluationRecord(
        run_id="2026-07-02",
        candidate_id="sha256:test",
        timestamp=datetime.fromisoformat("2026-07-02T00:00:00+00:00"),
        model="test",
        fit_score=5,
        novelty_score=4,
        practicality_score=4,
        noise_risk=1,
        evidence_quotes=[
            "Use HTML to understand code that the agent has written, to review code."
        ],
        recommendation=ScoutRecommendation.PROMOTE_TO_CURATOR,
        reason="Concrete HTML workflow pattern.",
        summary="HTML for agent outputs.",
    )
    assert validate_scouted([record], article_texts={"sha256:test": article}) == []


def test_validate_scouted_rejects_paraphrased_quote() -> None:
    article = "Use HTML to understand code that the agent has written."
    record = ScoutEvaluationRecord(
        run_id="2026-07-02",
        candidate_id="sha256:test",
        timestamp=datetime.fromisoformat("2026-07-02T00:00:00+00:00"),
        model="test",
        fit_score=5,
        novelty_score=4,
        practicality_score=4,
        noise_risk=1,
        evidence_quotes=["HTML helps you understand agent-written code."],
        recommendation=ScoutRecommendation.PROMOTE_TO_CURATOR,
        reason="Concrete HTML workflow pattern.",
        summary="HTML for agent outputs.",
    )
    errors = validate_scouted([record], article_texts={"sha256:test": article})
    assert errors == ["scout evidence quote not in article for sha256:test"]


def test_resolve_evidence_quote_maps_near_miss_to_article_line() -> None:
    article = (
        "Over the past several months I've been working on two interconnected problems: "
        "getting Claude to produce high-quality frontend designs, and getting it to build "
        "complete applications without human intervention."
    )
    quote = "getting Claude to build complete applications without human intervention"
    resolved = resolve_evidence_quote(quote=quote, article=article)
    assert resolved == article


def test_sanitize_evidence_quotes_drops_hallucinated_quotes() -> None:
    article = "Harness design with evaluator loops for long-running agents."
    quotes = [
        "evaluator loops",
        "The article explains evaluator loop patterns",
    ]
    assert sanitize_evidence_quotes(quotes=quotes, article=article) == [
        "evaluator loops"
    ]


def test_sanitize_evidence_quotes_drops_reworded_prompt_quotes() -> None:
    article = 'You can simply prompt it to " _make an HTML file_" or " _make an HTML artifact_."'
    quote = 'Ask Claude to "make an HTML file" or "make an HTML artifact".'
    assert sanitize_evidence_quotes(quotes=[quote], article=article) == []
