from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from .audit_models import (
    CuratorDecisionRecord,
    CuratorDecisionType,
    FetchRecord,
    FetchStatus,
    ScoutEvaluationRecord,
)
from .models import AuditStage, VaultPathEscapeError, resolve_vault_path

_QUOTE_CHAR_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u2032": "'",
    "\u2033": '"',
    "\u00a0": " ",
    "\u2014": "-",
    "\u2013": "-",
}

_RESOLVE_MIN_WORDS = 3
_RESOLVE_MIN_COVERAGE = 0.75
_DIGEST_SUMMARY_MAX_CHARS = 500


def _strip_markdown_inline(text: str) -> str:
    return re.sub(r"[_*`]+", "", text)


def normalize_quote_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", _strip_markdown_inline(text))
    for source, target in _QUOTE_CHAR_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    return re.sub(r"\s+", " ", normalized).strip()


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", normalize_quote_text(text).lower())


def _ordered_word_coverage(quote_words: list[str], line_words: list[str]) -> float:
    if not quote_words:
        return 0.0
    quote_index = 0
    matches = 0
    for word in line_words:
        if quote_index < len(quote_words) and word == quote_words[quote_index]:
            matches += 1
            quote_index += 1
    return matches / len(quote_words)


def resolve_evidence_quote(*, quote: str, article: str) -> str | None:
    if quote_found_in_article(quote=quote, article=article):
        return quote

    quote_words = _word_tokens(quote)
    if len(quote_words) < _RESOLVE_MIN_WORDS:
        return None

    matches: list[tuple[float, str]] = []
    for line in article.splitlines():
        stripped = line.strip()
        if len(stripped) < 20:
            continue
        coverage = _ordered_word_coverage(quote_words, _word_tokens(stripped))
        if coverage >= _RESOLVE_MIN_COVERAGE:
            matches.append((coverage, stripped))

    if not matches:
        return None

    matches.sort(key=lambda item: (-item[0], len(item[1])))
    top_coverage = matches[0][0]
    top_matches = [line for coverage, line in matches if coverage == top_coverage]
    if len(top_matches) != 1:
        return None

    resolved = top_matches[0]
    if resolved in article:
        return resolved
    for line in article.splitlines():
        if line.strip() == resolved:
            return resolved
    return None


def sanitize_digest_summary(
    summary: str, *, max_chars: int = _DIGEST_SUMMARY_MAX_CHARS
) -> str:
    cleaned = summary.strip()
    if len(cleaned) <= max_chars:
        return cleaned

    ellipsis = "..."
    cutoff = max_chars - len(ellipsis)
    truncated = cleaned[:cutoff].rstrip()
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space].rstrip()
    return f"{truncated}{ellipsis}"


def sanitize_evidence_quotes(*, quotes: list[str], article: str) -> list[str]:
    grounded: list[str] = []
    for quote in quotes:
        if not quote:
            continue
        if quote_found_in_article(quote=quote, article=article):
            grounded.append(quote)
            continue
        resolved = resolve_evidence_quote(quote=quote, article=article)
        if resolved and quote_found_in_article(quote=resolved, article=article):
            grounded.append(resolved)
    return grounded


def quote_found_in_article(*, quote: str, article: str) -> bool:
    if not quote:
        return True
    if not article:
        return False
    normalized_quote = normalize_quote_text(quote)
    normalized_article = normalize_quote_text(article)
    if normalized_quote in normalized_article:
        return True
    return quote in article


def validate_stage(stage: AuditStage, records: list[object]) -> list[str]:
    errors: list[str] = []
    if not records:
        errors.append(f"{stage.value}: no records")
    return errors


def validate_fetched(records: list[FetchRecord]) -> list[str]:
    errors: list[str] = []
    for record in records:
        if record.status != FetchStatus.OK:
            continue
        if not record.content_path:
            errors.append(f"fetched missing content_path for {record.candidate_id}")
        if not record.content_hash:
            errors.append(f"fetched missing content_hash for {record.candidate_id}")
    return errors


def validate_scouted(
    records: list[ScoutEvaluationRecord], *, article_texts: dict[str, str]
) -> list[str]:
    errors: list[str] = []
    for record in records:
        article = article_texts.get(record.candidate_id, "")
        for quote in record.evidence_quotes:
            if (
                quote
                and article
                and not quote_found_in_article(quote=quote, article=article)
            ):
                errors.append(
                    f"scout evidence quote not in article for {record.candidate_id}"
                )
        for field_name, value in (
            ("fit_score", record.fit_score),
            ("novelty_score", record.novelty_score),
            ("practicality_score", record.practicality_score),
            ("noise_risk", record.noise_risk),
        ):
            if not 1 <= value <= 5:
                errors.append(
                    f"scout {field_name} out of range for {record.candidate_id}"
                )
    return errors


def validate_curated(
    records: list[CuratorDecisionRecord], *, vault_root: Path
) -> list[str]:
    errors: list[str] = []
    generic_reasons = {"interesting article", "looks useful", "good read"}
    for record in records:
        reason = record.reason.strip().lower()
        if not reason:
            errors.append(f"curator missing reason for {record.candidate_id}")
        elif reason in generic_reasons:
            errors.append(f"curator generic reason for {record.candidate_id}")
        if record.decision == CuratorDecisionType.SURFACE:
            if not record.obsidian_recommendation.should_create_note:
                errors.append(
                    f"surface without obsidian note for {record.candidate_id}"
                )
            path = record.obsidian_recommendation.path
            if path:
                try:
                    resolve_vault_path(vault=vault_root, rel_path=path)
                except VaultPathEscapeError:
                    errors.append(
                        f"curator placement outside vault for {record.candidate_id}"
                    )
        if (
            record.decision == CuratorDecisionType.SURFACE
            and len(record.digest_summary) > 500
        ):
            errors.append(f"digest_summary too long for {record.candidate_id}")
    return errors
