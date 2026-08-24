"""Exact, human-recorded acceptance for a generated transcript note.

The review pack is deliberately boring: it binds every input that can change a
candidate to the exact bytes the operator inspected.  Export plans a note but
never mutates the canonical note or integration baselines.  Apply re-plans,
compares the complete candidate, and only then performs the existing atomic
integration write.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from ..cache_models import CacheEnvelope
from .cache import RunCache, atomic_write_text, sha256_of, stable_hash
from .chapters import ChapterList
from .integrate import IntegrationPlan, IntegrationReport, load_products, plan_integrate
from .minutes import MINUTES_CACHE_NAME, MinutesResult, MinutesReviewResult
from .models import RawTranscript, TranscriptProducts
from .note import MEETING_PREP_HEADING, parse_note, split_raw_frontmatter
from .polish import POLISHED_CACHE_NAME, PolishedChapterList, validate_polished_chapter

SCHEMA_VERSION = 1
REVIEW_POLICY_VERSION = "product-review-v1"
RENDERER_VERSION = "integrate-renderer-v1"
PIPELINE_REVISION = "transcription-product-acceptance-v1"
PACK_NAME = "product_review"
CANDIDATE_NAME = "product_review_candidate"
DECISION_NAME = "product_review_decision"


class ProductReviewError(RuntimeError):
    """Base class for review export, decision, and apply refusal errors."""


class ProductReviewRefusalError(ProductReviewError):
    """Raised when a candidate or accepted decision cannot be trusted."""


class ProductReviewExportError(ProductReviewError):
    """Raised when a review pack cannot be built without mutating the note."""


class ProductReviewPostWriteError(ProductReviewError):
    """Raised when an accepted candidate fails exact post-write verification."""


class SourceClipIdentity(CacheEnvelope):
    path: str
    offset_seconds: float
    duration_seconds: float
    sha256: str | None = None


class ReviewBinding(CacheEnvelope):
    """All exact identities that make a product candidate reviewable."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    note_path: str
    source_sha256: str
    revision_sha256: str
    review_policy_version: str
    policy_sha256: str
    transcript_sha256: str
    minutes_sha256: str
    note_base_sha256: str
    render_sha256: str
    pipeline_revision: str = PIPELINE_REVISION
    source_kind: Literal["audio", "text", "unknown"] = "unknown"
    renderer_version: str = RENDERER_VERSION
    source_clips: list[SourceClipIdentity] = Field(default_factory=list)
    resolved_transcript_sha256: str | None = None
    chapters_sha256: str | None = None
    polished_sha256: str | None = None
    minutes_review_sha256: str | None = None
    minutes_draft_sha256: str | None = None
    manifest_sha256s: dict[str, str] = Field(default_factory=dict)


class ProductReviewChecks(CacheEnvelope):
    """Mechanical safety checks; semantic warnings never masquerade as passes."""

    model_config = ConfigDict(extra="forbid")

    source_provenance: bool = False
    polished_provenance: bool = False
    no_cross_speaker_provenance: bool = False
    unknown_counts_carried: bool = False
    chapters_partition: bool = False
    temporal_geometry: bool = False
    protected_content: bool = False
    candidate_shape: bool = False
    stale_state: bool = False
    manifests_valid: bool = False
    safety_failures: list[str] = Field(default_factory=list)
    semantic_warnings: list[str] = Field(default_factory=list)
    passed: bool = False

    @model_validator(mode="after")
    def _derive_passed(self) -> ProductReviewChecks:
        self.passed = not self.safety_failures and all(
            getattr(self, field)
            for field in (
                "source_provenance",
                "polished_provenance",
                "no_cross_speaker_provenance",
                "unknown_counts_carried",
                "chapters_partition",
                "temporal_geometry",
                "protected_content",
                "candidate_shape",
                "stale_state",
                "manifests_valid",
            )
        )
        return self


class ProductReviewPack(CacheEnvelope):
    schema_version: Literal[1] = SCHEMA_VERSION
    review_id: str
    generated_at: datetime
    binding: ReviewBinding
    candidate_path: str
    pack_path: str | None = None
    checks: ProductReviewChecks


class ProductReviewDecision(CacheEnvelope):
    schema_version: Literal[1] = SCHEMA_VERSION
    review_id: str
    binding: ReviewBinding
    decision: Literal["accept", "reject"]
    reviewer: str
    reason: str | None = None
    decided_at: datetime

    @model_validator(mode="after")
    def _validate_rejection_reason(self) -> ProductReviewDecision:
        if not self.reviewer.strip():
            raise ValueError("reviewer must not be empty")
        if self.decision == "reject" and not (self.reason or "").strip():
            raise ValueError("a rejected review requires a non-empty reason")
        return self


def review_id_for(binding: ReviewBinding) -> str:
    """Derive a stable full- binding identity, never from a truncated run id."""
    return hashlib.sha256(
        binding.model_dump_json(exclude={"schema_version"}, by_alias=False).encode()
    ).hexdigest()


def _file_hash(path: Path) -> str:
    return sha256_of(path)


def _cache_file_hash(cache: RunCache, run_id: str, name: str) -> str | None:
    path = cache.run_dir(run_id) / f"{name}.json"
    return _file_hash(path) if path.exists() else None


def _manifest_hashes(cache: RunCache, run_id: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for stage in (
        "asr",
        "adapt",
        "chapterise",
        "polish",
        "minutes",
        "minutes-review",
    ):
        path = cache.run_dir(run_id) / f"{stage}.manifest.json"
        if path.exists():
            result[stage] = _file_hash(path)
    return result


def _source_identity(
    raw: RawTranscript, cache: RunCache, run_id: str
) -> tuple[str, str, list[SourceClipIdentity]]:
    clips: list[SourceClipIdentity] = []
    for clip in raw.clips:
        path = Path(clip.path)
        clips.append(
            SourceClipIdentity(
                path=clip.path,
                offset_seconds=clip.offset_seconds,
                duration_seconds=clip.duration_seconds,
                sha256=_file_hash(path) if path.exists() and path.is_file() else None,
            )
        )
    adapt_manifest = cache.load_manifest(run_id, "adapt")
    source_kind: str = (
        "audio"
        if raw.audio_sha256 or clips
        else "text"
        if adapt_manifest
        else "unknown"
    )
    identity = {
        "source_kind": source_kind,
        "clips": [item.model_dump(mode="json") for item in clips],
        "merged_sha256": raw.audio_sha256,
        "source_manifest_input_hash": (
            adapt_manifest.input_hashes.get("source_text") if adapt_manifest else None
        ),
    }
    return source_kind, stable_hash(identity), clips


def _stage_output_hash(cache: RunCache, run_id: str, name: str) -> str | None:
    path = cache.run_dir(run_id) / f"{name}.json"
    return _file_hash(path) if path.exists() else None


def _build_binding(
    note_path: Path,
    run_id: str,
    cache: RunCache,
    plan: IntegrationPlan,
) -> ReviewBinding:
    raw = cache.load(run_id, "resolved_transcript", RawTranscript)
    polished = cache.load(run_id, POLISHED_CACHE_NAME, PolishedChapterList)
    minutes = cache.load(run_id, MINUTES_CACHE_NAME, MinutesResult)
    if raw is None or polished is None or minutes is None:
        raise ProductReviewExportError(f"run {run_id!r} is missing product artefacts")
    source_kind, source_sha, clips = _source_identity(raw, cache, run_id)
    manifests = _manifest_hashes(cache, run_id)
    policy_sha = stable_hash(
        {
            "version": REVIEW_POLICY_VERSION,
            "renderer": RENDERER_VERSION,
            "pipeline": PIPELINE_REVISION,
        }
    )
    revision_sha = stable_hash(
        {
            "pipeline_revision": PIPELINE_REVISION,
            "renderer_version": RENDERER_VERSION,
            "review_policy_version": REVIEW_POLICY_VERSION,
            "manifests": manifests,
        }
    )
    return ReviewBinding(
        run_id=run_id,
        note_path=str(note_path),
        source_sha256=source_sha,
        revision_sha256=revision_sha,
        review_policy_version=REVIEW_POLICY_VERSION,
        policy_sha256=policy_sha,
        transcript_sha256=_stage_output_hash(cache, run_id, POLISHED_CACHE_NAME) or "",
        minutes_sha256=_stage_output_hash(cache, run_id, MINUTES_CACHE_NAME) or "",
        note_base_sha256=hashlib.sha256(note_path.read_bytes()).hexdigest(),
        render_sha256=hashlib.sha256(plan.candidate_text.encode("utf-8")).hexdigest(),
        pipeline_revision=PIPELINE_REVISION,
        source_kind=source_kind,  # type: ignore[arg-type]
        renderer_version=RENDERER_VERSION,
        source_clips=clips,
        resolved_transcript_sha256=_stage_output_hash(
            cache, run_id, "resolved_transcript"
        ),
        chapters_sha256=_stage_output_hash(cache, run_id, "chapters"),
        polished_sha256=_stage_output_hash(cache, run_id, POLISHED_CACHE_NAME),
        minutes_review_sha256=_stage_output_hash(cache, run_id, "minutes_review"),
        minutes_draft_sha256=_stage_output_hash(cache, run_id, "minutes_draft"),
        manifest_sha256s=manifests,
    )


def _validate_manifests(cache: RunCache, run_id: str) -> bool:
    for stage, name in (
        ("chapterise", "chapters"),
        ("polish", POLISHED_CACHE_NAME),
        ("minutes", MINUTES_CACHE_NAME),
        ("minutes-review", "minutes_review"),
    ):
        manifest = cache.load_manifest(run_id, stage)
        if manifest is None or manifest.output_hash is None:
            return False
        path = cache.run_dir(run_id) / f"{name}.json"
        if not path.exists():
            return False
        try:
            model_type = {
                "chapterise": ChapterList,
                "polish": PolishedChapterList,
                "minutes": MinutesResult,
                "minutes-review": MinutesReviewResult,
            }.get(stage)
            if model_type is not None:
                value = cache.load(run_id, name, model_type)
                if value is None or manifest.output_hash != stable_hash(
                    value.model_dump(mode="json")
                ):
                    return False
        except ValueError:
            return False
    return True


def _check_products(
    note_path: Path,
    run_id: str,
    cache: RunCache,
    plan: IntegrationPlan,
) -> ProductReviewChecks:
    failures: list[str] = []
    warnings: list[str] = []
    note = parse_note(note_path)
    raw = cache.load(run_id, "resolved_transcript", RawTranscript)
    chapter_list = cache.load(run_id, "chapters", ChapterList)
    polished = cache.load(run_id, POLISHED_CACHE_NAME, PolishedChapterList)
    minutes_review = cache.load(run_id, "minutes_review", MinutesReviewResult)
    source_ok = raw is not None and (
        bool(raw.clips)
        or bool(raw.audio_sha256)
        or cache.load_manifest(run_id, "adapt") is not None
    )
    if not source_ok:
        failures.append("source/provenance is missing")

    polished_ok = True
    cross_speaker_ok = True
    if raw is None or chapter_list is None or polished is None:
        polished_ok = False
        cross_speaker_ok = False
    else:
        if len(chapter_list.chapters) != len(polished.chapters):
            polished_ok = False
        for span, chapter in zip(
            chapter_list.chapters, polished.chapters, strict=False
        ):
            indices = range(span.start_utterance, span.end_utterance + 1)
            try:
                validate_polished_chapter(
                    chapter,
                    [raw.utterances[index] for index in indices],
                    source_indices=indices,
                )
            except ValueError, IndexError:
                polished_ok = False
                cross_speaker_ok = False
    if not polished_ok:
        failures.append("polished chapters lack exact source provenance")
    if not cross_speaker_ok:
        failures.append("cross-speaker or invalid polished provenance")

    unknown_ok = (
        raw is not None
        and polished is not None
        and all(
            turn.speaker == "Unknown"
            for chapter in polished.chapters
            for turn in chapter.turns
            if any(
                raw.utterances[index].speaker == "Unknown"
                for index in turn.source_turn_indices
                if 0 <= index < len(raw.utterances)
            )
        )
    )
    if not unknown_ok:
        failures.append("resolved Unknown provenance/counts are not carried through")

    chapter_ok = False
    geometry_ok = False
    if raw is not None and chapter_list is not None:
        spans = chapter_list.chapters
        expected = list(range(len(raw.utterances)))
        actual: list[int] = []
        for span in spans:
            actual.extend(range(span.start_utterance, span.end_utterance + 1))
        chapter_ok = actual == expected and all(
            span.start_utterance <= span.end_utterance for span in spans
        )
        duration_ok = len(spans) < 2 or all(
            raw.utterances[span.end_utterance].end
            - raw.utterances[span.start_utterance].start
            >= 60.0
            for span in spans
            if 0 <= span.start_utterance <= span.end_utterance < len(raw.utterances)
        )
        geometry_ok = (
            chapter_ok
            and duration_ok
            and all(
                span.start_seconds >= 0
                and (index := span.start_utterance) < len(raw.utterances)
                and raw.utterances[index].start >= span.start_seconds
                for span in spans
            )
        )
        for previous, current in zip(raw.utterances, raw.utterances[1:], strict=False):
            if current.start < previous.end:
                warnings.append(
                    "Genuine overlapping transcript spans need audio review; "
                    "do not infer the overlap's words from web lookup."
                )
                break
        if len(spans) < 2 and raw.utterances:
            warnings.append("Confirm chapter boundaries against the audio.")
    if not chapter_ok:
        failures.append("chapter index is not an exact partition")
    if not geometry_ok:
        failures.append("chapter temporal geometry is invalid")

    current_text = note_path.read_text()
    candidate = plan.candidate_text
    current_prefix, current_body = split_raw_frontmatter(current_text)
    candidate_prefix, candidate_body = split_raw_frontmatter(candidate)
    current_prep = next(
        (
            section.body
            for section in parse_note(note_path).sections
            if section.heading
            and section.heading.strip().casefold() == MEETING_PREP_HEADING
        ),
        None,
    )
    candidate_note = None
    candidate_check_path = cache.run_dir(run_id) / ".candidate-check.md"
    try:
        candidate_check_path.write_text(candidate)
        candidate_note = parse_note(candidate_check_path)
        candidate_prep = next(
            (
                section.body
                for section in candidate_note.sections
                if section.heading
                and section.heading.strip().casefold() == MEETING_PREP_HEADING
            ),
            None,
        )
    except Exception:
        candidate_prep = None
    finally:
        candidate_check_path.unlink(missing_ok=True)
    protected_ok = current_prefix == candidate_prefix and current_prep == candidate_prep
    if not protected_ok:
        failures.append("frontmatter or Meeting Prep changed")

    candidate_embed_count = (
        len(candidate_note.embeds) if candidate_note is not None else 0
    )
    shape_ok = (
        candidate_body.count("## Discussion Notes") == 1
        and candidate_body.count("## Chapters") == 1
        and candidate_body.count("## Transcript") == 1
        and candidate_embed_count >= 1
        and candidate_embed_count == len(note.embeds)
        and "[!summary]" in candidate
    )
    if not shape_ok:
        failures.append("candidate note shape/source embed is invalid")

    stale_ok = not (cache.run_dir(run_id) / "stale.json").exists()
    if not stale_ok:
        failures.append("run contains stale state")
    manifests_ok = _validate_manifests(cache, run_id) and minutes_review is not None
    if not manifests_ok:
        failures.append(
            "current product manifests or minutes review provenance are missing"
        )

    if minutes_review is not None:
        warnings.extend(minutes_review.issues)
        warnings.extend(minutes_review.warnings)
    warnings.append(
        "Semantic warnings require human audio or term confirmation; web lookup does not prove audio."
    )
    return ProductReviewChecks(
        source_provenance=source_ok,
        polished_provenance=polished_ok,
        no_cross_speaker_provenance=cross_speaker_ok,
        unknown_counts_carried=unknown_ok,
        chapters_partition=chapter_ok,
        temporal_geometry=geometry_ok,
        protected_content=protected_ok,
        candidate_shape=shape_ok,
        stale_state=stale_ok,
        manifests_valid=manifests_ok,
        safety_failures=failures,
        semantic_warnings=list(dict.fromkeys(warnings)),
    )


def export_product_review(
    note_path: Path, run_id: str, *, cache: RunCache
) -> ProductReviewPack:
    """Render and persist a review pack without touching the canonical note."""
    try:
        products = load_products(note_path, run_id, cache)
        plan = plan_integrate(note_path, products, cache=cache, run_id=run_id)
        binding = _build_binding(note_path, run_id, cache, plan)
        checks = _check_products(note_path, run_id, cache, plan)
        if not checks.passed:
            raise ProductReviewExportError(
                "product review refused: " + "; ".join(checks.safety_failures)
            )
        review_id = review_id_for(binding)
        candidate_path = cache.run_dir(run_id) / f"{CANDIDATE_NAME}.md"
        atomic_write_text(candidate_path, plan.candidate_text)
        pack = ProductReviewPack(
            review_id=review_id,
            generated_at=datetime.now(UTC),
            binding=binding,
            candidate_path=str(candidate_path),
            pack_path=str(cache.run_dir(run_id) / f"{PACK_NAME}.json"),
            checks=checks,
        )
        cache.store(run_id, PACK_NAME, pack)
        (cache.run_dir(run_id) / f"{DECISION_NAME}.json").unlink(missing_ok=True)
        return pack
    except ProductReviewError:
        raise
    except Exception as exc:
        raise ProductReviewExportError(
            f"could not export product review for {run_id!r}: {exc}"
        ) from exc


def decide_product_review(
    note_path: Path,
    run_id: str,
    *,
    review_id: str,
    decision: Literal["accept", "reject"],
    reviewer: str,
    reason: str | None,
    cache: RunCache,
) -> ProductReviewDecision:
    try:
        pack = cache.load(run_id, PACK_NAME, ProductReviewPack)
    except ValueError as exc:
        raise ProductReviewRefusalError("product review pack is malformed") from exc
    if pack is None:
        raise ProductReviewRefusalError(
            f"no product review pack exists for run {run_id!r}"
        )
    if (
        pack.review_id != review_id
        or pack.binding.run_id != run_id
        or pack.binding.note_path != str(note_path)
    ):
        raise ProductReviewRefusalError(
            "review decision does not match the exact note, run, and review pack"
        )
    result = ProductReviewDecision(
        review_id=review_id,
        binding=pack.binding,
        decision=decision,
        reviewer=reviewer,
        reason=reason,
        decided_at=datetime.now(UTC),
    )
    cache.store(run_id, DECISION_NAME, result)
    return result


def _same_file_hash(
    cache: RunCache, run_id: str, name: str, expected: str | None
) -> bool:
    return expected is not None and _cache_file_hash(cache, run_id, name) == expected


def validate_product_review_for_apply(
    note_path: Path,
    run_id: str,
    products: TranscriptProducts,
    *,
    cache: RunCache,
) -> tuple[ProductReviewPack, IntegrationPlan]:
    """Validate every binding before any note/baseline/fingerprint mutation."""
    try:
        pack = cache.load(run_id, PACK_NAME, ProductReviewPack)
        decision = cache.load(run_id, DECISION_NAME, ProductReviewDecision)
    except ValueError as exc:
        raise ProductReviewRefusalError(
            "product review pack or decision is malformed"
        ) from exc
    if pack is None or decision is None:
        raise ProductReviewRefusalError(
            "accepted product review pack and decision are required before integrate"
        )
    if decision.decision != "accept":
        raise ProductReviewRefusalError("product review was rejected")
    if decision.review_id != pack.review_id or decision.binding != pack.binding:
        raise ProductReviewRefusalError(
            "product review decision binding does not match the pack"
        )

    if pack.binding.run_id != run_id or pack.binding.note_path != str(note_path):
        raise ProductReviewRefusalError("product review belongs to another note or run")
    if (
        hashlib.sha256(note_path.read_bytes()).hexdigest()
        != pack.binding.note_base_sha256
    ):
        raise ProductReviewRefusalError(
            "current note changed since product review export"
        )
    if not _same_file_hash(
        cache, run_id, POLISHED_CACHE_NAME, pack.binding.polished_sha256
    ):
        raise ProductReviewRefusalError(
            "polished artefact changed since product review export"
        )
    if not _same_file_hash(
        cache, run_id, MINUTES_CACHE_NAME, pack.binding.minutes_sha256
    ):
        raise ProductReviewRefusalError(
            "minutes artefact changed since product review export"
        )
    for name, expected in (
        ("resolved_transcript", pack.binding.resolved_transcript_sha256),
        ("chapters", pack.binding.chapters_sha256),
        ("minutes_review", pack.binding.minutes_review_sha256),
        ("minutes_draft", pack.binding.minutes_draft_sha256),
    ):
        if expected is not None and not _same_file_hash(cache, run_id, name, expected):
            raise ProductReviewRefusalError(
                f"{name} artefact changed since product review export"
            )
    if _manifest_hashes(cache, run_id) != pack.binding.manifest_sha256s:
        raise ProductReviewRefusalError(
            "stage manifest changed since product review export"
        )
    candidate_path = Path(pack.candidate_path)
    if (
        candidate_path != cache.run_dir(run_id) / f"{CANDIDATE_NAME}.md"
        or not candidate_path.exists()
    ):
        raise ProductReviewRefusalError(
            "candidate path is stale or outside the run cache"
        )
    candidate_bytes = candidate_path.read_bytes()
    if hashlib.sha256(candidate_bytes).hexdigest() != pack.binding.render_sha256:
        raise ProductReviewRefusalError("review candidate bytes changed")
    plan = plan_integrate(note_path, products, cache=cache, run_id=run_id)
    cached_products = load_products(note_path, run_id, cache)
    if stable_hash(products.model_dump(mode="json")) != stable_hash(
        cached_products.model_dump(mode="json")
    ):
        raise ProductReviewRefusalError(
            "supplied products differ from the reviewed cache artefacts"
        )
    current_binding = _build_binding(note_path, run_id, cache, plan)
    if current_binding != pack.binding:
        raise ProductReviewRefusalError(
            "source, revision, policy, artefact, manifest, or note binding changed since export"
        )
    if plan.candidate_text.encode("utf-8") != candidate_bytes:
        raise ProductReviewRefusalError(
            "current renderer no longer produces the accepted candidate"
        )
    checks = _check_products(note_path, run_id, cache, plan)
    if not checks.passed:
        raise ProductReviewRefusalError(
            "product checks no longer pass: " + "; ".join(checks.safety_failures)
        )
    return pack, plan


def apply_accepted_product(
    note_path: Path,
    products: TranscriptProducts,
    *,
    run_id: str,
    cache: RunCache,
) -> IntegrationReport:
    """Apply only the exact accepted plan; the integration module owns writes."""
    from .integrate import apply_integration_plan

    _pack, plan = validate_product_review_for_apply(
        note_path, run_id, products, cache=cache
    )
    return apply_integration_plan(note_path, plan, cache=cache, run_id=run_id)
