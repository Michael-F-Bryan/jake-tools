"""Editorial review for transcript renders.

Capability validation proves structural integrity. Product review answers the separate
question an operator cares about: is this exact transcript readable, and do these
exact minutes preserve what was proposed, agreed, decided, and left open?
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..errors import TranscriptError
from .assignment import canonical_turn_set
from .components import (
    ChapterSetComponent,
    FindingKind,
    MinutesComponent,
    TextEditLedgerComponent,
    TextEditOperation,
)
from .document import NoDocumentYet, TranscriptDocumentV1, project_revision
from .ids import RenderId, mint_id
from .records import (
    ProductDisposition,
    ProductReviewRecord,
    RenderRecord,
)
from .store import BundleStore

PRODUCT_REVIEW_SCHEMA_VERSION = "v1"
_REVIEW_POLICY_VERSION = "private-meeting-v1"
_NUMBER_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?%?|zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|hundred|thousand)\b",
    re.IGNORECASE,
)


class ProductReviewError(TranscriptError):
    """Base class for product-review failures."""


class ProductReviewRequiredError(ProductReviewError):
    """The exact render has no accepted editorial product review."""


class InvalidProductReviewPackError(ProductReviewError):
    """A product-review pack is incomplete, stale, or bound to other bytes."""


class ProductReviewAlreadyDecidedError(ProductReviewError):
    """A render already has an immutable product-review decision."""


class ProductReviewSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str = Field(min_length=1)
    turn_ids: tuple[str, ...] = Field(min_length=1)
    text: str = Field(min_length=1)


class ProductReviewPack(BaseModel):
    """Editable review pack; the operator fills the final four fields."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = PRODUCT_REVIEW_SCHEMA_VERSION
    policy_version: str = _REVIEW_POLICY_VERSION
    bundle_id: str
    revision_id: str
    render_id: str
    render_output_sha256: str
    transcript_component_id: str
    transcript_content_hash: str
    minutes_component_id: str
    minutes_content_hash: str
    samples: tuple[ProductReviewSample, ...]
    rendered_body: str
    reviewer: str = ""
    transcript_disposition: ProductDisposition = ProductDisposition.PENDING
    minutes_disposition: ProductDisposition = ProductDisposition.PENDING
    blocking_findings: tuple[str, ...] = ()


def export_product_review_pack(
    store: BundleStore, *, render_id: RenderId, destination: Path
) -> Path:
    """Write a deterministic review pack for the exact bytes ``render_id`` names."""
    render = store.load_render(render_id)
    document, transcript, minutes = _bound_products(store, render)
    samples = _review_samples(document, transcript.turns, minutes)
    required = {"early", "middle", "late"}
    present = {sample.kind for sample in samples}
    missing = sorted(required - present)
    if missing:
        raise InvalidProductReviewPackError(
            f"cannot export product review: missing representative samples {missing}."
        )
    pack = ProductReviewPack(
        bundle_id=render.bundle_id,
        revision_id=render.revision_id,
        render_id=render.render_id,
        render_output_sha256=render.output_sha256,
        transcript_component_id=transcript.component_id,
        transcript_content_hash=transcript.content_hash,
        minutes_component_id=minutes.component_id,
        minutes_content_hash=minutes.content_hash,
        samples=samples,
        rendered_body=store.load_render_output(render_id).decode("utf-8"),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(pack.model_dump_json(indent=2), encoding="utf-8")
    return destination


def record_product_review(
    store: BundleStore, *, pack_path: Path
) -> ProductReviewRecord:
    """Validate and append one immutable decision for an exported review pack."""
    raw = pack_path.read_bytes()
    try:
        pack = ProductReviewPack.model_validate_json(raw)
    except ValueError as exc:
        raise InvalidProductReviewPackError(
            f"product review pack {pack_path} is invalid: {exc}"
        ) from exc
    if pack.schema_version != PRODUCT_REVIEW_SCHEMA_VERSION:
        raise InvalidProductReviewPackError(
            f"product review pack schema {pack.schema_version!r} is unsupported; "
            f"expected {PRODUCT_REVIEW_SCHEMA_VERSION!r}."
        )
    if pack.policy_version != _REVIEW_POLICY_VERSION:
        raise InvalidProductReviewPackError(
            f"product review policy {pack.policy_version!r} is unsupported; "
            f"expected {_REVIEW_POLICY_VERSION!r}."
        )
    if not pack.reviewer.strip():
        raise InvalidProductReviewPackError("product review requires a reviewer.")
    if ProductDisposition.PENDING in (
        pack.transcript_disposition,
        pack.minutes_disposition,
    ):
        raise InvalidProductReviewPackError(
            "product review must decide both transcript and minutes."
        )

    render = store.load_render(pack.render_id)
    _validate_pack_binding(store, render, pack)
    pack_sha256 = hashlib.sha256(raw).hexdigest()
    existing = [
        review
        for review in store.iter_product_reviews()
        if review.render_id == render.render_id
    ]
    if existing:
        identical = next(
            (review for review in existing if review.pack_sha256 == pack_sha256), None
        )
        if identical is not None:
            return identical
        raise ProductReviewAlreadyDecidedError(
            f"render {render.render_id} already has product review "
            f"{existing[-1].product_review_id}; render a new candidate after repairs."
        )

    record = ProductReviewRecord(
        product_review_id=mint_id("product_review"),
        bundle_id=render.bundle_id,
        revision_id=render.revision_id,
        render_id=render.render_id,
        render_output_sha256=render.output_sha256,
        transcript_component_id=pack.transcript_component_id,
        transcript_content_hash=pack.transcript_content_hash,
        minutes_component_id=pack.minutes_component_id,
        minutes_content_hash=pack.minutes_content_hash,
        transcript_disposition=pack.transcript_disposition,
        minutes_disposition=pack.minutes_disposition,
        pack_schema_version=pack.schema_version,
        review_policy_version=pack.policy_version,
        pack_sha256=pack_sha256,
        reviewer=pack.reviewer.strip(),
        blocking_findings=pack.blocking_findings,
        created_at=datetime.now(UTC),
    )
    return store.add_product_review(record)


def require_accepted_product_review(
    store: BundleStore, render: RenderRecord
) -> ProductReviewRecord:
    """Return the accepted review bound to ``render``, or refuse publication."""
    _document, transcript, minutes = _bound_products(store, render)
    accepted = [
        review
        for review in store.iter_product_reviews()
        if review.render_id == render.render_id
        and review.revision_id == render.revision_id
        and review.render_output_sha256 == render.output_sha256
        and review.transcript_component_id == transcript.component_id
        and review.transcript_content_hash == transcript.content_hash
        and review.minutes_component_id == minutes.component_id
        and review.minutes_content_hash == minutes.content_hash
        and review.review_policy_version == _REVIEW_POLICY_VERSION
        and review.transcript_disposition == ProductDisposition.ACCEPTED
        and review.minutes_disposition == ProductDisposition.ACCEPTED
        and not review.blocking_findings
    ]
    if accepted:
        return accepted[-1]
    raise ProductReviewRequiredError(
        f"render {render.render_id} has no accepted product review; export and decide "
        "the review before applying it."
    )


def _validate_pack_binding(
    store: BundleStore, render: RenderRecord, pack: ProductReviewPack
) -> None:
    if pack.bundle_id != render.bundle_id or pack.revision_id != render.revision_id:
        raise InvalidProductReviewPackError(
            "product review pack is bound to a different bundle or revision."
        )
    if pack.render_output_sha256 != render.output_sha256:
        raise InvalidProductReviewPackError(
            "product review pack render hash is stale or does not match the render."
        )
    if (
        hashlib.sha256(pack.rendered_body.encode("utf-8")).hexdigest()
        != render.output_sha256
    ):
        raise InvalidProductReviewPackError(
            "product review pack rendered body does not match the bound render bytes."
        )
    document, transcript, minutes = _bound_products(store, render)
    expected = (
        transcript.component_id,
        transcript.content_hash,
        minutes.component_id,
        minutes.content_hash,
    )
    actual = (
        pack.transcript_component_id,
        pack.transcript_content_hash,
        pack.minutes_component_id,
        pack.minutes_content_hash,
    )
    if actual != expected:
        raise InvalidProductReviewPackError(
            "product review pack is stale: its transcript or minutes binding changed."
        )
    expected_samples = _review_samples(document, transcript.turns, minutes)
    if pack.samples != expected_samples:
        raise InvalidProductReviewPackError(
            "product review samples were altered or no longer match the bound evidence."
        )
    required = {"early", "middle", "late"}
    if not required.issubset({sample.kind for sample in pack.samples}):
        raise InvalidProductReviewPackError(
            "product review pack lacks early, middle, or late transcript coverage."
        )


def _bound_products(store: BundleStore, render: RenderRecord):
    document = project_revision(store, render.revision_id)
    if isinstance(document, NoDocumentYet):
        raise InvalidProductReviewPackError(
            f"render {render.render_id} is not bound to an assembled document."
        )
    transcript = canonical_turn_set(document.components)
    chapters = document.components_of(ChapterSetComponent)
    minutes = document.components_of(MinutesComponent)
    if transcript is None or len(chapters) != 1 or len(minutes) != 1:
        raise InvalidProductReviewPackError(
            "product review requires one canonical transcript, chapter set, and minutes "
            "component."
        )
    for index in range(1, len(chapters[0].chapters)):
        previous = chapters[0].chapters[index - 1]
        current = chapters[0].chapters[index]
        if previous.end_ms > current.start_ms:
            raise InvalidProductReviewPackError(
                f"chapter {previous.title!r} overlaps chapter {current.title!r}; "
                "regenerate chapters before product review."
            )
    by_turn_id = {turn.turn_id: turn for turn in transcript.turns}
    for chapter in chapters[0].chapters:
        try:
            first_turn = by_turn_id[chapter.turn_ids[0]]
            final_turn = by_turn_id[chapter.turn_ids[-1]]
        except KeyError as exc:
            raise InvalidProductReviewPackError(
                f"chapter {chapter.title!r} references an unknown canonical turn."
            ) from exc
        if (
            chapter.start_ms != first_turn.start_ms
            or chapter.end_ms != final_turn.end_ms
        ):
            raise InvalidProductReviewPackError(
                f"chapter {chapter.title!r} is not snapped to its reconciled canonical "
                "turn edges; regenerate chapters before product review."
            )
    return document, transcript, minutes[0]


def _review_samples(
    document: TranscriptDocumentV1, turns, minutes: MinutesComponent
) -> tuple[ProductReviewSample, ...]:
    by_id = {turn.turn_id: turn for turn in turns}
    samples: list[ProductReviewSample] = []

    def add(kind: str, selected) -> None:
        selected = tuple(selected)
        if not selected:
            return
        samples.append(
            ProductReviewSample(
                kind=kind,
                turn_ids=tuple(turn.turn_id for turn in selected),
                text="\n".join(
                    f"{turn.speaker_label}: {turn.text}" for turn in selected
                ),
            )
        )

    width = min(5, len(turns))
    add("early", turns[:width])
    middle = max(0, (len(turns) - width) // 2)
    add("middle", turns[middle : middle + width])
    add("late", turns[-width:])

    for turn in turns:
        if _NUMBER_RE.search(turn.text):
            add("numeric", (turn,))

    for finding in minutes.findings:
        if finding.kind not in (FindingKind.DECISION, FindingKind.ACTION):
            continue
        evidence = tuple(
            by_id[turn_id] for turn_id in finding.evidence_turn_ids if turn_id in by_id
        )
        if evidence:
            sample = ProductReviewSample(
                kind=f"candidate-{finding.kind.value}",
                turn_ids=tuple(turn.turn_id for turn in evidence),
                text=(
                    f"Finding ({finding.commitment_status.value}): {finding.text}\n"
                    "Evidence:\n"
                    + "\n".join(
                        f"{turn.speaker_label}: {turn.text}" for turn in evidence
                    )
                ),
            )
            samples.append(sample)

    changed_ids = {
        turn_id
        for ledger in document.components_of(TextEditLedgerComponent)
        for entry in ledger.entries
        if entry.operation != TextEditOperation.IDENTITY
        for turn_id in entry.output_turn_ids
    }
    for turn_id in changed_ids:
        turn = by_id.get(turn_id)
        if turn is not None:
            add("changed-turn", (turn,))

    seen: set[tuple[str, tuple[str, ...]]] = set()
    unique: list[ProductReviewSample] = []
    for sample in samples:
        key = (sample.kind, sample.turn_ids)
        if key in seen:
            continue
        seen.add(key)
        unique.append(sample)
    return tuple(unique)


__all__ = [
    "InvalidProductReviewPackError",
    "ProductDisposition",
    "ProductReviewPack",
    "ProductReviewRequiredError",
    "export_product_review_pack",
    "record_product_review",
    "require_accepted_product_review",
]
