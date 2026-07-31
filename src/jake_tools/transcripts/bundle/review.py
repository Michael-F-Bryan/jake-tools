"""M8: the durable human checkpoint -- exporting a review pack and
applying the reviewer's decisions.

The review pack is a plain JSON file the operator edits and hands back.
It is deliberately a *file*, not an interactive prompt: the checkpoint has
to survive the process exiting, a day passing, and a different machine
picking the work back up, which is the whole point of the overhaul.

Binding is exact (M8). A pack carries the revision it was produced
against plus canonical hashes of that revision's turn and cluster
inventories, and :func:`apply_review` refuses a pack whose bindings no
longer match the head. There is no auto-rebase: re-export against the
current head is a deliberate, visible act, whereas silently re-pointing a
reviewer's decisions at turns they never saw is exactly the failure this
refuses.

Three application rules, all mechanical:

- **Idempotent re-apply.** The same ``review_id`` applied twice returns
  the first application's revision (via the ``reviews/`` registry), never
  a second revision.
- **Stale input.** A pack bound to a revision that is no longer the head,
  or to inventories that have changed, is refused with an explicit error.
- **A second, differing review against the same input revision is
  rejected in v1** (D3 defers branching); the refusal names the review
  that already applied.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import TranscriptError
from .assignment import (
    UNCLEAR_SPEAKER_LABEL,
    SpeakerContext,
    canonical_turns,
    cluster_inventory_hash,
    effective_assignment,
    speaker_display_name,
    turn_inventory_hash,
)
from .components import (
    ClusterDecision,
    MachineAttributionSetComponent,
    ProviderLabelDecision,
    ReviewDecision,
    ReviewDecisionKind,
    ReviewScope,
    SourceRangeDecision,
    SpeakerHypothesisSetComponent,
    SpeakerReviewComponent,
    SpeakerReviewComponentBody,
    TurnDecision,
)
from .document import NoDocumentYet, TranscriptDocumentV1, project_head
from .ids import ReviewId, RevisionId, RunId, mint_id
from .records import OperationRef, ReviewRecord, RevisionRecord
from .store import BundleStore

#: Bumped when the pack's own shape changes incompatibly. Stored on the
#: applied review (M8) so an old pack cannot be applied against a newer
#: reader that would misread its decisions.
PACK_SCHEMA_VERSION = "1"

#: How many of a cluster's turns the pack quotes as listening material.
#: A reviewer needs enough to recognise a voice's role in the
#: conversation; the full transcript is in the pack's own ``turns`` list
#: either way, so this only bounds the convenience excerpt.
_SAMPLE_TURNS_PER_ITEM = 6


class ReviewError(TranscriptError):
    """Base class for every error this module raises."""


class NoDocumentToReviewError(ReviewError):
    """The bundle has no assembled document yet (M18: assemble first)."""


class NothingToReviewError(ReviewError):
    """The document has no canonical turns, so there is nothing to review."""


class InvalidReviewPackError(ReviewError):
    """The pack is not a review pack this version can read."""


class StaleReviewError(ReviewError):
    """The pack's bound revision or inventories no longer match the head.

    No auto-rebase in v1 (M8): re-export against the current head. The
    decisions in the stale pack are not lost -- the file is still there --
    but they are not applied to turns the reviewer never saw.
    """


class CompetingReviewError(ReviewError):
    """A different review already applied against this input revision.

    D3 defers branching semantics until something demonstrates the need,
    so v1 rejects rather than inventing a merge order for two reviewers.
    """


class UnknownReviewTargetError(ReviewError):
    """A decision names a turn, cluster, participant, or label that does
    not exist in the document being reviewed."""


@dataclass(frozen=True)
class ReviewPackBinding:
    review_id: ReviewId
    input_revision_id: RevisionId
    turn_inventory_sha256: str
    cluster_inventory_sha256: str


def _cluster_ids(document: TranscriptDocumentV1) -> tuple[str, ...]:
    return tuple(
        cluster.cluster_id
        for attribution in document.components_of(MachineAttributionSetComponent)
        for cluster in attribution.clusters
    )


def build_review_pack(document: TranscriptDocumentV1) -> dict[str, Any]:
    """M8: the exported pack, bound to ``document``'s exact revision.

    Items -- the things the reviewer is asked to decide -- are one per
    voice cluster plus one per turn no cluster covers. That item list is
    stored on the applied review, which is what makes "a partial review"
    (items never addressed) mechanically distinguishable from "a complete
    review with unresolved items" (every item addressed, some as
    ``unclear-speaker``) without a flag anyone could set wrongly.
    """
    turns = canonical_turns(document.components)
    if not turns:
        raise NothingToReviewError(
            "this document has no canonical timed turns; run `transform normalise` "
            "first."
        )
    context = SpeakerContext.from_components(document.components)
    assignments = {turn.turn_id: effective_assignment(turn, context) for turn in turns}
    turns_by_cluster: dict[str, list[str]] = {}
    unattributed: list[str] = []
    for turn in turns:
        cluster_id = context.cluster_by_turn.get(turn.turn_id)
        if cluster_id is None:
            unattributed.append(turn.turn_id)
        else:
            turns_by_cluster.setdefault(cluster_id, []).append(turn.turn_id)

    text_by_turn = {turn.turn_id: turn for turn in turns}
    items: list[dict[str, Any]] = []
    for attribution in document.components_of(MachineAttributionSetComponent):
        for cluster in attribution.clusters:
            member_turn_ids = turns_by_cluster.get(cluster.cluster_id, [])
            hypothesis = context.hypothesis_by_cluster.get(cluster.cluster_id)
            items.append(
                {
                    "item_id": cluster.cluster_id,
                    "kind": "cluster",
                    "raw_label": cluster.raw_label,
                    "media_artefact_id": cluster.media_artefact_id,
                    "turn_count": len(member_turn_ids),
                    "total_ms": cluster.total_ms,
                    "hypothesis": (
                        None
                        if hypothesis is None
                        else {
                            "participant_id": hypothesis.participant_id,
                            "confidence": hypothesis.confidence,
                            "rationale": hypothesis.rationale,
                            "evidence_turn_ids": list(hypothesis.evidence_turn_ids),
                        }
                    ),
                    "sample_turns": [
                        {
                            "turn_id": turn_id,
                            "start_ms": text_by_turn[turn_id].start_ms,
                            "text": text_by_turn[turn_id].text,
                        }
                        for turn_id in member_turn_ids[:_SAMPLE_TURNS_PER_ITEM]
                    ],
                }
            )
    for turn_id in unattributed:
        items.append(
            {
                "item_id": turn_id,
                "kind": "unattributed-turn",
                "start_ms": text_by_turn[turn_id].start_ms,
                "text": text_by_turn[turn_id].text,
                "hypothesis": None,
            }
        )

    return {
        "pack_schema_version": PACK_SCHEMA_VERSION,
        "review_id": mint_id("review"),
        "bundle_id": document.bundle_id,
        "document_id": document.document_id,
        "input_revision_id": document.revision_id,
        "turn_inventory_sha256": turn_inventory_hash([turn.turn_id for turn in turns]),
        "cluster_inventory_sha256": cluster_inventory_hash(
            list(_cluster_ids(document))
        ),
        "reviewer": "",
        "remaining": None,
        "instructions": (
            "Fill in `reviewer`, then add one entry to `decisions` for each item in "
            "`items`. Each decision is {scope, kind, target..., participant_id, "
            "rationale}. scope is one of turn|source-range|cluster|provider-label; "
            "kind is assign (with participant_id) or unclear-speaker (without). "
            "Leaving an item out is a partial review, recorded honestly and "
            f"rendered as '{UNCLEAR_SPEAKER_LABEL}' -- but an undecided turn does "
            "not satisfy the meeting-note speaker gate. Set `remaining` to "
            f"'{REMAINING_UNCLEAR}' to record every item you did not decide as an "
            "explicit unclear-speaker decision; there is deliberately no way to "
            "bulk-assign a person."
        ),
        "participants": [
            {
                "participant_id": participant.participant_id,
                "display_name": participant.display_names[0],
                "status": participant.status.value,
                "room_proxy": participant.room_proxy,
            }
            for participant in sorted(
                context.participants.values(), key=lambda p: p.display_names[0]
            )
        ],
        "items": items,
        "turns": [
            {
                "turn_id": turn.turn_id,
                "source_artefact_id": turn.source_artefact_id,
                "cluster_id": context.cluster_by_turn.get(turn.turn_id),
                "start_ms": turn.start_ms,
                "end_ms": turn.end_ms,
                "text": turn.text,
                "current_speaker": speaker_display_name(
                    assignments[turn.turn_id], context
                ),
                "current_provenance": assignments[turn.turn_id].provenance.value,
            }
            for turn in turns
        ],
        "decisions": [],
    }


def export_review_pack(store: BundleStore, *, destination: Path) -> Path:
    """Write the head's review pack to ``destination`` (M8).

    Pure read plus one write outside the bundle, to a path the operator
    named. Nothing about the bundle changes -- exporting a pack is not an
    operation on the document, so it takes no lease and appends no
    revision.
    """
    document = project_head(store)
    if isinstance(document, NoDocumentYet):
        raise NoDocumentToReviewError(
            "this bundle has no assembled document yet (M18: run `transform "
            "assemble` first)."
        )
    pack = build_review_pack(document)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(pack, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def _decision_from_entry(entry: dict[str, Any]) -> ReviewDecision:
    """Parse one reviewer-authored decision into its typed form.

    Deliberately explicit rather than handing the raw dict to pydantic's
    discriminated union: the pack is hand-edited, so an unrecognised scope
    or a missing target must produce a message naming the field, not a
    validation trace about a union.
    """
    scope_value = entry.get("scope")
    try:
        scope = ReviewScope(scope_value)
    except ValueError as exc:
        raise InvalidReviewPackError(
            f"decision has scope {scope_value!r}; expected one of "
            f"{[member.value for member in ReviewScope]}."
        ) from exc
    kind_value = entry.get("kind")
    try:
        kind = ReviewDecisionKind(kind_value)
    except ValueError as exc:
        raise InvalidReviewPackError(
            f"decision has kind {kind_value!r}; expected one of "
            f"{[member.value for member in ReviewDecisionKind]}."
        ) from exc
    participant_id = entry.get("participant_id")
    rationale = str(entry.get("rationale", ""))
    common: dict[str, Any] = {
        "kind": kind,
        "participant_id": participant_id,
        "rationale": rationale,
    }

    def _required(field: str) -> Any:
        if field not in entry or entry[field] is None:
            raise InvalidReviewPackError(
                f"a {scope.value} decision requires {field!r}."
            )
        return entry[field]

    match scope:
        case ReviewScope.TURN:
            return TurnDecision(turn_id=_required("turn_id"), **common)
        case ReviewScope.CLUSTER:
            return ClusterDecision(cluster_id=_required("cluster_id"), **common)
        case ReviewScope.PROVIDER_LABEL:
            return ProviderLabelDecision(raw_label=_required("raw_label"), **common)
        case ReviewScope.SOURCE_RANGE:
            return SourceRangeDecision(
                source_artefact_id=_required("source_artefact_id"),
                start_ms=int(_required("start_ms")),
                end_ms=int(_required("end_ms")),
                **common,
            )


#: The pack field a reviewer sets to say "I considered the rest and they
#: are genuinely unclear". Only this one value is accepted -- there is no
#: bulk *assignment* affordance, because "assign the remainder to Michael"
#: is a guess wearing a reviewer's name, while "the remainder are unclear"
#: is the honest statement the M8 ladder already has a decision kind for.
REMAINING_UNCLEAR = "unclear-speaker"


def _expand_remaining_unclear(
    pack: dict[str, Any], decided: tuple[ReviewDecision, ...]
) -> tuple[ReviewDecision, ...]:
    """Turn ``"remaining": "unclear-speaker"`` into explicit decisions.

    Real diarisation on real room audio routinely leaves hundreds of
    turns with no voice evidence. Deciding each by hand is not review, it
    is data entry -- and without a decision those turns resolve at rung 8
    (``no-evidence``), which does not satisfy the meeting-note gate, so
    the note simply cannot be rendered.

    This closes that gap without weakening anything: each remaining item
    becomes a *real, stored* ``unclear-speaker`` decision at its own
    scope, indistinguishable from one typed by hand, and every one of
    them renders as "Unclear speaker" with a visible count. What it
    cannot do is assign anybody -- the only accepted value is
    ``unclear-speaker``.
    """
    remaining = pack.get("remaining")
    if remaining is None:
        return ()
    if remaining != REMAINING_UNCLEAR:
        raise InvalidReviewPackError(
            f"`remaining` is {remaining!r}; the only accepted value is "
            f"{REMAINING_UNCLEAR!r}. Bulk-assigning a participant to every "
            "undecided item would be a guess recorded as a human decision."
        )
    addressed = _addressed_item_ids(decided)
    expanded: list[ReviewDecision] = []
    for item in pack.get("items", []):
        item_id = str(item.get("item_id", ""))
        if not item_id or item_id in addressed:
            continue
        rationale = "not individually decided; recorded as unclear (bulk)"
        if item.get("kind") == "cluster":
            expanded.append(
                ClusterDecision(
                    kind=ReviewDecisionKind.UNCLEAR_SPEAKER,
                    cluster_id=item_id,
                    rationale=rationale,
                )
            )
        else:
            expanded.append(
                TurnDecision(
                    kind=ReviewDecisionKind.UNCLEAR_SPEAKER,
                    turn_id=item_id,
                    rationale=rationale,
                )
            )
    return tuple(expanded)


def _read_binding(pack: dict[str, Any]) -> ReviewPackBinding:
    version = pack.get("pack_schema_version")
    if version != PACK_SCHEMA_VERSION:
        raise InvalidReviewPackError(
            f"pack_schema_version is {version!r}; this build reads "
            f"{PACK_SCHEMA_VERSION!r}. Re-export the pack."
        )
    missing = [
        field
        for field in (
            "review_id",
            "input_revision_id",
            "turn_inventory_sha256",
            "cluster_inventory_sha256",
        )
        if not pack.get(field)
    ]
    if missing:
        raise InvalidReviewPackError(
            f"pack is missing required binding field(s): {missing}."
        )
    return ReviewPackBinding(
        review_id=pack["review_id"],
        input_revision_id=pack["input_revision_id"],
        turn_inventory_sha256=pack["turn_inventory_sha256"],
        cluster_inventory_sha256=pack["cluster_inventory_sha256"],
    )


def _check_targets_exist(
    decisions: tuple[ReviewDecision, ...], document: TranscriptDocumentV1
) -> None:
    context = SpeakerContext.from_components(document.components)
    known_turn_ids = {turn.turn_id for turn in canonical_turns(document.components)}
    known_cluster_ids = set(_cluster_ids(document))
    known_artefact_ids = {
        turn.source_artefact_id for turn in canonical_turns(document.components)
    }
    for decision in decisions:
        if (
            decision.participant_id is not None
            and decision.participant_id not in context.participants
        ):
            raise UnknownReviewTargetError(
                f"decision names participant {decision.participant_id!r}, which is "
                "not a declared participant of this document (F21)."
            )
        match decision:
            case TurnDecision() if decision.turn_id not in known_turn_ids:
                raise UnknownReviewTargetError(
                    f"decision names turn {decision.turn_id!r}, which is not a "
                    "canonical turn of this document."
                )
            case ClusterDecision() if decision.cluster_id not in known_cluster_ids:
                raise UnknownReviewTargetError(
                    f"decision names cluster {decision.cluster_id!r}, which is not a "
                    "voice cluster of this document."
                )
            case SourceRangeDecision() if (
                decision.source_artefact_id not in known_artefact_ids
            ):
                raise UnknownReviewTargetError(
                    f"decision names source artefact {decision.source_artefact_id!r}, "
                    "which contributes no canonical turn to this document."
                )
            case _:
                continue


@dataclass(frozen=True)
class ApplyReviewOutcome:
    review: ReviewRecord
    revision: RevisionRecord | None
    component: SpeakerReviewComponent | None
    already_applied: bool
    addressed_item_count: int
    total_item_count: int


@dataclass(frozen=True)
class PreparedReview:
    """A validated review, ready to commit -- nothing written yet.

    Splitting validation from commitment is what lets a refusal (stale
    pack, anonymous reviewer, invented cluster) surface as *itself*
    instead of as an executor crash that drags a run into ``failed``: the
    caller validates first, and only creates a run when there is genuine
    work to record.
    """

    body: SpeakerReviewComponentBody
    document: TranscriptDocumentV1
    pack_sha256: str


def prepare_review_application(
    store: BundleStore, *, pack_path: Path
) -> ReviewRecord | PreparedReview:
    """Validate a filled pack against the current head, writing nothing.

    Returns the existing :class:`~.records.ReviewRecord` when this review
    has already been applied (M8's idempotency), or a
    :class:`PreparedReview` when it is new and valid. Checks, in order:
    the pack is readable and this build's schema; the review has not
    already been applied; no *different* review has applied against the
    same input revision; the pack's bound revision is still the head; the
    turn and cluster inventories still hash the same; the reviewer is
    named; every decision names something that exists.
    """
    try:
        pack: dict[str, Any] = json.loads(pack_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidReviewPackError(
            f"cannot read review pack {pack_path}: {exc}"
        ) from exc
    pack_sha256 = hashlib.sha256(pack_path.read_bytes()).hexdigest()
    binding = _read_binding(pack)

    existing = store.find_review(binding.review_id)
    if existing is not None:
        return existing
    competing = next(
        (
            record
            for record in store.iter_reviews()
            if record.input_revision_id == binding.input_revision_id
        ),
        None,
    )
    if competing is not None:
        raise CompetingReviewError(
            f"review {competing.review_id} has already been applied against input "
            f"revision {binding.input_revision_id}; a second, differing review set "
            "against the same input revision is rejected in v1 (M8/D3). Re-export a "
            "pack against the current head to review again."
        )

    document = project_head(store)
    if isinstance(document, NoDocumentYet):
        raise NoDocumentToReviewError(
            "this bundle has no assembled document yet (M18: run `transform "
            "assemble` first)."
        )
    if document.revision_id != binding.input_revision_id:
        raise StaleReviewError(
            f"this pack is bound to revision {binding.input_revision_id}, but the "
            f"bundle head is now {document.revision_id}. No auto-rebase in v1 (M8) "
            "-- re-export the pack and review against the current head."
        )
    turns = canonical_turns(document.components)
    actual_turn_hash = turn_inventory_hash([turn.turn_id for turn in turns])
    actual_cluster_hash = cluster_inventory_hash(list(_cluster_ids(document)))
    if actual_turn_hash != binding.turn_inventory_sha256:
        raise StaleReviewError(
            "this pack's turn inventory no longer matches the document's "
            f"(pack {binding.turn_inventory_sha256}, document {actual_turn_hash}); "
            "re-export the pack."
        )
    if actual_cluster_hash != binding.cluster_inventory_sha256:
        raise StaleReviewError(
            "this pack's cluster inventory no longer matches the document's "
            f"(pack {binding.cluster_inventory_sha256}, document "
            f"{actual_cluster_hash}); re-export the pack."
        )

    entries = pack.get("decisions") or []
    if not isinstance(entries, list) or not entries:
        raise InvalidReviewPackError(
            "the pack's `decisions` list is empty; there is nothing to apply. An "
            "unreviewed document is left unreviewed rather than recorded as a "
            "review with no decisions."
        )
    decisions = tuple(_decision_from_entry(entry) for entry in entries)
    decisions += _expand_remaining_unclear(pack, decisions)
    _check_targets_exist(decisions, document)

    reviewer = str(pack.get("reviewer") or "").strip()
    if not reviewer:
        raise InvalidReviewPackError(
            "the pack's `reviewer` field is empty; M8 binds a review to a reviewer "
            "identity, and an anonymous review is not an auditable one."
        )
    pack_item_ids = tuple(
        str(item["item_id"]) for item in pack.get("items", []) if "item_id" in item
    )

    return PreparedReview(
        body=SpeakerReviewComponentBody(
            review_id=binding.review_id,
            input_revision_id=binding.input_revision_id,
            turn_inventory_hash=actual_turn_hash,
            cluster_inventory_hash=actual_cluster_hash,
            pack_schema_version=PACK_SCHEMA_VERSION,
            reviewer=reviewer,
            pack_item_ids=pack_item_ids,
            decisions=decisions,
        ),
        document=document,
        pack_sha256=pack_sha256,
    )


def commit_review(
    store: BundleStore, *, run_id: RunId, prepared: PreparedReview
) -> ApplyReviewOutcome:
    """Append the validated review's component, revision, and record.

    Called only with the lease held, and only after
    :func:`prepare_review_application` has already refused everything
    refusable -- so the three writes here are the whole of the mutation,
    in one place.
    """
    body = prepared.body
    document = prepared.document
    component = store.add_component(body)
    assert isinstance(component, SpeakerReviewComponent)
    revision = store.append_revision(
        operation=OperationRef(
            kind="review-apply",
            input_ids=(body.review_id, body.input_revision_id),
            rationale=f"speaker review applied by {body.reviewer} (M8)",
        ),
        parent_revision_ids=(document.revision_id,),
        component_ids=(component.component_id,),
        superseded_component_ids=tuple(
            prior.component_id
            for prior in document.components_of(SpeakerReviewComponent)
        ),
    )
    store.update_head(run_id=run_id, revision_id=revision.revision_id)
    review = store.add_review(
        ReviewRecord(
            review_id=body.review_id,
            bundle_id=document.bundle_id,
            input_revision_id=body.input_revision_id,
            result_revision_id=revision.revision_id,
            decision_component_id=component.component_id,
            pack_schema_version=PACK_SCHEMA_VERSION,
            pack_sha256=prepared.pack_sha256,
            reviewer=body.reviewer,
            created_at=revision.created_at,
        )
    )
    addressed = _addressed_item_ids(body.decisions)
    return ApplyReviewOutcome(
        review=review,
        revision=revision,
        component=component,
        already_applied=False,
        addressed_item_count=len(addressed.intersection(body.pack_item_ids)),
        total_item_count=len(body.pack_item_ids),
    )


def apply_review(
    store: BundleStore, *, run_id: RunId, pack_path: Path
) -> ApplyReviewOutcome:
    """Validate and commit a filled review pack in one call.

    The convenience shape for callers that already hold a lease.
    ``control.run_review_apply`` uses the two halves separately so a
    refusal never costs a run.
    """
    prepared = prepare_review_application(store, pack_path=pack_path)
    if isinstance(prepared, ReviewRecord):
        return ApplyReviewOutcome(
            review=prepared,
            revision=None,
            component=None,
            already_applied=True,
            addressed_item_count=0,
            total_item_count=0,
        )
    return commit_review(store, run_id=run_id, prepared=prepared)


def _addressed_item_ids(decisions: tuple[ReviewDecision, ...]) -> set[str]:
    """Which pack items a decision set actually speaks to.

    Only turn- and cluster-scoped decisions address an *item*, because
    those are the two things the pack lists as items. A source-range
    decision is extra reviewer precision covering whatever turns fall
    inside it -- genuinely useful, but it does not by itself demonstrate
    that the reviewer considered any particular item, so counting it here
    would overstate coverage.
    """
    addressed: set[str] = set()
    for decision in decisions:
        match decision:
            case TurnDecision():
                addressed.add(decision.turn_id)
            case ClusterDecision():
                addressed.add(decision.cluster_id)
            case _:
                continue
    return addressed


def review_is_complete(
    review: SpeakerReviewComponent,
) -> bool:
    """M8: did the reviewer address every item the pack asked about?

    A complete review may still contain ``unclear-speaker`` decisions --
    "I looked at all of them and cannot tell for two" is complete, and
    materially different from "I only got through half".
    """
    if not review.pack_item_ids:
        return False
    return set(review.pack_item_ids).issubset(_addressed_item_ids(review.decisions))


def hypothesis_sets(document: TranscriptDocumentV1) -> int:
    """How many proposal passes are live in this document (diagnostic)."""
    return len(document.components_of(SpeakerHypothesisSetComponent))
