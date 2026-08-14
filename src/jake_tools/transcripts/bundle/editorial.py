"""Truthful canonical-to-editorial transcript projection and validation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping

from ..errors import TranscriptError
from .assignment import SpeakerContext, turn_inventory_hash
from .components import (
    CanonicalTextSpan,
    ComponentRecord,
    EditorialAttribution,
    EditorialDerivationMode,
    EditorialNode,
    EditorialOperation,
    EditorialOperationKind,
    EditorialOperationLedgerComponentBody,
    EditorialSpeakerState,
    EditorialTranscriptComponentBody,
    SourceInterval,
    SpeakerReviewComponent,
    TimedTurn,
    TimedTurnSetComponent,
)
from .ids import ComponentId, mint_id

IDENTITY_EDITOR = "jake-tools:editorial-identity-v1"
IDENTITY_CONFIG_HASH = hashlib.sha256(
    json.dumps(
        {
            "editor": IDENTITY_EDITOR,
            "operation": "one-verbatim-node-per-reviewed-canonical-turn",
            "version": 1,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


class EditorialLineageError(TranscriptError):
    """Editorial output is not a legal derivation of its exact evidence."""


def _turns_by_id(canonical: TimedTurnSetComponent) -> dict[str, TimedTurn]:
    return {turn.turn_id: turn for turn in canonical.turns}


def _attribution_for(
    *, turn: TimedTurn, review: SpeakerReviewComponent, context: SpeakerContext
) -> EditorialAttribution:
    assignment = context.assignment_for(turn)
    if not assignment.is_reviewed:
        raise EditorialLineageError(
            f"canonical turn {turn.turn_id} has no effective human-reviewed "
            "attribution; editorial identity projection refuses to guess."
        )
    state = (
        EditorialSpeakerState.NAMED
        if assignment.participant_id is not None
        else EditorialSpeakerState.UNCLEAR
    )
    participant_ids = (
        (assignment.participant_id,) if assignment.participant_id is not None else ()
    )
    return EditorialAttribution(
        state=state,
        participant_ids=participant_ids,
        speaker_review_component_id=review.component_id,
        evidence_ref=(
            f"review:{review.review_id};turn:{turn.turn_id};"
            f"provenance:{assignment.provenance.value}"
        ),
    )


def identity_editorial_projection(
    *,
    canonical: TimedTurnSetComponent,
    review: SpeakerReviewComponent,
    components: Mapping[ComponentId, ComponentRecord] | None = None,
) -> tuple[EditorialTranscriptComponentBody, EditorialOperationLedgerComponentBody]:
    """Create one immutable, verbatim editorial node per reviewed turn."""
    expected_inventory = turn_inventory_hash([turn.turn_id for turn in canonical.turns])
    if review.turn_inventory_hash != expected_inventory:
        raise EditorialLineageError(
            "speaker review turn inventory does not match the canonical component."
        )
    resolved_components = dict(components or {})
    resolved_components[canonical.component_id] = canonical
    resolved_components[review.component_id] = review
    context = SpeakerContext.from_components(resolved_components)
    nodes: list[EditorialNode] = []
    operations: list[EditorialOperation] = []
    for display_order, turn in enumerate(canonical.turns):
        node_id = mint_id("editorial")
        span = CanonicalTextSpan(
            canonical_component_id=canonical.component_id,
            turn_id=turn.turn_id,
            start_char=0,
            end_char=len(turn.text),
        )
        node = EditorialNode(
            node_id=node_id,
            display_order=display_order,
            text=turn.text,
            source_spans=(span,),
            constituent_intervals=(
                SourceInterval(
                    turn_id=turn.turn_id,
                    start_ms=turn.start_ms,
                    end_ms=turn.end_ms,
                ),
            ),
            display_start_ms=turn.start_ms,
            display_end_ms=turn.end_ms,
            attribution=_attribution_for(turn=turn, review=review, context=context),
            derivation_mode=EditorialDerivationMode.VERBATIM,
            operation_ancestry=(f"identity:{node_id}",),
            creator=IDENTITY_EDITOR,
            config_hash=IDENTITY_CONFIG_HASH,
        )
        nodes.append(node)
        operations.append(
            EditorialOperation(
                operation_id=f"identity:{node_id}",
                operation=EditorialOperationKind.IDENTITY,
                input_node_ids=(node_id,),
                output_node_ids=(node_id,),
                source_spans=(span,),
                reason="identity projection from reviewed canonical evidence",
            )
        )
    body = EditorialTranscriptComponentBody(
        canonical_turn_set_component_id=canonical.component_id,
        canonical_turn_set_content_hash=canonical.content_hash,
        speaker_review_component_id=review.component_id,
        speaker_review_content_hash=review.content_hash,
        nodes=tuple(nodes),
    )
    ledger = EditorialOperationLedgerComponentBody(
        canonical_turn_set_component_id=canonical.component_id,
        speaker_review_component_id=review.component_id,
        creator=IDENTITY_EDITOR,
        config_hash=IDENTITY_CONFIG_HASH,
        operations=tuple(operations),
    )
    validate_editorial_lineage(body=body, canonical=canonical, review=review)
    return body, ledger


def validate_editorial_lineage(
    *,
    body: EditorialTranscriptComponentBody,
    canonical: TimedTurnSetComponent,
    review: SpeakerReviewComponent,
) -> None:
    """Prove structural legality and exact source accounting without semantics."""
    if (
        body.canonical_turn_set_component_id != canonical.component_id
        or body.canonical_turn_set_content_hash != canonical.content_hash
    ):
        raise EditorialLineageError(
            "editorial transcript has a stale canonical binding."
        )
    if (
        body.speaker_review_component_id != review.component_id
        or body.speaker_review_content_hash != review.content_hash
    ):
        raise EditorialLineageError(
            "editorial transcript has a stale speaker-review binding."
        )
    if review.turn_inventory_hash != turn_inventory_hash(
        [turn.turn_id for turn in canonical.turns]
    ):
        raise EditorialLineageError(
            "speaker review inventory does not match canonical turns."
        )

    turns = _turns_by_id(canonical)
    turn_order = {turn.turn_id: index for index, turn in enumerate(canonical.turns)}
    accounted: dict[str, list[tuple[int, int]]] = defaultdict(list)
    groups = {group.overlap_group_id: group for group in body.overlap_groups}
    node_orders: list[tuple[int, set[str], EditorialNode]] = []

    if [node.display_order for node in body.nodes] != list(range(len(body.nodes))):
        raise EditorialLineageError("editorial display order is not contiguous.")

    for node in body.nodes:
        span_turn_ids: set[str] = set()
        source_text: list[str] = []
        for span in node.source_spans:
            turn = turns.get(span.turn_id)
            if turn is None:
                raise EditorialLineageError(
                    f"editorial span cites unknown canonical turn {span.turn_id}."
                )
            if span.canonical_component_id != canonical.component_id:
                raise EditorialLineageError(
                    "editorial span cites the wrong canonical component."
                )
            if span.end_char > len(turn.text):
                raise EditorialLineageError(
                    f"editorial span [{span.start_char}, {span.end_char}) is out of "
                    f"bounds for canonical turn {turn.turn_id}."
                )
            accounted[turn.turn_id].append((span.start_char, span.end_char))
            span_turn_ids.add(turn.turn_id)
            source_text.append(turn.text[span.start_char : span.end_char])

        interval_turn_ids: set[str] = set()
        for interval in node.constituent_intervals:
            turn = turns.get(interval.turn_id)
            if turn is None:
                raise EditorialLineageError(
                    f"editorial interval cites unknown canonical turn {interval.turn_id}."
                )
            if (interval.start_ms, interval.end_ms) != (turn.start_ms, turn.end_ms):
                raise EditorialLineageError(
                    f"editorial interval for {turn.turn_id} is not its canonical interval."
                )
            interval_turn_ids.add(turn.turn_id)
        if interval_turn_ids != span_turn_ids:
            raise EditorialLineageError(
                f"editorial node {node.node_id} interval and text-span evidence disagree."
            )
        expected_envelope = (
            min(interval.start_ms for interval in node.constituent_intervals),
            max(interval.end_ms for interval in node.constituent_intervals),
        )
        if (node.display_start_ms, node.display_end_ms) != expected_envelope:
            raise EditorialLineageError(
                f"editorial node {node.node_id} has a false display envelope."
            )
        if (
            node.derivation_mode == EditorialDerivationMode.VERBATIM
            and node.text != "".join(source_text)
        ):
            raise EditorialLineageError(
                f"editorial node {node.node_id} is labelled verbatim but its bytes differ."
            )
        node_orders.append(
            (min(turn_order[turn_id] for turn_id in span_turn_ids), span_turn_ids, node)
        )

    for span in body.omitted_source_spans:
        turn = turns.get(span.turn_id)
        if turn is None:
            raise EditorialLineageError(
                f"omitted span cites unknown canonical turn {span.turn_id}."
            )
        if span.end_char > len(turn.text):
            raise EditorialLineageError("omitted canonical span is out of bounds.")
        accounted[turn.turn_id].append((span.start_char, span.end_char))

    for turn in canonical.turns:
        ranges = sorted(accounted[turn.turn_id])
        cursor = 0
        for start, end in ranges:
            if start != cursor:
                problem = "duplication" if start < cursor else "unaccounted characters"
                raise EditorialLineageError(
                    f"canonical turn {turn.turn_id} has {problem} at character {cursor}."
                )
            cursor = end
        if cursor != len(turn.text):
            raise EditorialLineageError(
                f"canonical turn {turn.turn_id} has unaccounted characters at {cursor}."
            )

    previous_order = -1
    previous_turn_ids: set[str] = set()
    previous_node: EditorialNode | None = None
    for canonical_order, turn_ids, node in node_orders:
        if canonical_order < previous_order:
            group = groups.get(node.overlap_group_id) if node.overlap_group_id else None
            authorised = (
                group is not None
                and previous_node is not None
                and previous_node.overlap_group_id == group.overlap_group_id
                and previous_turn_ids.union(turn_ids).issubset(set(group.turn_ids))
                and group.speaker_review_component_id == review.component_id
            )
            if not authorised:
                raise EditorialLineageError(
                    "display reorder requires one exact reviewed overlap group; timestamp "
                    "overlap alone is not authority."
                )
        previous_order = canonical_order
        previous_turn_ids = turn_ids
        previous_node = node
