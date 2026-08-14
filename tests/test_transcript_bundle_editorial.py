from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from jake_tools.transcripts.bundle.assignment import turn_inventory_hash
from jake_tools.transcripts.bundle.components import (
    EditorialDerivationMode,
    EditorialOperationLedgerComponent,
    EditorialSpeakerState,
    EditorialTranscriptComponent,
    ReviewDecisionKind,
    SpeakerReviewComponent,
    SpeakerReviewComponentBody,
    TimedTurn,
    TimedTurnSetComponent,
    TimedTurnSetComponentBody,
    TurnDecision,
    assemble_component_record,
    component_as_body,
)
from jake_tools.transcripts.bundle.editorial import (
    EditorialLineageError,
    identity_editorial_projection,
    validate_editorial_lineage,
)
from jake_tools.transcripts.bundle.ids import mint_id, source_domain

NOW = datetime.now(UTC)
HASH = hashlib.sha256(b"fixture").hexdigest()


def _stored(body: TimedTurnSetComponentBody | SpeakerReviewComponentBody):
    return assemble_component_record(
        body,
        component_id=mint_id("component"),
        content_hash=HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )


def _reviewed_source() -> tuple[TimedTurnSetComponent, SpeakerReviewComponent]:
    artefact_id = mint_id("artefact")
    participant_id = mint_id("participant")
    turns = (
        TimedTurn(
            turn_id=mint_id("turn"),
            source_segment_id=mint_id("seg"),
            source_artefact_id=artefact_id,
            speaker_label="SPEAKER_00",
            text="The CAN bus is not ready.",
            start_ms=100,
            end_ms=900,
        ),
        TimedTurn(
            turn_id=mint_id("turn"),
            source_segment_id=mint_id("seg"),
            source_artefact_id=artefact_id,
            speaker_label="SPEAKER_01",
            text="Maybe we should wait.",
            start_ms=850,
            end_ms=1_300,
        ),
    )
    canonical = _stored(
        TimedTurnSetComponentBody(
            source_artefact_ids=(artefact_id,),
            coordinate_domain=source_domain(artefact_id),
            turns=turns,
        )
    )
    assert isinstance(canonical, TimedTurnSetComponent)
    review = _stored(
        SpeakerReviewComponentBody(
            review_id=mint_id("review"),
            input_revision_id=mint_id("rev"),
            turn_inventory_hash=turn_inventory_hash([turn.turn_id for turn in turns]),
            cluster_inventory_hash=hashlib.sha256(b"clusters").hexdigest(),
            pack_schema_version="2",
            reviewer="Michael",
            pack_item_ids=("speaker-1", "speaker-2"),
            decisions=(
                TurnDecision(
                    turn_id=turns[0].turn_id,
                    kind=ReviewDecisionKind.ASSIGN,
                    participant_id=participant_id,
                    rationale="reviewed voice",
                ),
                TurnDecision(
                    turn_id=turns[1].turn_id,
                    kind=ReviewDecisionKind.UNCLEAR_SPEAKER,
                    rationale="overlapping speech",
                ),
            ),
        )
    )
    assert isinstance(review, SpeakerReviewComponent)
    return canonical, review


def test_identity_projection_accounts_for_every_canonical_character_without_mutation() -> (
    None
):
    canonical, review = _reviewed_source()
    before = canonical.model_dump_json()

    body, ledger = identity_editorial_projection(canonical=canonical, review=review)

    assert canonical.model_dump_json() == before
    assert body.canonical_turn_set_component_id == canonical.component_id
    assert body.canonical_turn_set_content_hash == canonical.content_hash
    assert body.speaker_review_component_id == review.component_id
    assert body.speaker_review_content_hash == review.content_hash
    assert [node.display_order for node in body.nodes] == [0, 1]
    assert [node.text for node in body.nodes] == [turn.text for turn in canonical.turns]
    assert [node.derivation_mode for node in body.nodes] == [
        EditorialDerivationMode.VERBATIM,
        EditorialDerivationMode.VERBATIM,
    ]
    assert body.nodes[0].attribution.state == EditorialSpeakerState.NAMED
    assert body.nodes[1].attribution.state == EditorialSpeakerState.UNCLEAR
    assert body.nodes[0].source_spans[0].start_char == 0
    assert body.nodes[0].source_spans[0].end_char == len(canonical.turns[0].text)
    assert body.nodes[0].constituent_intervals[0].start_ms == 100
    assert body.nodes[0].constituent_intervals[0].end_ms == 900
    assert body.nodes[0].display_start_ms == 100
    assert body.nodes[0].display_end_ms == 900
    assert len(ledger.operations) == 2
    assert all(operation.operation == "identity" for operation in ledger.operations)
    validate_editorial_lineage(body=body, canonical=canonical, review=review)


def test_editorial_components_round_trip_through_the_closed_component_dispatch() -> (
    None
):
    canonical, review = _reviewed_source()
    body, ledger = identity_editorial_projection(canonical=canonical, review=review)

    stored_body = assemble_component_record(
        body,
        component_id=mint_id("component"),
        content_hash=HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )
    stored_ledger = assemble_component_record(
        ledger,
        component_id=mint_id("component"),
        content_hash=HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )

    assert isinstance(stored_body, EditorialTranscriptComponent)
    assert isinstance(stored_ledger, EditorialOperationLedgerComponent)
    assert component_as_body(stored_body) == body
    assert component_as_body(stored_ledger) == ledger


def test_out_of_range_source_span_refuses() -> None:
    canonical, review = _reviewed_source()
    body, _ = identity_editorial_projection(canonical=canonical, review=review)
    node = body.nodes[0]
    bad_span = node.source_spans[0].model_copy(
        update={"end_char": len(canonical.turns[0].text) + 1}
    )
    bad_node = node.model_copy(update={"source_spans": (bad_span,)})
    bad_body = body.model_copy(update={"nodes": (bad_node, *body.nodes[1:])})

    with pytest.raises(EditorialLineageError, match="out of bounds"):
        validate_editorial_lineage(body=bad_body, canonical=canonical, review=review)


def test_generated_punctuation_cannot_be_labelled_verbatim() -> None:
    canonical, review = _reviewed_source()
    body, _ = identity_editorial_projection(canonical=canonical, review=review)
    node = body.nodes[0]
    changed = node.model_copy(update={"text": node.text + "!"})
    bad_body = body.model_copy(update={"nodes": (changed, *body.nodes[1:])})

    with pytest.raises(EditorialLineageError, match="verbatim"):
        validate_editorial_lineage(body=bad_body, canonical=canonical, review=review)


def test_timestamp_overlap_does_not_authorise_display_reorder() -> None:
    canonical, review = _reviewed_source()
    body, _ = identity_editorial_projection(canonical=canonical, review=review)
    reordered = (
        body.nodes[1].model_copy(update={"display_order": 0}),
        body.nodes[0].model_copy(update={"display_order": 1}),
    )
    bad_body = body.model_copy(update={"nodes": reordered})

    with pytest.raises(EditorialLineageError, match="reviewed overlap group"):
        validate_editorial_lineage(body=bad_body, canonical=canonical, review=review)
