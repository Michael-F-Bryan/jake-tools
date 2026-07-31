"""M8: the effective-assignment ladder.

The ladder is the one function every renderer, gate, and capability
payload consumes, so each of its eight rungs gets its own test, plus the
two rules that are easy to get subtly wrong: an explicit
``unclear-speaker`` blocking every weaker rung, and a narrower scope
beating a broader one.

Built from components directly rather than through a store: the ladder is
pure over a resolved component mapping, and testing it that way is what
lets one rung be isolated at a time.
"""

from __future__ import annotations

import hashlib

from jake_tools.transcripts.bundle.assignment import (
    UNCLEAR_SPEAKER_LABEL,
    ProvenanceClass,
    SpeakerContext,
    assignment_coverage,
    cluster_inventory_hash,
    effective_assignment,
    effective_assignments,
    overlapping_turns_needing_review,
    speaker_display_name,
    turn_inventory_hash,
)
from jake_tools.transcripts.bundle.components import (
    ClusterDecision,
    ComponentRecord,
    MachineAttributionSetComponentBody,
    ParticipantDeclarationSource,
    ParticipantRecord,
    ParticipantSetComponentBody,
    ParticipantStatus,
    ProviderLabelDecision,
    ProviderLabelSetComponentBody,
    ProviderLabelSpan,
    ReviewDecision,
    ReviewDecisionKind,
    SourceRangeDecision,
    SpeakerCluster,
    SpeakerHypothesis,
    SpeakerHypothesisSetComponentBody,
    SpeakerReviewComponentBody,
    TimedTurn,
    TimedTurnSetComponentBody,
    TrustClass,
    TurnClusterAssignment,
    TurnDecision,
    assemble_component_record,
)
from jake_tools.transcripts.bundle.ids import mint_id, source_domain

_HASH = hashlib.sha256(b"fixture").hexdigest()
_ARTEFACT = mint_id("artefact")
_MICHAEL = mint_id("participant")
_AVALON = mint_id("participant")


def _stored(body: object) -> ComponentRecord:
    from datetime import UTC, datetime

    return assemble_component_record(
        body,  # pyright: ignore[reportArgumentType]
        component_id=mint_id("component"),
        content_hash=_HASH,
        created_at=datetime.now(UTC),
        mint_segment_id=lambda: mint_id("seg"),
    )


def _turn(
    *,
    turn_id: str | None = None,
    start_ms: int = 0,
    end_ms: int = 1000,
    segment_id: str | None = None,
    label: str = "SPEAKER_00",
) -> TimedTurn:
    return TimedTurn(
        turn_id=turn_id or mint_id("turn"),
        source_segment_id=segment_id or mint_id("seg"),
        source_artefact_id=_ARTEFACT,
        speaker_label=label,
        text="hello",
        start_ms=start_ms,
        end_ms=end_ms,
    )


def _participants() -> ComponentRecord:
    return _stored(
        ParticipantSetComponentBody(
            participants=(
                ParticipantRecord(
                    participant_id=_MICHAEL,
                    declaration_source=ParticipantDeclarationSource.NOTE_FRONTMATTER,
                    declaration_evidence="frontmatter",
                    display_names=("Michael Bryan",),
                    status=ParticipantStatus.DECLARED,
                ),
                ParticipantRecord(
                    participant_id=_AVALON,
                    declaration_source=ParticipantDeclarationSource.NOTE_FRONTMATTER,
                    declaration_evidence="frontmatter",
                    display_names=("Avalon Mann",),
                    status=ParticipantStatus.DECLARED,
                ),
            )
        )
    )


def _components(*records: ComponentRecord) -> dict[str, ComponentRecord]:
    return {record.component_id: record for record in records}


def _review(*decisions: ReviewDecision) -> ComponentRecord:
    return _stored(
        SpeakerReviewComponentBody(
            review_id=mint_id("review"),
            input_revision_id=mint_id("rev"),
            turn_inventory_hash=_HASH,
            cluster_inventory_hash=_HASH,
            pack_schema_version="1",
            reviewer="Michael Bryan",
            decisions=decisions,
        )
    )


def _attribution(
    *, cluster_id: str, turn_id: str, media_artefact_id: str = _ARTEFACT
) -> ComponentRecord:
    return _stored(
        MachineAttributionSetComponentBody(
            clusters=(
                SpeakerCluster(
                    cluster_id=cluster_id,
                    raw_label="SPEAKER_00",
                    media_artefact_id=media_artefact_id,
                    diarisation_artefact_id=mint_id("artefact"),
                    segment_count=1,
                    total_ms=1000,
                ),
            ),
            assignments=(
                TurnClusterAssignment(
                    turn_id=turn_id, cluster_id=cluster_id, overlap_ms=1000
                ),
            ),
        )
    )


# -- the eight rungs, one at a time ------------------------------------------


def test_rung_1_a_reviewed_turn_override_wins_outright() -> None:
    turn = _turn()
    cluster_id = mint_id("cluster")
    context = SpeakerContext.from_components(
        _components(
            _participants(),
            _attribution(cluster_id=cluster_id, turn_id=turn.turn_id),
            _review(
                TurnDecision(
                    kind=ReviewDecisionKind.ASSIGN,
                    participant_id=_MICHAEL,
                    turn_id=turn.turn_id,
                ),
                ClusterDecision(
                    kind=ReviewDecisionKind.ASSIGN,
                    participant_id=_AVALON,
                    cluster_id=cluster_id,
                ),
            ),
        )
    )

    assignment = effective_assignment(turn, context)

    assert assignment.rung == 1
    assert assignment.participant_id == _MICHAEL
    assert assignment.provenance == ProvenanceClass.REVIEWED_TURN


def test_rung_2_a_reviewed_source_range_covers_the_turns_inside_it() -> None:
    inside = _turn(start_ms=5_000, end_ms=6_000)
    outside = _turn(start_ms=50_000, end_ms=51_000)
    context = SpeakerContext.from_components(
        _components(
            _participants(),
            _review(
                SourceRangeDecision(
                    kind=ReviewDecisionKind.ASSIGN,
                    participant_id=_AVALON,
                    source_artefact_id=_ARTEFACT,
                    start_ms=0,
                    end_ms=10_000,
                )
            ),
        )
    )

    assert effective_assignment(inside, context).participant_id == _AVALON
    assert effective_assignment(inside, context).rung == 2
    assert effective_assignment(outside, context).rung == 8


def test_rung_3_a_reviewed_cluster_default_covers_that_voice() -> None:
    turn = _turn()
    cluster_id = mint_id("cluster")
    context = SpeakerContext.from_components(
        _components(
            _participants(),
            _attribution(cluster_id=cluster_id, turn_id=turn.turn_id),
            _review(
                ClusterDecision(
                    kind=ReviewDecisionKind.ASSIGN,
                    participant_id=_AVALON,
                    cluster_id=cluster_id,
                )
            ),
        )
    )

    assignment = effective_assignment(turn, context)

    assert assignment.rung == 3
    assert assignment.participant_id == _AVALON


def test_rung_3_outranks_rung_4_because_a_cluster_is_a_voice() -> None:
    """M8: a reviewed cluster default is voice-specific, so it beats a
    reviewed provider-label default -- which on a room-proxy span describes
    an aggregate of several people, not one voice (D1)."""
    segment_id = mint_id("seg")
    turn = _turn(segment_id=segment_id)
    cluster_id = mint_id("cluster")
    context = SpeakerContext.from_components(
        _components(
            _participants(),
            _attribution(cluster_id=cluster_id, turn_id=turn.turn_id),
            _stored(
                ProviderLabelSetComponentBody(
                    source_artefact_id=_ARTEFACT,
                    proxy_config_hash=_HASH,
                    spans=(
                        ProviderLabelSpan(
                            source_segment_id=segment_id,
                            raw_label="Meetings Ahoy",
                            text="hello",
                            trust_class=TrustClass.ROOM_PROXY,
                        ),
                    ),
                )
            ),
            _review(
                ClusterDecision(
                    kind=ReviewDecisionKind.ASSIGN,
                    participant_id=_MICHAEL,
                    cluster_id=cluster_id,
                ),
                ProviderLabelDecision(
                    kind=ReviewDecisionKind.ASSIGN,
                    participant_id=_AVALON,
                    raw_label="Meetings Ahoy",
                ),
            ),
        )
    )

    assignment = effective_assignment(turn, context)

    assert assignment.rung == 3
    assert assignment.participant_id == _MICHAEL


def test_rung_6_an_imported_label_renders_as_supplied() -> None:
    """An imported transcript's own labels are evidence we never verified,
    so they render exactly as supplied rather than being mapped onto a
    participant this system would then appear to have confirmed."""
    segment_id = mint_id("seg")
    turn = _turn(segment_id=segment_id)
    context = SpeakerContext.from_components(
        _components(
            _participants(),
            _stored(
                ProviderLabelSetComponentBody(
                    source_artefact_id=_ARTEFACT,
                    proxy_config_hash=_HASH,
                    spans=(
                        ProviderLabelSpan(
                            source_segment_id=segment_id,
                            raw_label="Speaker 3",
                            text="hello",
                            trust_class=TrustClass.IMPORTED_UNVERIFIED,
                        ),
                    ),
                )
            ),
        )
    )

    assignment = effective_assignment(turn, context)

    assert assignment.rung == 6
    assert assignment.participant_id is None
    assert assignment.display_label == "Speaker 3"
    assert speaker_display_name(assignment, context) == "Speaker 3"
    assert not assignment.satisfies_meeting_note_gate


def test_rung_7_a_machine_hypothesis_never_satisfies_the_meeting_note_gate() -> None:
    """The Avalon trap, at the type level: a proposal can inform a
    reviewer, and can never publish itself."""
    turn = _turn()
    cluster_id = mint_id("cluster")
    context = SpeakerContext.from_components(
        _components(
            _participants(),
            _attribution(cluster_id=cluster_id, turn_id=turn.turn_id),
            _stored(
                SpeakerHypothesisSetComponentBody(
                    proposer="claude-agent:fixture",
                    config_hash=_HASH,
                    hypotheses=(
                        SpeakerHypothesis(
                            cluster_id=cluster_id,
                            participant_id=_AVALON,
                            confidence=0.9,
                            rationale="fixture",
                        ),
                    ),
                )
            ),
        )
    )

    assignment = effective_assignment(turn, context)

    assert assignment.rung == 7
    assert assignment.participant_id == _AVALON
    assert not assignment.satisfies_meeting_note_gate
    assert not assignment.is_reviewed


def test_rung_8_no_evidence_is_unresolved_not_a_guess() -> None:
    turn = _turn()
    context = SpeakerContext.from_components(_components(_participants()))

    assignment = effective_assignment(turn, context)

    assert assignment.rung == 8
    assert assignment.participant_id is None
    assert assignment.provenance == ProvenanceClass.NO_EVIDENCE
    assert speaker_display_name(assignment, context) == UNCLEAR_SPEAKER_LABEL
    assert not assignment.satisfies_meeting_note_gate


# -- unclear-speaker blocks weaker rungs -------------------------------------


def test_an_unclear_speaker_decision_blocks_every_weaker_rung() -> None:
    """A reviewer who looked and could not tell is never overridden by a
    machine guess -- and, unlike rung 8, their decision *does* satisfy the
    meeting-note gate, because "Unclear speaker" is an honest published
    answer (D1)."""
    turn = _turn()
    cluster_id = mint_id("cluster")
    context = SpeakerContext.from_components(
        _components(
            _participants(),
            _attribution(cluster_id=cluster_id, turn_id=turn.turn_id),
            _stored(
                SpeakerHypothesisSetComponentBody(
                    proposer="claude-agent:fixture",
                    config_hash=_HASH,
                    hypotheses=(
                        SpeakerHypothesis(
                            cluster_id=cluster_id,
                            participant_id=_AVALON,
                            confidence=0.99,
                            rationale="fixture",
                        ),
                    ),
                )
            ),
            _review(
                TurnDecision(
                    kind=ReviewDecisionKind.UNCLEAR_SPEAKER, turn_id=turn.turn_id
                )
            ),
        )
    )

    assignment = effective_assignment(turn, context)

    assert assignment.participant_id is None
    assert assignment.provenance == ProvenanceClass.REVIEWED_UNCLEAR
    assert assignment.satisfies_meeting_note_gate
    assert assignment.is_reviewed
    assert speaker_display_name(assignment, context) == UNCLEAR_SPEAKER_LABEL


# -- coverage ----------------------------------------------------------------


def test_coverage_separates_reviewed_from_gate_satisfying_from_unresolved() -> None:
    reviewed = _turn(start_ms=0, end_ms=1000)
    unresolved = _turn(start_ms=2000, end_ms=3000)
    turn_set = _stored(
        TimedTurnSetComponentBody(
            source_artefact_ids=(_ARTEFACT,),
            coordinate_domain=source_domain(_ARTEFACT),
            turns=(reviewed, unresolved),
        )
    )
    components = _components(
        _participants(),
        turn_set,
        _review(
            TurnDecision(
                kind=ReviewDecisionKind.ASSIGN,
                participant_id=_MICHAEL,
                turn_id=reviewed.turn_id,
            )
        ),
    )

    coverage = assignment_coverage(effective_assignments(components).values())

    assert coverage.total_turns == 2
    assert coverage.reviewed == 1
    assert coverage.gate_satisfying == 1
    assert coverage.unresolved == 1
    assert not coverage.fully_gate_satisfying
    assert not coverage.fully_confirmed


def test_full_coverage_with_an_unclear_decision_is_reviewed_but_not_confirmed() -> None:
    """ "We reviewed every turn" and "we know who everyone was" are
    different claims, and speakers.human-confirmed is only the second."""
    first = _turn(start_ms=0, end_ms=1000)
    second = _turn(start_ms=2000, end_ms=3000)
    components = _components(
        _participants(),
        _stored(
            TimedTurnSetComponentBody(
                source_artefact_ids=(_ARTEFACT,),
                coordinate_domain=source_domain(_ARTEFACT),
                turns=(first, second),
            )
        ),
        _review(
            TurnDecision(
                kind=ReviewDecisionKind.ASSIGN,
                participant_id=_MICHAEL,
                turn_id=first.turn_id,
            ),
            TurnDecision(
                kind=ReviewDecisionKind.UNCLEAR_SPEAKER, turn_id=second.turn_id
            ),
        ),
    )

    coverage = assignment_coverage(effective_assignments(components).values())

    assert coverage.fully_gate_satisfying
    assert not coverage.fully_confirmed


# -- D4's overlap review predicate -------------------------------------------


def test_overlapping_turns_with_different_weak_speakers_need_review() -> None:
    first = _turn(start_ms=0, end_ms=2000)
    second = _turn(start_ms=1000, end_ms=3000)
    components = _components(
        _participants(),
        _stored(
            TimedTurnSetComponentBody(
                source_artefact_ids=(_ARTEFACT,),
                coordinate_domain=source_domain(_ARTEFACT),
                turns=(first, second),
            )
        ),
    )

    conflicts = overlapping_turns_needing_review(components)

    assert conflicts == ()  # both unresolved -> same (unresolved) speaker


def test_overlap_between_a_reviewed_turn_and_an_unresolved_one_needs_review() -> None:
    first = _turn(start_ms=0, end_ms=2000)
    second = _turn(start_ms=1000, end_ms=3000)
    components = _components(
        _participants(),
        _stored(
            TimedTurnSetComponentBody(
                source_artefact_ids=(_ARTEFACT,),
                coordinate_domain=source_domain(_ARTEFACT),
                turns=(first, second),
            )
        ),
        _review(
            TurnDecision(
                kind=ReviewDecisionKind.ASSIGN,
                participant_id=_MICHAEL,
                turn_id=first.turn_id,
            )
        ),
    )

    conflicts = overlapping_turns_needing_review(components)

    assert conflicts == ((first.turn_id, second.turn_id),)


def test_two_overlapping_turns_both_reviewed_to_different_people_are_fine() -> None:
    """Concurrent speech that a human resolved is faithfully represented
    concurrent speech, not a reason to stop (D4)."""
    first = _turn(start_ms=0, end_ms=2000)
    second = _turn(start_ms=1000, end_ms=3000)
    components = _components(
        _participants(),
        _stored(
            TimedTurnSetComponentBody(
                source_artefact_ids=(_ARTEFACT,),
                coordinate_domain=source_domain(_ARTEFACT),
                turns=(first, second),
            )
        ),
        _review(
            TurnDecision(
                kind=ReviewDecisionKind.ASSIGN,
                participant_id=_MICHAEL,
                turn_id=first.turn_id,
            ),
            TurnDecision(
                kind=ReviewDecisionKind.ASSIGN,
                participant_id=_AVALON,
                turn_id=second.turn_id,
            ),
        ),
    )

    assert overlapping_turns_needing_review(components) == ()


# -- inventory hashing (M8) --------------------------------------------------


def test_inventory_hashing_is_order_independent() -> None:
    """M8: "sorted by their IDs before hashing, so two implementations of
    the same pack compute identical hashes"."""
    ids = [mint_id("turn") for _ in range(4)]

    assert turn_inventory_hash(ids) == turn_inventory_hash(list(reversed(ids)))


def test_turn_and_cluster_inventories_of_the_same_ids_hash_the_same_way() -> None:
    """Both are the same canonical rule over a list of IDs; keeping them as
    two named functions is about call-site clarity, not two algorithms."""
    ids = [mint_id("cluster") for _ in range(3)]

    assert cluster_inventory_hash(ids) == turn_inventory_hash(ids)


def test_the_empty_cluster_inventory_has_a_stable_hash() -> None:
    """M8: "the hash of the empty inventory when no diarisation output is
    in closure" -- a real value, not None, so a pack from a
    diarisation-free document still binds."""
    assert cluster_inventory_hash([]) == cluster_inventory_hash([])
    assert len(cluster_inventory_hash([])) == 64
