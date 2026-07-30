"""M8: the effective-assignment ladder -- the one function every consumer
of "who said this turn" goes through.

Kept deliberately free of the document projection, the store, and the LLM
seam: the capability registry has to run this to compute speaker coverage,
and ``document.py`` already imports ``registry.py``, so a ladder that
reached back for a ``TranscriptDocumentV1`` would close an import cycle
and drag ``claude_agent_sdk`` into every capability validation. Everything
here works over a resolved component mapping instead -- the same thing a
revision closure hands the registry.

:func:`effective_assignment` is the *one* function every
renderer, gate, and capability payload consumes. Its eight rungs are M8's
precedence order, in full:

1. reviewed turn override
2. reviewed source-range override
3. reviewed cluster default (voice-specific, so it outranks...)
4. reviewed provider-label default (...label defaults, which on a
   room-proxy span describe an aggregate, not a voice)
5. provider attribution with trust class ``per-participant-stream``
6. imported labels (``imported-unverified``) -- rendered as supplied
7. machine hypothesis -- never renders into a meeting note (M5)
8. unassigned -- ``participant_unresolved`` with provenance
   ``no-evidence``, rendered honestly as ``Unclear speaker``

An explicit ``unclear-speaker`` decision at any scope **blocks** every
weaker rung at that scope: a reviewer who looked and could not tell is
never overridden by a machine guess.

Rungs 4-6 are written but inert in v1: they read
:attr:`SpeakerContext.participant_by_provider_label`, which only a
provider-attribution producer can fill, and the local recording path has
none. That is deliberate (they are cheap, and every renderer consumes
this one function) -- but nothing in this module exists *solely* to feed
them, and no room-proxy config surface or imported-label path was built
to make them reachable.

The proposal stage that feeds rung 7 lives in ``speakers.py``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from ..errors import TranscriptError
from .components import (
    ClusterDecision,
    ComponentRecord,
    MachineAttributionSetComponent,
    ParticipantRecord,
    ParticipantSetComponent,
    ProviderLabelDecision,
    ProviderLabelSetComponent,
    ProviderLabelSpan,
    ReviewDecisionKind,
    SourceRangeDecision,
    SpeakerHypothesis,
    SpeakerHypothesisSetComponent,
    SpeakerReviewComponent,
    TimedTurn,
    TimedTurnSetComponent,
    TrustClass,
    TurnDecision,
)
from .ids import ClusterId, ComponentId, ParticipantId, SegmentId, TurnId

#: M1: the distinguished value the ladder *projects* for an unresolved
#: speaker. Never stored as a participant record -- the stored form of
#: reviewer uncertainty is the decision kind ``unclear-speaker``.
PARTICIPANT_UNRESOLVED = "participant_unresolved"

#: What an unresolved speaker is called in rendered output (D1).
UNCLEAR_SPEAKER_LABEL = "Unclear speaker"


class SpeakerError(TranscriptError):
    """Base class for every error the speaker modules raise."""


ComponentMap = Mapping[ComponentId, ComponentRecord]
"""One revision's resolved component graph -- what a closure resolves to,
and the only input the ladder ever reads."""


def _components_of[T](components: ComponentMap, kind: type[T]) -> tuple[T, ...]:
    return tuple(
        component for component in components.values() if isinstance(component, kind)
    )


class ProvenanceClass(StrEnum):
    """Which rung produced an effective assignment -- carried into the
    rendered label so a reader can always tell reviewed fact from machine
    guess (M8: "the winning provenance class carried into the rendered
    label")."""

    REVIEWED_TURN = "reviewed-turn"
    REVIEWED_SOURCE_RANGE = "reviewed-source-range"
    REVIEWED_CLUSTER = "reviewed-cluster"
    REVIEWED_PROVIDER_LABEL = "reviewed-provider-label"
    REVIEWED_UNCLEAR = "reviewed-unclear"
    PROVIDER_PER_PARTICIPANT_STREAM = "provider-per-participant-stream"
    IMPORTED_UNVERIFIED = "imported-unverified"
    MACHINE_HYPOTHESIS = "machine-hypothesis"
    NO_EVIDENCE = "no-evidence"


#: The provenance classes M5's meeting-note gate accepts: a trusted
#: provider stream, a reviewed decision, or the reviewer's explicit
#: ``unclear-speaker`` (which renders as "Unclear speaker" with visible
#: counts, D1). Everything else -- above all a bare machine hypothesis --
#: fails the gate.
GATE_SATISFYING_PROVENANCE: frozenset[ProvenanceClass] = frozenset(
    {
        ProvenanceClass.REVIEWED_TURN,
        ProvenanceClass.REVIEWED_SOURCE_RANGE,
        ProvenanceClass.REVIEWED_CLUSTER,
        ProvenanceClass.REVIEWED_PROVIDER_LABEL,
        ProvenanceClass.REVIEWED_UNCLEAR,
        ProvenanceClass.PROVIDER_PER_PARTICIPANT_STREAM,
    }
)

#: The subset that counts as *reviewed* for M5's assignment-state coverage.
#: A trusted provider stream satisfies the gate but is recorded as
#: provider-attributed, never relabelled "human-reviewed" (D1).
REVIEWED_PROVENANCE: frozenset[ProvenanceClass] = frozenset(
    {
        ProvenanceClass.REVIEWED_TURN,
        ProvenanceClass.REVIEWED_SOURCE_RANGE,
        ProvenanceClass.REVIEWED_CLUSTER,
        ProvenanceClass.REVIEWED_PROVIDER_LABEL,
        ProvenanceClass.REVIEWED_UNCLEAR,
    }
)


@dataclass(frozen=True)
class EffectiveAssignment:
    """One turn's resolved speaker, plus how it was resolved.

    ``participant_id`` is ``None`` exactly when the projection is
    :data:`PARTICIPANT_UNRESOLVED` -- either the reviewer said
    ``unclear-speaker`` (rung 1-4, provenance ``reviewed-unclear``) or no
    rung applied at all (rung 8, provenance ``no-evidence``). Those are
    different facts and stay distinguishable.

    ``display_label`` is set only by rung 6, where the honest rendering is
    the imported label exactly as supplied rather than a participant this
    system never confirmed.
    """

    turn_id: TurnId
    participant_id: ParticipantId | None
    provenance: ProvenanceClass
    rung: int
    display_label: str | None = None

    @property
    def resolved(self) -> bool:
        return self.participant_id is not None or self.display_label is not None

    @property
    def satisfies_meeting_note_gate(self) -> bool:
        return self.provenance in GATE_SATISFYING_PROVENANCE

    @property
    def is_reviewed(self) -> bool:
        return self.provenance in REVIEWED_PROVENANCE


@dataclass(frozen=True)
class SpeakerContext:
    """Everything the ladder reads, resolved once per document.

    Built by :meth:`from_document` so the eight rungs stay a pure function
    of typed inputs -- which is what makes them testable one rung at a
    time, and what stops a renderer from reaching past the ladder into the
    component graph for "just this one case".
    """

    turn_decisions: Mapping[TurnId, TurnDecision]
    range_decisions: tuple[SourceRangeDecision, ...]
    cluster_decisions: Mapping[ClusterId, ClusterDecision]
    provider_label_decisions: Mapping[str, ProviderLabelDecision]
    cluster_by_turn: Mapping[TurnId, ClusterId]
    provider_span_by_segment: Mapping[SegmentId, ProviderLabelSpan]
    participant_by_provider_label: Mapping[str, ParticipantId]
    hypothesis_by_cluster: Mapping[ClusterId, SpeakerHypothesis]
    participants: Mapping[ParticipantId, ParticipantRecord]

    def assignment_for(self, turn: TimedTurn) -> EffectiveAssignment:
        """:func:`effective_assignment` bound to this context -- the shape
        every caller that already holds a context wants, so none of them
        has to re-import the free function to ask the obvious question."""
        return effective_assignment(turn, self)

    @classmethod
    def from_components(cls, components: ComponentMap) -> SpeakerContext:
        turn_decisions: dict[TurnId, TurnDecision] = {}
        range_decisions: list[SourceRangeDecision] = []
        cluster_decisions: dict[ClusterId, ClusterDecision] = {}
        provider_label_decisions: dict[str, ProviderLabelDecision] = {}
        for review in _components_of(components, SpeakerReviewComponent):
            for decision in review.decisions:
                match decision:
                    case TurnDecision():
                        turn_decisions[decision.turn_id] = decision
                    case SourceRangeDecision():
                        range_decisions.append(decision)
                    case ClusterDecision():
                        cluster_decisions[decision.cluster_id] = decision
                    case ProviderLabelDecision():
                        provider_label_decisions[decision.raw_label] = decision

        cluster_by_turn: dict[TurnId, ClusterId] = {}
        for attribution in _components_of(components, MachineAttributionSetComponent):
            for assignment in attribution.assignments:
                cluster_by_turn[assignment.turn_id] = assignment.cluster_id

        provider_span_by_segment: dict[SegmentId, ProviderLabelSpan] = {}
        for label_set in _components_of(components, ProviderLabelSetComponent):
            for span in label_set.spans:
                provider_span_by_segment[span.source_segment_id] = span

        hypothesis_by_cluster: dict[ClusterId, SpeakerHypothesis] = {}
        for hypothesis_set in _components_of(components, SpeakerHypothesisSetComponent):
            for hypothesis in hypothesis_set.hypotheses:
                hypothesis_by_cluster[hypothesis.cluster_id] = hypothesis

        participants: dict[ParticipantId, ParticipantRecord] = {}
        for participant_set in _components_of(components, ParticipantSetComponent):
            for participant in participant_set.participants:
                participants[participant.participant_id] = participant

        return cls(
            turn_decisions=turn_decisions,
            range_decisions=tuple(range_decisions),
            cluster_decisions=cluster_decisions,
            provider_label_decisions=provider_label_decisions,
            cluster_by_turn=cluster_by_turn,
            provider_span_by_segment=provider_span_by_segment,
            # M19 forbids a label->participant edge derived from bare
            # string matching, and v1 has no component that records a
            # confirmed edge, so rungs 5-6 are correctly unreachable on
            # every path this phase can produce. This empty mapping is the
            # seam a provider-attribution producer would fill.
            participant_by_provider_label={},
            hypothesis_by_cluster=hypothesis_by_cluster,
            participants=participants,
        )


def _from_decision(
    turn_id: TurnId,
    decision: TurnDecision
    | SourceRangeDecision
    | ClusterDecision
    | ProviderLabelDecision,
    *,
    provenance: ProvenanceClass,
    rung: int,
) -> EffectiveAssignment:
    """Project one matched review decision onto a turn.

    ``unclear-speaker`` collapses to ``participant_unresolved`` with
    provenance ``reviewed-unclear`` regardless of which scope matched --
    the *blocking* behaviour M8 requires is simply that the ladder returns
    here rather than falling through to a weaker rung.
    """
    if decision.kind == ReviewDecisionKind.UNCLEAR_SPEAKER:
        return EffectiveAssignment(
            turn_id=turn_id,
            participant_id=None,
            provenance=ProvenanceClass.REVIEWED_UNCLEAR,
            rung=rung,
        )
    return EffectiveAssignment(
        turn_id=turn_id,
        participant_id=decision.participant_id,
        provenance=provenance,
        rung=rung,
    )


def effective_assignment(
    turn: TimedTurn, context: SpeakerContext
) -> EffectiveAssignment:
    """M8's eight-rung precedence, evaluated for one canonical turn.

    The single function every consumer goes through. Rungs are tried in
    order and the first match wins; there is no scoring, no blending, and
    no tie-break -- which is what makes the result reproducible and
    explainable ("this turn is Michael because *you* said so at turn
    scope", not "because the model was 0.62 sure").
    """
    decision = context.turn_decisions.get(turn.turn_id)
    if decision is not None:
        return _from_decision(
            turn.turn_id,
            decision,
            provenance=ProvenanceClass.REVIEWED_TURN,
            rung=1,
        )

    for range_decision in context.range_decisions:
        if range_decision.source_artefact_id != turn.source_artefact_id:
            continue
        if range_decision.start_ms <= turn.start_ms < range_decision.end_ms:
            return _from_decision(
                turn.turn_id,
                range_decision,
                provenance=ProvenanceClass.REVIEWED_SOURCE_RANGE,
                rung=2,
            )

    cluster_id = context.cluster_by_turn.get(turn.turn_id)
    if cluster_id is not None:
        cluster_decision = context.cluster_decisions.get(cluster_id)
        if cluster_decision is not None:
            return _from_decision(
                turn.turn_id,
                cluster_decision,
                provenance=ProvenanceClass.REVIEWED_CLUSTER,
                rung=3,
            )

    span = context.provider_span_by_segment.get(turn.source_segment_id)
    if span is not None:
        label_decision = context.provider_label_decisions.get(span.raw_label)
        if label_decision is not None:
            return _from_decision(
                turn.turn_id,
                label_decision,
                provenance=ProvenanceClass.REVIEWED_PROVIDER_LABEL,
                rung=4,
            )
        mapped = context.participant_by_provider_label.get(span.raw_label)
        if mapped is not None and span.trust_class == TrustClass.PER_PARTICIPANT_STREAM:
            return EffectiveAssignment(
                turn_id=turn.turn_id,
                participant_id=mapped,
                provenance=ProvenanceClass.PROVIDER_PER_PARTICIPANT_STREAM,
                rung=5,
            )
        if span.trust_class == TrustClass.IMPORTED_UNVERIFIED:
            return EffectiveAssignment(
                turn_id=turn.turn_id,
                participant_id=None,
                provenance=ProvenanceClass.IMPORTED_UNVERIFIED,
                rung=6,
                display_label=span.raw_label,
            )

    if cluster_id is not None:
        hypothesis = context.hypothesis_by_cluster.get(cluster_id)
        if hypothesis is not None and hypothesis.participant_id is not None:
            return EffectiveAssignment(
                turn_id=turn.turn_id,
                participant_id=hypothesis.participant_id,
                provenance=ProvenanceClass.MACHINE_HYPOTHESIS,
                rung=7,
            )

    return EffectiveAssignment(
        turn_id=turn.turn_id,
        participant_id=None,
        provenance=ProvenanceClass.NO_EVIDENCE,
        rung=8,
    )


def canonical_turn_set(components: ComponentMap) -> TimedTurnSetComponent | None:
    """The one canonical timed turn set, or ``None``.

    ``transcript.timed`` is a one-cardinality key (M4), so anything other
    than exactly one candidate is an ambiguity the registry reports as a
    failure -- and every consumer here treats it as "no canonical turns"
    rather than picking one arbitrarily.
    """
    turn_sets = _components_of(components, TimedTurnSetComponent)
    if len(turn_sets) != 1:
        return None
    return turn_sets[0]


def canonical_turns(components: ComponentMap) -> tuple[TimedTurn, ...]:
    """The canonical timed turns, in M6 canonical order.

    Returns empty rather than raising when there is no timed turn set: an
    untimed or notes-only document legitimately has none, and the callers
    here all treat "no turns" as a status, not an error.
    """
    turn_set = canonical_turn_set(components)
    return () if turn_set is None else turn_set.turns


def effective_assignments(
    components: ComponentMap,
) -> Mapping[TurnId, EffectiveAssignment]:
    context = SpeakerContext.from_components(components)
    return {
        turn.turn_id: effective_assignment(turn, context)
        for turn in canonical_turns(components)
    }


@dataclass(frozen=True)
class AssignmentCoverage:
    """M5: coverage fraction over canonical turns plus an unresolved count.

    ``gate_satisfying`` is what the meeting-note profile checks; ``reviewed``
    is what ``speakers.human-reviewed`` reports. They differ exactly where
    a trusted provider stream carries a turn -- gate-satisfying, but never
    counted as reviewed (D1).
    """

    total_turns: int
    reviewed: int
    gate_satisfying: int
    unresolved: int
    by_provenance: Mapping[ProvenanceClass, int]

    @property
    def fully_gate_satisfying(self) -> bool:
        return self.total_turns > 0 and self.gate_satisfying == self.total_turns

    @property
    def fully_confirmed(self) -> bool:
        """M4 ``speakers.human-confirmed``: every turn reviewed *and*
        resolved to a participant -- an explicit ``unclear-speaker`` is a
        legitimate review outcome but is not a confirmed assignment."""
        return (
            self.total_turns > 0
            and self.reviewed == self.total_turns
            and self.unresolved == 0
        )


def assignment_coverage(
    assignments: Iterable[EffectiveAssignment],
) -> AssignmentCoverage:
    by_provenance: dict[ProvenanceClass, int] = {}
    total = reviewed = gate_satisfying = unresolved = 0
    for assignment in assignments:
        total += 1
        by_provenance[assignment.provenance] = (
            by_provenance.get(assignment.provenance, 0) + 1
        )
        if assignment.is_reviewed:
            reviewed += 1
        if assignment.satisfies_meeting_note_gate:
            gate_satisfying += 1
        if not assignment.resolved:
            unresolved += 1
    return AssignmentCoverage(
        total_turns=total,
        reviewed=reviewed,
        gate_satisfying=gate_satisfying,
        unresolved=unresolved,
        by_provenance=by_provenance,
    )


def speaker_display_name(
    assignment: EffectiveAssignment, context: SpeakerContext
) -> str:
    """The label a renderer prints for one turn.

    Unresolved turns -- reviewed-unclear and no-evidence alike -- render as
    :data:`UNCLEAR_SPEAKER_LABEL`; a rung-6 imported label renders exactly
    as supplied. Nothing here ever falls back to a raw cluster label:
    ``SPEAKER_01`` in a meeting note reads as a speaker's name to a human
    and is precisely the machine-cluster-as-identity confusion F13
    forbids.
    """
    if assignment.display_label is not None:
        return assignment.display_label
    if assignment.participant_id is None:
        return UNCLEAR_SPEAKER_LABEL
    participant = context.participants.get(assignment.participant_id)
    if participant is None:
        return UNCLEAR_SPEAKER_LABEL
    return participant.display_names[0]


def overlapping_turns_needing_review(
    components: ComponentMap,
) -> tuple[tuple[TurnId, TurnId], ...]:
    """M6/D4's closed ``review_required`` predicate for concurrent speech.

    Fires iff two canonical turns overlap in time AND their effective
    speakers differ AND at least one of the two assignments is weaker than
    trusted/reviewed (rung 6, 7, or 8). Two overlapping turns that a human
    has already resolved to two different people are simply concurrent
    speech, faithfully represented -- not a reason to stop.
    """
    turns = canonical_turns(components)
    assignments = effective_assignments(components)
    conflicts: list[tuple[TurnId, TurnId]] = []
    for index, earlier in enumerate(turns):
        for later in turns[index + 1 :]:
            if later.start_ms >= earlier.end_ms:
                break
            first = assignments[earlier.turn_id]
            second = assignments[later.turn_id]
            if (first.participant_id, first.display_label) == (
                second.participant_id,
                second.display_label,
            ):
                continue
            if first.rung >= 6 or second.rung >= 6:
                conflicts.append((earlier.turn_id, later.turn_id))
    return tuple(conflicts)


def turn_inventory_hash(turn_ids: Sequence[str]) -> str:
    """M8: canonical inventory hashing -- IDs sorted, then hashed.

    Deliberately over IDs only, never over turn *text*: M7's identity
    remap keeps ``turn_id`` stable across text-only edits, so a review
    applied before routine polish must still bind afterwards. Hashing text
    here would invalidate every review the moment a filler word was
    removed.
    """
    return hashlib.sha256(
        json.dumps(sorted(turn_ids), separators=(",", ":")).encode()
    ).hexdigest()


def cluster_inventory_hash(cluster_ids: Sequence[str]) -> str:
    """M8's cluster inventory, same rule -- and the hash of the empty
    inventory when no diarisation output is in closure."""
    return hashlib.sha256(
        json.dumps(sorted(cluster_ids), separators=(",", ":")).encode()
    ).hexdigest()
