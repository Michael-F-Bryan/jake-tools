"""Capability registry v1 (M4).

A capability is a *proof*, never an assertion: it is emitted only by a
validator that actually inspected the referenced component(s), never by a
stage, a filename, or an operator's say-so. This module is the closed table
of what can be proven in v1 (:class:`CapabilityKey`), the typed payload
every proof is recorded in (:class:`CapabilityRecord`), and :func:`validate`
-- the single dispatcher every consumer goes through.

v1 can now *prove* ten things: ``notes.provider``/``notes.authored`` (M20),
``participants.declared`` (M19), ``transcript.untimed``/``transcript.timed``
(M6/M7/D6), ``speakers.provider-labels`` (M5, evidence only -- it
satisfies no product gate), ``media.recording`` (M3/M4/M11: an ingested
media artefact with duration, matched against a note's own recording
references), ``timeline.combined`` (M6: the assembled multi-recording
mapping), and ``inference.asr``/``inference.diarisation`` (M11: raw worker
output -- a component exists only for a *completed* stage, so a failed
sibling never blocks the other's own proof). Every other key is
registered with a stub validator that always returns ``not-attempted`` and
is structurally incapable of returning ``present-validated`` --
:func:`validate` raises if one ever tries, which is the "capabilities are
proofs" rule enforced as code, not just as prose.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ..errors import TranscriptError
from .assignment import (
    AssignmentCoverage,
    assignment_coverage,
    canonical_turn_set,
    canonical_turns,
    effective_assignments,
)
from .components import (
    AsrResultComponent,
    ChapterSetComponent,
    ClusterDecision,
    ComponentRecord,
    DiarisationResultComponent,
    MachineAttributionSetComponent,
    MediaRecordingComponent,
    MinutesComponent,
    NotesComponent,
    ParticipantSetComponent,
    ProviderLabelSetComponent,
    RecordingReferenceSetComponent,
    SpeakerReviewComponent,
    TextEditLedgerComponent,
    TextEditMode,
    TextEditOperation,
    TimedTurnSetComponent,
    TimelineCombinedComponent,
    TranscriptAbsenceDeclaration,
    TrustClass,
    TurnDecision,
    UntimedTurnSetComponent,
)
from .ids import ComponentId, RevisionId

_VALIDATOR_VERSION = "v1"


class RegistryError(TranscriptError):
    """Base class for every error this module raises."""


class StubEmittedPresentValidatedError(RegistryError):
    """A not-yet-implemented key's stub validator returned present-validated.

    Capabilities are proofs (M4): a stub -- registered precisely because
    v1 has no way to prove this key -- must never be allowed to fake one,
    even by accident. This is the safety net that catches it.
    """


class ValidatorKeyMismatchError(RegistryError):
    """A validator returned a record for a different key than it was
    registered under -- almost certainly a copy-paste bug in the registry
    table, caught here rather than silently mislabelling a proof.
    """


class PrerequisiteNotSatisfiedError(RegistryError):
    """A validator returned present-validated while a declared prerequisite
    key was not itself present-validated (M4: "prerequisites are
    validation preconditions"). This is a validator-authoring error, not
    a capability status -- raised, never silently downgraded to a status,
    so a validator can never claim more than its own prerequisites support.
    """


class RegistryConfigurationError(RegistryError):
    """The registry table itself is malformed.

    Raised at import time (from :func:`_build_registry`) rather than
    letting a bad table surface as a ``KeyError`` the first time
    :func:`validate` happens to need the missing ``prerequisite_statuses``
    entry -- e.g. a key declaring a prerequisite that the M4 table order
    places *after* it, which :func:`validate`'s single dispatch pass
    (table order) could never have computed yet.
    """


class CapabilityKey(StrEnum):
    """The M4 table: the closed, code-defined set of capabilities v1 knows
    about. Nothing outside this enum is a capability -- adding one is a
    contract change (a new dated CONTRACTS.md entry), not a runtime choice.
    """

    NOTES_PROVIDER = "notes.provider"
    NOTES_AUTHORED = "notes.authored"
    PARTICIPANTS_DECLARED = "participants.declared"
    MEDIA_RECORDING = "media.recording"
    TRANSCRIPT_UNTIMED = "transcript.untimed"
    TRANSCRIPT_TIMED = "transcript.timed"
    SPEAKERS_PROVIDER_LABELS = "speakers.provider-labels"
    SPEAKERS_PROVIDER_ATTRIBUTED = "speakers.provider-attributed"
    SPEAKERS_MACHINE_CLUSTERED = "speakers.machine-clustered"
    SPEAKERS_HUMAN_REVIEWED = "speakers.human-reviewed"
    SPEAKERS_HUMAN_CONFIRMED = "speakers.human-confirmed"
    TEXT_CORRECTED = "text.corrected"
    TEXT_POLISHED = "text.polished"
    CHAPTERS = "chapters"
    MINUTES = "minutes"
    TIMELINE_COMBINED = "timeline.combined"
    INFERENCE_ASR = "inference.asr"
    INFERENCE_DIARISATION = "inference.diarisation"


class CapabilityCardinality(StrEnum):
    """M4: declared per key -- ``many`` keys are proven per source/
    recording member; ``one`` keys are proven once for the whole document.
    """

    ONE = "one"
    MANY = "many"


#: M4: "many: media.recording, notes.provider, notes.authored,
#: speakers.provider-labels, inference.asr, inference.diarisation
#: (one per source/recording)." Every other key is ``one``.
_MANY_KEYS: frozenset[CapabilityKey] = frozenset(
    {
        CapabilityKey.MEDIA_RECORDING,
        CapabilityKey.NOTES_PROVIDER,
        CapabilityKey.NOTES_AUTHORED,
        CapabilityKey.SPEAKERS_PROVIDER_LABELS,
        CapabilityKey.INFERENCE_ASR,
        CapabilityKey.INFERENCE_DIARISATION,
    }
)


def cardinality_of(key: CapabilityKey) -> CapabilityCardinality:
    return (
        CapabilityCardinality.MANY if key in _MANY_KEYS else CapabilityCardinality.ONE
    )


class CapabilityStatus(StrEnum):
    """M4's six statuses. Only ``present-validated`` satisfies a consumer;
    none of the others silently converts to an empty component.
    """

    PRESENT_VALIDATED = "present-validated"
    ABSENT = "absent"
    NOT_APPLICABLE = "not-applicable"
    NOT_AVAILABLE_FROM_SOURCE = "not-available-from-source"
    FAILED = "failed"
    NOT_ATTEMPTED = "not-attempted"


class TrustClassCoverage(BaseModel):
    """M5: per-trust-class coverage breakdown on a speaker capability
    payload. Populated by ``speakers.provider-labels``' validator below.
    """

    model_config = ConfigDict(frozen=True)

    trust_class: TrustClass
    covered_turn_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)


class CapabilityMemberStatus(BaseModel):
    """One member's status within a ``many``-cardinality capability (M4).

    ``member_id`` identifies the member within its key's own scope (for
    the notes keys: the member's own ``component_id``, since a single
    source can produce several distinct notes components -- summary,
    decisions, actions -- that a renderer must be able to tell apart).
    """

    model_config = ConfigDict(frozen=True)

    member_id: str = Field(min_length=1)
    status: CapabilityStatus
    component_id: ComponentId | None = None
    detail: str = ""


class CapabilityRecord(BaseModel):
    """M4's typed capability payload: component ref(s), coverage set,
    input revision IDs, provenance class(es), validator version -- plus
    the M5 speaker-only fields (shape only, see :class:`TrustClassCoverage`).

    ``members`` is populated only for ``many``-cardinality keys; a ``one``
    key's proof lives entirely in the top-level ``status``.
    ``failure_detail`` names *why* a ``one``-cardinality key is not
    present-validated (e.g. "no participants" vs "two candidate
    components") so that distinction is diagnosable straight from the
    record -- a ``many`` key's equivalent detail lives per-member instead.
    """

    model_config = ConfigDict(frozen=True)

    key: CapabilityKey
    status: CapabilityStatus
    component_ids: tuple[ComponentId, ...] = ()
    coverage: tuple[str, ...] = ()
    input_revision_ids: tuple[RevisionId, ...] = ()
    provenance_classes: tuple[str, ...] = ()
    validator_version: str = Field(min_length=1)
    members: tuple[CapabilityMemberStatus, ...] = ()
    failure_detail: str = ""
    trust_class_coverage: tuple[TrustClassCoverage, ...] = ()
    proxy_config_hash: str | None = None


@dataclasses.dataclass(frozen=True)
class ValidationContext:
    """What a validator callable gets to look at.

    ``prerequisite_statuses`` carries the already-computed status of every
    key this key's :class:`RegistryEntry` names as a prerequisite (M4:
    "prerequisites are validation preconditions, not entailment") --
    populated for every key, even ones (like ``chapters`` in this phase)
    whose validator does not yet look at it.
    """

    revision_id: RevisionId
    components: Mapping[ComponentId, ComponentRecord]
    prerequisite_statuses: Mapping[CapabilityKey, CapabilityStatus]


CapabilityValidatorFn = Callable[[ValidationContext], CapabilityRecord]


@dataclasses.dataclass(frozen=True)
class RegistryEntry:
    key: CapabilityKey
    cardinality: CapabilityCardinality
    validator: CapabilityValidatorFn
    prerequisites: tuple[CapabilityKey, ...] = ()
    implemented: bool = False


# -- shared many-key aggregation (MAJOR 4) -----------------------------------


def _aggregate_many_key_status(
    member_statuses: Iterable[CapabilityStatus],
) -> CapabilityStatus:
    """M4: summarise a many-key's members without hiding a genuine failure
    or blocking on a merely-absent one.

    ``present-validated`` only if *every* member validates; ``failed`` if
    *any* member genuinely failed (never smoothed over). Otherwise -- no
    failures, but not everything validated either, e.g. M4's own worked
    example "two validated recordings and one reference-only embed" -- the
    aggregate degrades to the shared non-validated status when every
    remaining member agrees on one, so a caller reading only the
    top-level status still gets an honest label. A genuinely mixed bag of
    *different* non-failed statuses has no single honest label at the
    aggregate level, so it falls back to ``failed`` as the conservative
    choice; this never blocks anything by itself, because the seam
    (``document.capability_validating_seam``) gates on member-level
    ``failed`` only -- ``capability_members`` is the actual consumer path
    M4 mandates for many keys, the aggregate is a summary.
    """
    statuses = list(member_statuses)
    if not statuses:
        return CapabilityStatus.ABSENT
    if all(status == CapabilityStatus.PRESENT_VALIDATED for status in statuses):
        return CapabilityStatus.PRESENT_VALIDATED
    if any(status == CapabilityStatus.FAILED for status in statuses):
        return CapabilityStatus.FAILED
    degraded = {
        status for status in statuses if status != CapabilityStatus.PRESENT_VALIDATED
    }
    if len(degraded) == 1:
        return next(iter(degraded))
    return CapabilityStatus.FAILED


# -- real validators: notes.provider / notes.authored (M20) -----------------


def _duplicate_section_ids(components: Iterable[ComponentRecord]) -> frozenset[str]:
    """MAJOR 6: section IDs are one namespace across every notes component
    in the closure -- M10 evidence refs cite a section ID alone, not a
    (component, section) pair -- so uniqueness is checked globally, not
    just within a single component.
    """
    seen: set[str] = set()
    duplicates: set[str] = set()
    for component in components:
        if not isinstance(component, NotesComponent):
            continue
        for section in component.sections:
            if section.section_id in seen:
                duplicates.add(section.section_id)
            seen.add(section.section_id)
    return frozenset(duplicates)


def _validate_one_notes_component(
    component: NotesComponent, duplicate_section_ids: frozenset[str]
) -> tuple[CapabilityStatus, str]:
    section_ids = [section.section_id for section in component.sections]
    if len(set(section_ids)) != len(section_ids):
        return CapabilityStatus.FAILED, "duplicate section IDs within this component"
    collisions = duplicate_section_ids.intersection(section_ids)
    if collisions:
        return (
            CapabilityStatus.FAILED,
            f"section ID(s) shared with another notes component: {sorted(collisions)}",
        )
    return CapabilityStatus.PRESENT_VALIDATED, ""


def _make_notes_validator(
    *, key: CapabilityKey, authored: bool
) -> CapabilityValidatorFn:
    def _validate(context: ValidationContext) -> CapabilityRecord:
        matching = [
            component
            for component in context.components.values()
            if isinstance(component, NotesComponent) and component.authored == authored
        ]
        if not matching:
            return CapabilityRecord(
                key=key,
                status=CapabilityStatus.ABSENT,
                validator_version=_VALIDATOR_VERSION,
            )

        duplicate_section_ids = _duplicate_section_ids(context.components.values())
        members = []
        for component in matching:
            status, detail = _validate_one_notes_component(
                component, duplicate_section_ids
            )
            members.append(
                CapabilityMemberStatus(
                    member_id=component.component_id,
                    status=status,
                    component_id=component.component_id,
                    detail=detail,
                )
            )
        overall = _aggregate_many_key_status(member.status for member in members)
        return CapabilityRecord(
            key=key,
            status=overall,
            component_ids=tuple(component.component_id for component in matching),
            input_revision_ids=(context.revision_id,),
            provenance_classes=("authored",) if authored else ("provider",),
            validator_version=_VALIDATOR_VERSION,
            members=tuple(members),
        )

    return _validate


# -- real validator: participants.declared (M19) -----------------------------


def _validate_participants_declared(context: ValidationContext) -> CapabilityRecord:
    matching = [
        component
        for component in context.components.values()
        if isinstance(component, ParticipantSetComponent)
    ]
    if not matching:
        return CapabilityRecord(
            key=CapabilityKey.PARTICIPANTS_DECLARED,
            status=CapabilityStatus.ABSENT,
            validator_version=_VALIDATOR_VERSION,
        )
    if len(matching) > 1:
        # participants.declared is a `one`-cardinality key (M4): more than
        # one candidate component in the closure is an ambiguous input,
        # not a case to silently pick-first from.
        return CapabilityRecord(
            key=CapabilityKey.PARTICIPANTS_DECLARED,
            status=CapabilityStatus.FAILED,
            component_ids=tuple(component.component_id for component in matching),
            validator_version=_VALIDATOR_VERSION,
            failure_detail=(
                f"{len(matching)} participant-set components present for a "
                "one-cardinality key"
            ),
        )

    component = matching[0]
    provenance_classes = tuple(
        sorted(
            {
                participant.declaration_source.value
                for participant in component.participants
            }
        )
    )
    return CapabilityRecord(
        key=CapabilityKey.PARTICIPANTS_DECLARED,
        status=CapabilityStatus.PRESENT_VALIDATED,
        component_ids=(component.component_id,),
        input_revision_ids=(context.revision_id,),
        provenance_classes=provenance_classes,
        validator_version=_VALIDATOR_VERSION,
    )


# -- shared: D2/M5 "the source declares no transcript" signal ----------------


def _absence_declaration_status(
    components: Mapping[ComponentId, ComponentRecord],
) -> tuple[CapabilityStatus, tuple[ComponentId, ...], str]:
    """D2/M5: when no turn-set component exists, decide between the two
    honest reasons why -- ``absent`` (genuinely no evidence either way) vs
    ``not-available-from-source`` (the source explicitly states no
    transcript exists, carried as a :class:`TranscriptAbsenceDeclaration`
    in the closure) -- never silently defaulting to ``absent`` and losing
    that distinction (D2: "never absent, because the absence statement is
    itself the evidence the product must preserve"). More than one
    declaration in the closure is ambiguous, not a case to pick-first
    from.
    """
    declarations = [
        component
        for component in components.values()
        if isinstance(component, TranscriptAbsenceDeclaration)
    ]
    if len(declarations) > 1:
        return (
            CapabilityStatus.FAILED,
            tuple(component.component_id for component in declarations),
            (
                f"{len(declarations)} transcript-absence declarations present for "
                "a one-cardinality signal"
            ),
        )
    if declarations:
        return (
            CapabilityStatus.NOT_AVAILABLE_FROM_SOURCE,
            (declarations[0].component_id,),
            "",
        )
    return CapabilityStatus.ABSENT, (), ""


# -- real validators: transcript.untimed / transcript.timed (M6/M7/D6) ------
#
# Unlike notes section IDs (one namespace across the whole closure, M10),
# turn segment lineage is only ever compared within its own turn-set
# component in this phase -- each validator below checks that inline.


def _validate_transcript_untimed(context: ValidationContext) -> CapabilityRecord:
    matching = [
        component
        for component in context.components.values()
        if isinstance(component, UntimedTurnSetComponent)
    ]
    if not matching:
        status, component_ids, detail = _absence_declaration_status(context.components)
        return CapabilityRecord(
            key=CapabilityKey.TRANSCRIPT_UNTIMED,
            status=status,
            component_ids=component_ids,
            failure_detail=detail,
            validator_version=_VALIDATOR_VERSION,
        )
    if len(matching) > 1:
        # transcript.untimed is a `one`-cardinality key (M4): more than one
        # candidate untimed turn set in the closure is ambiguous, not a
        # case to silently pick-first from.
        return CapabilityRecord(
            key=CapabilityKey.TRANSCRIPT_UNTIMED,
            status=CapabilityStatus.FAILED,
            component_ids=tuple(component.component_id for component in matching),
            validator_version=_VALIDATOR_VERSION,
            failure_detail=(
                f"{len(matching)} untimed turn-set components present for a "
                "one-cardinality key"
            ),
        )

    component = matching[0]
    segment_ids = [turn.source_segment_id for turn in component.turns]
    if len(set(segment_ids)) != len(segment_ids):
        return CapabilityRecord(
            key=CapabilityKey.TRANSCRIPT_UNTIMED,
            status=CapabilityStatus.FAILED,
            component_ids=(component.component_id,),
            validator_version=_VALIDATOR_VERSION,
            failure_detail=(
                "duplicate source_segment_id across turns; segment lineage does "
                "not resolve to distinct raw segments"
            ),
        )
    # "No timing fields present" is enforced by UntimedTurn's own type
    # shape (components.py) -- there is no start_ms/end_ms field this
    # validator could find even if it looked, so there is nothing further
    # to check for that half of the M4 table entry here.
    return CapabilityRecord(
        key=CapabilityKey.TRANSCRIPT_UNTIMED,
        status=CapabilityStatus.PRESENT_VALIDATED,
        component_ids=(component.component_id,),
        input_revision_ids=(context.revision_id,),
        validator_version=_VALIDATOR_VERSION,
    )


def _validate_transcript_timed(context: ValidationContext) -> CapabilityRecord:
    matching = [
        component
        for component in context.components.values()
        if isinstance(component, TimedTurnSetComponent)
    ]
    if not matching:
        status, component_ids, detail = _absence_declaration_status(context.components)
        return CapabilityRecord(
            key=CapabilityKey.TRANSCRIPT_TIMED,
            status=status,
            component_ids=component_ids,
            failure_detail=detail,
            validator_version=_VALIDATOR_VERSION,
        )
    if len(matching) > 1:
        return CapabilityRecord(
            key=CapabilityKey.TRANSCRIPT_TIMED,
            status=CapabilityStatus.FAILED,
            component_ids=tuple(component.component_id for component in matching),
            validator_version=_VALIDATOR_VERSION,
            failure_detail=(
                f"{len(matching)} timed turn-set components present for a "
                "one-cardinality key"
            ),
        )

    component = matching[0]
    segment_ids = [turn.source_segment_id for turn in component.turns]
    if len(set(segment_ids)) != len(segment_ids):
        return CapabilityRecord(
            key=CapabilityKey.TRANSCRIPT_TIMED,
            status=CapabilityStatus.FAILED,
            component_ids=(component.component_id,),
            validator_version=_VALIDATOR_VERSION,
            failure_detail=(
                "duplicate source_segment_id across turns; segment lineage does "
                "not resolve to distinct raw segments"
            ),
        )
    # Canonical order and half-open spans are enforced by
    # TimedTurnSetComponentBody/TimedTurn's own validators (components.py)
    # at construction *and* re-checked on every load from disk (the
    # Component subclasses the Body, so it inherits those validators) --
    # a component that reached this point already carries both proofs.
    return CapabilityRecord(
        key=CapabilityKey.TRANSCRIPT_TIMED,
        status=CapabilityStatus.PRESENT_VALIDATED,
        component_ids=(component.component_id,),
        input_revision_ids=(context.revision_id,),
        validator_version=_VALIDATOR_VERSION,
    )


# -- real validator: speakers.provider-labels (M5) -- evidence only ---------


def _validate_speakers_provider_labels(context: ValidationContext) -> CapabilityRecord:
    matching = [
        component
        for component in context.components.values()
        if isinstance(component, ProviderLabelSetComponent)
    ]
    if not matching:
        return CapabilityRecord(
            key=CapabilityKey.SPEAKERS_PROVIDER_LABELS,
            status=CapabilityStatus.ABSENT,
            validator_version=_VALIDATOR_VERSION,
        )

    known_cue_segment_ids = {
        turn.source_segment_id
        for component in context.components.values()
        if isinstance(component, TimedTurnSetComponent)
        for turn in component.turns
    }
    proxy_hashes = {component.proxy_config_hash for component in matching}
    inconsistent_proxy = len(proxy_hashes) > 1

    members: list[CapabilityMemberStatus] = []
    trust_counts: dict[TrustClass, int] = {}
    for component in matching:
        if inconsistent_proxy:
            status = CapabilityStatus.FAILED
            detail = "inconsistent proxy_config_hash across provider-label-set members"
        else:
            unresolved = sorted(
                span.source_segment_id
                for span in component.spans
                if span.source_segment_id not in known_cue_segment_ids
            )
            if unresolved:
                status = CapabilityStatus.FAILED
                detail = (
                    f"span(s) do not resolve to a cue in this closure: {unresolved}"
                )
            else:
                status, detail = CapabilityStatus.PRESENT_VALIDATED, ""
                for span in component.spans:
                    trust_counts[span.trust_class] = (
                        trust_counts.get(span.trust_class, 0) + 1
                    )
        members.append(
            CapabilityMemberStatus(
                member_id=component.component_id,
                status=status,
                component_id=component.component_id,
                detail=detail,
            )
        )

    overall = _aggregate_many_key_status(member.status for member in members)
    trust_class_coverage = tuple(
        TrustClassCoverage(
            trust_class=trust_class, covered_turn_count=count, unresolved_count=0
        )
        for trust_class, count in sorted(
            trust_counts.items(), key=lambda pair: pair[0].value
        )
    )
    return CapabilityRecord(
        key=CapabilityKey.SPEAKERS_PROVIDER_LABELS,
        status=overall,
        component_ids=tuple(component.component_id for component in matching),
        input_revision_ids=(context.revision_id,),
        validator_version=_VALIDATOR_VERSION,
        members=tuple(members),
        trust_class_coverage=trust_class_coverage,
        proxy_config_hash=next(iter(proxy_hashes)) if len(proxy_hashes) == 1 else None,
    )


# -- real validator: media.recording (M3/M4/M11) -----------------------------


def _validate_media_recording(context: ValidationContext) -> CapabilityRecord:
    """M4: an ingested media artefact with duration proves this key, per
    member. A note's own recording references (:class:`RecordingReferenceSetComponent`,
    M3) also contribute members -- matched against an ingested
    :class:`MediaRecordingComponent` by resolved filesystem path -- so a
    reference whose target was never separately ingested still shows up
    as its own member, honestly ``not-available-from-source`` (M3: "a
    reference alone never satisfies media.recording"), exactly M4's own
    worked example ("two validated recordings and one reference-only
    embed").
    """
    media_components = [
        component
        for component in context.components.values()
        if isinstance(component, MediaRecordingComponent)
    ]
    reference_sets = [
        component
        for component in context.components.values()
        if isinstance(component, RecordingReferenceSetComponent)
    ]
    if not media_components and not reference_sets:
        return CapabilityRecord(
            key=CapabilityKey.MEDIA_RECORDING,
            status=CapabilityStatus.ABSENT,
            validator_version=_VALIDATOR_VERSION,
        )

    media_by_path = {component.media_path: component for component in media_components}
    referenced_paths: set[str] = set()
    members: list[CapabilityMemberStatus] = []
    for reference_set in reference_sets:
        for reference in reference_set.references:
            referenced_paths.add(reference.resolved_path)
            matched = media_by_path.get(reference.resolved_path)
            if matched is None:
                members.append(
                    CapabilityMemberStatus(
                        member_id=reference.resolved_path,
                        status=CapabilityStatus.NOT_AVAILABLE_FROM_SOURCE,
                        detail=(
                            "embed reference was never separately ingested as "
                            "media (M3)"
                        ),
                    )
                )
            else:
                members.append(
                    CapabilityMemberStatus(
                        member_id=reference.resolved_path,
                        status=CapabilityStatus.PRESENT_VALIDATED,
                        component_id=matched.component_id,
                    )
                )
    # Media ingested standalone (e.g. a bare `local-media` ingest with no
    # accompanying note reference, or a note ingested separately from its
    # media) is still a usable member in its own right (M3's
    # operator-assertion path).
    for media in media_components:
        if media.media_path not in referenced_paths:
            members.append(
                CapabilityMemberStatus(
                    member_id=media.media_path,
                    status=CapabilityStatus.PRESENT_VALIDATED,
                    component_id=media.component_id,
                )
            )

    overall = _aggregate_many_key_status(member.status for member in members)
    return CapabilityRecord(
        key=CapabilityKey.MEDIA_RECORDING,
        status=overall,
        component_ids=tuple(component.component_id for component in media_components),
        input_revision_ids=(context.revision_id,),
        validator_version=_VALIDATOR_VERSION,
        members=tuple(members),
    )


# -- real validator: timeline.combined (M6) ----------------------------------


def _validate_timeline_combined(context: ValidationContext) -> CapabilityRecord:
    matching = [
        component
        for component in context.components.values()
        if isinstance(component, TimelineCombinedComponent)
    ]
    if not matching:
        return CapabilityRecord(
            key=CapabilityKey.TIMELINE_COMBINED,
            status=CapabilityStatus.ABSENT,
            validator_version=_VALIDATOR_VERSION,
        )
    if len(matching) > 1:
        # timeline.combined is a `one`-cardinality key (M4): more than one
        # candidate combined-timeline component is ambiguous, not a case
        # to silently pick-first from.
        return CapabilityRecord(
            key=CapabilityKey.TIMELINE_COMBINED,
            status=CapabilityStatus.FAILED,
            component_ids=tuple(component.component_id for component in matching),
            validator_version=_VALIDATOR_VERSION,
            failure_detail=(
                f"{len(matching)} combined-timeline components present for a "
                "one-cardinality key"
            ),
        )

    component = matching[0]
    # Sequential, non-overlapping ordering and span-length preservation
    # are enforced by TimelineCombinedComponentBody/TimelineMappingSegment's
    # own validators (components.py) at construction *and* re-checked on
    # every load from disk (the Component subclasses the Body) -- a
    # component that reached this point already carries both proofs.
    return CapabilityRecord(
        key=CapabilityKey.TIMELINE_COMBINED,
        status=CapabilityStatus.PRESENT_VALIDATED,
        component_ids=(component.component_id,),
        input_revision_ids=(context.revision_id,),
        validator_version=_VALIDATOR_VERSION,
    )


# -- real validators: inference.asr / inference.diarisation (M11) -----------


def _make_inference_validator[T: AsrResultComponent | DiarisationResultComponent](
    *, key: CapabilityKey, component_type: type[T]
) -> CapabilityValidatorFn:
    """M11: a component of ``component_type`` exists only for a
    *completed* stage (see :class:`AsrResultComponentBody`'s docstring) --
    so unlike every other many-key validator here, every member this
    finds is trivially ``present-validated``; a recording with no
    completed stage output simply contributes no member at all (M11's
    partial-failure semantics: a failed sibling stage never blocks this
    one, because it never produced a component to begin with).

    ``component_type`` is a genuine ``type[T]`` (not a bare ``type``)
    purely so ``isinstance(component, component_type)`` below actually
    narrows ``matching``'s element type for the type checker -- an
    unparameterized ``type`` cannot narrow a discriminated union member
    at all.
    """

    def _validate(context: ValidationContext) -> CapabilityRecord:
        matching: list[T] = [
            component
            for component in context.components.values()
            if isinstance(component, component_type)
        ]
        if not matching:
            return CapabilityRecord(
                key=key,
                status=CapabilityStatus.ABSENT,
                validator_version=_VALIDATOR_VERSION,
            )
        members = tuple(
            CapabilityMemberStatus(
                member_id=component.media_artefact_id,
                status=CapabilityStatus.PRESENT_VALIDATED,
                component_id=component.component_id,
            )
            for component in matching
        )
        overall = _aggregate_many_key_status(member.status for member in members)
        return CapabilityRecord(
            key=key,
            status=overall,
            component_ids=tuple(component.component_id for component in matching),
            input_revision_ids=(context.revision_id,),
            provenance_classes=tuple(
                sorted({component.model_name for component in matching})
            ),
            validator_version=_VALIDATOR_VERSION,
            members=members,
        )

    return _validate


# -- real validators: the speaker capabilities (M4/M5/M7/M8) ----------------


def _dropped_turn_ids(components: Mapping[ComponentId, ComponentRecord]) -> set[str]:
    """Turn IDs an M7/M9 ledger legitimately retired.

    ``drop-empty`` discharges a turn's bindings. ``merge`` carries the
    utterance on the sole output turn and retires every other input ID.
    A vanished ID is a failure only when no ledger accounts for it.
    """
    retired: set[str] = set()
    for ledger in components.values():
        if not isinstance(ledger, TextEditLedgerComponent):
            continue
        for entry in ledger.entries:
            if entry.operation == TextEditOperation.DROP_EMPTY:
                retired.update(entry.input_turn_ids)
            elif entry.operation == TextEditOperation.MERGE:
                retired.update(set(entry.input_turn_ids) - set(entry.output_turn_ids))
    return retired


def _single_component[T](
    context: ValidationContext, kind: type[T], *, label: str
) -> tuple[T | None, CapabilityRecord | None]:
    """Resolve a one-cardinality key's single candidate component.

    Returns ``(None, record)`` with a ready-made ``absent``/``failed``
    record when there is not exactly one -- the shape every
    one-cardinality validator here already used, factored out because
    three more keys now need exactly it.
    """
    matching = [
        component
        for component in context.components.values()
        if isinstance(component, kind)
    ]
    if not matching:
        return None, CapabilityRecord(
            key=CapabilityKey(label),
            status=CapabilityStatus.ABSENT,
            validator_version=_VALIDATOR_VERSION,
        )
    if len(matching) > 1:
        return None, CapabilityRecord(
            key=CapabilityKey(label),
            status=CapabilityStatus.FAILED,
            component_ids=tuple(component.component_id for component in matching),  # pyright: ignore[reportAttributeAccessIssue]
            validator_version=_VALIDATOR_VERSION,
            failure_detail=(
                f"{len(matching)} candidate components present for the "
                "one-cardinality key {label}"
            ),
        )
    return matching[0], None


def _validate_speakers_machine_clustered(
    context: ValidationContext,
) -> CapabilityRecord:
    """M4: "diarisation clusters exist for the timed set".

    Proves three things, all by inspection: exactly one attribution set
    exists, every turn it names is a live canonical turn (or one a text
    pass accounted for), and every cluster it assigns to is one it
    declares (the component's own validator). The payload carries the
    coverage fraction M5 asks for. Nothing here maps a cluster to a
    person -- that is the whole point of the key (F13).
    """
    attribution, failure = _single_component(
        context,
        MachineAttributionSetComponent,
        label=CapabilityKey.SPEAKERS_MACHINE_CLUSTERED.value,
    )
    if failure is not None:
        return failure
    assert attribution is not None

    turns = canonical_turns(context.components)
    if not turns:
        return CapabilityRecord(
            key=CapabilityKey.SPEAKERS_MACHINE_CLUSTERED,
            status=CapabilityStatus.FAILED,
            component_ids=(attribution.component_id,),
            validator_version=_VALIDATOR_VERSION,
            failure_detail=(
                "machine clusters exist but no single canonical timed turn set "
                "does, so there is nothing the clusters are clusters *of*"
            ),
        )
    live_turn_ids = {turn.turn_id for turn in turns} | _dropped_turn_ids(
        context.components
    )
    named = {assignment.turn_id for assignment in attribution.assignments} | set(
        attribution.unattributed_turn_ids
    )
    unresolved = sorted(named - live_turn_ids)
    if unresolved:
        return CapabilityRecord(
            key=CapabilityKey.SPEAKERS_MACHINE_CLUSTERED,
            status=CapabilityStatus.FAILED,
            component_ids=(attribution.component_id,),
            validator_version=_VALIDATOR_VERSION,
            failure_detail=(
                f"attribution names turn ID(s) absent from the canonical turn set "
                f"and unaccounted for by any text-edit ledger: {unresolved}"
            ),
        )
    if not named:
        return CapabilityRecord(
            key=CapabilityKey.SPEAKERS_MACHINE_CLUSTERED,
            status=CapabilityStatus.FAILED,
            component_ids=(attribution.component_id,),
            validator_version=_VALIDATOR_VERSION,
            failure_detail=(
                "attribution set names no canonical turn at all -- clusters with "
                "nothing attributed to them prove nothing about the timed set"
            ),
        )
    return CapabilityRecord(
        key=CapabilityKey.SPEAKERS_MACHINE_CLUSTERED,
        status=CapabilityStatus.PRESENT_VALIDATED,
        component_ids=(attribution.component_id,),
        coverage=(
            f"attributed={len(attribution.assignments)}",
            f"unattributed={len(attribution.unattributed_turn_ids)}",
            f"total_turns={len(turns)}",
            f"clusters={len(attribution.clusters)}",
        ),
        input_revision_ids=(context.revision_id,),
        provenance_classes=("machine-clustered",),
        validator_version=_VALIDATOR_VERSION,
    )


def _review_target_failures(
    review: SpeakerReviewComponent, context: ValidationContext
) -> tuple[str, ...]:
    """Which of a review's decisions no longer resolve (M7's remap set).

    Turn-scoped decisions may legitimately lose their target to an M9
    ``drop-empty`` (bindings discharged); anything else vanishing means a
    structural change outside M7's closed remap set, which invalidates the
    review explicitly rather than carrying it silently onto other turns.
    """
    live_turn_ids = {
        turn.turn_id for turn in canonical_turns(context.components)
    } | _dropped_turn_ids(context.components)
    live_cluster_ids = {
        cluster.cluster_id
        for component in context.components.values()
        if isinstance(component, MachineAttributionSetComponent)
        for cluster in component.clusters
    }
    problems: list[str] = []
    for decision in review.decisions:
        match decision:
            case TurnDecision() if decision.turn_id not in live_turn_ids:
                problems.append(f"turn {decision.turn_id}")
            case ClusterDecision() if decision.cluster_id not in live_cluster_ids:
                problems.append(f"cluster {decision.cluster_id}")
            case _:
                continue
    return tuple(problems)


def _speaker_coverage_payload(
    context: ValidationContext,
) -> tuple[AssignmentCoverage, tuple[str, ...]]:
    assignments = effective_assignments(context.components)
    coverage = assignment_coverage(assignments.values())
    provenance_classes = tuple(
        sorted(
            provenance.value
            for provenance, count in coverage.by_provenance.items()
            if count
        )
    )
    return coverage, provenance_classes


def _validate_speakers_human_reviewed(context: ValidationContext) -> CapabilityRecord:
    """M4: "review applied; coverage + unresolved counts".

    Present-validated means an applied review exists whose decisions all
    still resolve. It deliberately does *not* require full coverage: a
    partial review is a real, recorded human act, and M8 keeps partial and
    complete distinguishable by coverage rather than by refusing one.
    ``speakers.human-confirmed`` is where full coverage matters.
    """
    review, failure = _single_component(
        context,
        SpeakerReviewComponent,
        label=CapabilityKey.SPEAKERS_HUMAN_REVIEWED.value,
    )
    if failure is not None:
        return failure
    assert review is not None

    problems = _review_target_failures(review, context)
    if problems:
        return CapabilityRecord(
            key=CapabilityKey.SPEAKERS_HUMAN_REVIEWED,
            status=CapabilityStatus.FAILED,
            component_ids=(review.component_id,),
            validator_version=_VALIDATOR_VERSION,
            failure_detail=(
                "applied review names target(s) that no longer exist and are not "
                f"accounted for by M7's remap set: {list(problems)}"
            ),
        )
    coverage, provenance_classes = _speaker_coverage_payload(context)
    return CapabilityRecord(
        key=CapabilityKey.SPEAKERS_HUMAN_REVIEWED,
        status=CapabilityStatus.PRESENT_VALIDATED,
        component_ids=(review.component_id,),
        coverage=(
            f"reviewed={coverage.reviewed}",
            f"total_turns={coverage.total_turns}",
            f"unresolved={coverage.unresolved}",
            f"gate_satisfying={coverage.gate_satisfying}",
        ),
        input_revision_ids=(review.input_revision_id, context.revision_id),
        provenance_classes=provenance_classes,
        validator_version=_VALIDATOR_VERSION,
    )


def _validate_speakers_human_confirmed(context: ValidationContext) -> CapabilityRecord:
    """M4: "full-coverage confirmed assignments".

    Every canonical turn must resolve to a participant *via a reviewed
    decision*. An explicit ``unclear-speaker`` is a legitimate and
    gate-satisfying review outcome (D1) but is not a confirmed assignment,
    so a document carrying one is honestly ``absent`` here rather than
    confirmed -- which is exactly the distinction that stops "we reviewed
    it" from being read as "we know who everyone was".
    """
    reviews = [
        component
        for component in context.components.values()
        if isinstance(component, SpeakerReviewComponent)
    ]
    if not reviews:
        return CapabilityRecord(
            key=CapabilityKey.SPEAKERS_HUMAN_CONFIRMED,
            status=CapabilityStatus.ABSENT,
            validator_version=_VALIDATOR_VERSION,
        )
    coverage, provenance_classes = _speaker_coverage_payload(context)
    if not coverage.fully_confirmed:
        return CapabilityRecord(
            key=CapabilityKey.SPEAKERS_HUMAN_CONFIRMED,
            status=CapabilityStatus.ABSENT,
            component_ids=tuple(review.component_id for review in reviews),
            validator_version=_VALIDATOR_VERSION,
            failure_detail=(
                f"{coverage.reviewed}/{coverage.total_turns} turns reviewed, "
                f"{coverage.unresolved} unresolved"
            ),
        )
    return CapabilityRecord(
        key=CapabilityKey.SPEAKERS_HUMAN_CONFIRMED,
        status=CapabilityStatus.PRESENT_VALIDATED,
        component_ids=tuple(review.component_id for review in reviews),
        coverage=(f"reviewed={coverage.reviewed}/{coverage.total_turns}",),
        input_revision_ids=(context.revision_id,),
        provenance_classes=provenance_classes,
        validator_version=_VALIDATOR_VERSION,
    )


# -- real validators: text.corrected / text.polished (M9) --------------------


def _turn_sets_since_reflow(
    components: Mapping[ComponentId, ComponentRecord], live: ComponentId
) -> frozenset[ComponentId]:
    """Turn sets reachable from live without crossing a reflow ledger.

    Reflow changes turn boundaries, so correction and polish proofs from
    before it are stale even when their surviving anchor IDs still happen to
    cover every live turn. The graph is deliberately one-to-many because
    content-addressed no-op correction and polish passes may share an output
    turn-set ID while retaining separate ledgers.
    """
    inputs_by_output: dict[ComponentId, set[ComponentId]] = {}
    for ledger in components.values():
        if not isinstance(ledger, TextEditLedgerComponent):
            continue
        if ledger.mode == TextEditMode.REFLOW:
            continue
        inputs_by_output.setdefault(ledger.output_turn_set_component_id, set()).add(
            ledger.input_turn_set_component_id
        )

    reachable: set[ComponentId] = set()
    pending = [live]
    while pending:
        current = pending.pop()
        if current in reachable:
            continue
        reachable.add(current)
        pending.extend(inputs_by_output.get(current, ()))
    return frozenset(reachable)


def _make_text_validator(
    *, key: CapabilityKey, mode: TextEditMode
) -> CapabilityValidatorFn:
    """M9: "a ledger that does not account for the full diff fails
    validation".

    Present-validated requires a ledger for this mode whose declared
    output turn set *is* the live canonical set, and whose entries account
    for every live turn exactly once. That is what makes the capability a
    proof rather than a label: a stage cannot claim ``text.polished`` by
    writing a ledger about turns that are no longer there.
    """

    def _validate(context: ValidationContext) -> CapabilityRecord:
        ledgers = [
            component
            for component in context.components.values()
            if isinstance(component, TextEditLedgerComponent) and component.mode == mode
        ]
        if not ledgers:
            return CapabilityRecord(
                key=key,
                status=CapabilityStatus.ABSENT,
                validator_version=_VALIDATOR_VERSION,
            )
        turn_set = canonical_turn_set(context.components)
        if turn_set is None:
            return CapabilityRecord(
                key=key,
                status=CapabilityStatus.FAILED,
                component_ids=tuple(ledger.component_id for ledger in ledgers),
                validator_version=_VALIDATOR_VERSION,
                failure_detail=(
                    "a text-edit ledger exists but no single canonical timed turn "
                    "set does, so the ledger describes nothing this document has"
                ),
            )
        recent = _turn_sets_since_reflow(context.components, turn_set.component_id)
        current = [
            ledger
            for ledger in ledgers
            if ledger.output_turn_set_component_id in recent
        ]
        if not current:
            return CapabilityRecord(
                key=key,
                status=CapabilityStatus.ABSENT,
                component_ids=tuple(ledger.component_id for ledger in ledgers),
                validator_version=_VALIDATOR_VERSION,
                failure_detail=(
                    "every ledger for this mode describes an earlier turn set; a "
                    "later pass has since replaced it, so this proof is stale rather "
                    "than current"
                ),
            )
        ledger = current[-1]
        live_turn_ids = {turn.turn_id for turn in turn_set.turns}
        accounted = {
            turn_id for entry in ledger.entries for turn_id in entry.output_turn_ids
        }
        unaccounted = sorted(live_turn_ids - accounted)
        if unaccounted:
            return CapabilityRecord(
                key=key,
                status=CapabilityStatus.FAILED,
                component_ids=(ledger.component_id,),
                validator_version=_VALIDATOR_VERSION,
                failure_detail=(
                    f"ledger does not account for {len(unaccounted)} live turn(s): "
                    f"{unaccounted[:5]}"
                ),
            )
        return CapabilityRecord(
            key=key,
            status=CapabilityStatus.PRESENT_VALIDATED,
            component_ids=(ledger.component_id,),
            coverage=(
                f"entries={len(ledger.entries)}",
                f"turns={len(live_turn_ids)}",
            ),
            input_revision_ids=(context.revision_id,),
            provenance_classes=(ledger.editor,),
            validator_version=_VALIDATOR_VERSION,
        )

    return _validate


# -- real validators: chapters / minutes (M10) -------------------------------


def _validate_chapters(context: ValidationContext) -> CapabilityRecord:
    """M4: "exact coverage of the ordered turn sequence".

    Exact means exact: the chapters' concatenated turn IDs must equal the
    canonical sequence, in order. A partition that merely covers the same
    *set* would let a chapter claim turns that appear elsewhere in the
    transcript, which is how a navigable table of contents quietly stops
    matching the thing it navigates.
    """
    chapter_set, failure = _single_component(
        context, ChapterSetComponent, label=CapabilityKey.CHAPTERS.value
    )
    if failure is not None:
        return failure
    assert chapter_set is not None

    turns = canonical_turns(context.components)
    if not turns:
        return CapabilityRecord(
            key=CapabilityKey.CHAPTERS,
            status=CapabilityStatus.FAILED,
            component_ids=(chapter_set.component_id,),
            validator_version=_VALIDATOR_VERSION,
            failure_detail=(
                "a chapter set exists but there is no single canonical timed turn "
                "sequence for it to cover"
            ),
        )
    covered = [
        turn_id for chapter in chapter_set.chapters for turn_id in chapter.turn_ids
    ]
    expected = [turn.turn_id for turn in turns]
    if covered != expected:
        return CapabilityRecord(
            key=CapabilityKey.CHAPTERS,
            status=CapabilityStatus.FAILED,
            component_ids=(chapter_set.component_id,),
            validator_version=_VALIDATOR_VERSION,
            failure_detail=(
                f"chapters cover {len(covered)} turn slot(s) but the canonical "
                f"sequence has {len(expected)}, or they are not in the same order"
            ),
        )
    return CapabilityRecord(
        key=CapabilityKey.CHAPTERS,
        status=CapabilityStatus.PRESENT_VALIDATED,
        component_ids=(chapter_set.component_id,),
        coverage=(
            f"chapters={len(chapter_set.chapters)}",
            f"turns={len(expected)}",
        ),
        input_revision_ids=(context.revision_id,),
        validator_version=_VALIDATOR_VERSION,
    )


def _validate_minutes(context: ValidationContext) -> CapabilityRecord:
    """M4/M10: "evidence-linked findings only".

    Checks each ref actually resolves: turn IDs against the canonical
    sequence (plus M7-accounted drops), notes section IDs against the M20
    notes components, and owners against participant records (F21). The
    component type already guarantees *some* ref exists on every claim;
    this is where "some ref" has to mean "a ref to something real".
    """
    minutes, failure = _single_component(
        context, MinutesComponent, label=CapabilityKey.MINUTES.value
    )
    if failure is not None:
        return failure
    assert minutes is not None

    live_turn_ids = {
        turn.turn_id for turn in canonical_turns(context.components)
    } | _dropped_turn_ids(context.components)
    live_section_ids = {
        section.section_id
        for component in context.components.values()
        if isinstance(component, NotesComponent)
        for section in component.sections
    }
    live_participant_ids = {
        participant.participant_id
        for component in context.components.values()
        if isinstance(component, ParticipantSetComponent)
        for participant in component.participants
    }
    problems: list[str] = []
    for claim in (minutes.summary, *minutes.findings):
        problems.extend(
            f"turn {turn_id}"
            for turn_id in claim.evidence_turn_ids
            if turn_id not in live_turn_ids
        )
        problems.extend(
            f"section {section_id}"
            for section_id in claim.evidence_section_ids
            if section_id not in live_section_ids
        )
    problems.extend(
        f"owner {finding.owner_participant_id}"
        for finding in minutes.findings
        if finding.owner_participant_id is not None
        and finding.owner_participant_id not in live_participant_ids
    )
    if problems:
        return CapabilityRecord(
            key=CapabilityKey.MINUTES,
            status=CapabilityStatus.FAILED,
            component_ids=(minutes.component_id,),
            validator_version=_VALIDATOR_VERSION,
            failure_detail=(
                f"minutes cite {len(problems)} ref(s) that do not resolve in this "
                f"document: {sorted(set(problems))[:5]}"
            ),
        )
    return CapabilityRecord(
        key=CapabilityKey.MINUTES,
        status=CapabilityStatus.PRESENT_VALIDATED,
        component_ids=(minutes.component_id,),
        coverage=(f"findings={len(minutes.findings)}",),
        input_revision_ids=(context.revision_id,),
        provenance_classes=tuple(
            sorted(
                {
                    claim.claim_status.value
                    for claim in (minutes.summary, *minutes.findings)
                }
            )
        ),
        validator_version=_VALIDATOR_VERSION,
    )


# -- stub validator: every other key -----------------------------------------


def _make_not_attempted_stub(key: CapabilityKey) -> CapabilityValidatorFn:
    """A validator that proves nothing and can prove nothing.

    It never inspects ``context`` -- there is no branch in it that could
    be made to return ``present-validated`` even by a bug in a caller,
    which is what makes the ``implemented=False`` safety net in
    :func:`validate` an actual guarantee rather than a convention.
    """

    def _validate(context: ValidationContext) -> CapabilityRecord:
        return CapabilityRecord(
            key=key,
            status=CapabilityStatus.NOT_ATTEMPTED,
            validator_version=_VALIDATOR_VERSION,
        )

    return _validate


_KEY_TABLE_ORDER: Mapping[CapabilityKey, int] = {
    key: index for index, key in enumerate(CapabilityKey)
}


def _check_prerequisite_ordering(
    entries: Mapping[CapabilityKey, RegistryEntry],
) -> None:
    """MAJOR 3's topological guard: every prerequisite must be declared
    *earlier* in the M4 table (:class:`CapabilityKey`'s own declaration
    order) than the key that depends on it.

    :func:`validate` dispatches in exactly that table order, building
    ``prerequisite_statuses`` from results it has *already* computed --
    a prerequisite declared later would still be missing from ``results``
    when its dependent runs, so :func:`validate` would raise a bare
    ``KeyError`` at the first bundle that ever exercised it rather than
    this failing loudly, at import time, against the table itself.
    Exposed as a standalone function (not inlined in
    :func:`_build_registry`) so it can be exercised directly against a
    synthetic, deliberately-misordered table without having to corrupt
    the real one.
    """
    for key, entry in entries.items():
        for prerequisite in entry.prerequisites:
            if prerequisite not in _KEY_TABLE_ORDER:
                raise RegistryConfigurationError(
                    f"capability {key.value!r} declares prerequisite "
                    f"{prerequisite!r}, which is not a member of CapabilityKey."
                )
            if _KEY_TABLE_ORDER[prerequisite] >= _KEY_TABLE_ORDER[key]:
                raise RegistryConfigurationError(
                    f"capability {key.value!r} declares prerequisite "
                    f"{prerequisite.value!r}, which is not declared earlier in the "
                    "M4 table order -- validate() dispatches in table order and "
                    "would KeyError computing prerequisite_statuses at runtime "
                    "instead of failing here, at import time."
                )


def _build_registry() -> dict[CapabilityKey, RegistryEntry]:
    entries: dict[CapabilityKey, RegistryEntry] = {}
    for key in CapabilityKey:
        entries[key] = RegistryEntry(
            key=key,
            cardinality=cardinality_of(key),
            validator=_make_not_attempted_stub(key),
        )

    entries[CapabilityKey.NOTES_PROVIDER] = RegistryEntry(
        key=CapabilityKey.NOTES_PROVIDER,
        cardinality=cardinality_of(CapabilityKey.NOTES_PROVIDER),
        validator=_make_notes_validator(
            key=CapabilityKey.NOTES_PROVIDER, authored=False
        ),
        implemented=True,
    )
    entries[CapabilityKey.NOTES_AUTHORED] = RegistryEntry(
        key=CapabilityKey.NOTES_AUTHORED,
        cardinality=cardinality_of(CapabilityKey.NOTES_AUTHORED),
        validator=_make_notes_validator(
            key=CapabilityKey.NOTES_AUTHORED, authored=True
        ),
        implemented=True,
    )
    entries[CapabilityKey.PARTICIPANTS_DECLARED] = RegistryEntry(
        key=CapabilityKey.PARTICIPANTS_DECLARED,
        cardinality=cardinality_of(CapabilityKey.PARTICIPANTS_DECLARED),
        validator=_validate_participants_declared,
        implemented=True,
    )
    entries[CapabilityKey.TRANSCRIPT_UNTIMED] = RegistryEntry(
        key=CapabilityKey.TRANSCRIPT_UNTIMED,
        cardinality=cardinality_of(CapabilityKey.TRANSCRIPT_UNTIMED),
        validator=_validate_transcript_untimed,
        implemented=True,
    )
    entries[CapabilityKey.TRANSCRIPT_TIMED] = RegistryEntry(
        key=CapabilityKey.TRANSCRIPT_TIMED,
        cardinality=cardinality_of(CapabilityKey.TRANSCRIPT_TIMED),
        validator=_validate_transcript_timed,
        implemented=True,
    )
    entries[CapabilityKey.SPEAKERS_PROVIDER_LABELS] = RegistryEntry(
        key=CapabilityKey.SPEAKERS_PROVIDER_LABELS,
        cardinality=cardinality_of(CapabilityKey.SPEAKERS_PROVIDER_LABELS),
        validator=_validate_speakers_provider_labels,
        implemented=True,
    )
    entries[CapabilityKey.MEDIA_RECORDING] = RegistryEntry(
        key=CapabilityKey.MEDIA_RECORDING,
        cardinality=cardinality_of(CapabilityKey.MEDIA_RECORDING),
        validator=_validate_media_recording,
        implemented=True,
    )
    entries[CapabilityKey.TIMELINE_COMBINED] = RegistryEntry(
        key=CapabilityKey.TIMELINE_COMBINED,
        cardinality=cardinality_of(CapabilityKey.TIMELINE_COMBINED),
        validator=_validate_timeline_combined,
        implemented=True,
    )
    entries[CapabilityKey.INFERENCE_ASR] = RegistryEntry(
        key=CapabilityKey.INFERENCE_ASR,
        cardinality=cardinality_of(CapabilityKey.INFERENCE_ASR),
        validator=_make_inference_validator(
            key=CapabilityKey.INFERENCE_ASR, component_type=AsrResultComponent
        ),
        implemented=True,
    )
    entries[CapabilityKey.INFERENCE_DIARISATION] = RegistryEntry(
        key=CapabilityKey.INFERENCE_DIARISATION,
        cardinality=cardinality_of(CapabilityKey.INFERENCE_DIARISATION),
        validator=_make_inference_validator(
            key=CapabilityKey.INFERENCE_DIARISATION,
            component_type=DiarisationResultComponent,
        ),
        implemented=True,
    )

    entries[CapabilityKey.SPEAKERS_MACHINE_CLUSTERED] = RegistryEntry(
        key=CapabilityKey.SPEAKERS_MACHINE_CLUSTERED,
        cardinality=cardinality_of(CapabilityKey.SPEAKERS_MACHINE_CLUSTERED),
        validator=_validate_speakers_machine_clustered,
        implemented=True,
    )
    entries[CapabilityKey.SPEAKERS_HUMAN_REVIEWED] = RegistryEntry(
        key=CapabilityKey.SPEAKERS_HUMAN_REVIEWED,
        cardinality=cardinality_of(CapabilityKey.SPEAKERS_HUMAN_REVIEWED),
        validator=_validate_speakers_human_reviewed,
        implemented=True,
    )
    entries[CapabilityKey.SPEAKERS_HUMAN_CONFIRMED] = RegistryEntry(
        key=CapabilityKey.SPEAKERS_HUMAN_CONFIRMED,
        cardinality=cardinality_of(CapabilityKey.SPEAKERS_HUMAN_CONFIRMED),
        validator=_validate_speakers_human_confirmed,
        implemented=True,
    )

    entries[CapabilityKey.TEXT_CORRECTED] = RegistryEntry(
        key=CapabilityKey.TEXT_CORRECTED,
        cardinality=cardinality_of(CapabilityKey.TEXT_CORRECTED),
        validator=_make_text_validator(
            key=CapabilityKey.TEXT_CORRECTED, mode=TextEditMode.CORRECT
        ),
        implemented=True,
    )
    entries[CapabilityKey.TEXT_POLISHED] = RegistryEntry(
        key=CapabilityKey.TEXT_POLISHED,
        cardinality=cardinality_of(CapabilityKey.TEXT_POLISHED),
        validator=_make_text_validator(
            key=CapabilityKey.TEXT_POLISHED, mode=TextEditMode.POLISH
        ),
        implemented=True,
    )
    entries[CapabilityKey.MINUTES] = RegistryEntry(
        key=CapabilityKey.MINUTES,
        cardinality=cardinality_of(CapabilityKey.MINUTES),
        validator=_validate_minutes,
        implemented=True,
    )

    # M4: the one sanctioned prerequisite -- chapters unavailable without a
    # validated timed turn set (D6). Data only in this phase: no validator
    # (real or stub) for either key inspects `prerequisite_statuses` yet.
    entries[CapabilityKey.CHAPTERS] = RegistryEntry(
        key=CapabilityKey.CHAPTERS,
        cardinality=cardinality_of(CapabilityKey.CHAPTERS),
        validator=_validate_chapters,
        prerequisites=(CapabilityKey.TRANSCRIPT_TIMED,),
        implemented=True,
    )
    _check_prerequisite_ordering(entries)
    return entries


REGISTRY: Mapping[CapabilityKey, RegistryEntry] = _build_registry()


def validate(
    revision_id: RevisionId,
    components: Mapping[ComponentId, ComponentRecord],
    *,
    registry: Mapping[CapabilityKey, RegistryEntry] = REGISTRY,
) -> Mapping[CapabilityKey, CapabilityRecord]:
    """Run every registered validator over one revision's resolved
    component graph (M4/M16).

    Takes the raw closure (a revision ID plus its resolved components)
    rather than a :class:`~.document.TranscriptDocumentV1` on purpose:
    the document *contains* this function's own output (its
    ``capabilities`` mapping), so a document can only be constructed
    *after* validation runs -- passing one in would be circular.

    Dispatches every :class:`CapabilityKey` in declaration order (M4 table
    order), so a key's ``prerequisites`` -- all declared earlier in the
    table, enforced at import time by :func:`_check_prerequisite_ordering`
    -- have already been validated by the time it runs. Raises
    :class:`StubEmittedPresentValidatedError` if a key registered with
    ``implemented=False`` ever returns ``present-validated``: capabilities
    are proofs, and a stub must never be allowed to fake one. Raises
    :class:`PrerequisiteNotSatisfiedError` if a validator returns
    ``present-validated`` while any of *its own declared* prerequisites
    was not itself ``present-validated`` (M4: prerequisites are
    validation preconditions -- centrally enforced here rather than left
    to each validator to remember to check ``prerequisite_statuses``).

    ``registry`` defaults to the module's own closed table; it is an
    explicit parameter (not a monkeypatch target) purely so tests can
    exercise these safety nets with a deliberately-broken validator or
    table without mutating shared, module-level state.
    """
    results: dict[CapabilityKey, CapabilityRecord] = {}
    for key in CapabilityKey:
        entry = registry[key]
        prerequisite_statuses = {
            prerequisite: results[prerequisite].status
            for prerequisite in entry.prerequisites
        }
        context = ValidationContext(
            revision_id=revision_id,
            components=components,
            prerequisite_statuses=prerequisite_statuses,
        )
        record = entry.validator(context)
        if record.key != key:
            raise ValidatorKeyMismatchError(
                f"validator registered for {key.value!r} returned a record for "
                f"{record.key.value!r}."
            )
        if (
            not entry.implemented
            and record.status == CapabilityStatus.PRESENT_VALIDATED
        ):
            raise StubEmittedPresentValidatedError(
                f"capability {key.value!r} has no v1 implementation and its stub "
                "validator must never emit present-validated."
            )
        if record.status == CapabilityStatus.PRESENT_VALIDATED:
            unsatisfied = {
                prerequisite.value: status.value
                for prerequisite, status in prerequisite_statuses.items()
                if status != CapabilityStatus.PRESENT_VALIDATED
            }
            if unsatisfied:
                raise PrerequisiteNotSatisfiedError(
                    f"validator for {key.value!r} returned present-validated while "
                    f"its prerequisite(s) were not: {unsatisfied}."
                )
        results[key] = record
    return results
