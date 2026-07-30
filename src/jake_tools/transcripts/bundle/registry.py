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
from .components import (
    AsrResultComponent,
    ComponentRecord,
    DiarisationResultComponent,
    MediaRecordingComponent,
    NotesComponent,
    ParticipantSetComponent,
    ProviderLabelSetComponent,
    RecordingReferenceSetComponent,
    TimedTurnSetComponent,
    TimelineCombinedComponent,
    TranscriptAbsenceDeclaration,
    TrustClass,
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

    # M4: the one sanctioned prerequisite -- chapters unavailable without a
    # validated timed turn set (D6). Data only in this phase: no validator
    # (real or stub) for either key inspects `prerequisite_statuses` yet.
    entries[CapabilityKey.CHAPTERS] = dataclasses.replace(
        entries[CapabilityKey.CHAPTERS],
        prerequisites=(CapabilityKey.TRANSCRIPT_TIMED,),
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
