"""Capability registry v1 (M4).

A capability is a *proof*, never an assertion: it is emitted only by a
validator that actually inspected the referenced component(s), never by a
stage, a filename, or an operator's say-so. This module is the closed table
of what can be proven in v1 (:class:`CapabilityKey`), the typed payload
every proof is recorded in (:class:`CapabilityRecord`), and :func:`validate`
-- the single dispatcher every consumer goes through.

v1 can only *prove* three things from the components that exist so far:
``notes.provider``/``notes.authored`` (M20) and ``participants.declared``
(M19). Every other key is registered with a stub validator that always
returns ``not-attempted`` and is structurally incapable of returning
``present-validated`` -- :func:`validate` raises if one ever tries, which
is the "capabilities are proofs" rule enforced as code, not just as prose.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ..errors import TranscriptError
from .components import ComponentRecord, NotesComponent, ParticipantSetComponent
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


class TrustClass(StrEnum):
    """M5 shape only -- no speaker validator in this phase populates this.

    Kept here (not deferred) because M5 names exactly these three values
    and the payload needs a typed home for them now; nothing in v1
    constructs one outside a test fixture.
    """

    PER_PARTICIPANT_STREAM = "per-participant-stream"
    ROOM_PROXY = "room-proxy"
    IMPORTED_UNVERIFIED = "imported-unverified"


class TrustClassCoverage(BaseModel):
    """M5: per-trust-class coverage breakdown on a speaker capability
    payload -- shape only, never populated in this phase.
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


# -- real validators: notes.provider / notes.authored (M20) -----------------


def _validate_one_notes_component(
    component: NotesComponent,
) -> tuple[CapabilityStatus, str]:
    if not component.sections:
        return CapabilityStatus.FAILED, "notes component has no sections"
    section_ids = [section.section_id for section in component.sections]
    if len(set(section_ids)) != len(section_ids):
        return CapabilityStatus.FAILED, "duplicate section IDs"
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

        members = []
        for component in matching:
            status, detail = _validate_one_notes_component(component)
            members.append(
                CapabilityMemberStatus(
                    member_id=component.component_id,
                    status=status,
                    component_id=component.component_id,
                    detail=detail,
                )
            )
        overall = (
            CapabilityStatus.PRESENT_VALIDATED
            if all(
                member.status == CapabilityStatus.PRESENT_VALIDATED
                for member in members
            )
            else CapabilityStatus.FAILED
        )
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
        )

    component = matching[0]
    if not component.participants:
        return CapabilityRecord(
            key=CapabilityKey.PARTICIPANTS_DECLARED,
            status=CapabilityStatus.FAILED,
            component_ids=(component.component_id,),
            validator_version=_VALIDATOR_VERSION,
        )
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

    # M4: the one sanctioned prerequisite -- chapters unavailable without a
    # validated timed turn set (D6). Data only in this phase: no validator
    # (real or stub) for either key inspects `prerequisite_statuses` yet.
    entries[CapabilityKey.CHAPTERS] = dataclasses.replace(
        entries[CapabilityKey.CHAPTERS],
        prerequisites=(CapabilityKey.TRANSCRIPT_TIMED,),
    )
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
    table -- have already been validated by the time it runs. Raises
    :class:`StubEmittedPresentValidatedError` if a key registered with
    ``implemented=False`` ever returns ``present-validated``: capabilities
    are proofs, and a stub must never be allowed to fake one.

    ``registry`` defaults to the module's own closed table; it is an
    explicit parameter (not a monkeypatch target) purely so tests can
    exercise this safety net with a deliberately-broken validator without
    mutating shared, module-level state.
    """
    results: dict[CapabilityKey, CapabilityRecord] = {}
    for key in CapabilityKey:
        entry = registry[key]
        context = ValidationContext(
            revision_id=revision_id,
            components=components,
            prerequisite_statuses={
                prerequisite: results[prerequisite].status
                for prerequisite in entry.prerequisites
            },
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
        results[key] = record
    return results
