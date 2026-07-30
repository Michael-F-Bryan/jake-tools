"""Typed, content-identified components (M1, M19, M20).

A component is immutable and identified by the SHA-256 of its own content
-- never by a caller-minted ID (M1). Every kind here is split into a
``*Body`` type (the hashed content, with no ``component_id``/``created_at``)
and a ``*Component`` record type (the body plus those two store-minted
fields) -- so "the canonical JSON body excluding component_id and
timestamps" (M1) is a structural fact about the ``Body`` type, not a
field-exclusion list someone could forget to update. :class:`.store.
BundleStore.add_component` is the only place a ``Body`` becomes a
``Component``.

v1 ships exactly the two component kinds the inference-free fixtures need:
the M20 notes component and an M19 participant set. ``ComponentKind`` is
deliberately closed to just these two -- expanding it is later phases'
job, once their components exist to back it.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from .ids import ArtefactId, ComponentId, ParticipantId, SourceId

# -- M19: participant records -------------------------------------------


class ParticipantDeclarationSource(StrEnum):
    """M19: how a participant record was declared -- never inferred."""

    CALENDAR = "calendar"
    TEAMS_ROSTER = "teams-roster"
    NOTE_FRONTMATTER = "note-frontmatter"
    OPERATOR = "operator"


class ParticipantStatus(StrEnum):
    """M19: presence evidence vs speech evidence.

    A declared attendee who never speaks stays ``declared`` --
    declaration is presence evidence, not speech evidence.
    """

    DECLARED = "declared"
    SPEAKING_EVIDENCED = "speaking-evidenced"


class ProviderAccountId(BaseModel):
    """M19: one provider account ID for a participant, scoped to the
    source it was observed on ("provider account IDs per source where
    available") -- a typed pair rather than a bare ``dict[str, str]``.
    """

    model_config = ConfigDict(frozen=True)

    source_id: SourceId
    account_id: str = Field(min_length=1)


class ParticipantRecord(BaseModel):
    """M19: a canonical participant, with declaration provenance.

    ``display_names`` are aliases only -- display-name string equality is
    never identity equality (M19); a label -> participant mapping is a
    separate, explicitly recorded edge (out of scope this phase: no
    speaker capability validators exist yet). ``room_proxy`` is set only
    from explicit operator config (M5); nothing in this module infers it.
    """

    model_config = ConfigDict(frozen=True)

    participant_id: ParticipantId
    declaration_source: ParticipantDeclarationSource
    declaration_evidence: str = Field(min_length=1)
    provider_account_ids: tuple[ProviderAccountId, ...] = ()
    display_names: tuple[str, ...] = Field(min_length=1)
    status: ParticipantStatus
    room_proxy: bool = False


# -- M20: notes component -------------------------------------------------


class NotesKind(StrEnum):
    """M20: what a notes component holds."""

    PROVIDER_SUMMARY = "provider-summary"
    PROVIDER_DECISIONS = "provider-decisions"
    PROVIDER_ACTIONS = "provider-actions"
    PROVIDER_DETAILS = "provider-details"
    AUTHORED_PREP = "authored-prep"


class NotesSection(BaseModel):
    """M20: one ordered, addressable section of a notes component.

    ``section_id`` uses the ``seg`` prefix (minted via
    :func:`.ids.mint_id` at component construction, per the M20 task
    scope) -- M10 evidence refs cite these IDs, which is what makes a
    notes-derived minutes finding mechanically auditable.
    """

    model_config = ConfigDict(frozen=True)

    section_id: Annotated[str, Field(pattern=r"^seg_.+$")]
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)


class ComponentKind(StrEnum):
    """The closed set of component kinds v1 can store (M1).

    Fail-closed by construction: a persisted component whose
    ``component_kind`` is not one of these two values does not match
    :data:`ComponentRecord`'s discriminated union, so
    :meth:`.store.BundleStore.load_component` raises rather than guessing.
    """

    NOTES = "notes"
    PARTICIPANT_SET = "participant-set"


class NotesComponentBody(BaseModel):
    """M20: the hashed content of a notes component -- no identity fields."""

    model_config = ConfigDict(frozen=True)

    component_kind: Literal[ComponentKind.NOTES] = ComponentKind.NOTES
    notes_kind: NotesKind
    source_artefact_id: ArtefactId
    authored: bool
    sections: tuple[NotesSection, ...]


class NotesComponent(NotesComponentBody):
    """M20: a stored notes component (body plus store-minted identity)."""

    component_id: ComponentId
    created_at: datetime


class ParticipantSetComponentBody(BaseModel):
    """M4/M19: the hashed content backing the ``participants.declared``
    capability -- a snapshot of participant records for one bundle.
    """

    model_config = ConfigDict(frozen=True)

    component_kind: Literal[ComponentKind.PARTICIPANT_SET] = (
        ComponentKind.PARTICIPANT_SET
    )
    participants: tuple[ParticipantRecord, ...] = ()


class ParticipantSetComponent(ParticipantSetComponentBody):
    component_id: ComponentId
    created_at: datetime


ComponentBody = Annotated[
    NotesComponentBody | ParticipantSetComponentBody,
    Field(discriminator="component_kind"),
]
ComponentRecord = Annotated[
    NotesComponent | ParticipantSetComponent,
    Field(discriminator="component_kind"),
]


class ComponentInputRefs(NamedTuple):
    """A component's declared input closure (M16 closure validation)."""

    artefact_ids: tuple[ArtefactId, ...]
    component_ids: tuple[ComponentId, ...]


def component_input_refs(
    record: NotesComponent | ParticipantSetComponent,
) -> ComponentInputRefs:
    """Every artefact/component this component declares as an input.

    :meth:`.store.BundleStore._validate_structural_closure` resolves these
    and requires each one to already be a member of the revision's own
    closure -- a component embedded in a document can never silently pull
    in an artefact or component that was never actually assembled into
    that document's lineage (F1).
    """
    match record:
        case NotesComponent():
            return ComponentInputRefs(
                artefact_ids=(record.source_artefact_id,), component_ids=()
            )
        case ParticipantSetComponent():
            return ComponentInputRefs(artefact_ids=(), component_ids=())


def assemble_component_record(
    body: NotesComponentBody | ParticipantSetComponentBody,
    *,
    component_id: ComponentId,
    created_at: datetime,
) -> NotesComponent | ParticipantSetComponent:
    """Attach store-minted identity to a body, producing its stored record.

    The only place a ``*Body`` becomes a ``*Component``; kept here (not in
    ``store.py``) so the closed kind set and its dispatch live next to each
    other -- adding a new component kind only ever touches this module.
    """
    fields = body.model_dump(mode="python")
    match body:
        case NotesComponentBody():
            return NotesComponent(
                **fields, component_id=component_id, created_at=created_at
            )
        case ParticipantSetComponentBody():
            return ParticipantSetComponent(
                **fields, component_id=component_id, created_at=created_at
            )
