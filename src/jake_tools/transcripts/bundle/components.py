"""Typed, content-identified components (M1, M19, M20).

A component is immutable and identified by the SHA-256 of its own content
-- never by a caller-minted ID (M1). Every kind here is split into a
``*Body`` type (the hashed content, with no ``component_id``/``created_at``)
and a ``*Component`` record type (the body plus those store-minted
fields) -- so "the canonical JSON body excluding component_id and
timestamps" (M1) is a structural fact about the ``Body`` type, not a
field-exclusion list someone could forget to update. :class:`.store.
BundleStore.add_component` is the only place a ``Body`` becomes a
``Component`` -- and, for notes, the only place a section gets its
``section_id``: a caller supplies section *content* only
(:class:`NotesSectionBody`), never an ID (see :class:`NotesSection`).

v1 ships exactly the two component kinds the inference-free fixtures need:
the M20 notes component and an M19 participant set. ``ComponentKind`` is
deliberately closed to just these two -- expanding it is later phases'
job, once their components exist to back it.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, NamedTuple, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .ids import ArtefactId, ComponentId, ParticipantId, SegmentId, SourceId
from .records import Sha256Hex

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


class NotesSectionBody(BaseModel):
    """M20: the hashed content of one notes section -- no identity field.

    ``section_id`` is minted by the store, never supplied here (see
    :class:`NotesSection`): a caller-chosen ID could collide across
    components or be arbitrary junk, which is exactly the class of bug
    this split closes off structurally rather than by convention.
    """

    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1)
    text: str = Field(min_length=1)


class NotesSection(NotesSectionBody):
    """M20: one ordered, addressable, store-identified notes section.

    ``section_id`` is a genuine :data:`.ids.SegmentId` (the ``seg``
    prefix's uuid7 pattern), minted by :meth:`.store.BundleStore.
    add_component` -- never caller-supplied free text. M10 evidence refs
    cite these IDs, which is what makes a notes-derived minutes finding
    mechanically auditable; a caller-controlled ID would make that
    auditability worthless.
    """

    section_id: SegmentId


class ComponentKind(StrEnum):
    """The closed set of component kinds v1 can store (M1).

    Fail-closed by construction: a persisted component whose
    ``component_kind`` is not one of these two values does not match
    :data:`ComponentRecord`'s discriminated union, so
    :meth:`.store.BundleStore.load_component` raises rather than guessing.
    """

    NOTES = "notes"
    PARTICIPANT_SET = "participant-set"


def _check_authored_matches_notes_kind(
    *, notes_kind: NotesKind, authored: bool
) -> None:
    expected = notes_kind == NotesKind.AUTHORED_PREP
    if authored != expected:
        raise ValueError(
            f"authored={authored!r} disagrees with notes_kind={notes_kind.value!r}: "
            "authored must be True iff notes_kind is authored-prep, and False for "
            "every provider-* kind."
        )


class NotesComponentBody(BaseModel):
    """M20: the hashed content of a notes component -- no identity fields."""

    model_config = ConfigDict(frozen=True)

    component_kind: Literal[ComponentKind.NOTES] = ComponentKind.NOTES
    notes_kind: NotesKind
    source_artefact_id: ArtefactId
    authored: bool
    sections: tuple[NotesSectionBody, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_authored_matches_notes_kind(self) -> Self:
        _check_authored_matches_notes_kind(
            notes_kind=self.notes_kind, authored=self.authored
        )
        return self


class NotesComponent(BaseModel):
    """M20: a stored notes component (body content plus store-minted identity).

    Deliberately *not* a subclass of :class:`NotesComponentBody`: its
    ``sections`` field holds identified :class:`NotesSection` entries
    rather than :class:`NotesSectionBody`, and a mutable-looking field
    override on a subclass is invariant under static typing even though
    both models are frozen at runtime (pyright's
    ``reportIncompatibleVariableOverride``) -- duplicating the shared
    fields here avoids fighting that rather than suppressing it. Content
    identity is still computed correctly (:func:`component_as_body`
    reconstructs the exact :class:`NotesComponentBody` this hashes as,
    section IDs stripped) rather than by relying on Python inheritance.
    Loaded directly from disk (not always built via
    :func:`assemble_component_record`), so it re-runs the same
    authored/notes_kind check independently -- a hand-edited or corrupted
    on-disk record must fail closed here too, not just at construction
    time via a body that was never actually re-validated.
    """

    model_config = ConfigDict(frozen=True)

    component_kind: Literal[ComponentKind.NOTES] = ComponentKind.NOTES
    notes_kind: NotesKind
    source_artefact_id: ArtefactId
    authored: bool
    sections: tuple[NotesSection, ...] = Field(min_length=1)
    component_id: ComponentId
    content_hash: Sha256Hex
    created_at: datetime

    @model_validator(mode="after")
    def _validate_authored_matches_notes_kind(self) -> Self:
        _check_authored_matches_notes_kind(
            notes_kind=self.notes_kind, authored=self.authored
        )
        return self


class ParticipantSetComponentBody(BaseModel):
    """M4/M19: the hashed content backing the ``participants.declared``
    capability -- a snapshot of participant records for one bundle.

    Participants are canonicalised by sorting on ``participant_id`` so the
    same set hashes identically regardless of the order the caller
    happened to list them in (content identity, M1) -- unlike notes
    sections, participant order carries no meaning M19 defines.
    """

    model_config = ConfigDict(frozen=True)

    component_kind: Literal[ComponentKind.PARTICIPANT_SET] = (
        ComponentKind.PARTICIPANT_SET
    )
    participants: tuple[ParticipantRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _sort_participants_by_id(self) -> Self:
        ordered = tuple(sorted(self.participants, key=lambda p: p.participant_id))
        if ordered != self.participants:
            object.__setattr__(self, "participants", ordered)
        return self


class ParticipantSetComponent(ParticipantSetComponentBody):
    component_id: ComponentId
    content_hash: Sha256Hex
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


def component_as_body(
    component: NotesComponentBody
    | NotesComponent
    | ParticipantSetComponentBody
    | ParticipantSetComponent,
) -> NotesComponentBody | ParticipantSetComponentBody:
    """The exact ``*Body`` a component's content hashes as (MINOR C).

    A bare ``*Body`` is returned unchanged. A stored ``*Component`` record
    is *reconstructed* field by field into its ``*Body`` -- not filtered
    by top-level field name (``model_dump(include=...)``), because that
    would still include store-minted identity nested *inside* an included
    field: ``NotesComponent.sections`` holds :class:`NotesSection` entries
    carrying each section's own minted ``section_id``, invisible to a
    field-name filter that only looks at the component's own top-level
    keys. Reconstructing drops it explicitly, so
    ``hash(component_as_body(record)) == hash(body)`` for the same
    content, which is what makes :meth:`.store.BundleStore.
    add_component`'s dedup check -- and :meth:`.store.BundleStore.
    load_component`'s content_hash verification -- correct.

    The subclass cases (:class:`NotesComponent`, :class:`ParticipantSetComponent`)
    are matched *before* their base ``*Body`` cases: ``ParticipantSetComponent``
    **is** a ``ParticipantSetComponentBody`` (ordinary subclass), so a
    base-first match would silently return the full record, identity
    fields and all.
    """
    match component:
        case NotesComponent():
            return NotesComponentBody(
                notes_kind=component.notes_kind,
                source_artefact_id=component.source_artefact_id,
                authored=component.authored,
                sections=tuple(
                    NotesSectionBody(title=section.title, text=section.text)
                    for section in component.sections
                ),
            )
        case NotesComponentBody():
            return component
        case ParticipantSetComponent():
            return ParticipantSetComponentBody(participants=component.participants)
        case ParticipantSetComponentBody():
            return component


def assemble_component_record(
    body: NotesComponentBody | ParticipantSetComponentBody,
    *,
    component_id: ComponentId,
    content_hash: Sha256Hex,
    created_at: datetime,
    mint_segment_id: Callable[[], str],
) -> NotesComponent | ParticipantSetComponent:
    """Attach store-minted identity to a body, producing its stored record.

    The only place a ``*Body`` becomes a ``*Component``; kept here (not in
    ``store.py``) so the closed kind set and its dispatch live next to
    each other -- adding a new component kind only ever touches this
    module. ``mint_segment_id`` is called once per notes section (never by
    this module directly minting an ID itself -- M1: the store mints,
    nowhere else) to attach each section's :data:`.ids.SegmentId`.
    """
    match body:
        case NotesComponentBody():
            sections = tuple(
                NotesSection(
                    section_id=mint_segment_id(), title=section.title, text=section.text
                )
                for section in body.sections
            )
            fields = body.model_dump(mode="python", exclude={"sections"})
            return NotesComponent(
                **fields,
                sections=sections,
                component_id=component_id,
                content_hash=content_hash,
                created_at=created_at,
            )
        case ParticipantSetComponentBody():
            fields = body.model_dump(mode="python")
            return ParticipantSetComponent(
                **fields,
                component_id=component_id,
                content_hash=content_hash,
                created_at=created_at,
            )
