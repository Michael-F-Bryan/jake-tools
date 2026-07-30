"""Typed, content-identified components (M1, M6, M7, M18, M19, M20).

A component is immutable and identified by the SHA-256 of its own content
-- never by a caller-minted ID (M1). Every kind here is split into a
``*Body`` type (the hashed content, with no ``component_id``/``created_at``)
and a ``*Component`` record type (the body plus those store-minted
fields) -- so "the canonical JSON body excluding component_id and
timestamps" (M1) is a structural fact about the ``Body`` type, not a
field-exclusion list someone could forget to update. :class:`.store.
BundleStore.add_component` is the only place a ``Body`` becomes a
``Component``.

Two ID-minting shapes coexist here, both store-only (M1: "IDs are minted
by the store at record creation, nowhere else"):

- Notes sections get their :class:`~.ids.SegmentId` minted *inside*
  ``add_component`` (via the injected ``mint_segment_id`` callback in
  :func:`assemble_component_record`) -- a caller supplies section
  *content* only (:class:`NotesSectionBody`), never an ID.
- Turn segment IDs (:attr:`TimedTurn.source_segment_id`,
  :attr:`UntimedTurn.source_segment_id`) and participant IDs
  (:attr:`ParticipantRecord.participant_id`) are minted by the *caller*
  (an adapter in ``adapters.py``, or a test) via
  :func:`.ids.mint_id` directly, then supplied as already-identified
  values -- the established precedent for :class:`ParticipantRecord`
  (every existing test constructs one with a caller-minted
  ``participant_id``; there is no store method that mints one for a
  caller). Turn sets follow the same shape: this is what lets M6's
  canonical order -- keyed on ``(start_ms, end_ms, source_segment_id)``,
  the third field included -- be validated in full at *body*
  construction time, rather than needing a second, post-mint order check
  for a field that would otherwise not exist yet.

v1's closed :class:`ComponentKind` set: the M20 notes component, the M19
participant set, the M6/M7 untimed and timed turn sets, the M5 provider
label set, the D2/M5 transcript-absence declaration, and the M18 assembly
manifest.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, NamedTuple, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .ids import ArtefactId, ComponentId, ParticipantId, SegmentId, SourceId
from .records import Sha256Hex

# -- M5: speaker trust classes --------------------------------------------


class TrustClass(StrEnum):
    """M5: the class of evidence backing one provider-label span.

    Derived from the mapped participant's record, never from the label
    text itself: ``per-participant-stream`` (D1: a genuine per-attendee
    audio stream -- trusted), ``room-proxy`` (D1: a room-proxy display
    name aggregates multiple real speakers -- unresolved evidence, takes
    the local-diarisation-and-review path), ``imported-unverified``
    (labels carried over from an already-imported transcript -- rendered
    as supplied, never counted as reviewed). Lives here (not in
    ``registry.py``) because :class:`ProviderLabelSpan` -- a component
    field -- needs it, and ``registry.py`` already imports domain types
    from this module, never the reverse.
    """

    PER_PARTICIPANT_STREAM = "per-participant-stream"
    ROOM_PROXY = "room-proxy"
    IMPORTED_UNVERIFIED = "imported-unverified"


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

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
    account_id: str = Field(min_length=1)


class ParticipantRecord(BaseModel):
    """M19: a canonical participant, with declaration provenance.

    ``display_names`` are aliases only -- display-name string equality is
    never identity equality (M19); a label -> participant mapping is a
    separate, explicitly recorded edge (M19's four-value edge-provenance
    enum, backing ``speakers.provider-attributed`` -- still a stub this
    phase; adapters instead confirm a raw label against a participant
    explicitly and record it via ``status``, see ``adapters.py``).
    ``room_proxy`` is set only from explicit operator config (M5);
    nothing in this module infers it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

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

    model_config = ConfigDict(frozen=True, extra="forbid")

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
    ``component_kind`` is not one of these values does not match
    :data:`ComponentRecord`'s discriminated union, so
    :meth:`.store.BundleStore.load_component` raises rather than guessing.
    """

    NOTES = "notes"
    PARTICIPANT_SET = "participant-set"
    UNTIMED_TURN_SET = "untimed-turn-set"
    TIMED_TURN_SET = "timed-turn-set"
    PROVIDER_LABEL_SET = "provider-label-set"
    TRANSCRIPT_ABSENCE_DECLARATION = "transcript-absence-declaration"
    ASSEMBLY_MANIFEST = "assembly-manifest"


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

    model_config = ConfigDict(frozen=True, extra="forbid")

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

    model_config = ConfigDict(frozen=True, extra="forbid")

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

    model_config = ConfigDict(frozen=True, extra="forbid")

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


# -- M6/M7: untimed and timed turn sets -----------------------------------


class UntimedTurn(BaseModel):
    """M6/M7/D6: one speaker-labelled turn with no timing evidence.

    ``source_segment_id`` is minted by the caller (an adapter) at parse
    time (M7), never by this module. The absence of any ``start_ms``/
    ``end_ms`` field is itself the enforcement mechanism behind
    ``transcript.untimed``'s "no timing fields present" check
    (``registry.py``) -- there is no value this type could hold that
    would smuggle timing in; the type simply has no such field to set.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_segment_id: SegmentId
    speaker_label: str = Field(min_length=1)
    text: str = Field(min_length=1)


class UntimedTurnSetComponentBody(BaseModel):
    """M6: turns preserved in exactly their supplied import order -- never
    reordered (unlike :class:`ParticipantSetComponentBody`, for which
    M19 defines no meaningful order).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_kind: Literal[ComponentKind.UNTIMED_TURN_SET] = (
        ComponentKind.UNTIMED_TURN_SET
    )
    source_artefact_id: ArtefactId
    turns: tuple[UntimedTurn, ...] = Field(min_length=1)


class UntimedTurnSetComponent(UntimedTurnSetComponentBody):
    component_id: ComponentId
    content_hash: Sha256Hex
    created_at: datetime


class TimedTurn(BaseModel):
    """M6: one canonical timed turn -- half-open ``[start_ms, end_ms)``,
    ``end_ms > start_ms`` strictly (zero-length cues are legal only in
    raw source evidence, never in a canonical timed turn -- M6).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_segment_id: SegmentId
    speaker_label: str = Field(min_length=1)
    text: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_half_open_span(self) -> Self:
        if self.end_ms <= self.start_ms:
            raise ValueError(
                f"turn span [{self.start_ms}, {self.end_ms}) is not half-open "
                "with end_ms > start_ms (M6); the adapter/normaliser must drop "
                "zero-length cues before constructing a canonical timed turn."
            )
        return self


class TimedTurnSetComponentBody(BaseModel):
    """M6: turns fixed in canonical order -- ``(start_ms, end_ms,
    source_segment_id)`` ascending -- at construction, never recomputed at
    render time. Enforced here (not silently re-sorted, unlike
    :class:`ParticipantSetComponentBody`'s order-independent participants)
    because M6 requires the order to be a *decision* the
    adapter/normaliser makes once, not an incidental byproduct of
    whatever order components happen to be re-serialised in.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_kind: Literal[ComponentKind.TIMED_TURN_SET] = ComponentKind.TIMED_TURN_SET
    source_artefact_id: ArtefactId
    turns: tuple[TimedTurn, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_canonical_order(self) -> Self:
        keys = [
            (turn.start_ms, turn.end_ms, turn.source_segment_id) for turn in self.turns
        ]
        if keys != sorted(keys):
            raise ValueError(
                "turns are not in M6 canonical order (start_ms, end_ms, "
                "source_segment_id ascending); the adapter/normaliser must sort "
                "before constructing this component."
            )
        return self


class TimedTurnSetComponent(TimedTurnSetComponentBody):
    component_id: ComponentId
    content_hash: Sha256Hex
    created_at: datetime


# -- M5: provider label set -------------------------------------------------


class ProviderLabelSpan(BaseModel):
    """M5: one raw provider label, preserved as evidence.

    ``source_segment_id`` is the cue-lineage reference: it names the
    :class:`TimedTurn` (in a sibling :class:`TimedTurnSetComponent`, same
    revision closure) this span's raw cue was normalised into --
    ``registry.py``'s ``speakers.provider-labels`` validator resolves it
    against every timed turn set in the closure ("spans resolve to
    cues"). ``trust_class`` is set by the adapter from D1/M5's rules
    (room-proxy config vs a genuine per-attendee stream), never inferred
    from the label text.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_segment_id: SegmentId
    raw_label: str = Field(min_length=1)
    text: str = Field(min_length=1)
    trust_class: TrustClass


class ProviderLabelSetComponentBody(BaseModel):
    """M5: raw provider labels for one source, plus the M5 room-proxy
    config hash that produced their trust classes -- a later change to
    that config must change this hash (M5: "changing the proxy list
    changes the hash and invalidates those capabilities").
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_kind: Literal[ComponentKind.PROVIDER_LABEL_SET] = (
        ComponentKind.PROVIDER_LABEL_SET
    )
    source_artefact_id: ArtefactId
    proxy_config_hash: Sha256Hex
    spans: tuple[ProviderLabelSpan, ...] = Field(min_length=1)


class ProviderLabelSetComponent(ProviderLabelSetComponentBody):
    component_id: ComponentId
    content_hash: Sha256Hex
    created_at: datetime


# -- D2/M5: transcript-absence declaration -----------------------------------


class TranscriptAbsenceDeclarationBody(BaseModel):
    """D2/M5: the source's own explicit statement that no transcript was
    available for this meeting -- the concrete, typed carrier that lets
    ``registry.py``'s ``transcript.timed``/``transcript.untimed``
    validators report ``not-available-from-source`` instead of a bare
    ``absent`` when it is present in the closure (D2: "the absence
    statement is itself the evidence the product must preserve"). Its
    mere presence is the signal; ``absent`` remains correct whenever no
    such declaration exists (e.g. a source that simply has no timed
    evidence, with no claim either way).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_kind: Literal[ComponentKind.TRANSCRIPT_ABSENCE_DECLARATION] = (
        ComponentKind.TRANSCRIPT_ABSENCE_DECLARATION
    )
    source_artefact_id: ArtefactId
    statement: str = Field(min_length=1)


class TranscriptAbsenceDeclaration(TranscriptAbsenceDeclarationBody):
    component_id: ComponentId
    content_hash: Sha256Hex
    created_at: datetime


# -- M18: assembly manifest --------------------------------------------------


class Disposition(StrEnum):
    """M18: the closed set of roles a selected artefact may play in an
    assembly revision. An artefact may carry more than one at once (e.g.
    an Obsidian source note is both ``destination`` and ``notes`` --
    M18's own worked example).
    """

    EVIDENCE_ONLY = "evidence-only"
    MEDIA = "media"
    TRANSCRIPT_CANDIDATE = "transcript-candidate"
    SELECTED_TRANSCRIPT = "selected-transcript"
    NOTES = "notes"
    DESTINATION = "destination"


class ArtefactSelection(BaseModel):
    """M18: one artefact's disposition set within an assembly revision.

    ``dispositions`` is a genuine *set* (M18: "non-empty disposition
    set") represented as a canonically sorted, deduplicated tuple --
    never a bare ``frozenset``/``set`` field, whose iteration order is
    process-randomised for ``str`` members (``PYTHONHASHSEED``) and would
    make this component's content hash non-deterministic across runs
    (the same failure mode :class:`ParticipantSetComponentBody` already
    solves for participant order).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    artefact_id: ArtefactId
    dispositions: tuple[Disposition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _sort_and_dedupe_dispositions(self) -> Self:
        deduped = set(self.dispositions)
        if len(deduped) != len(self.dispositions):
            raise ValueError(
                f"duplicate disposition(s) for artefact {self.artefact_id!r} -- "
                "M18 requires a set, not a bag."
            )
        ordered = tuple(sorted(deduped, key=lambda disposition: disposition.value))
        if ordered != self.dispositions:
            object.__setattr__(self, "dispositions", ordered)
        return self


class AssemblyManifestComponentBody(BaseModel):
    """M18: the record of exactly which artefacts one assembly revision
    selected, each with its disposition set, plus the operation's
    rationale (D5: "the selection revision records the chosen candidate
    and rationale"). Lives as a component (not a new ``RevisionRecord``
    field) because ``records.py`` is read-only to this slice and
    ``RevisionRecord.artefact_ids`` already carries the flat selected-ID
    list the store's own closure validation needs -- this component adds
    exactly the richer per-artefact disposition/rationale detail that
    field cannot.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_kind: Literal[ComponentKind.ASSEMBLY_MANIFEST] = (
        ComponentKind.ASSEMBLY_MANIFEST
    )
    selections: tuple[ArtefactSelection, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _no_duplicate_artefact_selections(self) -> Self:
        artefact_ids = [selection.artefact_id for selection in self.selections]
        if len(set(artefact_ids)) != len(artefact_ids):
            raise ValueError(
                "an assembly manifest may not select the same artefact twice (M18)."
            )
        return self


class AssemblyManifestComponent(AssemblyManifestComponentBody):
    component_id: ComponentId
    content_hash: Sha256Hex
    created_at: datetime


ComponentBody = Annotated[
    NotesComponentBody
    | ParticipantSetComponentBody
    | UntimedTurnSetComponentBody
    | TimedTurnSetComponentBody
    | ProviderLabelSetComponentBody
    | TranscriptAbsenceDeclarationBody
    | AssemblyManifestComponentBody,
    Field(discriminator="component_kind"),
]
ComponentRecord = Annotated[
    NotesComponent
    | ParticipantSetComponent
    | UntimedTurnSetComponent
    | TimedTurnSetComponent
    | ProviderLabelSetComponent
    | TranscriptAbsenceDeclaration
    | AssemblyManifestComponent,
    Field(discriminator="component_kind"),
]


class ComponentInputRefs(NamedTuple):
    """A component's declared input closure (M16 closure validation)."""

    artefact_ids: tuple[ArtefactId, ...]
    component_ids: tuple[ComponentId, ...]


def component_input_refs(record: ComponentRecord) -> ComponentInputRefs:
    """Every artefact/component this component declares as an input.

    :meth:`.store.BundleStore._validate_structural_closure` resolves these
    and requires each one to already be a member of the revision's own
    closure -- a component embedded in a document can never silently pull
    in an artefact or component that was never actually assembled into
    that document's lineage (F1).

    A :class:`ProviderLabelSetComponent` deliberately does *not* declare
    its sibling :class:`TimedTurnSetComponent` as a component ref here:
    that would make the turn set a *mandatory* dependency
    (``_resolve_component_closure`` hard-fails a revision whose declared
    component ref is missing from the closure), whereas a label set
    missing its cues is a *soft*, inspectable capability failure
    (``speakers.provider-labels`` reports ``failed`` with the unresolved
    span IDs named) -- the label evidence itself is still a perfectly
    loadable component either way.
    """
    match record:
        case NotesComponent():
            return ComponentInputRefs(
                artefact_ids=(record.source_artefact_id,), component_ids=()
            )
        case ParticipantSetComponent():
            return ComponentInputRefs(artefact_ids=(), component_ids=())
        case UntimedTurnSetComponent():
            return ComponentInputRefs(
                artefact_ids=(record.source_artefact_id,), component_ids=()
            )
        case TimedTurnSetComponent():
            return ComponentInputRefs(
                artefact_ids=(record.source_artefact_id,), component_ids=()
            )
        case ProviderLabelSetComponent():
            return ComponentInputRefs(
                artefact_ids=(record.source_artefact_id,), component_ids=()
            )
        case TranscriptAbsenceDeclaration():
            return ComponentInputRefs(
                artefact_ids=(record.source_artefact_id,), component_ids=()
            )
        case AssemblyManifestComponent():
            return ComponentInputRefs(
                artefact_ids=tuple(
                    selection.artefact_id for selection in record.selections
                ),
                component_ids=(),
            )


def component_as_body(component: ComponentBody | ComponentRecord) -> ComponentBody:
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
        case UntimedTurnSetComponent():
            return UntimedTurnSetComponentBody(
                source_artefact_id=component.source_artefact_id, turns=component.turns
            )
        case UntimedTurnSetComponentBody():
            return component
        case TimedTurnSetComponent():
            return TimedTurnSetComponentBody(
                source_artefact_id=component.source_artefact_id, turns=component.turns
            )
        case TimedTurnSetComponentBody():
            return component
        case ProviderLabelSetComponent():
            return ProviderLabelSetComponentBody(
                source_artefact_id=component.source_artefact_id,
                proxy_config_hash=component.proxy_config_hash,
                spans=component.spans,
            )
        case ProviderLabelSetComponentBody():
            return component
        case TranscriptAbsenceDeclaration():
            return TranscriptAbsenceDeclarationBody(
                source_artefact_id=component.source_artefact_id,
                statement=component.statement,
            )
        case TranscriptAbsenceDeclarationBody():
            return component
        case AssemblyManifestComponent():
            return AssemblyManifestComponentBody(
                selections=component.selections, rationale=component.rationale
            )
        case AssemblyManifestComponentBody():
            return component


def assemble_component_record(
    body: ComponentBody,
    *,
    component_id: ComponentId,
    content_hash: Sha256Hex,
    created_at: datetime,
    mint_segment_id: Callable[[], str],
) -> ComponentRecord:
    """Attach store-minted identity to a body, producing its stored record.

    The only place a ``*Body`` becomes a ``*Component``; kept here (not in
    ``store.py``) so the closed kind set and its dispatch live next to
    each other -- adding a new component kind only ever touches this
    module. ``mint_segment_id`` is called once per notes section (never by
    this module directly minting an ID itself -- M1: the store mints,
    nowhere else) to attach each section's :data:`.ids.SegmentId`.

    The turn-set, label-set, absence-declaration, and assembly-manifest
    kinds mint no identity of their own here: their per-item IDs
    (``source_segment_id``, ``participant_id``) are already present on
    the body, minted by the caller before construction (module docstring)
    -- so attaching store identity is the same "dump the body's fields,
    re-validate them onto the ``*Component`` subclass plus its three
    identity fields" shape :class:`ParticipantSetComponentBody` already
    established, not a new pattern.
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
        case UntimedTurnSetComponentBody():
            fields = body.model_dump(mode="python")
            return UntimedTurnSetComponent(
                **fields,
                component_id=component_id,
                content_hash=content_hash,
                created_at=created_at,
            )
        case TimedTurnSetComponentBody():
            fields = body.model_dump(mode="python")
            return TimedTurnSetComponent(
                **fields,
                component_id=component_id,
                content_hash=content_hash,
                created_at=created_at,
            )
        case ProviderLabelSetComponentBody():
            fields = body.model_dump(mode="python")
            return ProviderLabelSetComponent(
                **fields,
                component_id=component_id,
                content_hash=content_hash,
                created_at=created_at,
            )
        case TranscriptAbsenceDeclarationBody():
            fields = body.model_dump(mode="python")
            return TranscriptAbsenceDeclaration(
                **fields,
                component_id=component_id,
                content_hash=content_hash,
                created_at=created_at,
            )
        case AssemblyManifestComponentBody():
            fields = body.model_dump(mode="python")
            return AssemblyManifestComponent(
                **fields,
                component_id=component_id,
                content_hash=content_hash,
                created_at=created_at,
            )
