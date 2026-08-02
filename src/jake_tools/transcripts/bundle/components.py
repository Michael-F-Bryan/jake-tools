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

import itertools
from collections.abc import Callable, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, NamedTuple, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .ids import (
    ArtefactId,
    AttemptId,
    ClusterId,
    ComponentId,
    CoordinateDomain,
    ParticipantId,
    ReviewId,
    RevisionId,
    SegmentId,
    SourceId,
    TurnId,
)
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
    """M19/M22: how a participant record came to exist.

    ``note-inferred`` (M22) is a model's reading of the note, and is kept
    as a *distinct* value rather than folded into ``operator``: recording
    a guess as an operator assertion would be a lie about exactly the
    field M19 exists to protect. It is admissible only because inferring
    who was *present* is not inferring who *said what* -- M8's ladder
    still requires a reviewed decision before any turn carries a name, so
    an inferred attendee can never become an attribution on its own.
    """

    CALENDAR = "calendar"
    TEAMS_ROSTER = "teams-roster"
    NOTE_FRONTMATTER = "note-frontmatter"
    OPERATOR = "operator"
    NOTE_INFERRED = "note-inferred"


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
    MEDIA_RECORDING = "media-recording"
    RECORDING_REFERENCE_SET = "recording-reference-set"
    DESTINATION = "destination"
    TIMELINE_COMBINED = "timeline-combined"
    ASR_RESULT = "asr-result"
    DIARISATION_RESULT = "diarisation-result"
    NORMALISATION_LEDGER = "normalisation-ledger"
    MACHINE_ATTRIBUTION_SET = "machine-attribution-set"
    SPEAKER_HYPOTHESIS_SET = "speaker-hypothesis-set"
    SPEAKER_REVIEW = "speaker-review"
    TEXT_EDIT_LEDGER = "text-edit-ledger"
    CHAPTER_SET = "chapter-set"
    MINUTES = "minutes"


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
    time (M7), never by this module. ``turn_id`` is the M7 *editorial*
    node ID -- a different concept from the immutable source-segment ID,
    and the thing chapters, review decisions, and M10 minutes evidence
    all bind to. The absence of any ``start_ms``/``end_ms`` field is
    itself the enforcement mechanism behind ``transcript.untimed``'s "no
    timing fields present" check (``registry.py``) -- there is no value
    this type could hold that would smuggle timing in; the type simply
    has no such field to set.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    turn_id: TurnId
    source_segment_id: SegmentId
    speaker_label: str = Field(min_length=1)
    text: str = Field(min_length=1)


def _check_distinct_turn_ids(turn_ids: Sequence[str]) -> None:
    """M7: ``turn_id`` is the editorial node identity every binding
    (chapters, review decisions, minutes evidence) resolves against, so a
    duplicate inside one turn set would make those bindings ambiguous --
    refused at construction rather than discovered when a chapter
    silently covers two different turns.
    """
    duplicates = sorted(
        {turn_id for turn_id in turn_ids if turn_ids.count(turn_id) > 1}
    )
    if duplicates:
        raise ValueError(
            f"duplicate turn_id(s) within one turn set: {duplicates} -- M7 turn IDs "
            "identify one editorial node each."
        )


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

    @model_validator(mode="after")
    def _check_turn_ids(self) -> Self:
        _check_distinct_turn_ids([turn.turn_id for turn in self.turns])
        return self


class UntimedTurnSetComponent(UntimedTurnSetComponentBody):
    component_id: ComponentId
    content_hash: Sha256Hex
    created_at: datetime


class TimedTurn(BaseModel):
    """M6: one canonical timed turn -- half-open ``[start_ms, end_ms)``,
    ``end_ms > start_ms`` strictly (zero-length cues are legal only in
    raw source evidence, never in a canonical timed turn -- M6).

    ``start_ms``/``end_ms`` are expressed in the *set's* declared
    coordinate domain (:attr:`TimedTurnSetComponentBody.coordinate_domain`),
    not necessarily in ``source_artefact_id``'s own domain: a canonical
    multi-recording set lives in ``combined:<component_id>`` while each
    turn still records which recording its evidence came from, which is
    what M8's source-range review scope and M6's reversible mappings both
    need.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    turn_id: TurnId
    source_segment_id: SegmentId
    source_artefact_id: ArtefactId
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
    source_artefact_ids: tuple[ArtefactId, ...] = Field(min_length=1)
    coordinate_domain: CoordinateDomain
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
        _check_distinct_turn_ids([turn.turn_id for turn in self.turns])
        declared = set(self.source_artefact_ids)
        undeclared = sorted(
            {
                turn.source_artefact_id
                for turn in self.turns
                if turn.source_artefact_id not in declared
            }
        )
        if undeclared:
            raise ValueError(
                f"turn(s) cite source artefact(s) this set does not declare: "
                f"{undeclared} -- source_artefact_ids is the set's own input "
                "closure (M16) and must name every artefact its turns come from."
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


# -- M13: destination component (note snapshot, target identity, owned region) -


class OwnedRegionState(StrEnum):
    """M13: what the note's own content shows about a prior jake-tools
    write, detected only -- never written -- by the ingest adapter.
    ``markers-present`` means the explicit
    ``<!-- jake-tools:transcript:begin/end -->`` comment pair already
    exists; ``legacy-headings`` means no markers but at least one of
    ``merge.py``'s ``GENERATED_HEADINGS`` is present (the migration case
    M13 describes); ``none`` means neither -- a first-ever write.
    """

    MARKERS_PRESENT = "markers-present"
    LEGACY_HEADINGS = "legacy-headings"
    NONE = "none"


class DestinationComponentBody(BaseModel):
    """M13: the apply target's identity -- vault-relative path plus the
    note snapshot artefact (content hash) taken at ingest -- and the
    owned-region state observed in that snapshot. Detection only: render
    and apply (which would use this to write back) are a later slice
    (CONTRACTS.md M13 scope note); this component only records what was
    observed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_kind: Literal[ComponentKind.DESTINATION] = ComponentKind.DESTINATION
    note_artefact_id: ArtefactId
    vault_relative_path: str = Field(min_length=1)
    owned_region_state: OwnedRegionState


class DestinationComponent(DestinationComponentBody):
    component_id: ComponentId
    content_hash: Sha256Hex
    created_at: datetime


# -- M3: recording references from note embeds --------------------------


class RecordingReference(BaseModel):
    """M3: one embed link found in a note, plus the local path it resolves
    to. A reference alone never satisfies ``media.recording`` (M3) -- the
    ``media.recording`` validator (``registry.py``) matches
    ``resolved_path`` against a separately-ingested
    :class:`MediaRecordingComponent`'s own ``media_path`` to decide
    whether this reference is backed by ingested media or is
    reference-only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_link: str = Field(min_length=1)
    resolved_path: str = Field(min_length=1)


class RecordingReferenceSetComponentBody(BaseModel):
    """M3/M6: every recording embed in one note, in note-embed (textual)
    order -- the M6 default ordering for the combined timeline. Order is
    preserved exactly as found, never re-sorted (unlike
    :class:`ParticipantSetComponentBody`, for which M19 defines no
    meaningful order) -- this is deliberately *not*
    ``obsidian.py``'s ``SourceNote.recordings`` (which re-sorts by file
    creation time for the older scribe-based recipe); the adapter reads
    ``extract_recording_links``/``resolve_recording_path`` directly to
    keep the note's own textual order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_kind: Literal[ComponentKind.RECORDING_REFERENCE_SET] = (
        ComponentKind.RECORDING_REFERENCE_SET
    )
    note_artefact_id: ArtefactId
    references: tuple[RecordingReference, ...] = Field(min_length=1)


class RecordingReferenceSetComponent(RecordingReferenceSetComponentBody):
    component_id: ComponentId
    content_hash: Sha256Hex
    created_at: datetime


# -- M4/M11: ingested media recording ------------------------------------


class MediaRecordingComponentBody(BaseModel):
    """M4: the proof behind ``media.recording`` -- an **ingested** media
    artefact with a known duration, which is what defines its
    ``source:<artefact_id>`` coordinate domain (M6: "origin = media start
    = 0"). ``media_path`` is the same resolved filesystem path used as the
    artefact's own ``acquisition_locator`` at ingest -- the join key the
    ``media.recording`` validator uses to match this component against a
    note's own :class:`RecordingReferenceSetComponent` entries (M3).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_kind: Literal[ComponentKind.MEDIA_RECORDING] = (
        ComponentKind.MEDIA_RECORDING
    )
    source_artefact_id: ArtefactId
    media_path: str = Field(min_length=1)
    duration_ms: int = Field(gt=0)
    codec: str = Field(min_length=1)
    sample_rate_hz: int = Field(gt=0)
    channels: int = Field(gt=0)


class MediaRecordingComponent(MediaRecordingComponentBody):
    component_id: ComponentId
    content_hash: Sha256Hex
    created_at: datetime


# -- M6: combined timeline ------------------------------------------------


class TimelineMappingSegment(BaseModel):
    """M6: one piecewise mapping ``{artefact, [src_start, src_end) ->
    [dst_start, dst_end)}``. Both spans are half-open with a strictly
    positive length, and the mapping is a pure shift -- it must preserve
    span length exactly, never stretch or compress time.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    artefact_id: ArtefactId
    source_start_ms: int = Field(ge=0)
    source_end_ms: int = Field(ge=0)
    combined_start_ms: int = Field(ge=0)
    combined_end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_spans(self) -> Self:
        if self.source_end_ms <= self.source_start_ms:
            raise ValueError(
                f"source span [{self.source_start_ms}, {self.source_end_ms}) is not "
                "half-open with end > start (M6)."
            )
        if self.combined_end_ms <= self.combined_start_ms:
            raise ValueError(
                f"combined span [{self.combined_start_ms}, {self.combined_end_ms}) is "
                "not half-open with end > start (M6)."
            )
        source_len = self.source_end_ms - self.source_start_ms
        combined_len = self.combined_end_ms - self.combined_start_ms
        if source_len != combined_len:
            raise ValueError(
                f"mapping segment for {self.artefact_id} does not preserve span "
                f"length (source {source_len}ms, combined {combined_len}ms) -- a "
                "combined-timeline mapping is a pure shift (M6)."
            )
        return self


class TimelineGapRecord(BaseModel):
    """M6: an explicit inter-recording gap in the combined domain.
    Zero-length gaps are legal (adjacent recordings with no declared
    real-world break); ``combined_end_ms < combined_start_ms`` never is.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    after_artefact_id: ArtefactId
    before_artefact_id: ArtefactId
    combined_start_ms: int = Field(ge=0)
    combined_end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_span(self) -> Self:
        if self.combined_end_ms < self.combined_start_ms:
            raise ValueError(
                f"gap [{self.combined_start_ms}, {self.combined_end_ms}) has end < "
                "start (M6)."
            )
        return self


class TimelineCombinedComponentBody(BaseModel):
    """M6: the assembled multi-recording timeline. ``segments`` must
    already be in ascending ``combined_start_ms`` order and must not
    overlap -- v1's assembly policy is sequential, non-overlapping
    fragments (M6); an overlapping pair is refused here, at construction,
    rather than silently flattened. This is the enforcement mechanism
    behind M6's "overlapping recordings are unsupported in v1 --
    assembly fails with an explicit error."
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_kind: Literal[ComponentKind.TIMELINE_COMBINED] = (
        ComponentKind.TIMELINE_COMBINED
    )
    segments: tuple[TimelineMappingSegment, ...] = Field(min_length=1)
    gaps: tuple[TimelineGapRecord, ...] = ()

    @model_validator(mode="after")
    def _check_sequential_non_overlapping(self) -> Self:
        ordered = sorted(self.segments, key=lambda segment: segment.combined_start_ms)
        if list(ordered) != list(self.segments):
            raise ValueError(
                "segments must be listed in ascending combined_start_ms order (M6)."
            )
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            if later.combined_start_ms < earlier.combined_end_ms:
                raise ValueError(
                    f"overlapping recordings in the combined timeline: "
                    f"{earlier.artefact_id} ends at {earlier.combined_end_ms}ms but "
                    f"{later.artefact_id} starts at {later.combined_start_ms}ms -- "
                    "unsupported in v1 (M6); assembly must fail explicitly rather "
                    "than silently flatten this."
                )
        return self


class TimelineCombinedComponent(TimelineCombinedComponentBody):
    component_id: ComponentId
    content_hash: Sha256Hex
    created_at: datetime


# -- M11: worker inference results (ASR / diarisation) -------------------


class AsrResultComponentBody(BaseModel):
    """M4/M11: the proof behind ``inference.asr`` for one media artefact.
    Only ever constructed for a *completed* ASR stage (M11's partial-
    failure semantics: a failed stage never becomes a component -- it
    stays absent for this member until a retry succeeds, which is also
    what keeps a failed sibling stage from blocking this one's own
    capability). ``request_fingerprint`` is the caller's own
    deterministic hash over the inputs/config it controls (audio hash,
    declared model identity, audio-preparation config) -- computed
    *before* invoking the worker, so resume-by-hash matching never
    depends on anything the worker itself resolves at runtime (e.g. a
    locally-cached model revision). ``worker_config_hash`` is the
    worker's own reported stage config hash, kept for provenance/audit
    only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_kind: Literal[ComponentKind.ASR_RESULT] = ComponentKind.ASR_RESULT
    media_artefact_id: ArtefactId
    result_artefact_id: ArtefactId
    attempt_id: AttemptId
    request_fingerprint: Sha256Hex
    worker_config_hash: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)


class AsrResultComponent(AsrResultComponentBody):
    component_id: ComponentId
    content_hash: Sha256Hex
    created_at: datetime


class DiarisationResultComponentBody(BaseModel):
    """M4/M11: the ``inference.diarisation`` proof, mirroring
    :class:`AsrResultComponentBody` exactly (see its docstring)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_kind: Literal[ComponentKind.DIARISATION_RESULT] = (
        ComponentKind.DIARISATION_RESULT
    )
    media_artefact_id: ArtefactId
    result_artefact_id: ArtefactId
    attempt_id: AttemptId
    request_fingerprint: Sha256Hex
    worker_config_hash: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)


class DiarisationResultComponent(DiarisationResultComponentBody):
    component_id: ComponentId
    content_hash: Sha256Hex
    created_at: datetime


# -- M7: normalisation lineage and coverage ledgers ------------------------


class DropReason(StrEnum):
    """M7's closed lineage vocabulary for what normalisation removes.

    Deliberately only the two dispositions normalisation itself produces:
    ``split_from``/``merged_from`` are created where splits and merges
    actually happen (the M9 text transforms), not here.
    """

    DROPPED_AS_DUPLICATE = "dropped_as_duplicate"
    DROPPED_AS_EMPTY = "dropped_as_empty"


class DroppedSegmentRecord(BaseModel):
    """M7: one raw source segment normalisation removed, and why.

    A duplicate must name the segment that was *retained* in its place --
    that reference is what makes the raw-segment ledger auditable rather
    than merely a count; an empty (zero-length) segment has no retained
    counterpart and must not claim one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_segment_id: SegmentId
    source_artefact_id: ArtefactId
    reason: DropReason
    retained_source_segment_id: SegmentId | None = None
    detail: str = ""

    @model_validator(mode="after")
    def _check_retained_ref_matches_reason(self) -> Self:
        if self.reason == DropReason.DROPPED_AS_DUPLICATE:
            if self.retained_source_segment_id is None:
                raise ValueError(
                    "a dropped_as_duplicate record must name the retained segment "
                    "it was a duplicate of (M7)."
                )
        elif self.retained_source_segment_id is not None:
            raise ValueError(
                f"reason {self.reason.value!r} must not name a retained segment -- "
                "only dropped_as_duplicate has one (M7)."
            )
        return self


class CoverageLedger(BaseModel):
    """M7: one coverage universe's accounting. ``accounted + dropped ==
    total`` exactly -- a ledger that does not add up is the failure mode
    this type exists to make unrepresentable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    total: int = Field(ge=0)
    accounted: int = Field(ge=0)
    dropped: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_ledger_balances(self) -> Self:
        if self.accounted + self.dropped != self.total:
            raise ValueError(
                f"coverage ledger does not balance: accounted={self.accounted} + "
                f"dropped={self.dropped} != total={self.total} (M7)."
            )
        return self


class NormalisationLedgerComponentBody(BaseModel):
    """M7: the audit trail behind one normalisation pass.

    Two of M7's three coverage universes live here, never conflated:
    ``raw_source_segments`` (every raw ASR token this pass saw, and what
    became of it) and ``canonical_turns`` (the editorial sequence it
    produced). The third -- rendered turns -- is a render-time concern and
    lives on the render record (M17).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_kind: Literal[ComponentKind.NORMALISATION_LEDGER] = (
        ComponentKind.NORMALISATION_LEDGER
    )
    source_artefact_ids: tuple[ArtefactId, ...] = Field(min_length=1)
    config_hash: Sha256Hex
    raw_source_segments: CoverageLedger
    canonical_turns: CoverageLedger
    dropped: tuple[DroppedSegmentRecord, ...] = ()

    @model_validator(mode="after")
    def _check_dropped_matches_ledger(self) -> Self:
        if len(self.dropped) != self.raw_source_segments.dropped:
            raise ValueError(
                f"{len(self.dropped)} dropped-segment record(s) but the raw-segment "
                f"ledger claims {self.raw_source_segments.dropped} dropped -- the "
                "ledger must account for exactly the records it carries (M7)."
            )
        return self


class NormalisationLedgerComponent(NormalisationLedgerComponentBody):
    component_id: ComponentId
    content_hash: Sha256Hex
    created_at: datetime


# -- M4/M5: machine voice clusters and their per-turn attribution -----------


class SpeakerCluster(BaseModel):
    """M5: one machine voice cluster, scoped to a single diarisation
    output over a single media artefact. ``raw_label`` (``SPEAKER_00``) is
    retained as a *field*, never as identity (F13): a rerun mints new
    cluster IDs, and the same raw label on a different recording is a
    different cluster.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cluster_id: ClusterId
    raw_label: str = Field(min_length=1)
    media_artefact_id: ArtefactId
    diarisation_artefact_id: ArtefactId
    segment_count: int = Field(ge=1)
    total_ms: int = Field(ge=0)


class TurnClusterAssignment(BaseModel):
    """M7: one canonical turn's machine hypothesis, with the overlap that
    justified it (maximal-overlap token->segment rule)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    turn_id: TurnId
    cluster_id: ClusterId
    overlap_ms: int = Field(ge=0)


class MachineAttributionSetComponentBody(BaseModel):
    """M4's ``speakers.machine-clustered`` proof: clusters exist for the
    timed set, and each canonical turn is either attributed to exactly one
    of them or explicitly listed as unattributed.

    Machine clusters are **not** participant identities (F13, corpus §6):
    nothing here names a participant, and no rung above M8's rung 7 can
    read this component -- which is the structural reason a bare machine
    hypothesis can never satisfy the meeting-note speaker gate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_kind: Literal[ComponentKind.MACHINE_ATTRIBUTION_SET] = (
        ComponentKind.MACHINE_ATTRIBUTION_SET
    )
    clusters: tuple[SpeakerCluster, ...] = Field(min_length=1)
    assignments: tuple[TurnClusterAssignment, ...] = ()
    unattributed_turn_ids: tuple[TurnId, ...] = ()

    @model_validator(mode="after")
    def _check_cluster_and_turn_identity(self) -> Self:
        cluster_ids = [cluster.cluster_id for cluster in self.clusters]
        if len(set(cluster_ids)) != len(cluster_ids):
            raise ValueError("duplicate cluster_id within one attribution set (M1).")
        undeclared = sorted(
            {
                assignment.cluster_id
                for assignment in self.assignments
                if assignment.cluster_id not in set(cluster_ids)
            }
        )
        if undeclared:
            raise ValueError(
                f"assignment(s) name cluster(s) this set does not declare: {undeclared}."
            )
        turn_ids = [assignment.turn_id for assignment in self.assignments] + list(
            self.unattributed_turn_ids
        )
        _check_distinct_turn_ids(turn_ids)
        return self


class MachineAttributionSetComponent(MachineAttributionSetComponentBody):
    component_id: ComponentId
    content_hash: Sha256Hex
    created_at: datetime


# -- M8 rung 7: machine hypotheses (proposals, never assignments) -----------


class SpeakerHypothesis(BaseModel):
    """M8 rung 7: the machine's *proposed* participant for one cluster.

    ``participant_id`` may be ``None`` -- "no candidate the evidence
    supports" is a first-class outcome, not a gap to be filled by
    guessing. A hypothesis renders into a transcript honestly labelled as
    a hypothesis, and never into a meeting note (M5).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cluster_id: ClusterId
    participant_id: ParticipantId | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_turn_ids: tuple[TurnId, ...] = ()
    rationale: str = Field(min_length=1)


class SpeakerHypothesisSetComponentBody(BaseModel):
    """M8: one proposal pass's hypotheses, at most one per cluster."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_kind: Literal[ComponentKind.SPEAKER_HYPOTHESIS_SET] = (
        ComponentKind.SPEAKER_HYPOTHESIS_SET
    )
    proposer: str = Field(min_length=1)
    config_hash: Sha256Hex
    hypotheses: tuple[SpeakerHypothesis, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _one_hypothesis_per_cluster(self) -> Self:
        cluster_ids = [hypothesis.cluster_id for hypothesis in self.hypotheses]
        if len(set(cluster_ids)) != len(cluster_ids):
            raise ValueError(
                "more than one hypothesis for the same cluster -- rung 7 resolves "
                "one candidate per cluster, so competing proposals must be "
                "reconciled by the proposer, not left for the ladder to guess (M8)."
            )
        return self


class SpeakerHypothesisSetComponent(SpeakerHypothesisSetComponentBody):
    component_id: ComponentId
    content_hash: Sha256Hex
    created_at: datetime


# -- M8: the applied human review ------------------------------------------


class ReviewScope(StrEnum):
    """M8's four decision scopes, in precedence order (rungs 1-4)."""

    TURN = "turn"
    SOURCE_RANGE = "source-range"
    CLUSTER = "cluster"
    PROVIDER_LABEL = "provider-label"


class ReviewDecisionKind(StrEnum):
    """M8: an assignment, or the reviewer's explicit uncertainty.

    ``unclear-speaker`` is a *decision*, not an absence: at its scope it
    blocks every weaker rung, so a reviewer who has genuinely looked and
    cannot tell is never overridden by a machine guess.
    """

    ASSIGN = "assign"
    UNCLEAR_SPEAKER = "unclear-speaker"


class _ReviewDecisionBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ReviewDecisionKind
    participant_id: ParticipantId | None = None
    rationale: str = ""

    @model_validator(mode="after")
    def _check_participant_matches_kind(self) -> Self:
        if self.kind == ReviewDecisionKind.ASSIGN:
            if self.participant_id is None:
                raise ValueError("an 'assign' decision must name a participant (M8).")
        elif self.participant_id is not None:
            raise ValueError(
                "an 'unclear-speaker' decision must not name a participant -- it is "
                "projected as participant_unresolved, never stored as a record (M1)."
            )
        return self


class TurnDecision(_ReviewDecisionBase):
    """M8 rung 1: the narrowest, highest-precedence override."""

    scope: Literal[ReviewScope.TURN] = ReviewScope.TURN
    turn_id: TurnId


class SourceRangeDecision(_ReviewDecisionBase):
    """M8 rung 2: a half-open ``[start_ms, end_ms)`` window in one source
    recording's own domain -- deliberately *source* coordinates, so a
    reviewer's range decision survives a combined timeline being rebuilt.
    """

    scope: Literal[ReviewScope.SOURCE_RANGE] = ReviewScope.SOURCE_RANGE
    source_artefact_id: ArtefactId
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_half_open_span(self) -> Self:
        if self.end_ms <= self.start_ms:
            raise ValueError(
                f"review range [{self.start_ms}, {self.end_ms}) is not half-open "
                "with end_ms > start_ms (M6)."
            )
        return self


class ClusterDecision(_ReviewDecisionBase):
    """M8 rung 3: a voice-specific default for one machine cluster."""

    scope: Literal[ReviewScope.CLUSTER] = ReviewScope.CLUSTER
    cluster_id: ClusterId


class ProviderLabelDecision(_ReviewDecisionBase):
    """M8 rung 4: a default for one raw provider label. Outranked by rung
    3 because a room-proxy label describes an aggregate, not a voice (D1).
    """

    scope: Literal[ReviewScope.PROVIDER_LABEL] = ReviewScope.PROVIDER_LABEL
    raw_label: str = Field(min_length=1)


ReviewDecision = Annotated[
    TurnDecision | SourceRangeDecision | ClusterDecision | ProviderLabelDecision,
    Field(discriminator="scope"),
]


class SpeakerReviewComponentBody(BaseModel):
    """M8: one applied ``ReviewDecisionSet``, bound to the exact revision
    and inventories it was produced against.

    ``pack_item_ids`` is the full list of items the exported pack asked
    the reviewer to address; which of them a decision actually targets is
    derived, never stored -- that is what makes "a partial review" (some
    items never addressed) mechanically distinguishable from "a complete
    review with unresolved items" (every item addressed, some as
    ``unclear-speaker``) without a flag anyone could set wrongly (M8).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_kind: Literal[ComponentKind.SPEAKER_REVIEW] = ComponentKind.SPEAKER_REVIEW
    review_id: ReviewId
    input_revision_id: RevisionId
    turn_inventory_hash: Sha256Hex
    cluster_inventory_hash: Sha256Hex
    pack_schema_version: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    pack_item_ids: tuple[str, ...] = ()
    decisions: tuple[ReviewDecision, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_decision_targets(self) -> Self:
        targets: list[tuple[str, str]] = []
        ranges: list[SourceRangeDecision] = []
        for decision in self.decisions:
            match decision:
                case TurnDecision():
                    targets.append((decision.scope.value, decision.turn_id))
                case ClusterDecision():
                    targets.append((decision.scope.value, decision.cluster_id))
                case ProviderLabelDecision():
                    targets.append((decision.scope.value, decision.raw_label))
                case SourceRangeDecision():
                    ranges.append(decision)
        duplicates = sorted({t for t in targets if targets.count(t) > 1})
        if duplicates:
            raise ValueError(
                f"more than one decision for the same scope+target: {duplicates} -- "
                "M8's ladder resolves one decision per scope, so a conflict must be "
                "rejected at review validation, not resolved by precedence."
            )
        for first, second in itertools.combinations(ranges, 2):
            if first.source_artefact_id != second.source_artefact_id:
                continue
            if first.start_ms < second.end_ms and second.start_ms < first.end_ms:
                raise ValueError(
                    f"overlapping source-range decisions on {first.source_artefact_id}: "
                    f"[{first.start_ms}, {first.end_ms}) and [{second.start_ms}, "
                    f"{second.end_ms}) -- rejected at review validation (M8)."
                )
        return self


class SpeakerReviewComponent(SpeakerReviewComponentBody):
    component_id: ComponentId
    content_hash: Sha256Hex
    created_at: datetime


# -- M9: text-edit ledger ---------------------------------------------------


class TextEditMode(StrEnum):
    """Transcript text/shape passes, kept distinct in provenance."""

    REFLOW = "reflow"
    CORRECT = "correct"
    POLISH = "polish"


class TextEditOperation(StrEnum):
    """M7's closed v1 remap set, as it appears in an M9 ledger.

    ``split`` is admitted by the type but refused by the polish/correct
    transforms in v1 (no fixture needs it and M7's remap set does not
    define how bindings follow a split); it exists here so a later slice
    that does need it records the lineage rather than inventing a new one.
    """

    IDENTITY = "identity"
    TEXT_EDIT = "text-edit"
    MERGE = "merge"
    SPLIT = "split"
    DROP_EMPTY = "drop-empty"


class RemovalReason(StrEnum):
    """M9's closed removal-reason enum. Nothing else is a legal reason to
    remove words from transcript text."""

    FILLER = "filler"
    STUTTER_REPEAT = "stutter-repeat"
    FALSE_START = "false-start"
    NON_LEXICAL = "non-lexical"
    DUPLICATE = "duplicate"


class TextEditEntry(BaseModel):
    """M9: one accounted-for change (or non-change) between the input and
    output turn sets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: TextEditOperation
    input_turn_ids: tuple[TurnId, ...] = Field(min_length=1)
    output_turn_ids: tuple[TurnId, ...] = ()
    old_text_sha256: Sha256Hex
    new_text_sha256: Sha256Hex | None = None
    removal_reasons: tuple[RemovalReason, ...] = ()
    evidence_ref: str = ""

    @model_validator(mode="after")
    def _check_operation_shape(self) -> Self:
        match self.operation:
            case TextEditOperation.DROP_EMPTY:
                if self.output_turn_ids or self.new_text_sha256 is not None:
                    raise ValueError(
                        "a drop-empty entry produces no output turn and no new text."
                    )
                if not self.removal_reasons:
                    raise ValueError(
                        "a drop-empty entry must record why the turn emptied (M9's "
                        "closed removal-reason enum)."
                    )
            case TextEditOperation.MERGE:
                if len(self.input_turn_ids) < 2 or len(self.output_turn_ids) != 1:
                    raise ValueError(
                        "a merge entry takes two or more input turns and produces "
                        "exactly one output turn (M7)."
                    )
            case TextEditOperation.SPLIT:
                if len(self.input_turn_ids) != 1 or len(self.output_turn_ids) < 2:
                    raise ValueError(
                        "a split entry takes one input turn and produces two or "
                        "more output turns (M7)."
                    )
            case TextEditOperation.IDENTITY | TextEditOperation.TEXT_EDIT:
                if len(self.input_turn_ids) != 1 or len(self.output_turn_ids) != 1:
                    raise ValueError(
                        f"a {self.operation.value} entry maps exactly one turn to "
                        "one turn (M7: text-only edits retain the turn_id)."
                    )
                if self.input_turn_ids != self.output_turn_ids:
                    raise ValueError(
                        f"a {self.operation.value} entry must retain the turn_id "
                        "(M7 identity remap); a changed ID is a split or a merge."
                    )
        if self.operation != TextEditOperation.DROP_EMPTY and (
            self.new_text_sha256 is None
        ):
            raise ValueError(
                f"a {self.operation.value} entry must record the resulting text hash."
            )
        if (
            self.operation == TextEditOperation.IDENTITY
            and self.new_text_sha256 != self.old_text_sha256
        ):
            raise ValueError(
                "an identity entry's text hash must be unchanged -- a changed hash "
                "is a text-edit and must be recorded as one (M9)."
            )
        return self


class TextEditLedgerComponentBody(BaseModel):
    """M9: the complete diff accounting behind one ``text.corrected`` /
    ``text.polished`` proof.

    ``input_turn_set_component_id`` and ``output_turn_set_component_id``
    are recorded for audit but deliberately **not** declared as component
    input refs (:func:`component_input_refs`): the input turn set is
    superseded by this very transform (M21), so a hard ref to it would
    dangle the moment the correction landed, and the output set may itself
    be superseded by a later pass. Component-to-component hard refs are
    only safe to declare for components nothing supersedes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_kind: Literal[ComponentKind.TEXT_EDIT_LEDGER] = (
        ComponentKind.TEXT_EDIT_LEDGER
    )
    mode: TextEditMode
    input_turn_set_component_id: ComponentId
    output_turn_set_component_id: ComponentId
    editor: str = Field(min_length=1)
    config_hash: Sha256Hex
    entries: tuple[TextEditEntry, ...] = Field(min_length=1)


class TextEditLedgerComponent(TextEditLedgerComponentBody):
    component_id: ComponentId
    content_hash: Sha256Hex
    created_at: datetime


# -- M10: chapters ----------------------------------------------------------


class ChapterRecord(BaseModel):
    """M10: one chapter, citing the exact ordered turn range it covers.

    Titles and summaries derive only from covered turns -- ``turn_ids`` is
    both the citation and the coverage claim the ``chapters`` validator
    checks against the canonical turn sequence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = Field(min_length=1)
    summary: str = ""
    turn_ids: tuple[TurnId, ...] = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_span(self) -> Self:
        if self.end_ms <= self.start_ms:
            raise ValueError(
                f"chapter {self.title!r} span [{self.start_ms}, {self.end_ms}) is "
                "not half-open with end_ms > start_ms (M6)."
            )
        return self


class ChapterSetComponentBody(BaseModel):
    """M4's ``chapters`` proof: an ordered partition of the canonical turn
    sequence. Disjointness is enforced here; *exact* coverage needs the
    turn set too and is enforced by the registry validator.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_kind: Literal[ComponentKind.CHAPTER_SET] = ComponentKind.CHAPTER_SET
    chapters: tuple[ChapterRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_disjoint_and_ordered(self) -> Self:
        seen: set[str] = set()
        for chapter in self.chapters:
            overlap = sorted(seen.intersection(chapter.turn_ids))
            if overlap:
                raise ValueError(
                    f"chapter {chapter.title!r} covers turn(s) already covered by an "
                    f"earlier chapter: {overlap} -- every canonical turn belongs to "
                    "exactly one chapter (M7)."
                )
            seen.update(chapter.turn_ids)
        starts = [chapter.start_ms for chapter in self.chapters]
        if starts != sorted(starts):
            raise ValueError("chapters must be listed in ascending start_ms order.")
        return self


class ChapterSetComponent(ChapterSetComponentBody):
    component_id: ComponentId
    content_hash: Sha256Hex
    created_at: datetime


# -- M10/M20: minutes -------------------------------------------------------


class FindingKind(StrEnum):
    DECISION = "decision"
    ACTION = "action"
    RISK = "risk"
    QUESTION = "question"


class ClaimStatus(StrEnum):
    """M10: where a claim came from. Rendered visibly distinct (D2, F22),
    which is why it is a stored field and not a rendering heuristic."""

    TRANSCRIPT_DERIVED = "transcript-derived"
    NOTES_DERIVED = "notes-derived"
    MIXED = "mixed"


class CommitmentStatus(StrEnum):
    """The meeting's actual level of commitment to a finding."""

    PROPOSED = "proposed"
    TENTATIVE = "tentative"
    AGREED = "agreed"
    DECIDED = "decided"
    UNRESOLVED = "unresolved"
    DISCUSSED = "discussed"


class _EvidencedClaim(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    claim_status: ClaimStatus
    evidence_turn_ids: tuple[TurnId, ...] = ()
    evidence_section_ids: tuple[SegmentId, ...] = ()

    @model_validator(mode="after")
    def _check_evidence_matches_claim_status(self) -> Self:
        has_turns = bool(self.evidence_turn_ids)
        has_sections = bool(self.evidence_section_ids)
        if not has_turns and not has_sections:
            raise ValueError(
                "every minutes claim carries at least one evidence ref -- turn IDs "
                "and/or notes section IDs (M10); an unsourced claim is invalid, not "
                "merely unattributed."
            )
        expected = {
            (True, False): ClaimStatus.TRANSCRIPT_DERIVED,
            (False, True): ClaimStatus.NOTES_DERIVED,
            (True, True): ClaimStatus.MIXED,
        }[(has_turns, has_sections)]
        if self.claim_status != expected:
            raise ValueError(
                f"claim_status {self.claim_status.value!r} disagrees with the "
                f"evidence actually cited (expected {expected.value!r}) -- the "
                "status is a fact about the refs, not a label a stage may choose."
            )
        return self


class MinutesSummary(_EvidencedClaim):
    """M9/M10: even the summary is evidence-linked. Editorial synthesis is
    legal in a minutes component precisely *because* every claim in it
    carries refs -- prose with no ref is the failure mode this closes."""


class MinutesFinding(_EvidencedClaim):
    """M10: one decision, action, risk, or question.

    ``owner_participant_id`` may only ever be a participant record's ID
    (F21): owners are never invented and never inferred from a display
    name appearing in the text.
    """

    kind: FindingKind
    commitment_status: CommitmentStatus = CommitmentStatus.DISCUSSED
    owner_participant_id: ParticipantId | None = None
    due: str = ""

    @model_validator(mode="before")
    @classmethod
    def _default_legacy_modality(cls, data: object) -> object:
        if not isinstance(data, dict) or "commitment_status" in data:
            return data
        normalised = dict(data)
        kind = normalised.get("kind")
        if kind in (FindingKind.DECISION, FindingKind.DECISION.value):
            status = CommitmentStatus.DECIDED
        elif kind in (FindingKind.ACTION, FindingKind.ACTION.value):
            status = CommitmentStatus.AGREED
        elif kind in (FindingKind.QUESTION, FindingKind.QUESTION.value):
            status = CommitmentStatus.UNRESOLVED
        else:
            status = CommitmentStatus.DISCUSSED
        normalised["commitment_status"] = status
        return normalised

    @model_validator(mode="after")
    def _check_commitment_kind(self) -> Self:
        if self.kind == FindingKind.DECISION and self.commitment_status not in (
            CommitmentStatus.AGREED,
            CommitmentStatus.DECIDED,
        ):
            raise ValueError("a decision must be evidenced as agreed or decided.")
        if self.kind == FindingKind.ACTION and self.commitment_status not in (
            CommitmentStatus.AGREED,
            CommitmentStatus.DECIDED,
        ):
            raise ValueError("an action must be evidenced as agreed or decided.")
        if (
            self.kind == FindingKind.QUESTION
            and self.commitment_status == CommitmentStatus.DECIDED
        ):
            raise ValueError("a decided matter is not an open question.")
        return self


class MinutesComponentBody(BaseModel):
    """M4's ``minutes`` proof: evidence-linked findings only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_kind: Literal[ComponentKind.MINUTES] = ComponentKind.MINUTES
    summary: MinutesSummary
    findings: tuple[MinutesFinding, ...] = ()
    author: str = Field(min_length=1)
    config_hash: Sha256Hex


class MinutesComponent(MinutesComponentBody):
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
    | AssemblyManifestComponentBody
    | DestinationComponentBody
    | RecordingReferenceSetComponentBody
    | MediaRecordingComponentBody
    | TimelineCombinedComponentBody
    | AsrResultComponentBody
    | DiarisationResultComponentBody
    | NormalisationLedgerComponentBody
    | MachineAttributionSetComponentBody
    | SpeakerHypothesisSetComponentBody
    | SpeakerReviewComponentBody
    | TextEditLedgerComponentBody
    | ChapterSetComponentBody
    | MinutesComponentBody,
    Field(discriminator="component_kind"),
]
ComponentRecord = Annotated[
    NotesComponent
    | ParticipantSetComponent
    | UntimedTurnSetComponent
    | TimedTurnSetComponent
    | ProviderLabelSetComponent
    | TranscriptAbsenceDeclaration
    | AssemblyManifestComponent
    | DestinationComponent
    | RecordingReferenceSetComponent
    | MediaRecordingComponent
    | TimelineCombinedComponent
    | AsrResultComponent
    | DiarisationResultComponent
    | NormalisationLedgerComponent
    | MachineAttributionSetComponent
    | SpeakerHypothesisSetComponent
    | SpeakerReviewComponent
    | TextEditLedgerComponent
    | ChapterSetComponent
    | MinutesComponent,
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
                artefact_ids=record.source_artefact_ids, component_ids=()
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
        case DestinationComponent():
            return ComponentInputRefs(
                artefact_ids=(record.note_artefact_id,), component_ids=()
            )
        case RecordingReferenceSetComponent():
            return ComponentInputRefs(
                artefact_ids=(record.note_artefact_id,), component_ids=()
            )
        case MediaRecordingComponent():
            return ComponentInputRefs(
                artefact_ids=(record.source_artefact_id,), component_ids=()
            )
        case TimelineCombinedComponent():
            return ComponentInputRefs(
                artefact_ids=tuple(segment.artefact_id for segment in record.segments),
                component_ids=(),
            )
        case AsrResultComponent():
            return ComponentInputRefs(
                artefact_ids=(record.media_artefact_id, record.result_artefact_id),
                component_ids=(),
            )
        case DiarisationResultComponent():
            return ComponentInputRefs(
                artefact_ids=(record.media_artefact_id, record.result_artefact_id),
                component_ids=(),
            )
        case NormalisationLedgerComponent():
            return ComponentInputRefs(
                artefact_ids=record.source_artefact_ids, component_ids=()
            )
        case MachineAttributionSetComponent():
            return ComponentInputRefs(
                artefact_ids=tuple(
                    sorted(
                        {
                            artefact_id
                            for cluster in record.clusters
                            for artefact_id in (
                                cluster.media_artefact_id,
                                cluster.diarisation_artefact_id,
                            )
                        }
                    )
                ),
                component_ids=(),
            )
        # The four kinds below bind to *turn/cluster IDs*, never to the
        # components those IDs happen to live in (M8: "chapters and review
        # decisions bind to a revision plus its turn/cluster inventory
        # hashes"). That is deliberate and load-bearing under M21: a text
        # pass supersedes the turn-set component while retaining every
        # turn_id (M7 identity remap), so an applied review, a chapter
        # set, or a minutes component that declared a hard ref to the old
        # component ID would dangle the instant routine polish ran. ID-level
        # binding survives it; the registry validators below resolve those
        # IDs against whatever turn set is live in the closure and report a
        # soft failure if one no longer resolves.
        case SpeakerHypothesisSetComponent():
            return ComponentInputRefs(artefact_ids=(), component_ids=())
        case SpeakerReviewComponent():
            return ComponentInputRefs(artefact_ids=(), component_ids=())
        case TextEditLedgerComponent():
            return ComponentInputRefs(artefact_ids=(), component_ids=())
        case ChapterSetComponent():
            return ComponentInputRefs(artefact_ids=(), component_ids=())
        case MinutesComponent():
            return ComponentInputRefs(artefact_ids=(), component_ids=())


def _strip_stored_identity[BodyT: BaseModel](
    body_type: type[BodyT], component: BaseModel
) -> BodyT:
    """Rebuild ``body_type`` from a ``*Component`` record that subclasses it.

    Projects exactly ``body_type``'s own declared fields, so the record's
    three store-minted fields (``component_id``/``content_hash``/
    ``created_at``) are dropped by construction rather than by an
    exclusion list that a new identity field could silently escape. Only
    valid for kinds whose record is a plain subclass with no minted
    identity nested *inside* a body field -- :class:`NotesComponent` is
    the standing counter-example and is reconstructed explicitly instead.
    """
    return body_type.model_validate(
        {name: getattr(component, name) for name in body_type.model_fields}
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
                source_artefact_ids=component.source_artefact_ids,
                coordinate_domain=component.coordinate_domain,
                turns=component.turns,
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
        case DestinationComponent():
            return DestinationComponentBody(
                note_artefact_id=component.note_artefact_id,
                vault_relative_path=component.vault_relative_path,
                owned_region_state=component.owned_region_state,
            )
        case DestinationComponentBody():
            return component
        case RecordingReferenceSetComponent():
            return RecordingReferenceSetComponentBody(
                note_artefact_id=component.note_artefact_id,
                references=component.references,
            )
        case RecordingReferenceSetComponentBody():
            return component
        case MediaRecordingComponent():
            return MediaRecordingComponentBody(
                source_artefact_id=component.source_artefact_id,
                media_path=component.media_path,
                duration_ms=component.duration_ms,
                codec=component.codec,
                sample_rate_hz=component.sample_rate_hz,
                channels=component.channels,
            )
        case MediaRecordingComponentBody():
            return component
        case TimelineCombinedComponent():
            return TimelineCombinedComponentBody(
                segments=component.segments, gaps=component.gaps
            )
        case TimelineCombinedComponentBody():
            return component
        case AsrResultComponent():
            return AsrResultComponentBody(
                media_artefact_id=component.media_artefact_id,
                result_artefact_id=component.result_artefact_id,
                attempt_id=component.attempt_id,
                request_fingerprint=component.request_fingerprint,
                worker_config_hash=component.worker_config_hash,
                model_name=component.model_name,
                model_version=component.model_version,
            )
        case AsrResultComponentBody():
            return component
        case DiarisationResultComponent():
            return DiarisationResultComponentBody(
                media_artefact_id=component.media_artefact_id,
                result_artefact_id=component.result_artefact_id,
                attempt_id=component.attempt_id,
                request_fingerprint=component.request_fingerprint,
                worker_config_hash=component.worker_config_hash,
                model_name=component.model_name,
                model_version=component.model_version,
            )
        case DiarisationResultComponentBody():
            return component
        # Every kind below is a plain ``*Body`` subclass with no minted
        # identity nested inside a field, so :func:`_strip_stored_identity`
        # reconstructs it exactly -- the explicit field-by-field cases
        # above exist for the kinds where that is *not* true
        # (``NotesComponent.sections``) or where the record deliberately
        # is not a subclass at all.
        case NormalisationLedgerComponent():
            return _strip_stored_identity(NormalisationLedgerComponentBody, component)
        case NormalisationLedgerComponentBody():
            return component
        case MachineAttributionSetComponent():
            return _strip_stored_identity(MachineAttributionSetComponentBody, component)
        case MachineAttributionSetComponentBody():
            return component
        case SpeakerHypothesisSetComponent():
            return _strip_stored_identity(SpeakerHypothesisSetComponentBody, component)
        case SpeakerHypothesisSetComponentBody():
            return component
        case SpeakerReviewComponent():
            return _strip_stored_identity(SpeakerReviewComponentBody, component)
        case SpeakerReviewComponentBody():
            return component
        case TextEditLedgerComponent():
            return _strip_stored_identity(TextEditLedgerComponentBody, component)
        case TextEditLedgerComponentBody():
            return component
        case ChapterSetComponent():
            return _strip_stored_identity(ChapterSetComponentBody, component)
        case ChapterSetComponentBody():
            return component
        case MinutesComponent():
            return _strip_stored_identity(MinutesComponentBody, component)
        case MinutesComponentBody():
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
        case DestinationComponentBody():
            fields = body.model_dump(mode="python")
            return DestinationComponent(
                **fields,
                component_id=component_id,
                content_hash=content_hash,
                created_at=created_at,
            )
        case RecordingReferenceSetComponentBody():
            fields = body.model_dump(mode="python")
            return RecordingReferenceSetComponent(
                **fields,
                component_id=component_id,
                content_hash=content_hash,
                created_at=created_at,
            )
        case MediaRecordingComponentBody():
            fields = body.model_dump(mode="python")
            return MediaRecordingComponent(
                **fields,
                component_id=component_id,
                content_hash=content_hash,
                created_at=created_at,
            )
        case TimelineCombinedComponentBody():
            fields = body.model_dump(mode="python")
            return TimelineCombinedComponent(
                **fields,
                component_id=component_id,
                content_hash=content_hash,
                created_at=created_at,
            )
        case AsrResultComponentBody():
            fields = body.model_dump(mode="python")
            return AsrResultComponent(
                **fields,
                component_id=component_id,
                content_hash=content_hash,
                created_at=created_at,
            )
        case DiarisationResultComponentBody():
            fields = body.model_dump(mode="python")
            return DiarisationResultComponent(
                **fields,
                component_id=component_id,
                content_hash=content_hash,
                created_at=created_at,
            )
        case NormalisationLedgerComponentBody():
            return NormalisationLedgerComponent(
                **body.model_dump(mode="python"),
                component_id=component_id,
                content_hash=content_hash,
                created_at=created_at,
            )
        case MachineAttributionSetComponentBody():
            return MachineAttributionSetComponent(
                **body.model_dump(mode="python"),
                component_id=component_id,
                content_hash=content_hash,
                created_at=created_at,
            )
        case SpeakerHypothesisSetComponentBody():
            return SpeakerHypothesisSetComponent(
                **body.model_dump(mode="python"),
                component_id=component_id,
                content_hash=content_hash,
                created_at=created_at,
            )
        case SpeakerReviewComponentBody():
            return SpeakerReviewComponent(
                **body.model_dump(mode="python"),
                component_id=component_id,
                content_hash=content_hash,
                created_at=created_at,
            )
        case TextEditLedgerComponentBody():
            return TextEditLedgerComponent(
                **body.model_dump(mode="python"),
                component_id=component_id,
                content_hash=content_hash,
                created_at=created_at,
            )
        case ChapterSetComponentBody():
            return ChapterSetComponent(
                **body.model_dump(mode="python"),
                component_id=component_id,
                content_hash=content_hash,
                created_at=created_at,
            )
        case MinutesComponentBody():
            return MinutesComponent(
                **body.model_dump(mode="python"),
                component_id=component_id,
                content_hash=content_hash,
                created_at=created_at,
            )
