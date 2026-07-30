from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from jake_tools.transcripts.bundle.components import (
    ArtefactSelection,
    AsrResultComponent,
    AsrResultComponentBody,
    AssemblyManifestComponent,
    AssemblyManifestComponentBody,
    ComponentBody,
    ComponentInputRefs,
    ComponentKind,
    ComponentRecord,
    DestinationComponent,
    DestinationComponentBody,
    DiarisationResultComponent,
    DiarisationResultComponentBody,
    Disposition,
    MediaRecordingComponent,
    MediaRecordingComponentBody,
    NotesComponent,
    NotesComponentBody,
    NotesKind,
    NotesSection,
    NotesSectionBody,
    OwnedRegionState,
    ParticipantDeclarationSource,
    ParticipantRecord,
    ParticipantSetComponent,
    ParticipantSetComponentBody,
    ParticipantStatus,
    ProviderLabelSetComponent,
    ProviderLabelSetComponentBody,
    ProviderLabelSpan,
    RecordingReference,
    RecordingReferenceSetComponent,
    RecordingReferenceSetComponentBody,
    TimedTurn,
    TimedTurnSetComponent,
    TimedTurnSetComponentBody,
    TimelineCombinedComponent,
    TimelineCombinedComponentBody,
    TimelineMappingSegment,
    TranscriptAbsenceDeclaration,
    TranscriptAbsenceDeclarationBody,
    TrustClass,
    UntimedTurn,
    UntimedTurnSetComponent,
    UntimedTurnSetComponentBody,
    assemble_component_record,
    component_as_body,
    component_input_refs,
)
from jake_tools.transcripts.bundle.ids import mint_id, source_domain

NOW = datetime.now(UTC)
_FAKE_HASH = hashlib.sha256(b"fixture").hexdigest()


def _notes_body(
    *, section_count: int = 1, authored: bool = False
) -> NotesComponentBody:
    return NotesComponentBody(
        notes_kind=NotesKind.AUTHORED_PREP if authored else NotesKind.PROVIDER_SUMMARY,
        source_artefact_id=mint_id("artefact"),
        authored=authored,
        sections=tuple(
            NotesSectionBody(title=f"Section {i}", text="text")
            for i in range(section_count)
        ),
    )


def _participant(
    *, participant_id: str | None = None, name: str = "Jake"
) -> ParticipantRecord:
    return ParticipantRecord(
        participant_id=participant_id or mint_id("participant"),
        declaration_source=ParticipantDeclarationSource.OPERATOR,
        declaration_evidence=f"cli: --participant '{name}'",
        display_names=(name,),
        status=ParticipantStatus.DECLARED,
    )


def _assemble_notes(body: NotesComponentBody) -> NotesComponent:
    record = assemble_component_record(
        body,
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )
    assert isinstance(record, NotesComponent)
    return record


def _assemble_participants(
    body: ParticipantSetComponentBody,
) -> ParticipantSetComponent:
    record = assemble_component_record(
        body,
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )
    assert isinstance(record, ParticipantSetComponent)
    return record


# -- NotesSectionBody / NotesSection / NotesComponentBody --------------------


def test_notes_section_body_carries_no_section_id() -> None:
    """MAJOR 6: a caller supplies section content only -- the section ID
    is minted by the store, never accepted here."""
    body = NotesSectionBody(title="T", text="x")

    assert not hasattr(body, "section_id")


def test_notes_section_requires_a_genuine_segment_id() -> None:
    """MAJOR 6: section_id is a real SegmentId (uuid7 pattern), not
    caller-supplied free text -- a path-traversal-shaped string is
    rejected the same as any other malformed ID."""
    with pytest.raises(ValidationError, match="section_id"):
        NotesSection(section_id="seg_../../../etc/passwd", title="T", text="x")


def test_notes_section_accepts_a_real_minted_segment_id() -> None:
    section = NotesSection(section_id=mint_id("seg"), title="T", text="x")

    assert section.section_id.startswith("seg_")


def test_notes_section_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        NotesSectionBody(title="T", text="")


def test_notes_component_body_carries_no_identity_fields() -> None:
    """M1: the body type structurally cannot carry component_id/created_at
    -- content identity is computed from exactly what this type holds."""
    body = _notes_body()

    assert not hasattr(body, "component_id")
    assert not hasattr(body, "created_at")
    assert body.component_kind == ComponentKind.NOTES


def test_notes_component_body_rejects_empty_sections() -> None:
    """MINOR 11: an empty notes component is permanent junk -- rejected
    at construction, not merely flagged by a validator later."""
    with pytest.raises(ValidationError):
        NotesComponentBody(
            notes_kind=NotesKind.PROVIDER_SUMMARY,
            source_artefact_id=mint_id("artefact"),
            authored=False,
            sections=(),
        )


def test_notes_component_body_rejects_authored_true_for_a_provider_kind() -> None:
    """MINOR 10: authored must be True iff notes_kind is authored-prep."""
    with pytest.raises(ValidationError, match="authored"):
        NotesComponentBody(
            notes_kind=NotesKind.PROVIDER_SUMMARY,
            source_artefact_id=mint_id("artefact"),
            authored=True,
            sections=(NotesSectionBody(title="T", text="x"),),
        )


def test_notes_component_body_rejects_authored_false_for_authored_prep() -> None:
    with pytest.raises(ValidationError, match="authored"):
        NotesComponentBody(
            notes_kind=NotesKind.AUTHORED_PREP,
            source_artefact_id=mint_id("artefact"),
            authored=False,
            sections=(NotesSectionBody(title="T", text="x"),),
        )


def test_notes_component_body_accepts_authored_prep_with_authored_true() -> None:
    body = _notes_body(authored=True)

    assert body.authored is True
    assert body.notes_kind == NotesKind.AUTHORED_PREP


def test_notes_component_re_validates_authored_matches_kind_independently() -> None:
    """NotesComponent is not a subclass of NotesComponentBody (it is
    loaded straight from disk too) -- it must re-run the same check
    rather than relying on a body that may never have existed."""
    with pytest.raises(ValidationError, match="authored"):
        NotesComponent(
            notes_kind=NotesKind.PROVIDER_SUMMARY,
            source_artefact_id=mint_id("artefact"),
            authored=True,
            sections=(NotesSection(section_id=mint_id("seg"), title="T", text="x"),),
            component_id=mint_id("component"),
            content_hash=_FAKE_HASH,
            created_at=NOW,
        )


# -- ParticipantRecord (M19) --------------------------------------------------


def test_participant_record_requires_at_least_one_display_name() -> None:
    with pytest.raises(ValidationError):
        ParticipantRecord(
            participant_id=mint_id("participant"),
            declaration_source=ParticipantDeclarationSource.CALENDAR,
            declaration_evidence="calendar-event:abc",
            display_names=(),
            status=ParticipantStatus.DECLARED,
        )


def test_participant_record_defaults_room_proxy_to_false() -> None:
    record = _participant()

    assert record.room_proxy is False
    assert record.status == ParticipantStatus.DECLARED


def test_participant_record_room_proxy_is_only_settable_explicitly() -> None:
    """M5: room_proxy is set only from explicit operator config -- never
    inferred -- which this model enforces by simply never defaulting to
    True and never deriving it from any other field."""
    record = ParticipantRecord(
        participant_id=mint_id("participant"),
        declaration_source=ParticipantDeclarationSource.OPERATOR,
        declaration_evidence="cli: --room-proxy 'Meetings Ahoy'",
        display_names=("Meetings Ahoy",),
        status=ParticipantStatus.DECLARED,
        room_proxy=True,
    )

    assert record.room_proxy is True


# -- ParticipantSetComponentBody (MINOR 7, MINOR 11) --------------------------


def test_participant_set_component_body_rejects_an_empty_set() -> None:
    with pytest.raises(ValidationError):
        ParticipantSetComponentBody(participants=())


def test_participant_set_component_body_is_order_independent() -> None:
    """MINOR 7: the same set of participants, supplied in two different
    orders, must produce the exact same (sorted-by-id) tuple -- so
    add_component's content-hash dedup treats them as identical."""
    a = _participant(participant_id=mint_id("participant"), name="A")
    b = _participant(participant_id=mint_id("participant"), name="B")

    forward = ParticipantSetComponentBody(participants=(a, b))
    reverse = ParticipantSetComponentBody(participants=(b, a))

    assert forward.participants == reverse.participants
    assert [p.participant_id for p in forward.participants] == sorted(
        [a.participant_id, b.participant_id]
    )


# -- assemble_component_record / component_input_refs / component_as_body ----


def test_assemble_component_record_attaches_identity_and_mints_section_ids() -> None:
    body = _notes_body()
    component_id = mint_id("component")

    record = assemble_component_record(
        body,
        component_id=component_id,
        content_hash=_FAKE_HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )

    assert isinstance(record, NotesComponent)
    assert record.component_id == component_id
    assert record.content_hash == _FAKE_HASH
    assert record.created_at == NOW
    assert len(record.sections) == len(body.sections)
    assert all(section.section_id.startswith("seg_") for section in record.sections)
    # Content (title/text) carries over unchanged; only the ID is new.
    assert [s.title for s in record.sections] == [s.title for s in body.sections]


def test_assemble_component_record_mints_a_distinct_id_per_section() -> None:
    body = _notes_body(section_count=3)

    record = _assemble_notes(body)

    section_ids = [section.section_id for section in record.sections]
    assert len(set(section_ids)) == 3


def test_assemble_component_record_attaches_identity_to_a_participant_set_body() -> (
    None
):
    body = ParticipantSetComponentBody(participants=(_participant(),))
    component_id = mint_id("component")

    record = assemble_component_record(
        body,
        component_id=component_id,
        content_hash=_FAKE_HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )

    assert isinstance(record, ParticipantSetComponent)
    assert record.participants == body.participants
    assert record.content_hash == _FAKE_HASH


def test_component_input_refs_names_the_notes_source_artefact() -> None:
    body = _notes_body()
    record = _assemble_notes(body)

    refs = component_input_refs(record)

    assert refs == ComponentInputRefs(
        artefact_ids=(body.source_artefact_id,), component_ids=()
    )


def test_component_input_refs_for_a_participant_set_has_no_refs() -> None:
    record = _assemble_participants(
        ParticipantSetComponentBody(participants=(_participant(),))
    )

    assert component_input_refs(record) == ComponentInputRefs(
        artefact_ids=(), component_ids=()
    )


def test_component_as_body_maps_records_back_to_their_equivalent_body() -> None:
    """MINOR C: hash(body) == hash(component_as_body(record)) requires the
    reconstruction to be a genuine equal Body, not merely the right type --
    including notes sections, whose minted section_id must be stripped."""
    notes_body = _notes_body()
    notes_record = _assemble_notes(notes_body)
    participant_body = ParticipantSetComponentBody(participants=(_participant(),))
    participant_record = _assemble_participants(participant_body)

    assert component_as_body(notes_record) == notes_body
    assert component_as_body(participant_record) == participant_body
    # Bare bodies map to themselves (identity, not just equality).
    assert component_as_body(notes_body) is notes_body
    assert component_as_body(participant_body) is participant_body


def test_component_as_body_strips_each_section_id_not_just_the_component_id() -> None:
    """A field-name include=/exclude= filter would miss this: section_id
    lives nested inside `sections`, not at the component's own top level."""
    record = _assemble_notes(_notes_body(section_count=2))

    reconstructed = component_as_body(record)

    assert isinstance(reconstructed, NotesComponentBody)
    assert not any(hasattr(section, "section_id") for section in reconstructed.sections)


# -- UntimedTurn / UntimedTurnSetComponentBody (M6/D6) ------------------------


def test_untimed_turn_carries_no_timing_fields() -> None:
    """The type itself is the enforcement mechanism behind
    transcript.untimed's 'no timing fields present' check -- there is no
    start_ms/end_ms field for a caller to accidentally set."""
    turn = UntimedTurn(
        turn_id=mint_id("turn"),
        source_segment_id=mint_id("seg"),
        speaker_label="A",
        text="hi",
    )

    assert not hasattr(turn, "start_ms")
    assert not hasattr(turn, "end_ms")


def test_untimed_turn_rejects_an_unknown_field_instead_of_silently_dropping_it() -> (
    None
):
    """MINOR 3 (adversarial review): extra="forbid" on every component/
    body/turn model -- a caller passing start_ms to an UntimedTurn must
    fail loudly, not silently construct a timing-free turn as if nothing
    were wrong. Fail-closed beats silent-ignore."""
    with pytest.raises(ValidationError, match="start_ms"):
        UntimedTurn(
            turn_id=mint_id("turn"),
            source_segment_id=mint_id("seg"),
            speaker_label="A",
            text="hi",
            start_ms=100,  # pyright: ignore[reportCallIssue]
        )


def test_notes_section_body_rejects_an_unknown_field() -> None:
    with pytest.raises(ValidationError):
        NotesSectionBody(
            title="T",
            text="x",
            section_id=mint_id("seg"),  # pyright: ignore[reportCallIssue]
        )


def test_participant_record_rejects_an_unknown_field() -> None:
    with pytest.raises(ValidationError):
        ParticipantRecord(
            participant_id=mint_id("participant"),
            declaration_source=ParticipantDeclarationSource.OPERATOR,
            declaration_evidence="test",
            display_names=("A",),
            status=ParticipantStatus.DECLARED,
            unexpected_field="surprise",  # pyright: ignore[reportCallIssue]
        )


def test_untimed_turn_set_body_preserves_supplied_import_order() -> None:
    """M6: untimed turn sets are never reordered."""
    turns = tuple(
        UntimedTurn(
            turn_id=mint_id("turn"),
            source_segment_id=mint_id("seg"),
            speaker_label=label,
            text="x",
        )
        for label in ("C", "A", "B")
    )

    body = UntimedTurnSetComponentBody(
        source_artefact_id=mint_id("artefact"), turns=turns
    )

    assert body.turns == turns


def test_untimed_turn_set_body_rejects_an_empty_turn_list() -> None:
    with pytest.raises(ValidationError):
        UntimedTurnSetComponentBody(source_artefact_id=mint_id("artefact"), turns=())


# -- TimedTurn / TimedTurnSetComponentBody (M6) -------------------------------


def test_timed_turn_rejects_a_zero_length_span() -> None:
    """M6: zero-length cues are legal only in raw source evidence, never
    in a canonical timed turn."""
    with pytest.raises(ValidationError, match="half-open"):
        TimedTurn(
            turn_id=mint_id("turn"),
            source_segment_id=mint_id("seg"),
            source_artefact_id=mint_id("artefact"),
            speaker_label="A",
            text="x",
            start_ms=1000,
            end_ms=1000,
        )


def test_timed_turn_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError, match="half-open"):
        TimedTurn(
            turn_id=mint_id("turn"),
            source_segment_id=mint_id("seg"),
            source_artefact_id=mint_id("artefact"),
            speaker_label="A",
            text="x",
            start_ms=1000,
            end_ms=500,
        )


#: One artefact ID shared by every helper-built timed turn below, so a
#: set built from them satisfies TimedTurnSetComponentBody's "every turn's
#: source artefact must be declared" check without each test restating it.
_TURN_ARTEFACT_ID = mint_id("artefact")


def _timed_turn(
    *, start_ms: int, end_ms: int, segment_id: str | None = None
) -> TimedTurn:
    return TimedTurn(
        turn_id=mint_id("turn"),
        source_segment_id=segment_id or mint_id("seg"),
        source_artefact_id=_TURN_ARTEFACT_ID,
        speaker_label="A",
        text="x",
        start_ms=start_ms,
        end_ms=end_ms,
    )


def _timed_turn_set_body(turns: tuple[TimedTurn, ...]) -> TimedTurnSetComponentBody:
    return TimedTurnSetComponentBody(
        source_artefact_ids=(_TURN_ARTEFACT_ID,),
        coordinate_domain=source_domain(_TURN_ARTEFACT_ID),
        turns=turns,
    )


def test_timed_turn_set_body_accepts_turns_already_in_canonical_order() -> None:
    turns = (
        _timed_turn(start_ms=0, end_ms=100),
        _timed_turn(start_ms=100, end_ms=200),
        _timed_turn(start_ms=200, end_ms=300),
    )

    body = _timed_turn_set_body(turns)

    assert body.turns == turns


def test_timed_turn_set_body_rejects_turns_out_of_canonical_order() -> None:
    """M6: canonical order is enforced at construction, not silently fixed
    -- the adapter/normaliser is responsible for sorting first."""
    turns = (
        _timed_turn(start_ms=200, end_ms=300),
        _timed_turn(start_ms=0, end_ms=100),
    )

    with pytest.raises(ValidationError, match="canonical order"):
        _timed_turn_set_body(turns)


def test_timed_turn_set_body_breaks_ties_on_source_segment_id() -> None:
    """M6: for turns sharing (start_ms, end_ms), source_segment_id is the
    tertiary sort key -- supplying them in ascending segment-id order for
    a genuine tie is accepted; descending is rejected."""
    first, second = sorted((mint_id("seg"), mint_id("seg")))
    ascending = (
        _timed_turn(start_ms=0, end_ms=100, segment_id=first),
        _timed_turn(start_ms=0, end_ms=100, segment_id=second),
    )
    descending = tuple(reversed(ascending))

    accepted = _timed_turn_set_body(ascending)
    assert accepted.turns == ascending

    with pytest.raises(ValidationError, match="canonical order"):
        _timed_turn_set_body(descending)


def test_timed_turn_set_component_re_validates_canonical_order_on_load() -> None:
    """TimedTurnSetComponent subclasses the Body, so it inherits the same
    order check -- a hand-edited/corrupted on-disk record must fail
    closed here too, not just at Body-construction time."""
    turns = (
        _timed_turn(start_ms=200, end_ms=300),
        _timed_turn(start_ms=0, end_ms=100),
    )

    with pytest.raises(ValidationError, match="canonical order"):
        TimedTurnSetComponent(
            source_artefact_ids=(_TURN_ARTEFACT_ID,),
            coordinate_domain=source_domain(_TURN_ARTEFACT_ID),
            turns=turns,
            component_id=mint_id("component"),
            content_hash=_FAKE_HASH,
            created_at=NOW,
        )


# -- ProviderLabelSpan / ProviderLabelSetComponentBody (M5) -------------------


def test_provider_label_set_body_requires_a_proxy_config_hash() -> None:
    with pytest.raises(ValidationError):
        ProviderLabelSetComponentBody(
            source_artefact_id=mint_id("artefact"),
            proxy_config_hash="not-a-sha256",
            spans=(
                ProviderLabelSpan(
                    source_segment_id=mint_id("seg"),
                    raw_label="Michael BRYAN",
                    text="hi",
                    trust_class=TrustClass.PER_PARTICIPANT_STREAM,
                ),
            ),
        )


def test_provider_label_set_body_accepts_a_real_proxy_config_hash() -> None:
    body = ProviderLabelSetComponentBody(
        source_artefact_id=mint_id("artefact"),
        proxy_config_hash=_FAKE_HASH,
        spans=(
            ProviderLabelSpan(
                source_segment_id=mint_id("seg"),
                raw_label="Michael BRYAN",
                text="hi",
                trust_class=TrustClass.PER_PARTICIPANT_STREAM,
            ),
        ),
    )

    assert body.spans[0].trust_class == TrustClass.PER_PARTICIPANT_STREAM


# -- TranscriptAbsenceDeclarationBody (D2/M5) ---------------------------------


def test_transcript_absence_declaration_requires_a_non_empty_statement() -> None:
    with pytest.raises(ValidationError):
        TranscriptAbsenceDeclarationBody(
            source_artefact_id=mint_id("artefact"), statement=""
        )


# -- Disposition / ArtefactSelection (M18) ------------------------------------


def test_artefact_selection_rejects_an_empty_disposition_set() -> None:
    with pytest.raises(ValidationError):
        ArtefactSelection(artefact_id=mint_id("artefact"), dispositions=())


def test_artefact_selection_rejects_a_duplicate_disposition() -> None:
    with pytest.raises(ValidationError, match="duplicate disposition"):
        ArtefactSelection(
            artefact_id=mint_id("artefact"),
            dispositions=(Disposition.NOTES, Disposition.NOTES),
        )


def test_artefact_selection_canonicalises_disposition_order() -> None:
    """So the same set, supplied in a different order, hashes identically
    (mirrors ParticipantSetComponentBody's participant-order handling)."""
    forward = ArtefactSelection(
        artefact_id=mint_id("artefact"),
        dispositions=(Disposition.SELECTED_TRANSCRIPT, Disposition.MEDIA),
    )
    reverse = ArtefactSelection(
        artefact_id=forward.artefact_id,
        dispositions=(Disposition.MEDIA, Disposition.SELECTED_TRANSCRIPT),
    )

    assert forward.dispositions == reverse.dispositions


def test_assembly_manifest_body_rejects_selecting_the_same_artefact_twice() -> None:
    artefact_id = mint_id("artefact")
    with pytest.raises(ValidationError, match="same artefact twice"):
        AssemblyManifestComponentBody(
            selections=(
                ArtefactSelection(
                    artefact_id=artefact_id, dispositions=(Disposition.NOTES,)
                ),
                ArtefactSelection(
                    artefact_id=artefact_id, dispositions=(Disposition.MEDIA,)
                ),
            ),
            rationale="test",
        )


def test_assembly_manifest_body_requires_a_non_empty_rationale() -> None:
    with pytest.raises(ValidationError):
        AssemblyManifestComponentBody(
            selections=(
                ArtefactSelection(
                    artefact_id=mint_id("artefact"), dispositions=(Disposition.NOTES,)
                ),
            ),
            rationale="",
        )


# -- assemble_component_record / component_as_body / component_input_refs ---
# for the five new component kinds -------------------------------------------


def test_assemble_component_record_attaches_identity_to_an_untimed_turn_set() -> None:
    body = UntimedTurnSetComponentBody(
        source_artefact_id=mint_id("artefact"),
        turns=(
            UntimedTurn(
                turn_id=mint_id("turn"),
                source_segment_id=mint_id("seg"),
                speaker_label="A",
                text="x",
            ),
        ),
    )

    record = assemble_component_record(
        body,
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )

    assert isinstance(record, UntimedTurnSetComponent)
    assert record.turns == body.turns
    assert component_as_body(record) == body


def test_assemble_component_record_attaches_identity_to_a_timed_turn_set() -> None:
    body = _timed_turn_set_body((_timed_turn(start_ms=0, end_ms=100),))

    record = assemble_component_record(
        body,
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )

    assert isinstance(record, TimedTurnSetComponent)
    assert record.turns == body.turns
    assert component_as_body(record) == body


def test_assemble_component_record_attaches_identity_to_a_provider_label_set() -> None:
    body = ProviderLabelSetComponentBody(
        source_artefact_id=mint_id("artefact"),
        proxy_config_hash=_FAKE_HASH,
        spans=(
            ProviderLabelSpan(
                source_segment_id=mint_id("seg"),
                raw_label="Sam Lintern",
                text="hi",
                trust_class=TrustClass.PER_PARTICIPANT_STREAM,
            ),
        ),
    )

    record = assemble_component_record(
        body,
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )

    assert isinstance(record, ProviderLabelSetComponent)
    assert record.spans == body.spans
    assert component_as_body(record) == body


def test_assemble_component_record_attaches_identity_to_an_absence_declaration() -> (
    None
):
    body = TranscriptAbsenceDeclarationBody(
        source_artefact_id=mint_id("artefact"),
        statement="No Gemini transcript was available for this meeting.",
    )

    record = assemble_component_record(
        body,
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )

    assert isinstance(record, TranscriptAbsenceDeclaration)
    assert component_as_body(record) == body


def test_assemble_component_record_attaches_identity_to_an_assembly_manifest() -> None:
    body = AssemblyManifestComponentBody(
        selections=(
            ArtefactSelection(
                artefact_id=mint_id("artefact"),
                dispositions=(Disposition.SELECTED_TRANSCRIPT,),
            ),
        ),
        rationale="only candidate available",
    )

    record = assemble_component_record(
        body,
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )

    assert isinstance(record, AssemblyManifestComponent)
    assert component_as_body(record) == body


def test_component_input_refs_for_the_new_kinds() -> None:
    artefact_id = mint_id("artefact")
    untimed = assemble_component_record(
        UntimedTurnSetComponentBody(
            source_artefact_id=artefact_id,
            turns=(
                UntimedTurn(
                    turn_id=mint_id("turn"),
                    source_segment_id=mint_id("seg"),
                    speaker_label="A",
                    text="x",
                ),
            ),
        ),
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )
    assert component_input_refs(untimed) == ComponentInputRefs(
        artefact_ids=(artefact_id,), component_ids=()
    )

    manifest_artefact_id = mint_id("artefact")
    manifest = assemble_component_record(
        AssemblyManifestComponentBody(
            selections=(
                ArtefactSelection(
                    artefact_id=manifest_artefact_id,
                    dispositions=(Disposition.NOTES,),
                ),
            ),
            rationale="test",
        ),
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )
    assert component_input_refs(manifest) == ComponentInputRefs(
        artefact_ids=(manifest_artefact_id,), component_ids=()
    )


# -- Phase 3A: destination / recording-reference-set / media-recording ------


def _mint(body: ComponentBody) -> ComponentRecord:
    return assemble_component_record(
        body,
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )


def test_destination_component_body_round_trips_through_as_body_and_input_refs() -> (
    None
):
    note_artefact_id = mint_id("artefact")
    body = DestinationComponentBody(
        note_artefact_id=note_artefact_id,
        vault_relative_path="2 Areas/Home Loan/note.md",
        owned_region_state=OwnedRegionState.NONE,
    )
    record = _mint(body)

    assert isinstance(record, DestinationComponent)
    assert component_as_body(record) == body
    assert component_input_refs(record) == ComponentInputRefs(
        artefact_ids=(note_artefact_id,), component_ids=()
    )


def test_recording_reference_set_preserves_note_embed_order_not_sorted() -> None:
    note_artefact_id = mint_id("artefact")
    body = RecordingReferenceSetComponentBody(
        note_artefact_id=note_artefact_id,
        references=(
            RecordingReference(raw_link="[[z.m4a]]", resolved_path="/tmp/z.m4a"),
            RecordingReference(raw_link="[[a.m4a]]", resolved_path="/tmp/a.m4a"),
        ),
    )
    record = _mint(body)

    assert isinstance(record, RecordingReferenceSetComponent)
    assert [r.raw_link for r in record.references] == ["[[z.m4a]]", "[[a.m4a]]"]
    assert component_as_body(record) == body
    assert component_input_refs(record) == ComponentInputRefs(
        artefact_ids=(note_artefact_id,), component_ids=()
    )


def test_recording_reference_set_body_rejects_an_empty_reference_list() -> None:
    with pytest.raises(ValidationError):
        RecordingReferenceSetComponentBody(
            note_artefact_id=mint_id("artefact"), references=()
        )


def test_media_recording_component_body_requires_a_positive_duration() -> None:
    with pytest.raises(ValidationError):
        MediaRecordingComponentBody(
            source_artefact_id=mint_id("artefact"),
            media_path="/tmp/x.m4a",
            duration_ms=0,
            codec="opus",
            sample_rate_hz=48000,
            channels=1,
        )


def test_media_recording_component_round_trips_through_as_body_and_input_refs() -> None:
    source_artefact_id = mint_id("artefact")
    body = MediaRecordingComponentBody(
        source_artefact_id=source_artefact_id,
        media_path="/tmp/x.m4a",
        duration_ms=50_000,
        codec="opus",
        sample_rate_hz=48000,
        channels=1,
    )
    record = _mint(body)

    assert isinstance(record, MediaRecordingComponent)
    assert component_as_body(record) == body
    assert component_input_refs(record) == ComponentInputRefs(
        artefact_ids=(source_artefact_id,), component_ids=()
    )


# -- Phase 3A: combined timeline ---------------------------------------------


def test_timeline_combined_component_round_trips_and_names_every_segment_artefact() -> (
    None
):
    first_artefact_id = mint_id("artefact")
    second_artefact_id = mint_id("artefact")
    body = TimelineCombinedComponentBody(
        segments=(
            TimelineMappingSegment(
                artefact_id=first_artefact_id,
                source_start_ms=0,
                source_end_ms=1000,
                combined_start_ms=0,
                combined_end_ms=1000,
            ),
            TimelineMappingSegment(
                artefact_id=second_artefact_id,
                source_start_ms=0,
                source_end_ms=500,
                combined_start_ms=1000,
                combined_end_ms=1500,
            ),
        )
    )
    record = _mint(body)

    assert isinstance(record, TimelineCombinedComponent)
    assert component_as_body(record) == body
    assert component_input_refs(record) == ComponentInputRefs(
        artefact_ids=(first_artefact_id, second_artefact_id), component_ids=()
    )


# -- Phase 3A: ASR / diarisation result components ---------------------------


def test_asr_result_component_round_trips_and_names_both_artefacts() -> None:
    media_artefact_id = mint_id("artefact")
    result_artefact_id = mint_id("artefact")
    body = AsrResultComponentBody(
        media_artefact_id=media_artefact_id,
        result_artefact_id=result_artefact_id,
        attempt_id=mint_id("attempt"),
        request_fingerprint=_FAKE_HASH,
        worker_config_hash="worker-hash",
        model_name="mlx-community/parakeet-tdt-0.6b-v2",
        model_version="unpinned",
    )
    record = _mint(body)

    assert isinstance(record, AsrResultComponent)
    assert component_as_body(record) == body
    assert component_input_refs(record) == ComponentInputRefs(
        artefact_ids=(media_artefact_id, result_artefact_id), component_ids=()
    )


def test_diarisation_result_component_round_trips_and_names_both_artefacts() -> None:
    media_artefact_id = mint_id("artefact")
    result_artefact_id = mint_id("artefact")
    body = DiarisationResultComponentBody(
        media_artefact_id=media_artefact_id,
        result_artefact_id=result_artefact_id,
        attempt_id=mint_id("attempt"),
        request_fingerprint=_FAKE_HASH,
        worker_config_hash="worker-hash",
        model_name="pyannote/speaker-diarization-community-1",
        model_version="unpinned",
    )
    record = _mint(body)

    assert isinstance(record, DiarisationResultComponent)
    assert component_as_body(record) == body
    assert component_input_refs(record) == ComponentInputRefs(
        artefact_ids=(media_artefact_id, result_artefact_id), component_ids=()
    )


def test_asr_result_component_requires_a_genuine_attempt_id() -> None:
    with pytest.raises(ValidationError):
        AsrResultComponentBody(
            media_artefact_id=mint_id("artefact"),
            result_artefact_id=mint_id("artefact"),
            attempt_id="not-a-real-attempt-id",
            request_fingerprint=_FAKE_HASH,
            worker_config_hash="worker-hash",
            model_name="x",
            model_version="y",
        )
