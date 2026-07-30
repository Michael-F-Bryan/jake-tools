from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from jake_tools.transcripts.bundle.components import (
    ComponentInputRefs,
    ComponentKind,
    NotesComponent,
    NotesComponentBody,
    NotesKind,
    NotesSection,
    NotesSectionBody,
    ParticipantDeclarationSource,
    ParticipantRecord,
    ParticipantSetComponent,
    ParticipantSetComponentBody,
    ParticipantStatus,
    assemble_component_record,
    component_as_body,
    component_input_refs,
)
from jake_tools.transcripts.bundle.ids import mint_id

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
