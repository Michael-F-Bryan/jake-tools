from __future__ import annotations

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
    ParticipantDeclarationSource,
    ParticipantRecord,
    ParticipantSetComponent,
    ParticipantSetComponentBody,
    ParticipantStatus,
    assemble_component_record,
    component_input_refs,
)
from jake_tools.transcripts.bundle.ids import mint_id

NOW = datetime.now(UTC)


def _notes_body(*, section_count: int = 1) -> NotesComponentBody:
    return NotesComponentBody(
        notes_kind=NotesKind.PROVIDER_SUMMARY,
        source_artefact_id=mint_id("artefact"),
        authored=False,
        sections=tuple(
            NotesSection(section_id=mint_id("seg"), title=f"Section {i}", text="text")
            for i in range(section_count)
        ),
    )


def _participant() -> ParticipantRecord:
    return ParticipantRecord(
        participant_id=mint_id("participant"),
        declaration_source=ParticipantDeclarationSource.OPERATOR,
        declaration_evidence="cli: --participant 'Jake'",
        display_names=("Jake",),
        status=ParticipantStatus.DECLARED,
    )


# -- NotesSection / NotesComponentBody --------------------------------------


def test_notes_section_requires_a_seg_prefixed_id() -> None:
    with pytest.raises(ValidationError, match="section_id"):
        NotesSection(section_id="not-a-seg-id", title="T", text="x")


def test_notes_section_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        NotesSection(section_id=mint_id("seg"), title="T", text="")


def test_notes_component_body_carries_no_identity_fields() -> None:
    """M1: the body type structurally cannot carry component_id/created_at
    -- content identity is computed from exactly what this type holds."""
    body = _notes_body()

    assert not hasattr(body, "component_id")
    assert not hasattr(body, "created_at")
    assert body.component_kind == ComponentKind.NOTES


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


# -- assemble_component_record / component_input_refs ------------------------


def test_assemble_component_record_attaches_identity_to_a_notes_body() -> None:
    body = _notes_body()
    component_id = mint_id("component")

    record = assemble_component_record(body, component_id=component_id, created_at=NOW)

    assert isinstance(record, NotesComponent)
    assert record.component_id == component_id
    assert record.created_at == NOW
    assert record.sections == body.sections


def test_assemble_component_record_attaches_identity_to_a_participant_set_body() -> (
    None
):
    body = ParticipantSetComponentBody(participants=(_participant(),))
    component_id = mint_id("component")

    record = assemble_component_record(body, component_id=component_id, created_at=NOW)

    assert isinstance(record, ParticipantSetComponent)
    assert record.participants == body.participants


def test_component_input_refs_names_the_notes_source_artefact() -> None:
    body = _notes_body()
    record = assemble_component_record(
        body, component_id=mint_id("component"), created_at=NOW
    )

    refs = component_input_refs(record)

    assert refs == ComponentInputRefs(
        artefact_ids=(body.source_artefact_id,), component_ids=()
    )


def test_component_input_refs_for_a_participant_set_has_no_refs() -> None:
    record = assemble_component_record(
        ParticipantSetComponentBody(participants=(_participant(),)),
        component_id=mint_id("component"),
        created_at=NOW,
    )

    assert component_input_refs(record) == ComponentInputRefs(
        artefact_ids=(), component_ids=()
    )
