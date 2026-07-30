from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from jake_tools.transcripts.bundle.components import (
    NotesComponentBody,
    NotesKind,
    NotesSection,
    ParticipantDeclarationSource,
    ParticipantRecord,
    ParticipantSetComponentBody,
    ParticipantStatus,
    assemble_component_record,
)
from jake_tools.transcripts.bundle.ids import mint_id
from jake_tools.transcripts.bundle.registry import (
    REGISTRY,
    CapabilityKey,
    CapabilityRecord,
    CapabilityStatus,
    StubEmittedPresentValidatedError,
    ValidationContext,
    ValidatorKeyMismatchError,
    cardinality_of,
    validate,
)

NOW = datetime.now(UTC)


def _notes_component(*, authored: bool, section_count: int = 1):
    body = NotesComponentBody(
        notes_kind=NotesKind.AUTHORED_PREP if authored else NotesKind.PROVIDER_SUMMARY,
        source_artefact_id=mint_id("artefact"),
        authored=authored,
        sections=tuple(
            NotesSection(section_id=mint_id("seg"), title=f"Section {i}", text="text")
            for i in range(section_count)
        ),
    )
    return assemble_component_record(
        body, component_id=mint_id("component"), created_at=NOW
    )


def _empty_sections_notes_component(*, authored: bool = False):
    body = NotesComponentBody(
        notes_kind=NotesKind.PROVIDER_SUMMARY,
        source_artefact_id=mint_id("artefact"),
        authored=authored,
        sections=(),
    )
    return assemble_component_record(
        body, component_id=mint_id("component"), created_at=NOW
    )


def _duplicate_section_id_notes_component():
    shared_id = mint_id("seg")
    body = NotesComponentBody(
        notes_kind=NotesKind.PROVIDER_SUMMARY,
        source_artefact_id=mint_id("artefact"),
        authored=False,
        sections=(
            NotesSection(section_id=shared_id, title="A", text="a"),
            NotesSection(section_id=shared_id, title="B", text="b"),
        ),
    )
    return assemble_component_record(
        body, component_id=mint_id("component"), created_at=NOW
    )


def _participant_set_component(*, participant_count: int = 1):
    participants = tuple(
        ParticipantRecord(
            participant_id=mint_id("participant"),
            declaration_source=ParticipantDeclarationSource.OPERATOR,
            declaration_evidence=f"cli: --participant 'P{i}'",
            display_names=(f"P{i}",),
            status=ParticipantStatus.DECLARED,
        )
        for i in range(participant_count)
    )
    body = ParticipantSetComponentBody(participants=participants)
    return assemble_component_record(
        body, component_id=mint_id("component"), created_at=NOW
    )


# -- registry table -----------------------------------------------------------


def test_registry_declares_every_m4_key() -> None:
    assert set(REGISTRY) == set(CapabilityKey)


def test_many_cardinality_keys_match_m4() -> None:
    many_keys = {key for key in CapabilityKey if cardinality_of(key) == "many"}

    assert many_keys == {
        CapabilityKey.MEDIA_RECORDING,
        CapabilityKey.NOTES_PROVIDER,
        CapabilityKey.NOTES_AUTHORED,
        CapabilityKey.SPEAKERS_PROVIDER_LABELS,
        CapabilityKey.INFERENCE_ASR,
        CapabilityKey.INFERENCE_DIARISATION,
    }


def test_chapters_prerequisite_data_is_present() -> None:
    """M4: 'the one sanctioned prerequisite is chapters -> transcript.timed'
    -- encoded as data on the registry entry, checked here independent of
    whether any validator currently reads it."""
    entry = REGISTRY[CapabilityKey.CHAPTERS]

    assert entry.prerequisites == (CapabilityKey.TRANSCRIPT_TIMED,)


def test_no_other_key_declares_a_prerequisite() -> None:
    for key, entry in REGISTRY.items():
        if key is CapabilityKey.CHAPTERS:
            continue
        assert entry.prerequisites == (), (
            f"{key.value} unexpectedly declares a prerequisite"
        )


# -- stub validators ------------------------------------------------------


def test_every_unimplemented_key_stub_returns_not_attempted() -> None:
    implemented = {
        CapabilityKey.NOTES_PROVIDER,
        CapabilityKey.NOTES_AUTHORED,
        CapabilityKey.PARTICIPANTS_DECLARED,
    }
    results = validate(mint_id("rev"), {})

    for key in CapabilityKey:
        if key in implemented:
            continue
        assert results[key].status == CapabilityStatus.NOT_ATTEMPTED, key.value


def test_stub_validator_cannot_be_made_to_emit_present_validated() -> None:
    """'Capabilities are proofs; a stub must never fake one' -- enforced by
    validate() itself: a stub registered as unimplemented that somehow
    returned present-validated is rejected, not silently accepted.

    Uses validate()'s injectable ``registry`` parameter (dependency
    injection) to swap in a deliberately-lying validator for one key,
    rather than mutating the shared module-level REGISTRY."""

    def _lying_stub(context: ValidationContext) -> CapabilityRecord:
        return CapabilityRecord(
            key=CapabilityKey.MEDIA_RECORDING,
            status=CapabilityStatus.PRESENT_VALIDATED,
            validator_version="v1",
        )

    lying_registry = dict(REGISTRY)
    lying_registry[CapabilityKey.MEDIA_RECORDING] = dataclasses.replace(
        REGISTRY[CapabilityKey.MEDIA_RECORDING], validator=_lying_stub
    )

    with pytest.raises(StubEmittedPresentValidatedError):
        validate(mint_id("rev"), {}, registry=lying_registry)


def test_validate_raises_if_a_validator_returns_a_record_for_the_wrong_key() -> None:
    """A registry-table mistake (e.g. a copy-pasted validator closure)
    must be caught, not silently mislabel one key's proof as another's."""

    def _mislabelled(context: ValidationContext) -> CapabilityRecord:
        return CapabilityRecord(
            key=CapabilityKey.INFERENCE_ASR,
            status=CapabilityStatus.NOT_ATTEMPTED,
            validator_version="v1",
        )

    mislabelled_registry = dict(REGISTRY)
    mislabelled_registry[CapabilityKey.MEDIA_RECORDING] = dataclasses.replace(
        REGISTRY[CapabilityKey.MEDIA_RECORDING], validator=_mislabelled
    )

    with pytest.raises(ValidatorKeyMismatchError):
        validate(mint_id("rev"), {}, registry=mislabelled_registry)


# -- notes.provider / notes.authored (M20) ------------------------------------


def test_notes_provider_is_absent_with_no_notes_components() -> None:
    results = validate(mint_id("rev"), {})

    assert results[CapabilityKey.NOTES_PROVIDER].status == CapabilityStatus.ABSENT
    assert results[CapabilityKey.NOTES_PROVIDER].members == ()


def test_notes_provider_is_present_validated_for_a_genuinely_valid_component() -> None:
    component = _notes_component(authored=False)

    results = validate(mint_id("rev"), {component.component_id: component})

    record = results[CapabilityKey.NOTES_PROVIDER]
    assert record.status == CapabilityStatus.PRESENT_VALIDATED
    assert record.component_ids == (component.component_id,)
    assert [member.status for member in record.members] == [
        CapabilityStatus.PRESENT_VALIDATED
    ]
    # notes.authored must not pick up a provider-flavoured component.
    assert results[CapabilityKey.NOTES_AUTHORED].status == CapabilityStatus.ABSENT


def test_notes_authored_is_present_validated_for_a_genuinely_valid_component() -> None:
    component = _notes_component(authored=True)

    results = validate(mint_id("rev"), {component.component_id: component})

    assert (
        results[CapabilityKey.NOTES_AUTHORED].status
        == CapabilityStatus.PRESENT_VALIDATED
    )
    assert results[CapabilityKey.NOTES_PROVIDER].status == CapabilityStatus.ABSENT


def test_notes_provider_fails_for_a_component_with_no_sections() -> None:
    component = _empty_sections_notes_component()

    results = validate(mint_id("rev"), {component.component_id: component})

    record = results[CapabilityKey.NOTES_PROVIDER]
    assert record.status == CapabilityStatus.FAILED
    assert record.members[0].status == CapabilityStatus.FAILED
    assert "no sections" in record.members[0].detail


def test_notes_provider_fails_for_duplicate_section_ids() -> None:
    component = _duplicate_section_id_notes_component()

    results = validate(mint_id("rev"), {component.component_id: component})

    record = results[CapabilityKey.NOTES_PROVIDER]
    assert record.status == CapabilityStatus.FAILED
    assert "duplicate" in record.members[0].detail


def test_notes_provider_per_member_query_on_mixed_member_statuses() -> None:
    """M4: 'consumers query many keys per member, never by aggregate status
    alone' -- one valid and one broken provider-notes component in the
    same closure must each keep their own status."""
    good = _notes_component(authored=False)
    broken = _empty_sections_notes_component(authored=False)

    results = validate(
        mint_id("rev"), {good.component_id: good, broken.component_id: broken}
    )

    record = results[CapabilityKey.NOTES_PROVIDER]
    # Aggregate reflects "not every member validates" -- but per-member
    # detail is what a consumer must actually use (M4).
    assert record.status == CapabilityStatus.FAILED
    statuses_by_member = {member.member_id: member.status for member in record.members}
    assert statuses_by_member[good.component_id] == CapabilityStatus.PRESENT_VALIDATED
    assert statuses_by_member[broken.component_id] == CapabilityStatus.FAILED


# -- participants.declared (M19) -----------------------------------------


def test_participants_declared_is_absent_with_no_participant_set() -> None:
    results = validate(mint_id("rev"), {})

    assert (
        results[CapabilityKey.PARTICIPANTS_DECLARED].status == CapabilityStatus.ABSENT
    )


def test_participants_declared_is_present_validated_with_provenance() -> None:
    component = _participant_set_component(participant_count=2)

    results = validate(mint_id("rev"), {component.component_id: component})

    record = results[CapabilityKey.PARTICIPANTS_DECLARED]
    assert record.status == CapabilityStatus.PRESENT_VALIDATED
    assert record.provenance_classes == (ParticipantDeclarationSource.OPERATOR.value,)


def test_participants_declared_fails_for_an_empty_participant_set() -> None:
    component = _participant_set_component(participant_count=0)

    results = validate(mint_id("rev"), {component.component_id: component})

    assert (
        results[CapabilityKey.PARTICIPANTS_DECLARED].status == CapabilityStatus.FAILED
    )


def test_participants_declared_fails_for_more_than_one_component() -> None:
    """participants.declared is a `one`-cardinality key: two candidate
    components in the same closure is ambiguous input, not a pick-first."""
    first = _participant_set_component()
    second = _participant_set_component()

    results = validate(
        mint_id("rev"), {first.component_id: first, second.component_id: second}
    )

    assert (
        results[CapabilityKey.PARTICIPANTS_DECLARED].status == CapabilityStatus.FAILED
    )
