from __future__ import annotations

import dataclasses
import hashlib
from datetime import UTC, datetime

import pytest

from jake_tools.transcripts.bundle.components import (
    NotesComponent,
    NotesComponentBody,
    NotesKind,
    NotesSection,
    NotesSectionBody,
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
    PrerequisiteNotSatisfiedError,
    RegistryConfigurationError,
    RegistryEntry,
    StubEmittedPresentValidatedError,
    ValidationContext,
    ValidatorKeyMismatchError,
    _aggregate_many_key_status,  # noqa: PLC2701 - white-box unit test of the pure helper
    _check_prerequisite_ordering,  # noqa: PLC2701
    cardinality_of,
    validate,
)

NOW = datetime.now(UTC)
_FAKE_HASH = hashlib.sha256(b"fixture").hexdigest()


def _notes_component(*, authored: bool, section_count: int = 1) -> NotesComponent:
    body = NotesComponentBody(
        notes_kind=NotesKind.AUTHORED_PREP if authored else NotesKind.PROVIDER_SUMMARY,
        source_artefact_id=mint_id("artefact"),
        authored=authored,
        sections=tuple(
            NotesSectionBody(title=f"Section {i}", text="text")
            for i in range(section_count)
        ),
    )
    record = assemble_component_record(
        body,
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )
    assert isinstance(record, NotesComponent)
    return record


def _notes_component_with_duplicate_section_id() -> NotesComponent:
    """Simulates a corrupted/adversarial on-disk record: section IDs are
    store-minted (MAJOR 6) so this can't happen via assemble_component_record
    -- construct the stored record directly, the way a hand-edited file
    (or a genuine mint collision) would produce it."""
    shared_id = mint_id("seg")
    return NotesComponent(
        notes_kind=NotesKind.PROVIDER_SUMMARY,
        source_artefact_id=mint_id("artefact"),
        authored=False,
        sections=(
            NotesSection(section_id=shared_id, title="A", text="a"),
            NotesSection(section_id=shared_id, title="B", text="b"),
        ),
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
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
        body,
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )


# -- registry table contract ---------------------------------------------


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
    -- encoded as data on the registry entry."""
    entry = REGISTRY[CapabilityKey.CHAPTERS]

    assert entry.prerequisites == (CapabilityKey.TRANSCRIPT_TIMED,)


def test_no_other_key_declares_a_prerequisite() -> None:
    for key, entry in REGISTRY.items():
        if key is CapabilityKey.CHAPTERS:
            continue
        assert entry.prerequisites == (), (
            f"{key.value} unexpectedly declares a prerequisite"
        )


# -- MAJOR 3: prerequisite enforcement is a real, executable behaviour ------


def test_validate_raises_when_present_validated_is_claimed_without_its_prerequisite() -> (
    None
):
    """The prerequisite data (test above) is worthless if validate() never
    reads it -- this is the genuine enforcement test fix 3 adds: an
    implemented=True validator that claims present-validated while its
    own declared prerequisite is not itself present-validated is a
    validator-authoring bug, caught centrally rather than trusted."""

    def _lying_chapters(context: ValidationContext) -> CapabilityRecord:
        return CapabilityRecord(
            key=CapabilityKey.CHAPTERS,
            status=CapabilityStatus.PRESENT_VALIDATED,
            validator_version="v1",
        )

    lying_registry = dict(REGISTRY)
    lying_registry[CapabilityKey.CHAPTERS] = dataclasses.replace(
        REGISTRY[CapabilityKey.CHAPTERS], validator=_lying_chapters, implemented=True
    )

    with pytest.raises(PrerequisiteNotSatisfiedError, match="transcript.timed"):
        validate(mint_id("rev"), {}, registry=lying_registry)


def test_validate_allows_present_validated_when_prerequisite_is_also_satisfied() -> (
    None
):
    """The positive case: a key may claim present-validated once its
    declared prerequisite genuinely is too."""

    def _satisfied_transcript_timed(context: ValidationContext) -> CapabilityRecord:
        return CapabilityRecord(
            key=CapabilityKey.TRANSCRIPT_TIMED,
            status=CapabilityStatus.PRESENT_VALIDATED,
            validator_version="v1",
        )

    def _satisfied_chapters(context: ValidationContext) -> CapabilityRecord:
        return CapabilityRecord(
            key=CapabilityKey.CHAPTERS,
            status=CapabilityStatus.PRESENT_VALIDATED,
            validator_version="v1",
        )

    satisfied_registry = dict(REGISTRY)
    satisfied_registry[CapabilityKey.TRANSCRIPT_TIMED] = dataclasses.replace(
        REGISTRY[CapabilityKey.TRANSCRIPT_TIMED],
        validator=_satisfied_transcript_timed,
        implemented=True,
    )
    satisfied_registry[CapabilityKey.CHAPTERS] = dataclasses.replace(
        REGISTRY[CapabilityKey.CHAPTERS],
        validator=_satisfied_chapters,
        implemented=True,
    )

    results = validate(mint_id("rev"), {}, registry=satisfied_registry)

    assert results[CapabilityKey.CHAPTERS].status == CapabilityStatus.PRESENT_VALIDATED


def test_check_prerequisite_ordering_rejects_a_prerequisite_declared_later_in_the_table() -> (
    None
):
    """MAJOR 3's topological guard: a key whose prerequisite is declared
    *after* it in CapabilityKey's own order must fail loudly here, at
    table-construction time -- not as a bare KeyError the first time
    validate() tries to read a prerequisite_statuses entry that does not
    exist yet. Exercised against a synthetic table so the real one is
    never put in this state."""
    bad_entries: dict[CapabilityKey, RegistryEntry] = dict(REGISTRY)
    bad_entries[CapabilityKey.NOTES_PROVIDER] = dataclasses.replace(
        REGISTRY[CapabilityKey.NOTES_PROVIDER],
        # CHAPTERS is declared *after* NOTES_PROVIDER in CapabilityKey.
        prerequisites=(CapabilityKey.CHAPTERS,),
    )

    with pytest.raises(RegistryConfigurationError, match="notes.provider"):
        _check_prerequisite_ordering(bad_entries)


def test_check_prerequisite_ordering_accepts_the_real_registry() -> None:
    """The real table's only prerequisite (chapters -> transcript.timed)
    is correctly ordered -- this is what lets REGISTRY import cleanly."""
    _check_prerequisite_ordering(REGISTRY)


# -- MAJOR 4: many-key aggregation -------------------------------------------


def test_aggregate_many_key_status_is_present_validated_only_if_all_members_are() -> (
    None
):
    assert (
        _aggregate_many_key_status(
            [CapabilityStatus.PRESENT_VALIDATED, CapabilityStatus.PRESENT_VALIDATED]
        )
        == CapabilityStatus.PRESENT_VALIDATED
    )


def test_aggregate_many_key_status_is_failed_if_any_member_genuinely_failed() -> None:
    assert (
        _aggregate_many_key_status(
            [CapabilityStatus.PRESENT_VALIDATED, CapabilityStatus.FAILED]
        )
        == CapabilityStatus.FAILED
    )


def test_aggregate_many_key_status_degrades_to_the_shared_non_failed_status() -> None:
    """M4's own worked example: 'two validated recordings and one
    reference-only embed' -- the aggregate must be the embed's own
    status, not a fabricated failure."""
    assert (
        _aggregate_many_key_status(
            [
                CapabilityStatus.PRESENT_VALIDATED,
                CapabilityStatus.PRESENT_VALIDATED,
                CapabilityStatus.NOT_AVAILABLE_FROM_SOURCE,
            ]
        )
        == CapabilityStatus.NOT_AVAILABLE_FROM_SOURCE
    )


def test_aggregate_many_key_status_is_absent_with_no_members() -> None:
    assert _aggregate_many_key_status([]) == CapabilityStatus.ABSENT


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


def test_notes_provider_fails_for_duplicate_section_ids_within_one_component() -> None:
    component = _notes_component_with_duplicate_section_id()

    results = validate(mint_id("rev"), {component.component_id: component})

    record = results[CapabilityKey.NOTES_PROVIDER]
    assert record.status == CapabilityStatus.FAILED
    assert "duplicate" in record.members[0].detail


def test_notes_provider_fails_for_section_ids_shared_across_two_components() -> None:
    """MAJOR 6: section IDs are one namespace across every notes component
    in the closure, not just within a single component."""
    shared_id = mint_id("seg")
    component_a = NotesComponent(
        notes_kind=NotesKind.PROVIDER_SUMMARY,
        source_artefact_id=mint_id("artefact"),
        authored=False,
        sections=(NotesSection(section_id=shared_id, title="A", text="a"),),
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
    )
    component_b = NotesComponent(
        notes_kind=NotesKind.PROVIDER_SUMMARY,
        source_artefact_id=mint_id("artefact"),
        authored=False,
        sections=(NotesSection(section_id=shared_id, title="B", text="b"),),
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
    )

    results = validate(
        mint_id("rev"),
        {component_a.component_id: component_a, component_b.component_id: component_b},
    )

    record = results[CapabilityKey.NOTES_PROVIDER]
    statuses_by_member = {member.member_id: member.status for member in record.members}
    assert statuses_by_member[component_a.component_id] == CapabilityStatus.FAILED
    assert statuses_by_member[component_b.component_id] == CapabilityStatus.FAILED
    assert "shared with another" in record.members[0].detail


def test_notes_provider_per_member_query_on_mixed_member_statuses() -> None:
    """M4: 'consumers query many keys per member, never by aggregate status
    alone' -- one valid and one broken provider-notes component in the
    same closure must each keep their own status."""
    good = _notes_component(authored=False)
    broken = _notes_component_with_duplicate_section_id()

    results = validate(
        mint_id("rev"), {good.component_id: good, broken.component_id: broken}
    )

    record = results[CapabilityKey.NOTES_PROVIDER]
    # Aggregate reflects a genuine failure -- but per-member detail is
    # what a consumer must actually use (M4).
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


def test_participants_declared_fails_for_more_than_one_component() -> None:
    """participants.declared is a `one`-cardinality key: two candidate
    components in the same closure is ambiguous input, not a pick-first.
    MINOR 9: the record's own failure_detail names the reason."""
    first = _participant_set_component()
    second = _participant_set_component()

    results = validate(
        mint_id("rev"), {first.component_id: first, second.component_id: second}
    )

    record = results[CapabilityKey.PARTICIPANTS_DECLARED]
    assert record.status == CapabilityStatus.FAILED
    assert "2 participant-set components" in record.failure_detail
