from __future__ import annotations

import dataclasses
import hashlib
from datetime import UTC, datetime

import pytest

from jake_tools.transcripts.bundle.components import (
    AsrResultComponent,
    AsrResultComponentBody,
    DiarisationResultComponent,
    DiarisationResultComponentBody,
    MediaRecordingComponent,
    MediaRecordingComponentBody,
    NotesComponent,
    NotesComponentBody,
    NotesKind,
    NotesSection,
    NotesSectionBody,
    ParticipantDeclarationSource,
    ParticipantRecord,
    ParticipantSetComponentBody,
    ParticipantStatus,
    ProviderLabelSetComponent,
    ProviderLabelSetComponentBody,
    ProviderLabelSpan,
    RecordingReference,
    RecordingReferenceSetComponentBody,
    TimedTurn,
    TimedTurnSetComponent,
    TimedTurnSetComponentBody,
    TimelineCombinedComponentBody,
    TimelineMappingSegment,
    TranscriptAbsenceDeclaration,
    TranscriptAbsenceDeclarationBody,
    TrustClass,
    UntimedTurn,
    UntimedTurnSetComponent,
    UntimedTurnSetComponentBody,
    assemble_component_record,
)
from jake_tools.transcripts.bundle.ids import mint_id, source_domain
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


_IMPLEMENTED_KEYS = {
    CapabilityKey.NOTES_PROVIDER,
    CapabilityKey.NOTES_AUTHORED,
    CapabilityKey.PARTICIPANTS_DECLARED,
    CapabilityKey.TRANSCRIPT_UNTIMED,
    CapabilityKey.TRANSCRIPT_TIMED,
    CapabilityKey.SPEAKERS_PROVIDER_LABELS,
    CapabilityKey.MEDIA_RECORDING,
    CapabilityKey.TIMELINE_COMBINED,
    CapabilityKey.INFERENCE_ASR,
    CapabilityKey.INFERENCE_DIARISATION,
    CapabilityKey.SPEAKERS_MACHINE_CLUSTERED,
    CapabilityKey.SPEAKERS_HUMAN_REVIEWED,
    CapabilityKey.SPEAKERS_HUMAN_CONFIRMED,
    CapabilityKey.TEXT_CORRECTED,
    CapabilityKey.TEXT_POLISHED,
    CapabilityKey.CHAPTERS,
    CapabilityKey.MINUTES,
}


def test_every_unimplemented_key_stub_returns_not_attempted() -> None:
    results = validate(mint_id("rev"), {})

    for key in CapabilityKey:
        if key in _IMPLEMENTED_KEYS:
            continue
        assert results[key].status == CapabilityStatus.NOT_ATTEMPTED, key.value


def test_every_implemented_key_reports_absent_on_an_empty_closure() -> None:
    """The mirror of the stub check above: a real validator over an empty
    component set must report a genuine, evidence-based status -- never
    the stub's not-attempted (that would mean it never actually looked)."""
    results = validate(mint_id("rev"), {})

    for key in _IMPLEMENTED_KEYS:
        assert results[key].status == CapabilityStatus.ABSENT, key.value


def test_stub_validator_cannot_be_made_to_emit_present_validated() -> None:
    """'Capabilities are proofs; a stub must never fake one' -- enforced by
    validate() itself: a stub registered as unimplemented that somehow
    returned present-validated is rejected, not silently accepted.

    Uses validate()'s injectable ``registry`` parameter (dependency
    injection) to swap in a deliberately-lying validator for one key,
    rather than mutating the shared module-level REGISTRY."""

    stub_key = CapabilityKey.SPEAKERS_PROVIDER_ATTRIBUTED

    def _lying_stub(context: ValidationContext) -> CapabilityRecord:
        return CapabilityRecord(
            key=stub_key,
            status=CapabilityStatus.PRESENT_VALIDATED,
            validator_version="v1",
        )

    lying_registry = dict(REGISTRY)
    lying_registry[stub_key] = dataclasses.replace(
        REGISTRY[stub_key], validator=_lying_stub
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


# -- transcript.untimed / transcript.timed (M6/M7/D6) -----------------------


def _untimed_turn_set_component(*, turn_count: int = 1) -> UntimedTurnSetComponent:
    body = UntimedTurnSetComponentBody(
        source_artefact_id=mint_id("artefact"),
        turns=tuple(
            UntimedTurn(
                turn_id=mint_id("turn"),
                source_segment_id=mint_id("seg"),
                speaker_label="A",
                text="x",
            )
            for _ in range(turn_count)
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
    return record


def _timed_turn_set_component(*, turn_count: int = 1) -> TimedTurnSetComponent:
    artefact_id = mint_id("artefact")
    turns = []
    start_ms = 0
    for _ in range(turn_count):
        turns.append(
            TimedTurn(
                turn_id=mint_id("turn"),
                source_segment_id=mint_id("seg"),
                source_artefact_id=artefact_id,
                speaker_label="A",
                text="x",
                start_ms=start_ms,
                end_ms=start_ms + 100,
            )
        )
        start_ms += 100
    body = TimedTurnSetComponentBody(
        source_artefact_ids=(artefact_id,),
        coordinate_domain=source_domain(artefact_id),
        turns=tuple(turns),
    )
    record = assemble_component_record(
        body,
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )
    assert isinstance(record, TimedTurnSetComponent)
    return record


def _absence_declaration_component() -> TranscriptAbsenceDeclaration:
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
    return record


def test_transcript_untimed_is_absent_with_no_evidence_either_way() -> None:
    results = validate(mint_id("rev"), {})

    assert results[CapabilityKey.TRANSCRIPT_UNTIMED].status == CapabilityStatus.ABSENT


def test_transcript_untimed_is_present_validated_with_a_turn_set() -> None:
    component = _untimed_turn_set_component(turn_count=3)

    results = validate(mint_id("rev"), {component.component_id: component})

    record = results[CapabilityKey.TRANSCRIPT_UNTIMED]
    assert record.status == CapabilityStatus.PRESENT_VALIDATED
    assert record.component_ids == (component.component_id,)


def test_transcript_untimed_fails_for_more_than_one_turn_set() -> None:
    first = _untimed_turn_set_component()
    second = _untimed_turn_set_component()

    results = validate(
        mint_id("rev"), {first.component_id: first, second.component_id: second}
    )

    assert results[CapabilityKey.TRANSCRIPT_UNTIMED].status == CapabilityStatus.FAILED


def test_transcript_untimed_reports_not_available_from_source_with_a_declaration() -> (
    None
):
    """D2/M5: the Gemini-notes-only case -- no turn set, but the source
    explicitly says no transcript exists. Distinct from plain absent."""
    declaration = _absence_declaration_component()

    results = validate(mint_id("rev"), {declaration.component_id: declaration})

    record = results[CapabilityKey.TRANSCRIPT_UNTIMED]
    assert record.status == CapabilityStatus.NOT_AVAILABLE_FROM_SOURCE
    assert record.component_ids == (declaration.component_id,)


def test_transcript_timed_reports_not_available_from_source_with_a_declaration() -> (
    None
):
    declaration = _absence_declaration_component()

    results = validate(mint_id("rev"), {declaration.component_id: declaration})

    assert (
        results[CapabilityKey.TRANSCRIPT_TIMED].status
        == CapabilityStatus.NOT_AVAILABLE_FROM_SOURCE
    )


def test_transcript_timed_is_absent_with_no_evidence_either_way() -> None:
    results = validate(mint_id("rev"), {})

    assert results[CapabilityKey.TRANSCRIPT_TIMED].status == CapabilityStatus.ABSENT


def test_transcript_timed_is_present_validated_with_a_turn_set() -> None:
    component = _timed_turn_set_component(turn_count=3)

    results = validate(mint_id("rev"), {component.component_id: component})

    record = results[CapabilityKey.TRANSCRIPT_TIMED]
    assert record.status == CapabilityStatus.PRESENT_VALIDATED
    assert record.component_ids == (component.component_id,)


def test_transcript_timed_fails_for_more_than_one_turn_set() -> None:
    first = _timed_turn_set_component()
    second = _timed_turn_set_component()

    results = validate(
        mint_id("rev"), {first.component_id: first, second.component_id: second}
    )

    assert results[CapabilityKey.TRANSCRIPT_TIMED].status == CapabilityStatus.FAILED


def test_transcript_capability_fails_with_two_conflicting_absence_declarations() -> (
    None
):
    first = _absence_declaration_component()
    second = _absence_declaration_component()

    results = validate(
        mint_id("rev"), {first.component_id: first, second.component_id: second}
    )

    assert results[CapabilityKey.TRANSCRIPT_UNTIMED].status == CapabilityStatus.FAILED
    assert results[CapabilityKey.TRANSCRIPT_TIMED].status == CapabilityStatus.FAILED


def test_chapters_prerequisite_is_satisfied_once_transcript_timed_validates() -> None:
    """The M4 prerequisite wiring (chapters -> transcript.timed), end to
    end: a present-validated timed turn set satisfies the precondition but
    proves nothing by itself -- with no chapter set in the closure,
    chapters is honestly absent, not present-validated."""
    component = _timed_turn_set_component()

    results = validate(mint_id("rev"), {component.component_id: component})

    assert (
        results[CapabilityKey.TRANSCRIPT_TIMED].status
        == CapabilityStatus.PRESENT_VALIDATED
    )
    assert results[CapabilityKey.CHAPTERS].status == CapabilityStatus.ABSENT


# -- speakers.provider-labels (M5, evidence only) ----------------------------


def _provider_label_set_component(
    *, cue_segment_ids: tuple[str, ...], proxy_config_hash: str = _FAKE_HASH
) -> ProviderLabelSetComponent:
    body = ProviderLabelSetComponentBody(
        source_artefact_id=mint_id("artefact"),
        proxy_config_hash=proxy_config_hash,
        spans=tuple(
            ProviderLabelSpan(
                source_segment_id=segment_id,
                raw_label="Michael BRYAN",
                text="hi",
                trust_class=TrustClass.PER_PARTICIPANT_STREAM,
            )
            for segment_id in cue_segment_ids
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
    return record


def test_speakers_provider_labels_is_absent_with_no_label_set() -> None:
    results = validate(mint_id("rev"), {})

    assert (
        results[CapabilityKey.SPEAKERS_PROVIDER_LABELS].status
        == CapabilityStatus.ABSENT
    )


def test_speakers_provider_labels_is_present_validated_when_spans_resolve_to_cues() -> (
    None
):
    turn_set = _timed_turn_set_component(turn_count=2)
    cue_segment_ids = tuple(turn.source_segment_id for turn in turn_set.turns)
    label_set = _provider_label_set_component(cue_segment_ids=cue_segment_ids)

    results = validate(
        mint_id("rev"),
        {turn_set.component_id: turn_set, label_set.component_id: label_set},
    )

    record = results[CapabilityKey.SPEAKERS_PROVIDER_LABELS]
    assert record.status == CapabilityStatus.PRESENT_VALIDATED
    assert record.component_ids == (label_set.component_id,)
    assert record.proxy_config_hash == _FAKE_HASH
    assert len(record.trust_class_coverage) == 1
    assert (
        record.trust_class_coverage[0].trust_class == TrustClass.PER_PARTICIPANT_STREAM
    )
    assert record.trust_class_coverage[0].covered_turn_count == 2


def test_speakers_provider_labels_fails_when_a_span_does_not_resolve_to_a_cue() -> None:
    """M5: 'spans resolve to cues' -- a span whose segment ID names no
    turn anywhere in the closure is unresolved evidence, not silently
    accepted."""
    label_set = _provider_label_set_component(cue_segment_ids=(mint_id("seg"),))

    results = validate(mint_id("rev"), {label_set.component_id: label_set})

    record = results[CapabilityKey.SPEAKERS_PROVIDER_LABELS]
    assert record.status == CapabilityStatus.FAILED
    assert record.members[0].status == CapabilityStatus.FAILED


def test_speakers_provider_labels_fails_on_inconsistent_proxy_config_hash() -> None:
    """M5: the proxy-config hash is a recorded config input -- two label
    sets in the same closure disagreeing about it is a data-integrity
    problem, never silently resolved by picking one."""
    turn_set = _timed_turn_set_component(turn_count=1)
    cue_segment_id = turn_set.turns[0].source_segment_id
    first = _provider_label_set_component(
        cue_segment_ids=(cue_segment_id,), proxy_config_hash=_FAKE_HASH
    )
    other_hash = hashlib.sha256(b"different-proxy-config").hexdigest()
    second = _provider_label_set_component(
        cue_segment_ids=(cue_segment_id,), proxy_config_hash=other_hash
    )

    results = validate(
        mint_id("rev"),
        {
            turn_set.component_id: turn_set,
            first.component_id: first,
            second.component_id: second,
        },
    )

    record = results[CapabilityKey.SPEAKERS_PROVIDER_LABELS]
    assert record.status == CapabilityStatus.FAILED
    assert record.proxy_config_hash is None
    assert all(member.status == CapabilityStatus.FAILED for member in record.members)
    assert all(
        "inconsistent proxy_config_hash" in member.detail for member in record.members
    )


# -- media.recording (Phase 3A, M3/M4) ---------------------------------------


def _media_recording_component(
    *, media_path: str = "/tmp/meeting.m4a"
) -> MediaRecordingComponent:
    body = MediaRecordingComponentBody(
        source_artefact_id=mint_id("artefact"),
        media_path=media_path,
        duration_ms=50_000,
        codec="opus",
        sample_rate_hz=48000,
        channels=1,
    )
    record = assemble_component_record(
        body,
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )
    assert isinstance(record, MediaRecordingComponent)
    return record


def _recording_reference_set_component(*, resolved_path: str = "/tmp/meeting.m4a"):
    body = RecordingReferenceSetComponentBody(
        note_artefact_id=mint_id("artefact"),
        references=(
            RecordingReference(raw_link="[[meeting.m4a]]", resolved_path=resolved_path),
        ),
    )
    return assemble_component_record(
        body,
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )


def test_media_recording_is_absent_with_no_evidence_either_way() -> None:
    results = validate(mint_id("rev"), {})

    assert results[CapabilityKey.MEDIA_RECORDING].status == CapabilityStatus.ABSENT


def test_media_recording_is_present_validated_for_standalone_ingested_media() -> None:
    """A local-media ingest with no accompanying note reference is still
    a usable member on its own (M3's operator-assertion path)."""
    media = _media_recording_component()

    results = validate(mint_id("rev"), {media.component_id: media})

    record = results[CapabilityKey.MEDIA_RECORDING]
    assert record.status == CapabilityStatus.PRESENT_VALIDATED
    assert record.members[0].member_id == media.media_path
    assert record.members[0].component_id == media.component_id


def test_media_recording_matches_a_reference_to_its_ingested_media_by_path() -> None:
    media = _media_recording_component(media_path="/tmp/meeting.m4a")
    reference_set = _recording_reference_set_component(resolved_path="/tmp/meeting.m4a")

    results = validate(
        mint_id("rev"),
        {media.component_id: media, reference_set.component_id: reference_set},
    )

    record = results[CapabilityKey.MEDIA_RECORDING]
    assert record.status == CapabilityStatus.PRESENT_VALIDATED
    assert (
        len(record.members) == 1
    )  # the reference and the media are one member, not two
    assert record.members[0].component_id == media.component_id


def test_media_recording_reports_not_available_from_source_for_a_reference_only_embed() -> (
    None
):
    """M3: "a reference alone never satisfies media.recording" -- the
    embed is still visible as its own member, honestly not-available."""
    reference_set = _recording_reference_set_component(
        resolved_path="/tmp/never-ingested.m4a"
    )

    results = validate(mint_id("rev"), {reference_set.component_id: reference_set})

    record = results[CapabilityKey.MEDIA_RECORDING]
    # M4's own worked example: a reference-only embed degrades the
    # aggregate status honestly rather than reporting present-validated.
    assert record.status == CapabilityStatus.NOT_AVAILABLE_FROM_SOURCE
    assert record.members[0].status == CapabilityStatus.NOT_AVAILABLE_FROM_SOURCE
    assert record.members[0].component_id is None


def test_media_recording_per_member_query_on_mixed_reference_and_standalone_media() -> (
    None
):
    """M4's own worked example: "two validated recordings and one
    reference-only embed still has two usable media.recording members" --
    consumers must query per member, never by aggregate status alone."""
    matched_media = _media_recording_component(media_path="/tmp/a.m4a")
    standalone_media = _media_recording_component(media_path="/tmp/b.m4a")
    reference_set = RecordingReferenceSetComponentBody(
        note_artefact_id=mint_id("artefact"),
        references=(
            RecordingReference(raw_link="[[a]]", resolved_path="/tmp/a.m4a"),
            RecordingReference(
                raw_link="[[c]]", resolved_path="/tmp/never-ingested.m4a"
            ),
        ),
    )
    reference_set_record = assemble_component_record(
        reference_set,
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )

    results = validate(
        mint_id("rev"),
        {
            matched_media.component_id: matched_media,
            standalone_media.component_id: standalone_media,
            reference_set_record.component_id: reference_set_record,
        },
    )

    record = results[CapabilityKey.MEDIA_RECORDING]
    statuses = {member.member_id: member.status for member in record.members}
    assert statuses["/tmp/a.m4a"] == CapabilityStatus.PRESENT_VALIDATED
    assert statuses["/tmp/b.m4a"] == CapabilityStatus.PRESENT_VALIDATED
    assert (
        statuses["/tmp/never-ingested.m4a"]
        == CapabilityStatus.NOT_AVAILABLE_FROM_SOURCE
    )
    usable = [
        m for m in record.members if m.status == CapabilityStatus.PRESENT_VALIDATED
    ]
    assert len(usable) == 2


# -- timeline.combined (Phase 3A, M6) ----------------------------------------


def _timeline_combined_component():
    body = TimelineCombinedComponentBody(
        segments=(
            TimelineMappingSegment(
                artefact_id=mint_id("artefact"),
                source_start_ms=0,
                source_end_ms=1000,
                combined_start_ms=0,
                combined_end_ms=1000,
            ),
        )
    )
    return assemble_component_record(
        body,
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )


def test_timeline_combined_is_absent_with_no_timeline_component() -> None:
    results = validate(mint_id("rev"), {})

    assert results[CapabilityKey.TIMELINE_COMBINED].status == CapabilityStatus.ABSENT


def test_timeline_combined_is_present_validated_with_one_component() -> None:
    timeline = _timeline_combined_component()

    results = validate(mint_id("rev"), {timeline.component_id: timeline})

    record = results[CapabilityKey.TIMELINE_COMBINED]
    assert record.status == CapabilityStatus.PRESENT_VALIDATED
    assert record.component_ids == (timeline.component_id,)


def test_timeline_combined_fails_for_more_than_one_component() -> None:
    """timeline.combined is a `one`-cardinality key: two candidate
    combined-timeline components in one closure is ambiguous."""
    first = _timeline_combined_component()
    second = _timeline_combined_component()

    results = validate(
        mint_id("rev"), {first.component_id: first, second.component_id: second}
    )

    record = results[CapabilityKey.TIMELINE_COMBINED]
    assert record.status == CapabilityStatus.FAILED


# -- inference.asr / inference.diarisation (Phase 3A, M11) -------------------


def _asr_result_component(
    *, media_artefact_id: str | None = None, model_name: str = "parakeet"
) -> AsrResultComponent:
    body = AsrResultComponentBody(
        media_artefact_id=media_artefact_id or mint_id("artefact"),
        result_artefact_id=mint_id("artefact"),
        attempt_id=mint_id("attempt"),
        request_fingerprint=_FAKE_HASH,
        worker_config_hash="cfg",
        model_name=model_name,
        model_version="unpinned",
    )
    record = assemble_component_record(
        body,
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )
    assert isinstance(record, AsrResultComponent)
    return record


def _diarisation_result_component(
    *, media_artefact_id: str | None = None
) -> DiarisationResultComponent:
    body = DiarisationResultComponentBody(
        media_artefact_id=media_artefact_id or mint_id("artefact"),
        result_artefact_id=mint_id("artefact"),
        attempt_id=mint_id("attempt"),
        request_fingerprint=_FAKE_HASH,
        worker_config_hash="cfg",
        model_name="pyannote",
        model_version="unpinned",
    )
    record = assemble_component_record(
        body,
        component_id=mint_id("component"),
        content_hash=_FAKE_HASH,
        created_at=NOW,
        mint_segment_id=lambda: mint_id("seg"),
    )
    assert isinstance(record, DiarisationResultComponent)
    return record


def test_inference_asr_is_absent_with_no_result_component() -> None:
    results = validate(mint_id("rev"), {})

    assert results[CapabilityKey.INFERENCE_ASR].status == CapabilityStatus.ABSENT


def test_inference_asr_is_present_validated_per_recording() -> None:
    """inference.asr is a `many` key (M4: one per source/recording) -- a
    completed component always validates, since one only ever exists for
    a completed stage (M11's partial-failure semantics live in whether
    the component was created at all, not in its status field)."""
    first = _asr_result_component()
    second = _asr_result_component()

    results = validate(
        mint_id("rev"), {first.component_id: first, second.component_id: second}
    )

    record = results[CapabilityKey.INFERENCE_ASR]
    assert record.status == CapabilityStatus.PRESENT_VALIDATED
    assert len(record.members) == 2
    assert record.provenance_classes == ("parakeet",)


def test_inference_diarisation_partial_failure_never_touches_inference_asr() -> None:
    """The structural half of M11's partial-failure rule: a diarisation
    that failed for some recording never even creates a component, so
    inference.asr for the SAME recording is completely unaffected."""
    media_artefact_id = mint_id("artefact")
    asr = _asr_result_component(media_artefact_id=media_artefact_id)
    # No DiarisationResultComponent at all for this media -- the failed
    # stage's own absence *is* the honest signal (see components.py's
    # AsrResultComponentBody docstring).

    results = validate(mint_id("rev"), {asr.component_id: asr})

    assert (
        results[CapabilityKey.INFERENCE_ASR].status
        == CapabilityStatus.PRESENT_VALIDATED
    )
    assert (
        results[CapabilityKey.INFERENCE_DIARISATION].status == CapabilityStatus.ABSENT
    )


def test_inference_diarisation_is_present_validated_per_recording() -> None:
    diarisation = _diarisation_result_component()

    results = validate(mint_id("rev"), {diarisation.component_id: diarisation})

    record = results[CapabilityKey.INFERENCE_DIARISATION]
    assert record.status == CapabilityStatus.PRESENT_VALIDATED
    assert record.members[0].member_id == diarisation.media_artefact_id
