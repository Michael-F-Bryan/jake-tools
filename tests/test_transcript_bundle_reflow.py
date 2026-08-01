"""Deterministic turn reflow with explicit source-turn lineage."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from bundle_pipeline import (
    Recording,
    SpokenSegment,
    prepare_transcribed_bundle,
    tokens_from_utterances,
)
from bundle_stage_agent import StagePlan, stage_agent

from jake_tools.transcripts.bundle.assignment import SpeakerContext
from jake_tools.transcripts.bundle.components import (
    ChapterSetComponent,
    ComponentRecord,
    MinutesComponent,
    ParticipantDeclarationSource,
    ParticipantRecord,
    ParticipantSetComponentBody,
    ParticipantStatus,
    ReviewDecisionKind,
    SpeakerReviewComponentBody,
    TextEditMode,
    TextEditOperation,
    TimedTurn,
    TimedTurnSetComponent,
    TurnDecision,
    assemble_component_record,
)
from jake_tools.transcripts.bundle.control import (
    run_chapter_transform,
    run_minutes_transform,
    run_normalise_transform,
    run_reflow_transform,
    run_review_apply,
    run_speakers_propose,
    run_text_transform,
)
from jake_tools.transcripts.bundle.document import TranscriptDocumentV1, project_head
from jake_tools.transcripts.bundle.ids import mint_id
from jake_tools.transcripts.bundle.reflow import (
    reflow_turns,
    reflow_until_stable,
)
from jake_tools.transcripts.bundle.registry import CapabilityKey, CapabilityStatus
from jake_tools.transcripts.bundle.review import export_review_pack
from jake_tools.transcripts.bundle.store import BundleStore

_HASH = hashlib.sha256(b"fixture").hexdigest()
_ARTEFACT = mint_id("artefact")
_MICHAEL = mint_id("participant")
_AVALON = mint_id("participant")


def _stored(body: object) -> ComponentRecord:
    return assemble_component_record(
        body,  # pyright: ignore[reportArgumentType]
        component_id=mint_id("component"),
        content_hash=_HASH,
        created_at=datetime.now(UTC),
        mint_segment_id=lambda: mint_id("seg"),
    )


def _turn(
    text: str,
    *,
    start_ms: int,
    end_ms: int,
    label: str,
) -> TimedTurn:
    return TimedTurn(
        turn_id=mint_id("turn"),
        source_segment_id=mint_id("seg"),
        source_artefact_id=_ARTEFACT,
        speaker_label=label,
        text=text,
        start_ms=start_ms,
        end_ms=end_ms,
    )


def _context(assignments: dict[str, str | None]) -> SpeakerContext:
    participants = _stored(
        ParticipantSetComponentBody(
            participants=(
                ParticipantRecord(
                    participant_id=_MICHAEL,
                    declaration_source=ParticipantDeclarationSource.NOTE_FRONTMATTER,
                    declaration_evidence="fixture",
                    display_names=("Michael Bryan",),
                    status=ParticipantStatus.DECLARED,
                ),
                ParticipantRecord(
                    participant_id=_AVALON,
                    declaration_source=ParticipantDeclarationSource.NOTE_FRONTMATTER,
                    declaration_evidence="fixture",
                    display_names=("Avalon Mann",),
                    status=ParticipantStatus.DECLARED,
                ),
            )
        )
    )
    decisions = tuple(
        TurnDecision(
            kind=(
                ReviewDecisionKind.ASSIGN
                if participant_id is not None
                else ReviewDecisionKind.UNCLEAR_SPEAKER
            ),
            participant_id=participant_id,
            turn_id=turn_id,
        )
        for turn_id, participant_id in assignments.items()
    )
    review = _stored(
        SpeakerReviewComponentBody(
            review_id=mint_id("review"),
            input_revision_id=mint_id("rev"),
            turn_inventory_hash=_HASH,
            cluster_inventory_hash=_HASH,
            pack_schema_version="1",
            reviewer="Michael Bryan",
            decisions=decisions,
        )
    )
    return SpeakerContext.from_components(
        {
            participants.component_id: participants,
            review.component_id: review,
        }
    )


def test_reflow_merges_adjacent_turns_from_the_same_reviewed_speaker() -> None:
    first = _turn("We need", start_ms=0, end_ms=800, label="SPEAKER_00")
    second = _turn("to act.", start_ms=900, end_ms=1600, label="SPEAKER_00")

    result = reflow_turns(
        (first, second),
        _context({first.turn_id: _MICHAEL, second.turn_id: _MICHAEL}),
    )

    assert [turn.text for turn in result.turns] == ["We need to act."]
    assert result.turns[0].turn_id == first.turn_id
    assert result.turns[0].end_ms == second.end_ms
    assert result.entries[0].operation == TextEditOperation.MERGE
    assert result.entries[0].input_turn_ids == (first.turn_id, second.turn_id)
    assert result.entries[0].output_turn_ids == (first.turn_id,)


def test_reflow_repairs_an_unmistakable_unattributed_word_suffix() -> None:
    first = _turn("a driver's lic", start_ms=0, end_ms=800, label="SPEAKER_00")
    suffix = _turn("ense.", start_ms=800, end_ms=1100, label="unattributed")

    result = reflow_turns(
        (first, suffix),
        _context({first.turn_id: _MICHAEL, suffix.turn_id: None}),
    )

    assert [turn.text for turn in result.turns] == ["a driver's license."]
    assert result.entries[0].evidence_ref == "lexical-boundary-repair"


def test_reflow_keeps_an_overlapping_interjection_but_joins_its_speaker_continuation() -> (
    None
):
    first = _turn(
        "A Clark mast is one of the more",
        start_ms=0,
        end_ms=900,
        label="SPEAKER_00",
    )
    interjection = _turn(
        "The hard part was standing up.",
        start_ms=1000,
        end_ms=2200,
        label="SPEAKER_01",
    )
    continuation = _turn(
        "simple things, yeah.",
        start_ms=1100,
        end_ms=1800,
        label="SPEAKER_00",
    )

    result = reflow_turns(
        (first, interjection, continuation),
        _context(
            {
                first.turn_id: _MICHAEL,
                interjection.turn_id: _AVALON,
                continuation.turn_id: _MICHAEL,
            }
        ),
    )

    assert [turn.text for turn in result.turns] == [
        "A Clark mast is one of the more simple things, yeah.",
        "The hard part was standing up.",
    ]
    assert result.entries[0].input_turn_ids == (
        first.turn_id,
        continuation.turn_id,
    )
    assert result.entries[0].evidence_ref == "overlapping-same-speaker-continuation"


def test_reflow_does_not_merge_two_named_speakers() -> None:
    first = _turn("We should", start_ms=0, end_ms=800, label="SPEAKER_00")
    second = _turn("not assume that.", start_ms=800, end_ms=1600, label="SPEAKER_01")

    result = reflow_turns(
        (first, second),
        _context({first.turn_id: _MICHAEL, second.turn_id: _AVALON}),
    )

    assert result.turns == (first, second)
    assert all(
        entry.operation == TextEditOperation.IDENTITY for entry in result.entries
    )


def test_reflow_converges_when_one_merge_creates_a_new_adjacent_pair() -> None:
    first = _turn("First", start_ms=0, end_ms=900, label="SPEAKER_00")
    interjection = _turn("Second", start_ms=1000, end_ms=2200, label="SPEAKER_01")
    continuation = _turn("continued.", start_ms=1100, end_ms=1800, label="SPEAKER_00")
    second_continuation = _turn(
        "also continued.", start_ms=2300, end_ms=2900, label="SPEAKER_01"
    )
    context = _context(
        {
            first.turn_id: _MICHAEL,
            interjection.turn_id: _AVALON,
            continuation.turn_id: _MICHAEL,
            second_continuation.turn_id: _AVALON,
        }
    )

    passes = reflow_until_stable(
        (first, interjection, continuation, second_continuation), context
    )

    assert len(passes) == 2
    assert [turn.text for turn in passes[-1].turns] == [
        "First continued.",
        "Second also continued.",
    ]
    assert sum(result.merged_turn_count for result in passes) == 2


def _reviewed_store(bundle_path: Path) -> BundleStore:
    recording = Recording(
        name="meeting.m4a",
        duration_ms=10_000,
        tokens=tokens_from_utterances(
            [
                (1000, 3000, "a driver's lic"),
                (3050, 3400, "ense."),
                (6000, 7000, "Agreed."),
            ]
        ),
        segments=(
            SpokenSegment(900, 3000, "SPEAKER_00"),
            SpokenSegment(5900, 7100, "SPEAKER_01"),
        ),
    )
    prepared = prepare_transcribed_bundle(bundle_path, recordings=[recording])
    run_normalise_transform(prepared.store)
    asyncio.run(
        run_speakers_propose(
            prepared.store,
            agent=stage_agent(StagePlan(proposals={0: 0, 1: 1})),
            model="fixture-model",
        )
    )
    pack_path = export_review_pack(
        prepared.store,
        destination=bundle_path / "speaker-review.json",
    )
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    participants = {
        p["display_name"]: p["participant_id"] for p in pack["participants"]
    }
    assigned = {
        "SPEAKER_00": participants["Michael Bryan"],
        "SPEAKER_01": participants["Avalon Mann"],
    }
    pack["reviewer"] = "Michael Bryan"
    pack["decisions"] = [
        {
            "scope": "cluster",
            "kind": "assign",
            "cluster_id": item["item_id"],
            "participant_id": assigned[item["raw_label"]],
            "rationale": "reviewed fixture",
        }
        for item in pack["items"]
        if item["kind"] == "cluster"
    ]
    pack_path.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    run_review_apply(prepared.store, pack_path=pack_path)
    return prepared.store


def test_reflow_transform_appends_a_durable_merge_ledger(tmp_path: Path) -> None:
    store = _reviewed_store(tmp_path / "bundle")

    outcome = run_reflow_transform(store)

    assert outcome.merged_turn_count == 1
    assert outcome.revision is not None
    assert outcome.ledger is not None
    assert outcome.ledger.mode.value == "reflow"
    document = project_head(store)
    assert isinstance(document, TranscriptDocumentV1)
    turn_set = document.components_of(TimedTurnSetComponent)[0]
    assert [turn.text for turn in turn_set.turns] == [
        "a driver's license.",
        "Agreed.",
    ]
    assert outcome.ledger.entries[0].operation == TextEditOperation.MERGE


def test_reflow_transform_is_idempotent_once_no_merge_remains(tmp_path: Path) -> None:
    store = _reviewed_store(tmp_path / "bundle")
    first = run_reflow_transform(store)
    assert first.revision is not None
    head_after_first = store.load_manifest().head_revision_id

    second = run_reflow_transform(store)

    assert second.merged_turn_count == 0
    assert second.revision is None
    assert second.ledger is None
    assert store.load_manifest().head_revision_id == head_after_first


def test_reflow_invalidates_earlier_correction_and_polish_proofs(
    tmp_path: Path,
) -> None:
    store = _reviewed_store(tmp_path / "bundle")
    agent = stage_agent(StagePlan())
    asyncio.run(
        run_text_transform(
            store,
            mode=TextEditMode.CORRECT,
            agent=agent,
            model="fixture-model",
        )
    )
    asyncio.run(
        run_text_transform(
            store,
            mode=TextEditMode.POLISH,
            agent=agent,
            model="fixture-model",
        )
    )
    before = project_head(store)
    assert isinstance(before, TranscriptDocumentV1)
    assert (
        before.capability_status(CapabilityKey.TEXT_CORRECTED)
        == CapabilityStatus.PRESENT_VALIDATED
    )
    assert (
        before.capability_status(CapabilityKey.TEXT_POLISHED)
        == CapabilityStatus.PRESENT_VALIDATED
    )

    run_reflow_transform(store)

    after = project_head(store)
    assert isinstance(after, TranscriptDocumentV1)
    assert (
        after.capability_status(CapabilityKey.TEXT_CORRECTED) == CapabilityStatus.ABSENT
    )
    assert (
        after.capability_status(CapabilityKey.TEXT_POLISHED) == CapabilityStatus.ABSENT
    )


def test_reflow_supersedes_chapters_and_minutes_built_from_old_turns(
    tmp_path: Path,
) -> None:
    store = _reviewed_store(tmp_path / "bundle")
    agent = stage_agent(StagePlan())
    asyncio.run(run_chapter_transform(store, agent=agent, model="fixture-model"))
    asyncio.run(run_minutes_transform(store, agent=agent, model="fixture-model"))
    before = project_head(store)
    assert isinstance(before, TranscriptDocumentV1)
    assert before.components_of(ChapterSetComponent)
    assert before.components_of(MinutesComponent)

    run_reflow_transform(store)

    after = project_head(store)
    assert isinstance(after, TranscriptDocumentV1)
    assert not after.components_of(ChapterSetComponent)
    assert not after.components_of(MinutesComponent)
