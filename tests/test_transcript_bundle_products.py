"""M9/M10: the text passes, chapters, and minutes.

Every test drives the real transform against a real bundle, with the LLM
boundary -- and only that boundary -- faked. The validation, ledgers,
gates, and capability proofs downstream of it are the things under test,
so they all run for real.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest
from bundle_pipeline import (
    Recording,
    SpokenSegment,
    prepare_transcribed_bundle,
    tokens_from_utterances,
)
from bundle_stage_agent import StagePlan, stage_agent

from jake_tools.transcripts.bundle.components import (
    ChapterSetComponent,
    MinutesComponent,
    TextEditLedgerComponent,
    TextEditMode,
    TextEditOperation,
    TimedTurnSetComponent,
)
from jake_tools.transcripts.bundle.control import (
    ExecutorFailedError,
    run_chapter_transform,
    run_minutes_transform,
    run_normalise_transform,
    run_text_transform,
)
from jake_tools.transcripts.bundle.document import TranscriptDocumentV1, project_head
from jake_tools.transcripts.bundle.products import (
    InvalidChapterPlanError,
    project_chapter_boundaries,
)
from jake_tools.transcripts.bundle.registry import CapabilityKey, CapabilityStatus
from jake_tools.transcripts.bundle.store import BundleStore

_RECORDING = Recording(
    name="meeting.m4a",
    duration_ms=60_000,
    tokens=tokens_from_utterances(
        [
            (1000, 4000, "So we should look at the autonomy platforms next week."),
            (5000, 9000, "I think we should start with the middleware question."),
            (10000, 10600, "Um yeah"),
            (12500, 16500, "Agreed, and we will report back on Friday."),
            (18000, 22000, "One risk is that the vendor lead time is unclear."),
        ]
    ),
    segments=(
        SpokenSegment(900, 4100, "SPEAKER_00"),
        SpokenSegment(4900, 9100, "SPEAKER_01"),
        SpokenSegment(9900, 10700, "SPEAKER_00"),
        SpokenSegment(12400, 16600, "SPEAKER_00"),
        SpokenSegment(17900, 22100, "SPEAKER_01"),
    ),
)


def _normalised(tmp_path: Path) -> BundleStore:
    prepared = prepare_transcribed_bundle(tmp_path / "bundle", recordings=[_RECORDING])
    run_normalise_transform(prepared.store)
    return prepared.store


def _head(store: BundleStore) -> TranscriptDocumentV1:
    document = project_head(store)
    assert isinstance(document, TranscriptDocumentV1)
    return document


def _turns(store: BundleStore) -> tuple[str, ...]:
    turn_set = _head(store).components_of(TimedTurnSetComponent)[0]
    return tuple(turn.text for turn in turn_set.turns)


def _turn_ids(store: BundleStore) -> tuple[str, ...]:
    turn_set = _head(store).components_of(TimedTurnSetComponent)[0]
    return tuple(turn.turn_id for turn in turn_set.turns)


def _fake_turn(index: int):
    from jake_tools.transcripts.bundle.components import TimedTurn
    from jake_tools.transcripts.bundle.ids import mint_id

    return TimedTurn(
        turn_id=mint_id("turn"),
        source_segment_id=mint_id("seg"),
        source_artefact_id=mint_id("artefact"),
        speaker_label="SPEAKER_00",
        text=f"turn {index}",
        start_ms=index * 1000,
        end_ms=index * 1000 + 500,
    )


# -- M9: the text passes -----------------------------------------------------


def test_a_correct_pass_rewrites_text_and_keeps_turn_identity(
    tmp_path: Path,
) -> None:
    """M7's identity remap is what lets polish run after a review without
    invalidating it, so it is the first thing to pin down."""
    store = _normalised(tmp_path)
    before = _turn_ids(store)

    asyncio.run(
        run_text_transform(
            store,
            mode=TextEditMode.CORRECT,
            agent=stage_agent(
                StagePlan(correct_text=lambda text: text.replace("we", "we"))
            ),
            model="fixture-model",
        )
    )

    assert _turn_ids(store) == before


def test_a_polish_pass_records_every_turn_in_its_ledger(tmp_path: Path) -> None:
    """M9: "a ledger that does not account for the full diff fails
    validation" -- including the turns nothing happened to."""
    store = _normalised(tmp_path)

    asyncio.run(
        run_text_transform(
            store,
            mode=TextEditMode.POLISH,
            agent=stage_agent(
                StagePlan(polish_text=lambda text: text.replace("So we", "We"))
            ),
            model="fixture-model",
        )
    )

    ledger = _head(store).components_of(TextEditLedgerComponent)[0]
    accounted = {
        turn_id for entry in ledger.entries for turn_id in entry.output_turn_ids
    }
    assert accounted == set(_turn_ids(store))
    assert any(
        entry.operation == TextEditOperation.TEXT_EDIT for entry in ledger.entries
    )
    assert any(
        entry.operation == TextEditOperation.IDENTITY for entry in ledger.entries
    )


def test_a_polish_pass_that_drops_a_filler_turn_records_the_reason(
    tmp_path: Path,
) -> None:
    store = _normalised(tmp_path)
    # A short middle turn. Dropping the first or last would move the
    # transcript's own time-span coverage (verify.py's coverage gate
    # refuses that), and dropping a long one would trip the
    # content-retention gate -- polish removes filler, not substance.
    filler_id = _turn_ids(store)[2]

    def _empty_the_first(prompt: str) -> dict[str, Any] | None:
        if "Polish these transcript turns" not in prompt:
            return None
        return {
            "turns": [
                {
                    "turn_id": turn_id,
                    "text": "" if turn_id == filler_id else text,
                    "removal_reasons": ["filler"] if turn_id == filler_id else [],
                }
                for turn_id, text in zip(_turn_ids(store), _turns(store), strict=True)
            ]
        }

    asyncio.run(
        run_text_transform(
            store,
            mode=TextEditMode.POLISH,
            agent=stage_agent(StagePlan(override=_empty_the_first)),
            model="fixture-model",
        )
    )

    ledger = _head(store).components_of(TextEditLedgerComponent)[0]
    drop = next(
        entry
        for entry in ledger.entries
        if entry.operation == TextEditOperation.DROP_EMPTY
    )
    assert drop.input_turn_ids == (filler_id,)
    assert drop.removal_reasons
    assert filler_id not in _turn_ids(store)


def test_a_correct_pass_may_not_empty_a_turn(tmp_path: Path) -> None:
    """Correction fixes mis-transcriptions; removing a turn is a different
    act with different lineage, and conflating them would let "correct"
    quietly delete evidence."""
    store = _normalised(tmp_path)

    def _empty_everything(prompt: str) -> dict[str, Any] | None:
        if "Correct mis-transcriptions" not in prompt:
            return None
        return {
            "turns": [
                {"turn_id": turn_id, "text": "", "removal_reasons": []}
                for turn_id in _turn_ids(store)
            ]
        }

    with pytest.raises(ExecutorFailedError, match="emptied"):
        asyncio.run(
            run_text_transform(
                store,
                mode=TextEditMode.CORRECT,
                agent=stage_agent(StagePlan(override=_empty_everything)),
                model="fixture-model",
            )
        )


def test_an_edit_naming_a_turn_that_does_not_exist_is_refused(
    tmp_path: Path,
) -> None:
    """A stage that invents editorial nodes is not editing this
    transcript, so its whole result is refused rather than partly kept."""
    store = _normalised(tmp_path)

    def _invent(prompt: str) -> dict[str, Any] | None:
        if "Polish these transcript turns" not in prompt:
            return None
        return {
            "turns": [
                {
                    "turn_id": "turn_019fb000-0000-7000-8000-000000000000",
                    "text": "invented",
                    "removal_reasons": [],
                }
            ]
        }

    with pytest.raises(ExecutorFailedError, match="not in the input turn set"):
        asyncio.run(
            run_text_transform(
                store,
                mode=TextEditMode.POLISH,
                agent=stage_agent(StagePlan(override=_invent)),
                model="fixture-model",
            )
        )


def test_an_over_aggressive_polish_is_refused_not_published(tmp_path: Path) -> None:
    """verify.py's retention gate, reused as M9's validator: a polish that
    guts the transcript is not "a bit terse", it is a transcript that no
    longer says what was said."""
    store = _normalised(tmp_path)

    with pytest.raises(ExecutorFailedError, match="UnfaithfulEditError"):
        asyncio.run(
            run_text_transform(
                store,
                mode=TextEditMode.POLISH,
                agent=stage_agent(StagePlan(polish_text=lambda text: "x")),
                model="fixture-model",
            )
        )


def test_a_text_pass_makes_its_own_capability_real(tmp_path: Path) -> None:
    store = _normalised(tmp_path)

    asyncio.run(
        run_text_transform(
            store,
            mode=TextEditMode.CORRECT,
            agent=stage_agent(StagePlan()),
            model="fixture-model",
        )
    )

    document = _head(store)
    assert (
        document.capability_status(CapabilityKey.TEXT_CORRECTED)
        == CapabilityStatus.PRESENT_VALIDATED
    )
    assert (
        document.capability_status(CapabilityKey.TEXT_POLISHED)
        == CapabilityStatus.ABSENT
    ), "the two passes are distinguishable in provenance"


def test_polish_after_correct_leaves_one_canonical_turn_set(tmp_path: Path) -> None:
    store = _normalised(tmp_path)
    agent = stage_agent(
        StagePlan(
            correct_text=lambda text: text.replace("autonomy", "autonomy"),
            polish_text=lambda text: text.replace("So we", "We"),
        )
    )

    asyncio.run(
        run_text_transform(
            store, mode=TextEditMode.CORRECT, agent=agent, model="fixture-model"
        )
    )
    asyncio.run(
        run_text_transform(
            store, mode=TextEditMode.POLISH, agent=agent, model="fixture-model"
        )
    )

    document = _head(store)
    assert len(document.components_of(TimedTurnSetComponent)) == 1
    assert (
        document.capability_status(CapabilityKey.TRANSCRIPT_TIMED)
        == CapabilityStatus.PRESENT_VALIDATED
    )
    assert _turns(store)[0].startswith("We should look")


# -- M10: chapters -----------------------------------------------------------


def test_chapter_boundaries_project_onto_an_exact_partition() -> None:
    turns = tuple(_fake_turn(index) for index in range(5))

    ranges = project_chapter_boundaries(turns, [0, 2])

    assert ranges == ((0, 2), (2, 5))


def test_a_plan_that_forgets_the_first_turn_still_covers_it() -> None:
    """M7 requires every canonical turn to be in exactly one chapter, so a
    plan starting at turn 2 gets a chapter prepended rather than leaving a
    coverage gap at the front of the transcript."""
    turns = tuple(_fake_turn(index) for index in range(4))

    ranges = project_chapter_boundaries(turns, [2])

    assert ranges == ((0, 2), (2, 4))


def test_out_of_order_chapter_boundaries_are_refused() -> None:
    turns = tuple(_fake_turn(index) for index in range(4))

    with pytest.raises(InvalidChapterPlanError, match="order"):
        project_chapter_boundaries(turns, [2, 1])


def test_an_unknown_boundary_is_ignored_but_the_rest_still_partition() -> None:
    """A turn a later polish pass dropped shifts every ordinal after it, so
    a boundary can fall outside the sequence. Losing the whole plan over
    one is worse than losing that one boundary -- coverage stays exact
    either way, because the partition is derived from what survived."""
    turns = tuple(_fake_turn(index) for index in range(4))

    ranges = project_chapter_boundaries(turns, [0, 99, 2])

    assert ranges == ((0, 2), (2, 4))


def test_a_plan_where_no_boundary_is_canonical_is_refused() -> None:
    """Ignoring one bad ID is tolerance; ignoring all of them would mean
    silently inventing a single-chapter plan the stage never proposed."""
    turns = tuple(_fake_turn(index) for index in range(3))

    with pytest.raises(InvalidChapterPlanError, match="does not describe"):
        project_chapter_boundaries(turns, [99])


def test_chaptering_covers_every_canonical_turn_exactly_once(
    tmp_path: Path,
) -> None:
    store = _normalised(tmp_path)

    asyncio.run(
        run_chapter_transform(
            store,
            agent=stage_agent(StagePlan(chapter_starts=(0, 2))),
            model="fixture-model",
        )
    )

    document = _head(store)
    chapters = document.components_of(ChapterSetComponent)[0]
    covered = [turn_id for chapter in chapters.chapters for turn_id in chapter.turn_ids]
    assert covered == list(_turn_ids(store))
    assert (
        document.capability_status(CapabilityKey.CHAPTERS)
        == CapabilityStatus.PRESENT_VALIDATED
    )


# -- M10: minutes ------------------------------------------------------------


def test_minutes_findings_carry_resolvable_evidence(tmp_path: Path) -> None:
    store = _normalised(tmp_path)

    outcome = asyncio.run(
        run_minutes_transform(
            store, agent=stage_agent(StagePlan()), model="fixture-model"
        )
    )

    finding = outcome.minutes.findings[0]
    assert set(finding.evidence_turn_ids) <= set(_turn_ids(store))
    assert (
        _head(store).capability_status(CapabilityKey.MINUTES)
        == CapabilityStatus.PRESENT_VALIDATED
    )


def test_an_unsourced_finding_is_dropped_and_reported(tmp_path: Path) -> None:
    """M10: "a finding with no evidence ref is invalid -- the transform
    must drop it or fail". Dropping silently would let the count of
    findings imply a completeness the evidence does not support, so the
    drop is reported."""
    store = _normalised(tmp_path)

    def _one_good_one_unsourced(prompt: str) -> dict[str, Any] | None:
        if "Write terse minutes" not in prompt:
            return None
        turn_ids = _turn_ids(store)
        return {
            "summary": {
                "text": "A tasking discussion.",
                "evidence_turn_ids": [turn_ids[0]],
                "evidence_section_ids": [],
            },
            "findings": [
                {
                    "kind": "decision",
                    "text": "Report back on Friday.",
                    "evidence_turn_ids": [turn_ids[2]],
                    "evidence_section_ids": [],
                    "owner_participant_id": None,
                    "due": "",
                },
                {
                    "kind": "action",
                    "text": "Something nobody actually said.",
                    "evidence_turn_ids": [],
                    "evidence_section_ids": [],
                    "owner_participant_id": None,
                    "due": "",
                },
            ],
        }

    outcome = asyncio.run(
        run_minutes_transform(
            store,
            agent=stage_agent(StagePlan(override=_one_good_one_unsourced)),
            model="fixture-model",
        )
    )

    assert len(outcome.minutes.findings) == 1
    assert outcome.dropped_unsourced_findings == ("Something nobody actually said.",)


def test_minutes_whose_every_finding_is_unsourced_are_refused(
    tmp_path: Path,
) -> None:
    """Publishing the summary alone would imply the findings were
    considered and found absent."""
    store = _normalised(tmp_path)

    def _all_unsourced(prompt: str) -> dict[str, Any] | None:
        if "Write terse minutes" not in prompt:
            return None
        return {
            "summary": {
                "text": "A tasking discussion.",
                "evidence_turn_ids": [_turn_ids(store)[0]],
                "evidence_section_ids": [],
            },
            "findings": [
                {
                    "kind": "decision",
                    "text": "Invented decision.",
                    "evidence_turn_ids": ["turn_019fb000-0000-7000-8000-000000000000"],
                    "evidence_section_ids": [],
                    "owner_participant_id": None,
                    "due": "",
                }
            ],
        }

    with pytest.raises(ExecutorFailedError, match="NoEvidencedFindingsError"):
        asyncio.run(
            run_minutes_transform(
                store,
                agent=stage_agent(StagePlan(override=_all_unsourced)),
                model="fixture-model",
            )
        )


def test_an_invented_owner_drops_the_assignee_not_the_finding(
    tmp_path: Path,
) -> None:
    """F21: owners come only from participant records. The finding is
    still evidenced, so it survives -- it just has no confirmed owner,
    which is more useful than losing the decision entirely."""
    store = _normalised(tmp_path)

    def _invented_owner(prompt: str) -> dict[str, Any] | None:
        if "Write terse minutes" not in prompt:
            return None
        turn_ids = _turn_ids(store)
        return {
            "summary": {
                "text": "A tasking discussion.",
                "evidence_turn_ids": [turn_ids[0]],
                "evidence_section_ids": [],
            },
            "findings": [
                {
                    "kind": "action",
                    "text": "Report back on Friday.",
                    "evidence_turn_ids": [turn_ids[2]],
                    "evidence_section_ids": [],
                    "owner_participant_id": (
                        "participant_019fb000-0000-7000-8000-000000000000"
                    ),
                    "due": "Friday",
                }
            ],
        }

    outcome = asyncio.run(
        run_minutes_transform(
            store,
            agent=stage_agent(StagePlan(override=_invented_owner)),
            model="fixture-model",
        )
    )

    assert len(outcome.minutes.findings) == 1
    assert outcome.minutes.findings[0].owner_participant_id is None
    assert (
        _head(store).capability_status(CapabilityKey.MINUTES)
        == CapabilityStatus.PRESENT_VALIDATED
    )


def test_minutes_cite_authored_note_sections_when_the_claim_comes_from_them(
    tmp_path: Path,
) -> None:
    """D2/F22: a notes-derived claim stays distinguishable from a
    transcript-derived one all the way to the render."""
    store = _normalised(tmp_path)

    def _notes_derived(prompt: str) -> dict[str, Any] | None:
        if "Write terse minutes" not in prompt:
            return None
        section_ids = re.findall(r"seg_[0-9a-f-]{36}", prompt)
        return {
            "summary": {
                "text": "A tasking discussion.",
                "evidence_turn_ids": [],
                "evidence_section_ids": section_ids[:1],
            },
            "findings": [],
        }

    outcome = asyncio.run(
        run_minutes_transform(
            store,
            agent=stage_agent(StagePlan(override=_notes_derived)),
            model="fixture-model",
        )
    )

    assert outcome.minutes.summary.claim_status.value == "notes-derived"


def test_re_running_a_product_supersedes_rather_than_duplicating(
    tmp_path: Path,
) -> None:
    store = _normalised(tmp_path)
    agent = stage_agent(StagePlan())

    asyncio.run(run_chapter_transform(store, agent=agent, model="fixture-model"))
    asyncio.run(run_minutes_transform(store, agent=agent, model="fixture-model"))
    asyncio.run(
        run_chapter_transform(
            store, agent=stage_agent(StagePlan(chapter_starts=(0, 1))), model="m"
        )
    )

    document = _head(store)
    assert len(document.components_of(ChapterSetComponent)) == 1
    assert len(document.components_of(MinutesComponent)) == 1
