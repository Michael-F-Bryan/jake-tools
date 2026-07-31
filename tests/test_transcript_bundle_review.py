"""M8: the durable review checkpoint -- pack export and apply.

The checkpoint is the whole point of the overhaul, so these tests are
about the properties that make it trustworthy rather than about the pack's
field names: it binds to an exact revision, it refuses stale input without
rebasing, re-applying is a no-op, and a competing review is rejected
rather than merged.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
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

from jake_tools.transcripts.bundle.components import SpeakerReviewComponent
from jake_tools.transcripts.bundle.control import (
    run_normalise_transform,
    run_review_apply,
    run_speakers_propose,
)
from jake_tools.transcripts.bundle.document import TranscriptDocumentV1, project_head
from jake_tools.transcripts.bundle.records import RunState
from jake_tools.transcripts.bundle.registry import CapabilityKey, CapabilityStatus
from jake_tools.transcripts.bundle.review import (
    CompetingReviewError,
    InvalidReviewPackError,
    StaleReviewError,
    UnknownReviewTargetError,
    export_review_pack,
    review_is_complete,
)
from jake_tools.transcripts.bundle.store import BundleStore

_RECORDING = Recording(
    name="meeting.m4a",
    duration_ms=30_000,
    tokens=tokens_from_utterances(
        [
            (1000, 2000, "Are you self employed?"),
            (3000, 5000, "Yes, I am."),
        ]
    ),
    segments=(
        SpokenSegment(900, 2100, "SPEAKER_00"),
        SpokenSegment(2900, 5100, "SPEAKER_01"),
    ),
)


def _bundle_at_review_checkpoint(tmp_path: Path) -> BundleStore:
    prepared = prepare_transcribed_bundle(tmp_path / "bundle", recordings=[_RECORDING])
    run_normalise_transform(prepared.store)
    asyncio.run(
        run_speakers_propose(
            prepared.store,
            agent=stage_agent(StagePlan(proposals={0: 1, 1: 0})),
            model="fixture-model",
        )
    )
    return prepared.store


def _head(store: BundleStore) -> TranscriptDocumentV1:
    document = project_head(store)
    assert isinstance(document, TranscriptDocumentV1)
    return document


def _fill(
    pack_path: Path,
    *,
    reviewer: str = "Michael Bryan",
    decide: Callable[[dict[str, Any], dict[str, str]], list[dict[str, Any]]]
    | None = None,
) -> Path:
    """Fill a pack the way a reviewer would: name yourself, then decide."""
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    participants = {
        participant["display_name"]: participant["participant_id"]
        for participant in pack["participants"]
    }
    pack["reviewer"] = reviewer
    if decide is None:
        pack["decisions"] = [
            {
                "scope": "cluster",
                "kind": "assign",
                "cluster_id": item["item_id"],
                "participant_id": participants["Michael Bryan"],
                "rationale": "reviewed by ear",
            }
            for item in pack["items"]
            if item["kind"] == "cluster"
        ]
    else:
        pack["decisions"] = decide(pack, participants)
    pack_path.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    return pack_path


# -- the checkpoint itself ---------------------------------------------------


def test_proposing_stops_in_review_required_with_the_lease_released(
    tmp_path: Path,
) -> None:
    """The operator must be able to walk away: a durable state with the
    lease still held would block every later command on this bundle."""
    store = _bundle_at_review_checkpoint(tmp_path)

    manifest = store.load_manifest()
    states = {store.load_run(run_id).state for run_id in manifest.run_ids}
    assert RunState.REVIEW_REQUIRED in states
    assert store.load_lease() is None


def test_the_waiting_run_records_what_to_do_next(tmp_path: Path) -> None:
    store = _bundle_at_review_checkpoint(tmp_path)

    waiting = next(
        store.load_run(run_id)
        for run_id in store.load_manifest().run_ids
        if store.load_run(run_id).state == RunState.REVIEW_REQUIRED
    )

    assert waiting.next_action is not None
    assert waiting.next_action.kind == "products"


# -- pack export -------------------------------------------------------------


def test_the_pack_binds_to_the_exact_revision_and_inventories(
    tmp_path: Path,
) -> None:
    store = _bundle_at_review_checkpoint(tmp_path)

    pack_path = export_review_pack(store, destination=tmp_path / "review.json")

    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    assert pack["input_revision_id"] == _head(store).revision_id
    assert len(pack["turn_inventory_sha256"]) == 64
    assert len(pack["cluster_inventory_sha256"]) == 64


def test_the_pack_lists_one_item_per_cluster_with_its_proposal(
    tmp_path: Path,
) -> None:
    store = _bundle_at_review_checkpoint(tmp_path)

    pack = json.loads(
        export_review_pack(store, destination=tmp_path / "review.json").read_text()
    )

    clusters = [item for item in pack["items"] if item["kind"] == "cluster"]
    assert len(clusters) == 2
    assert all(item["hypothesis"] is not None for item in clusters)


def test_exporting_takes_no_lease_and_changes_nothing(tmp_path: Path) -> None:
    store = _bundle_at_review_checkpoint(tmp_path)
    before = store.load_manifest()

    export_review_pack(store, destination=tmp_path / "review.json")

    assert store.load_lease() is None
    assert store.load_manifest() == before


# -- apply -------------------------------------------------------------------


def test_applying_a_filled_pack_makes_human_reviewed_real(tmp_path: Path) -> None:
    store = _bundle_at_review_checkpoint(tmp_path)
    pack_path = _fill(export_review_pack(store, destination=tmp_path / "r.json"))

    outcome = run_review_apply(store, pack_path=pack_path)

    assert not outcome.already_applied
    document = _head(store)
    assert (
        document.capability_status(CapabilityKey.SPEAKERS_HUMAN_REVIEWED)
        == CapabilityStatus.PRESENT_VALIDATED
    )


def test_re_applying_the_same_pack_is_idempotent(tmp_path: Path) -> None:
    """M8: "the application registry returns the existing result revision;
    no duplicate"."""
    store = _bundle_at_review_checkpoint(tmp_path)
    pack_path = _fill(export_review_pack(store, destination=tmp_path / "r.json"))
    first = run_review_apply(store, pack_path=pack_path)

    second = run_review_apply(store, pack_path=pack_path)

    assert second.already_applied
    assert second.review.result_revision_id == first.review.result_revision_id
    assert _head(store).revision_id == first.review.result_revision_id


def test_a_second_differing_review_of_the_same_revision_is_rejected(
    tmp_path: Path,
) -> None:
    """D3 defers branching until something demonstrates the need, so v1
    refuses rather than inventing a merge order for two reviewers."""
    store = _bundle_at_review_checkpoint(tmp_path)
    first = _fill(export_review_pack(store, destination=tmp_path / "first.json"))
    run_review_apply(store, pack_path=first)
    second = tmp_path / "second.json"
    second.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")
    pack = json.loads(second.read_text(encoding="utf-8"))
    pack["review_id"] = pack["review_id"].replace("review_0", "review_1", 1)
    second.write_text(json.dumps(pack, indent=2), encoding="utf-8")

    with pytest.raises(CompetingReviewError):
        run_review_apply(store, pack_path=second)


def test_a_pack_bound_to_a_superseded_revision_is_refused(tmp_path: Path) -> None:
    """No auto-rebase (M8): decisions made against turns that no longer
    exist are not silently re-pointed at whatever is there now."""
    store = _bundle_at_review_checkpoint(tmp_path)
    pack_path = _fill(export_review_pack(store, destination=tmp_path / "r.json"))
    run_normalise_transform(store)

    with pytest.raises(StaleReviewError):
        run_review_apply(store, pack_path=pack_path)


def test_a_pack_with_no_reviewer_is_refused(tmp_path: Path) -> None:
    """M8 binds a review to a reviewer identity; an anonymous review is not
    an auditable one."""
    store = _bundle_at_review_checkpoint(tmp_path)
    pack_path = _fill(
        export_review_pack(store, destination=tmp_path / "r.json"), reviewer="   "
    )

    with pytest.raises(InvalidReviewPackError, match="reviewer"):
        run_review_apply(store, pack_path=pack_path)


def test_a_pack_with_no_decisions_is_refused(tmp_path: Path) -> None:
    """An unreviewed document is left unreviewed, rather than recorded as
    a review that decided nothing -- which would make human-reviewed
    present-validated on evidence nobody looked at."""
    store = _bundle_at_review_checkpoint(tmp_path)
    pack_path = _fill(
        export_review_pack(store, destination=tmp_path / "r.json"),
        decide=lambda pack, participants: [],
    )

    with pytest.raises(InvalidReviewPackError, match="decisions"):
        run_review_apply(store, pack_path=pack_path)


def test_a_decision_naming_an_unknown_participant_is_refused(
    tmp_path: Path,
) -> None:
    """F21: speakers come only from participant records."""
    store = _bundle_at_review_checkpoint(tmp_path)
    pack_path = _fill(
        export_review_pack(store, destination=tmp_path / "r.json"),
        decide=lambda pack, participants: [
            {
                "scope": "cluster",
                "kind": "assign",
                "cluster_id": pack["items"][0]["item_id"],
                "participant_id": "participant_019fb000-0000-7000-8000-000000000000",
                "rationale": "invented",
            }
        ],
    )

    with pytest.raises(UnknownReviewTargetError, match="participant"):
        run_review_apply(store, pack_path=pack_path)


def test_a_decision_naming_an_unknown_cluster_is_refused(tmp_path: Path) -> None:
    store = _bundle_at_review_checkpoint(tmp_path)
    pack_path = _fill(
        export_review_pack(store, destination=tmp_path / "r.json"),
        decide=lambda pack, participants: [
            {
                "scope": "cluster",
                "kind": "assign",
                "cluster_id": "cluster_019fb000-0000-7000-8000-000000000000",
                "participant_id": participants["Michael Bryan"],
                "rationale": "invented",
            }
        ],
    )

    with pytest.raises(UnknownReviewTargetError, match="cluster"):
        run_review_apply(store, pack_path=pack_path)


def test_two_overlapping_source_range_decisions_are_rejected(
    tmp_path: Path,
) -> None:
    """M8: "overlapping conflicting range overrides are rejected at review
    validation, not resolved by precedence"."""
    store = _bundle_at_review_checkpoint(tmp_path)

    def _decide(pack: dict[str, Any], participants: dict[str, str]):
        source_artefact_id = pack["turns"][0]["source_artefact_id"]
        return [
            {
                "scope": "source-range",
                "kind": "assign",
                "source_artefact_id": source_artefact_id,
                "start_ms": 0,
                "end_ms": 4000,
                "participant_id": participants["Michael Bryan"],
                "rationale": "first",
            },
            {
                "scope": "source-range",
                "kind": "assign",
                "source_artefact_id": source_artefact_id,
                "start_ms": 3000,
                "end_ms": 6000,
                "participant_id": participants["Avalon Mann"],
                "rationale": "second, overlapping",
            },
        ]

    pack_path = _fill(
        export_review_pack(store, destination=tmp_path / "r.json"), decide=_decide
    )

    with pytest.raises(Exception, match="overlapping"):
        run_review_apply(store, pack_path=pack_path)


# -- partial vs complete -----------------------------------------------------


def test_a_partial_review_is_distinguishable_from_a_complete_one(
    tmp_path: Path,
) -> None:
    """M8: "a partial review is distinguishable from a complete review with
    unresolved items (coverage list, not a flag)"."""
    store = _bundle_at_review_checkpoint(tmp_path)
    pack_path = _fill(
        export_review_pack(store, destination=tmp_path / "r.json"),
        decide=lambda pack, participants: [
            {
                "scope": "cluster",
                "kind": "assign",
                "cluster_id": pack["items"][0]["item_id"],
                "participant_id": participants["Michael Bryan"],
                "rationale": "only decided one of them",
            }
        ],
    )

    outcome = run_review_apply(store, pack_path=pack_path)

    assert outcome.addressed_item_count == 1
    assert outcome.total_item_count == 2
    review = _head(store).components_of(SpeakerReviewComponent)[0]
    assert not review_is_complete(review)


def test_a_complete_review_with_an_unclear_item_is_still_complete(
    tmp_path: Path,
) -> None:
    store = _bundle_at_review_checkpoint(tmp_path)
    pack_path = _fill(
        export_review_pack(store, destination=tmp_path / "r.json"),
        decide=lambda pack, participants: [
            {
                "scope": "cluster",
                "kind": "assign",
                "cluster_id": pack["items"][0]["item_id"],
                "participant_id": participants["Michael Bryan"],
                "rationale": "recognised the voice",
            },
            {
                "scope": "cluster",
                "kind": "unclear-speaker",
                "cluster_id": pack["items"][1]["item_id"],
                "rationale": "genuinely could not tell",
            },
        ],
    )

    run_review_apply(store, pack_path=pack_path)

    review = _head(store).components_of(SpeakerReviewComponent)[0]
    assert review_is_complete(review)
    assert (
        _head(store).capability_status(CapabilityKey.SPEAKERS_HUMAN_CONFIRMED)
        == CapabilityStatus.ABSENT
    ), "an unclear item is reviewed but not confirmed"


def test_the_avalon_trap_can_be_resolved_as_unclear_rather_than_guessed(
    tmp_path: Path,
) -> None:
    """The corpus's material-failure case: confidently assigning the
    ambiguous fragment is worse than leaving it explicitly unresolved.
    Both are reachable, and the unresolved one still passes the gate."""
    store = _bundle_at_review_checkpoint(tmp_path)
    pack_path = _fill(
        export_review_pack(store, destination=tmp_path / "r.json"),
        decide=lambda pack, participants: [
            {
                "scope": "cluster",
                "kind": "unclear-speaker",
                "cluster_id": item["item_id"],
                "rationale": "acoustically ambiguous",
            }
            for item in pack["items"]
            if item["kind"] == "cluster"
        ],
    )

    run_review_apply(store, pack_path=pack_path)

    document = _head(store)
    assert (
        document.capability_status(CapabilityKey.SPEAKERS_HUMAN_REVIEWED)
        == CapabilityStatus.PRESENT_VALIDATED
    )
    assert (
        document.capability_status(CapabilityKey.SPEAKERS_HUMAN_CONFIRMED)
        == CapabilityStatus.ABSENT
    )


def test_applying_a_review_uses_a_run_of_its_own(tmp_path: Path) -> None:
    """The run waiting in review_required released its lease precisely so
    this could happen later; M2 allows one *active* run, not one run per
    bundle lifetime."""
    store = _bundle_at_review_checkpoint(tmp_path)
    before = len(store.load_manifest().run_ids)
    pack_path = _fill(export_review_pack(store, destination=tmp_path / "r.json"))

    run_review_apply(store, pack_path=pack_path)

    assert len(store.load_manifest().run_ids) == before + 1
    assert store.load_lease() is None
