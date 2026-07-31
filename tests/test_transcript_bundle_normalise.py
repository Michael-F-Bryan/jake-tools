"""M7: the normalisation transform.

Behaviour-first: every test here drives ``normalise_transcript`` over a
bundle built the way production builds one (``bundle_pipeline``), and
asserts on the canonical turns, the machine attribution, and the lineage
ledger it produced -- not on internal call sequences. The pure helpers
(``assign_tokens_to_segments``) are exercised directly where a property is
easier to pin down than to arrange end to end.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from bundle_pipeline import (
    Recording,
    SpokenSegment,
    SpokenToken,
    prepare_transcribed_bundle,
    tokens_from_utterances,
)

from jake_tools.transcripts.bundle.components import (
    DropReason,
    MachineAttributionSetComponent,
    NormalisationLedgerComponent,
    TimedTurnSetComponent,
    TimelineCombinedComponent,
)
from jake_tools.transcripts.bundle.document import TranscriptDocumentV1, project_head
from jake_tools.transcripts.bundle.ids import mint_id
from jake_tools.transcripts.bundle.normalise import (
    NoCombinedTimelineError,
    NoInferenceEvidenceError,
    NormaliseOutcome,
    RawToken,
    assign_tokens_to_segments,
    normalise_transcript,
)
from jake_tools.transcripts.bundle.records import OperationRef, RunState
from jake_tools.transcripts.bundle.registry import CapabilityKey, CapabilityStatus
from jake_tools.transcripts.bundle.store import BundleStore
from jake_tools.transcripts.bundle.worker_contract import WireDiarisationSegment


def _head(store: BundleStore) -> TranscriptDocumentV1:
    document = project_head(store)
    assert isinstance(document, TranscriptDocumentV1)
    return document


def _normalise(store: BundleStore) -> NormaliseOutcome:
    run = store.create_run(next_action=OperationRef(kind="normalise", rationale="test"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    try:
        return normalise_transcript(store, run_id=run.run_id)
    finally:
        store.release_lease(run_id=run.run_id, new_state=RunState.COMPLETED)


def _raw_token(start_ms: int, end_ms: int, text: str = " word") -> RawToken:
    return RawToken(
        source_segment_id=mint_id("seg"),
        source_artefact_id=mint_id("artefact"),
        start_ms=start_ms,
        end_ms=end_ms,
        text=text,
    )


def _segment(start_ms: int, end_ms: int, label: str) -> SpokenSegment:
    return SpokenSegment(start_ms=start_ms, end_ms=end_ms, speaker_label=label)


def _wire_segment(start_ms: int, end_ms: int, label: str) -> WireDiarisationSegment:
    """A diarisation segment in the worker's own wire shape.

    The pure-function tests below call ``assign_tokens_to_segments``
    directly, which reads what the worker actually wrote -- not the
    fixture builder's convenience dataclass.
    """
    return WireDiarisationSegment(start_ms=start_ms, end_ms=end_ms, speaker_label=label)


# -- the maximal-overlap rule (M7) -------------------------------------------


def test_a_token_is_assigned_to_the_segment_it_overlaps_most() -> None:
    token = _raw_token(1000, 2000)
    segments = [
        _wire_segment(0, 1200, "SPEAKER_00"),
        _wire_segment(1200, 3000, "SPEAKER_01"),
    ]

    assert assign_tokens_to_segments([token], segments) == (1,)


def test_a_tie_resolves_to_the_preceding_segment() -> None:
    """M7: "ties to the preceding segment". A coin-flip here would make the
    canonical set depend on iteration order, which is exactly the kind of
    non-determinism a review binding cannot survive."""
    token = _raw_token(1000, 2000)
    segments = [
        _wire_segment(0, 1500, "SPEAKER_00"),
        _wire_segment(1500, 3000, "SPEAKER_01"),
    ]

    assert assign_tokens_to_segments([token], segments) == (0,)


def test_a_token_no_segment_covers_is_unassigned_rather_than_nearest() -> None:
    token = _raw_token(5000, 6000)
    segments = [_wire_segment(0, 1000, "SPEAKER_00")]

    assert assign_tokens_to_segments([token], segments) == (None,)


def test_no_diarisation_segments_leaves_every_token_unassigned() -> None:
    tokens = [_raw_token(0, 100), _raw_token(100, 200)]

    assert assign_tokens_to_segments(tokens, []) == (None, None)


# -- canonical turns -----------------------------------------------------


def _single_recording(tmp_path: Path, **overrides: object) -> Recording:
    defaults: dict[str, object] = {
        "name": "meeting.m4a",
        "duration_ms": 30_000,
        "tokens": tokens_from_utterances(
            [(1000, 2000, "Yes."), (3000, 5000, "Are you self employed?")]
        ),
        "segments": (
            _segment(900, 2100, "SPEAKER_00"),
            _segment(2900, 5100, "SPEAKER_01"),
        ),
    }
    defaults.update(overrides)
    return Recording(**defaults)  # pyright: ignore[reportArgumentType]


def test_tokens_become_one_turn_per_diarisation_segment(tmp_path: Path) -> None:
    prepared = prepare_transcribed_bundle(
        tmp_path / "bundle", recordings=[_single_recording(tmp_path)]
    )

    outcome = _normalise(prepared.store)

    assert [turn.text for turn in outcome.turn_set.turns] == [
        "Yes.",
        "Are you self employed?",
    ]
    assert [turn.speaker_label for turn in outcome.turn_set.turns] == [
        "SPEAKER_00",
        "SPEAKER_01",
    ]


def test_overlapping_speech_becomes_two_whole_turns_not_interleaved_fragments(
    tmp_path: Path,
) -> None:
    """The corpus's simultaneous 00:58 utterances. Grouping by adjacency in
    the time-sorted token stream would shatter both speakers' sentences
    into fragments; grouping by segment keeps them whole and overlapping,
    which is what D4 says concurrent speech is."""
    recording = _single_recording(
        tmp_path,
        tokens=tokens_from_utterances(
            [(13000, 14000, "Yes, I am."), (13800, 14800, "Okay then.")]
        ),
        segments=(
            _segment(12950, 14050, "SPEAKER_01"),
            _segment(13750, 14850, "SPEAKER_00"),
        ),
    )
    prepared = prepare_transcribed_bundle(tmp_path / "bundle", recordings=[recording])

    outcome = _normalise(prepared.store)

    texts = [turn.text for turn in outcome.turn_set.turns]
    assert "Yes, I am." in texts
    assert "Okay then." in texts
    first, second = outcome.turn_set.turns[0], outcome.turn_set.turns[1]
    assert second.start_ms < first.end_ms, "the two turns should overlap in time"


def test_zero_length_and_duplicate_raw_tokens_are_dropped_with_lineage(
    tmp_path: Path,
) -> None:
    """M11 keeps raw output faithful (parakeet really does emit duplicate
    zero-length tokens); M6 forbids them in a canonical turn. Both facts
    hold at once only if normalisation drops them *with a record*."""
    recording = _single_recording(
        tmp_path,
        tokens=(
            SpokenToken(1000, 1000, "ghost"),
            SpokenToken(1000, 1500, " Yes."),
            SpokenToken(1000, 1500, " Yes."),
        ),
        segments=(_segment(900, 2000, "SPEAKER_00"),),
    )
    prepared = prepare_transcribed_bundle(tmp_path / "bundle", recordings=[recording])

    outcome = _normalise(prepared.store)

    assert outcome.dropped_empty_count == 1
    assert outcome.dropped_duplicate_count == 1
    assert [turn.text for turn in outcome.turn_set.turns] == ["Yes."]
    duplicate = next(
        record
        for record in outcome.ledger.dropped
        if record.reason == DropReason.DROPPED_AS_DUPLICATE
    )
    assert duplicate.retained_source_segment_id is not None


def test_the_raw_segment_ledger_accounts_for_every_token(tmp_path: Path) -> None:
    recording = _single_recording(
        tmp_path,
        tokens=(
            SpokenToken(1000, 1000, "ghost"),
            SpokenToken(1000, 1500, " Yes."),
        ),
        segments=(_segment(900, 2000, "SPEAKER_00"),),
    )
    prepared = prepare_transcribed_bundle(tmp_path / "bundle", recordings=[recording])

    outcome = _normalise(prepared.store)

    ledger = outcome.ledger.raw_source_segments
    assert ledger.total == 2
    assert ledger.dropped == 1
    assert ledger.accounted == 1


def test_multiple_recordings_share_one_canonical_set_in_combined_time(
    tmp_path: Path,
) -> None:
    """transcript.timed is one-cardinality (M4), so three recordings do not
    make three turn sets -- they make one, in the combined timeline's own
    coordinate domain, with the second recording's turns shifted past the
    first's duration."""
    first = Recording(
        name="part-1.m4a",
        duration_ms=10_000,
        tokens=tokens_from_utterances([(1000, 2000, "First part.")]),
        segments=(_segment(900, 2100, "SPEAKER_00"),),
    )
    second = Recording(
        name="part-2.m4a",
        duration_ms=10_000,
        tokens=tokens_from_utterances([(1000, 2000, "Second part.")]),
        segments=(_segment(900, 2100, "SPEAKER_00"),),
    )
    prepared = prepare_transcribed_bundle(
        tmp_path / "bundle", recordings=[first, second]
    )

    outcome = _normalise(prepared.store)

    assert len(outcome.turn_set.source_artefact_ids) == 2
    assert outcome.turn_set.coordinate_domain.startswith("combined:")
    starts = [turn.start_ms for turn in outcome.turn_set.turns]
    assert starts == [1000, 11000], "part 2 must be shifted past part 1's duration"


def test_each_recording_gets_its_own_clusters_even_for_the_same_raw_label(
    tmp_path: Path,
) -> None:
    """M5: clusters are scoped to one diarisation output over one media
    artefact. ``SPEAKER_00`` on part 1 and ``SPEAKER_00`` on part 2 are not
    known to be the same voice, and v1 never assumes they are."""
    recordings = [
        Recording(
            name=f"part-{index}.m4a",
            duration_ms=10_000,
            tokens=tokens_from_utterances([(1000, 2000, f"Part {index}.")]),
            segments=(_segment(900, 2100, "SPEAKER_00"),),
        )
        for index in (1, 2)
    ]
    prepared = prepare_transcribed_bundle(tmp_path / "bundle", recordings=recordings)

    outcome = _normalise(prepared.store)

    assert outcome.attribution is not None
    clusters = outcome.attribution.clusters
    assert len(clusters) == 2
    assert len({cluster.cluster_id for cluster in clusters}) == 2
    assert {cluster.raw_label for cluster in clusters} == {"SPEAKER_00"}


# -- capabilities and re-running ---------------------------------------------


def test_normalisation_makes_the_timed_and_clustered_capabilities_real(
    tmp_path: Path,
) -> None:
    prepared = prepare_transcribed_bundle(
        tmp_path / "bundle", recordings=[_single_recording(tmp_path)]
    )

    _normalise(prepared.store)

    document = _head(prepared.store)
    assert (
        document.capability_status(CapabilityKey.TRANSCRIPT_TIMED)
        == CapabilityStatus.PRESENT_VALIDATED
    )
    assert (
        document.capability_status(CapabilityKey.SPEAKERS_MACHINE_CLUSTERED)
        == CapabilityStatus.PRESENT_VALIDATED
    )
    assert (
        document.capability_status(CapabilityKey.SPEAKERS_HUMAN_REVIEWED)
        == CapabilityStatus.ABSENT
    ), "clusters are not a review"


def test_re_normalising_supersedes_rather_than_duplicating(tmp_path: Path) -> None:
    """Two canonical turn sets in one closure would make transcript.timed
    ambiguous and fail validation, so a second pass must retract the
    first (M21)."""
    prepared = prepare_transcribed_bundle(
        tmp_path / "bundle", recordings=[_single_recording(tmp_path)]
    )
    _normalise(prepared.store)

    _normalise(prepared.store)

    document = _head(prepared.store)
    assert len(document.components_of(TimedTurnSetComponent)) == 1
    assert len(document.components_of(MachineAttributionSetComponent)) == 1
    assert len(document.components_of(NormalisationLedgerComponent)) == 1


def test_a_recording_whose_diarisation_failed_still_yields_canonical_turns(
    tmp_path: Path,
) -> None:
    """M11's partial-failure rule: losing diarisation costs the speaker
    evidence, never the transcript. Every turn is honestly unattributed,
    which M8 rung 8 renders as "Unclear speaker" rather than a guess."""
    prepared = prepare_transcribed_bundle(
        tmp_path / "bundle",
        recordings=[_single_recording(tmp_path)],
        with_diarisation=False,
    )

    outcome = _normalise(prepared.store)

    assert outcome.attribution is None
    assert outcome.unattributed_turn_count == len(outcome.turn_set.turns)
    document = _head(prepared.store)
    assert (
        document.capability_status(CapabilityKey.SPEAKERS_MACHINE_CLUSTERED)
        == CapabilityStatus.ABSENT
    )


def test_normalising_without_a_timeline_is_refused(tmp_path: Path) -> None:
    """Refusing beats falling back to per-recording source coordinates,
    which would silently produce a set whose timestamps mean different
    things per turn."""
    prepared = prepare_transcribed_bundle(
        tmp_path / "bundle", recordings=[_single_recording(tmp_path)]
    )
    store = prepared.store
    document = _head(store)
    timeline = document.components_of(TimelineCombinedComponent)[0]
    run = store.create_run(
        next_action=OperationRef(kind="retract", rationale="drop the timeline")
    )
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    revision = store.append_revision(
        operation=OperationRef(kind="retract", rationale="retract the timeline"),
        parent_revision_ids=(document.revision_id,),
        superseded_component_ids=(timeline.component_id,),
    )
    store.update_head(run_id=run.run_id, revision_id=revision.revision_id)
    store.release_lease(run_id=run.run_id, new_state=RunState.COMPLETED)

    with pytest.raises(NoCombinedTimelineError):
        _normalise(store)


def test_normalising_a_document_with_no_asr_evidence_is_refused(
    tmp_path: Path,
) -> None:
    """A recording with no completed ASR stage has nothing to normalise;
    refusing names that, rather than writing an empty canonical set that
    would read as "this meeting had no speech"."""
    prepared = prepare_transcribed_bundle(
        tmp_path / "bundle",
        recordings=[_single_recording(tmp_path, tokens=(), segments=())],
    )

    with pytest.raises(NoInferenceEvidenceError):
        _normalise(prepared.store)
