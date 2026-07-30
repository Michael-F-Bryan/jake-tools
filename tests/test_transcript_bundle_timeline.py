"""M6 combined-timeline tests: pure mapping functions (seeded property
tests, no new dependency -- the same ``random.Random(seed)`` loop
convention ``inference-worker/tests/test_asr_merge_property.py`` already
uses for "no hypothesis" property testing), the refusal invariants on
:class:`~.components.TimelineCombinedComponentBody`, and
:func:`transform_timeline`'s store-level orchestration.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest
from fixtures_bundle import registered_source_and_artefact

from jake_tools.transcripts.bundle.assemble import assemble
from jake_tools.transcripts.bundle.components import (
    ArtefactSelection,
    Disposition,
    MediaRecordingComponentBody,
    RecordingReference,
    RecordingReferenceSetComponentBody,
    TimelineGapRecord,
    TimelineMappingSegment,
)
from jake_tools.transcripts.bundle.ids import mint_id
from jake_tools.transcripts.bundle.records import OperationRef, RunState
from jake_tools.transcripts.bundle.store import BundleStore
from jake_tools.transcripts.bundle.timeline import (
    AmbiguousNoteEmbedOrderError,
    MediaMember,
    NoDocumentToTransformError,
    NoTimelineMediaError,
    TimelineRangeError,
    build_combined_timeline,
    map_combined_to_source,
    map_source_to_combined,
    transform_timeline,
)

_SEED_COUNT = 200


def _random_members(rng: random.Random) -> list[MediaMember]:
    count = rng.randint(1, 6)
    return [
        MediaMember(
            artefact_id=mint_id("artefact"), duration_ms=rng.randint(1, 200_000)
        )
        for _ in range(count)
    ]


# -- seeded property tests -------------------------------------------------


def test_round_trip_identity_over_many_seeds() -> None:
    """source -> combined -> source recovers the exact (artefact, ms) pair
    for every in-range instant, across many randomly generated member
    sequences."""
    for seed in range(_SEED_COUNT):
        rng = random.Random(seed)
        members = _random_members(rng)
        body = build_combined_timeline(members)

        for member in members:
            for _ in range(5):
                source_ms = rng.randint(0, member.duration_ms - 1)
                combined_ms = map_source_to_combined(
                    body.segments, artefact_id=member.artefact_id, source_ms=source_ms
                )
                back_artefact, back_source_ms = map_combined_to_source(
                    body.segments, combined_ms=combined_ms
                )
                assert (back_artefact, back_source_ms) == (
                    member.artefact_id,
                    source_ms,
                ), f"seed {seed}"


def test_boundary_exactness_over_many_seeds() -> None:
    """The first and last valid millisecond of every member map exactly
    (start -> segment start; duration-1 -> segment end-1); the exclusive
    upper boundary (source_ms == duration) is refused, matching every
    other half-open span in this codebase."""
    for seed in range(_SEED_COUNT):
        rng = random.Random(seed)
        members = _random_members(rng)
        body = build_combined_timeline(members)

        offset = 0
        for member in members:
            start_combined = map_source_to_combined(
                body.segments, artefact_id=member.artefact_id, source_ms=0
            )
            assert start_combined == offset, f"seed {seed}"
            end_combined = map_source_to_combined(
                body.segments,
                artefact_id=member.artefact_id,
                source_ms=member.duration_ms - 1,
            )
            assert end_combined == offset + member.duration_ms - 1, f"seed {seed}"
            offset += member.duration_ms


def test_out_of_range_refusal_over_many_seeds() -> None:
    """Negative offsets, the exclusive upper boundary, and anything past
    it are all refused -- never clamped, never silently wrapped to a
    neighbour."""
    for seed in range(_SEED_COUNT):
        rng = random.Random(seed)
        members = _random_members(rng)
        body = build_combined_timeline(members)
        member = members[0]

        for bad_source_ms in (member.duration_ms, member.duration_ms + 1000):
            with pytest.raises(TimelineRangeError):
                map_source_to_combined(
                    body.segments,
                    artefact_id=member.artefact_id,
                    source_ms=bad_source_ms,
                )

        total_duration = sum(m.duration_ms for m in members)
        with pytest.raises(TimelineRangeError):
            map_combined_to_source(body.segments, combined_ms=total_duration)
        with pytest.raises(TimelineRangeError):
            map_combined_to_source(body.segments, combined_ms=total_duration + 500)


def test_map_source_to_combined_refuses_an_artefact_not_in_the_timeline() -> None:
    rng = random.Random(0)
    members = _random_members(rng)
    body = build_combined_timeline(members)

    with pytest.raises(TimelineRangeError):
        map_source_to_combined(
            body.segments, artefact_id=mint_id("artefact"), source_ms=0
        )


def test_map_combined_to_source_refuses_inside_a_real_gap() -> None:
    """A non-zero gap (an explicit, declared inter-recording break) is not
    "owned" by either neighbour -- map_combined_to_source must refuse
    inside it, not silently attribute it to whichever segment is nearest.
    """
    first = mint_id("artefact")
    second = mint_id("artefact")
    segments = (
        TimelineMappingSegment(
            artefact_id=first,
            source_start_ms=0,
            source_end_ms=1000,
            combined_start_ms=0,
            combined_end_ms=1000,
        ),
        TimelineMappingSegment(
            artefact_id=second,
            source_start_ms=0,
            source_end_ms=1000,
            combined_start_ms=1500,
            combined_end_ms=2500,
        ),
    )

    with pytest.raises(TimelineRangeError):
        map_combined_to_source(segments, combined_ms=1200)


# -- build_combined_timeline: gap records -----------------------------------


def test_build_combined_timeline_emits_one_gap_per_boundary() -> None:
    members = [
        MediaMember(artefact_id=mint_id("artefact"), duration_ms=1000),
        MediaMember(artefact_id=mint_id("artefact"), duration_ms=2000),
        MediaMember(artefact_id=mint_id("artefact"), duration_ms=500),
    ]

    body = build_combined_timeline(members)

    assert len(body.gaps) == 2
    assert body.gaps[0].after_artefact_id == members[0].artefact_id
    assert body.gaps[0].before_artefact_id == members[1].artefact_id
    assert body.gaps[0].combined_start_ms == body.gaps[0].combined_end_ms == 1000
    assert body.gaps[1].combined_start_ms == body.gaps[1].combined_end_ms == 3000


def test_build_combined_timeline_single_member_has_no_gaps() -> None:
    body = build_combined_timeline(
        [MediaMember(artefact_id=mint_id("artefact"), duration_ms=1000)]
    )
    assert body.gaps == ()
    assert len(body.segments) == 1


def test_build_combined_timeline_refuses_empty_members() -> None:
    with pytest.raises(NoTimelineMediaError):
        build_combined_timeline([])


# -- TimelineCombinedComponentBody: the enforcement mechanism itself --------


def test_overlapping_segments_are_refused_at_construction() -> None:
    """The actual M6 enforcement mechanism: overlap is refused by the
    component's own model_validator, not merely by the (non-overlap-
    producing) builder."""
    from jake_tools.transcripts.bundle.components import TimelineCombinedComponentBody

    first = mint_id("artefact")
    second = mint_id("artefact")
    with pytest.raises(ValueError, match="[Oo]verlap"):
        TimelineCombinedComponentBody(
            segments=(
                TimelineMappingSegment(
                    artefact_id=first,
                    source_start_ms=0,
                    source_end_ms=1000,
                    combined_start_ms=0,
                    combined_end_ms=1000,
                ),
                TimelineMappingSegment(
                    artefact_id=second,
                    source_start_ms=0,
                    source_end_ms=1000,
                    combined_start_ms=500,
                    combined_end_ms=1500,
                ),
            )
        )


def test_out_of_order_segments_are_refused_at_construction() -> None:
    from jake_tools.transcripts.bundle.components import TimelineCombinedComponentBody

    with pytest.raises(ValueError, match="ascending"):
        TimelineCombinedComponentBody(
            segments=(
                TimelineMappingSegment(
                    artefact_id=mint_id("artefact"),
                    source_start_ms=0,
                    source_end_ms=1000,
                    combined_start_ms=1000,
                    combined_end_ms=2000,
                ),
                TimelineMappingSegment(
                    artefact_id=mint_id("artefact"),
                    source_start_ms=0,
                    source_end_ms=1000,
                    combined_start_ms=0,
                    combined_end_ms=1000,
                ),
            )
        )


def test_mapping_segment_refuses_a_span_that_changes_length() -> None:
    """A combined-timeline mapping is a pure shift (M6) -- stretching or
    compressing time is refused, not silently applied."""
    with pytest.raises(ValueError, match="preserve span length"):
        TimelineMappingSegment(
            artefact_id=mint_id("artefact"),
            source_start_ms=0,
            source_end_ms=1000,
            combined_start_ms=0,
            combined_end_ms=2000,
        )


def test_gap_record_refuses_a_reversed_span() -> None:
    with pytest.raises(ValueError, match="end < start"):
        TimelineGapRecord(
            after_artefact_id=mint_id("artefact"),
            before_artefact_id=mint_id("artefact"),
            combined_start_ms=1000,
            combined_end_ms=500,
        )


# -- transform_timeline: store-level orchestration --------------------------


def _bundle(tmp_path: Path) -> BundleStore:
    store = BundleStore(tmp_path / "bundle")
    store.create_bundle()
    return store


def test_transform_timeline_refuses_when_nothing_is_assembled_yet(
    tmp_path: Path,
) -> None:
    store = _bundle(tmp_path)
    run = store.create_run(next_action=OperationRef(kind="timeline"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())

    with pytest.raises(NoDocumentToTransformError):
        transform_timeline(store, run_id=run.run_id)


def _assemble_media_only(store: BundleStore, *, durations_ms: list[int]) -> list[str]:
    """Ingest N synthetic media artefacts (no note), add a
    MediaRecordingComponent for each, assemble them all with disposition
    ``media``, and move the head. Returns the artefact IDs in the order
    assembled (also the order `transform_timeline`'s explicit ``order``
    argument will need, since there is no note-embed reference here)."""
    artefact_ids: list[str] = []
    component_ids: list[str] = []
    for index, duration_ms in enumerate(durations_ms):
        artefact = registered_source_and_artefact(
            store,
            content=f"media-{index}".encode(),
            kind="audio",
            acquisition_locator=f"/tmp/media-{index}.wav",
        )
        component = store.add_component(
            MediaRecordingComponentBody(
                source_artefact_id=artefact.artefact_id,
                media_path=f"/tmp/media-{index}.wav",
                duration_ms=duration_ms,
                codec="opus",
                sample_rate_hz=48000,
                channels=1,
            )
        )
        artefact_ids.append(artefact.artefact_id)
        component_ids.append(component.component_id)

    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    assemble(
        store,
        run_id=run.run_id,
        selections=tuple(
            ArtefactSelection(
                artefact_id=artefact_id, dispositions=(Disposition.MEDIA,)
            )
            for artefact_id in artefact_ids
        ),
        rationale="synthetic media for timeline tests",
        component_ids=tuple(component_ids),
    )
    store.release_lease(run_id=run.run_id, new_state=RunState.COMPLETED)
    return artefact_ids


def test_transform_timeline_with_explicit_order(tmp_path: Path) -> None:
    store = _bundle(tmp_path)
    artefact_ids = _assemble_media_only(store, durations_ms=[1000, 2000])

    run = store.create_run(next_action=OperationRef(kind="timeline"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    transform_timeline(store, run_id=run.run_id, order=tuple(reversed(artefact_ids)))
    store.release_lease(run_id=run.run_id, new_state=RunState.COMPLETED)

    from jake_tools.transcripts.bundle.components import TimelineCombinedComponent
    from jake_tools.transcripts.bundle.document import NoDocumentYet, project_head

    document = project_head(store)
    assert not isinstance(document, NoDocumentYet)
    (timeline,) = [
        c
        for c in document.components.values()
        if isinstance(c, TimelineCombinedComponent)
    ]
    assert [segment.artefact_id for segment in timeline.segments] == list(
        reversed(artefact_ids)
    )


def test_transform_timeline_refuses_when_no_media_resolves(tmp_path: Path) -> None:
    """An assembled document with a note-embed reference but no ingested
    media at all has nothing to build a timeline from."""
    store = _bundle(tmp_path)
    note_artefact = registered_source_and_artefact(
        store, content=b"note text", kind="obsidian-note"
    )
    reference_set = store.add_component(
        RecordingReferenceSetComponentBody(
            note_artefact_id=note_artefact.artefact_id,
            references=(
                RecordingReference(raw_link="[[x.m4a]]", resolved_path="/tmp/x.m4a"),
            ),
        )
    )
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    assemble(
        store,
        run_id=run.run_id,
        selections=(
            ArtefactSelection(
                artefact_id=note_artefact.artefact_id,
                dispositions=(Disposition.NOTES,),
            ),
        ),
        rationale="note only, no media ingested",
        component_ids=(reference_set.component_id,),
    )
    store.release_lease(run_id=run.run_id, new_state=RunState.COMPLETED)

    run2 = store.create_run(next_action=OperationRef(kind="timeline"))
    store.acquire_lease(run_id=run2.run_id, pid=os.getpid())
    with pytest.raises(NoTimelineMediaError):
        transform_timeline(store, run_id=run2.run_id)


def test_transform_timeline_ambiguous_default_order_needs_explicit_override(
    tmp_path: Path,
) -> None:
    """Two recording-reference-set components in one closure make "the"
    note-embed default ambiguous -- refused, not silently picked from."""
    store = _bundle(tmp_path)
    artefact_ids = _assemble_media_only(store, durations_ms=[1000, 2000])
    note_a = registered_source_and_artefact(
        store, content=b"note a", kind="obsidian-note", acquisition_locator="/tmp/a.md"
    )
    note_b = registered_source_and_artefact(
        store, content=b"note b", kind="obsidian-note", acquisition_locator="/tmp/b.md"
    )
    reference_set_a = store.add_component(
        RecordingReferenceSetComponentBody(
            note_artefact_id=note_a.artefact_id,
            references=(
                RecordingReference(raw_link="[[x]]", resolved_path="/tmp/media-0.wav"),
            ),
        )
    )
    reference_set_b = store.add_component(
        RecordingReferenceSetComponentBody(
            note_artefact_id=note_b.artefact_id,
            references=(
                RecordingReference(raw_link="[[y]]", resolved_path="/tmp/media-1.wav"),
            ),
        )
    )
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    assemble(
        store,
        run_id=run.run_id,
        selections=(
            ArtefactSelection(
                artefact_id=note_a.artefact_id, dispositions=(Disposition.NOTES,)
            ),
            ArtefactSelection(
                artefact_id=note_b.artefact_id, dispositions=(Disposition.NOTES,)
            ),
        ),
        rationale="two notes, both referencing already-assembled media",
        component_ids=(reference_set_a.component_id, reference_set_b.component_id),
    )
    store.release_lease(run_id=run.run_id, new_state=RunState.COMPLETED)

    run2 = store.create_run(next_action=OperationRef(kind="timeline"))
    store.acquire_lease(run_id=run2.run_id, pid=os.getpid())
    with pytest.raises(AmbiguousNoteEmbedOrderError):
        transform_timeline(store, run_id=run2.run_id)

    # Explicit order sidesteps the ambiguity entirely.
    transform_timeline(store, run_id=run2.run_id, order=tuple(artefact_ids))
