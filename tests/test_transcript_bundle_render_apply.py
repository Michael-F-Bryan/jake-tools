"""M17/M13: rendering the meeting note, and writing it into the note.

The apply half is the only place this engine writes outside the repo, so
its tests are the ones that matter most: authored content above *and
below* the owned region must survive byte-identically, a note that changed
since the render must be refused, and a legacy note whose generated
sections have authored content between them must fail closed rather than
absorb it.

Nothing here touches the vault. Every target is a file this test created.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from bundle_pipeline import (
    Recording,
    SpokenSegment,
    prepare_transcribed_bundle,
    tokens_from_utterances,
)
from bundle_stage_agent import StagePlan, stage_agent

from jake_tools.transcripts.bundle.apply import (
    END_MARKER,
    AmbiguousOwnedRegionError,
    LegacyMigrationRefusedError,
    StaleRenderError,
    StaleTargetError,
    apply_render,
    begin_marker,
    compose_note,
    locate_owned_region,
)
from jake_tools.transcripts.bundle.components import (
    ChapterSetComponent,
    ChapterSetComponentBody,
    DestinationComponentBody,
    OwnedRegionState,
)
from jake_tools.transcripts.bundle.control import (
    render_document,
    run_chapter_transform,
    run_minutes_transform,
    run_normalise_transform,
    run_review_apply,
    run_speakers_propose,
)
from jake_tools.transcripts.bundle.document import TranscriptDocumentV1, project_head
from jake_tools.transcripts.bundle.product_review import (
    InvalidProductReviewPackError,
    ProductDisposition,
    export_product_review_pack,
    record_product_review,
)
from jake_tools.transcripts.bundle.records import ApplyState, OperationRef, RunState
from jake_tools.transcripts.bundle.render import (
    SpeakerGateFailedError,
    render_meeting_note,
)
from jake_tools.transcripts.bundle.review import export_review_pack
from jake_tools.transcripts.bundle.store import BundleStore
from jake_tools.transcripts.errors import TranscriptError

_BUNDLE_ID = "bundle_019fb000-0000-7000-8000-000000000001"

_RECORDING = Recording(
    name="meeting.m4a",
    duration_ms=40_000,
    tokens=tokens_from_utterances(
        [
            (1000, 5000, "We should look at the autonomy platforms next week."),
            (7000, 11000, "I think the middleware question comes first."),
            (13000, 17000, "Agreed, and we will report back on Friday."),
        ]
    ),
    segments=(
        SpokenSegment(900, 5100, "SPEAKER_00"),
        SpokenSegment(6900, 11100, "SPEAKER_01"),
        SpokenSegment(12900, 17100, "SPEAKER_00"),
    ),
)


def _head(store: BundleStore) -> TranscriptDocumentV1:
    document = project_head(store)
    assert isinstance(document, TranscriptDocumentV1)
    return document


def _reviewed_bundle(tmp_path: Path, *, assign: bool = True) -> BundleStore:
    """A bundle carrying everything the meeting-note profile requires."""
    prepared = prepare_transcribed_bundle(tmp_path / "bundle", recordings=[_RECORDING])
    store = prepared.store
    run_normalise_transform(store)
    agent = stage_agent(StagePlan(proposals={0: 1, 1: 0}, chapter_starts=(0, 2)))
    asyncio.run(run_speakers_propose(store, agent=agent, model="fixture-model"))

    pack_path = export_review_pack(store, destination=tmp_path / "review.json")
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    participants = {
        participant["display_name"]: participant["participant_id"]
        for participant in pack["participants"]
    }
    pack["reviewer"] = "Michael Bryan"
    if assign:
        names = ["Michael Bryan", "Avalon Mann"]
        pack["decisions"] = [
            {
                "scope": "cluster",
                "kind": "assign",
                "cluster_id": item["item_id"],
                "participant_id": participants[names[index % 2]],
                "rationale": "recognised the voice",
            }
            for index, item in enumerate(pack["items"])
            if item["kind"] == "cluster"
        ]
        pack["remaining"] = "unclear-speaker"
    else:
        # One decision by hand, the rest via `remaining` -- the shape a
        # reviewer actually uses when diarisation left hundreds of turns
        # with no voice evidence.
        pack["decisions"] = [
            {
                "scope": "cluster",
                "kind": "unclear-speaker",
                "cluster_id": pack["items"][0]["item_id"],
                "rationale": "could not tell",
            }
        ]
        pack["remaining"] = "unclear-speaker"
    pack_path.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    run_review_apply(store, pack_path=pack_path)

    asyncio.run(run_chapter_transform(store, agent=agent, model="fixture-model"))
    asyncio.run(run_minutes_transform(store, agent=agent, model="fixture-model"))
    return store


# -- M5: the speaker gate ----------------------------------------------------


def test_rendering_refuses_while_any_turn_is_an_unreviewed_guess(
    tmp_path: Path,
) -> None:
    """The whole review loop exists for this refusal: a meeting note either
    says who spoke because someone decided, or does not claim to know."""
    prepared = prepare_transcribed_bundle(tmp_path / "bundle", recordings=[_RECORDING])
    run_normalise_transform(prepared.store)
    agent = stage_agent(StagePlan(proposals={0: 1, 1: 0}))
    asyncio.run(
        run_speakers_propose(prepared.store, agent=agent, model="fixture-model")
    )

    with pytest.raises(SpeakerGateFailedError, match="unreviewed"):
        render_meeting_note(_head(prepared.store))


def test_an_explicit_unclear_decision_satisfies_the_gate(tmp_path: Path) -> None:
    """D1: a meeting note may ship with visible unresolved counts. What it
    may not do is quietly attribute."""
    store = _reviewed_bundle(tmp_path, assign=False)

    result = render_meeting_note(_head(store))

    assert "Unclear speaker" in result.body
    assert result.coverage.unresolved > 0


# -- M17: the render ---------------------------------------------------------


def test_rendering_the_same_revision_twice_is_byte_identical(
    tmp_path: Path,
) -> None:
    """Corpus §8's determinism check, which is also what makes the apply
    step's "nothing changed" comparison meaningful."""
    store = _reviewed_bundle(tmp_path)

    first = render_document(store)
    second = render_document(store)

    assert first.output_sha256 == second.output_sha256
    assert store.load_render_output(first.render_id) == store.load_render_output(
        second.render_id
    )


def test_the_render_record_carries_its_complete_identity(tmp_path: Path) -> None:
    """M17: profile, profile version, renderer version, template hash, every
    parameter, and the capability closure it consumed."""
    store = _reviewed_bundle(tmp_path)

    record = render_document(store)

    assert record.profile == "meeting-note"
    assert record.profile_version and record.renderer_version
    assert len(record.template_sha256) == 64
    assert record.parameters["timezone"]
    assert "chapters" in record.input_capability_keys
    assert "minutes" in record.input_capability_keys


def test_the_dumc_render_places_summary_before_nested_discussion_notes(
    tmp_path: Path,
) -> None:
    store = _reviewed_bundle(tmp_path)

    body = store.load_render_output(render_document(store).render_id).decode("utf-8")

    assert body.startswith("> [!summary]\n")
    assert body.index("> [!summary]") < body.index("## Discussion Notes")
    assert "- Decisions\n\t- Agreed to proceed." in body
    assert "## Meeting Notes" not in body
    assert "## Chapters" in body
    assert "## Transcript" in body
    assert "Michael Bryan" in body


def test_a_render_without_the_transcript_body_omits_only_that(
    tmp_path: Path,
) -> None:
    store = _reviewed_bundle(tmp_path)

    record = render_document(store, include_transcript=False)
    body = store.load_render_output(record.render_id).decode("utf-8")

    assert "## Chapters" in body
    assert "## Transcript" not in body


# -- M13: the owned region, as pure functions --------------------------------


def test_a_first_write_appends_after_everything_the_author_has() -> None:
    note = "---\ntitle: x\n---\n\n## Prep\n\n- a point\n"

    region = locate_owned_region(note, bundle_id=_BUNDLE_ID)
    composed = compose_note(region, bundle_id=_BUNDLE_ID, body="## Meeting Notes\n")

    assert composed.startswith(note.rstrip())
    assert begin_marker(_BUNDLE_ID) in composed
    assert composed.rstrip().endswith(END_MARKER)


def test_re_composing_an_existing_region_is_stable() -> None:
    """Growing the file by a newline per apply would make re-apply never a
    no-op, which is the difference between idempotent and not."""
    note = "## Prep\n\n- a point\n"
    once = compose_note(
        locate_owned_region(note, bundle_id=_BUNDLE_ID),
        bundle_id=_BUNDLE_ID,
        body="## Meeting Notes\n",
    )

    twice = compose_note(
        locate_owned_region(once, bundle_id=_BUNDLE_ID),
        bundle_id=_BUNDLE_ID,
        body="## Meeting Notes\n",
    )

    assert twice == once


def test_authored_content_below_the_region_survives() -> None:
    """The specific bug M13 names: the legacy merge path truncates
    everything after the first generated heading."""
    note = (
        "## Prep\n\n- a point\n\n"
        f"{begin_marker(_BUNDLE_ID)}\nold generated\n{END_MARKER}\n\n"
        "## Follow-up\n\nauthored, below the region\n"
    )

    composed = compose_note(
        locate_owned_region(note, bundle_id=_BUNDLE_ID),
        bundle_id=_BUNDLE_ID,
        body="new generated",
    )

    assert "authored, below the region" in composed
    assert "old generated" not in composed
    assert composed.startswith("## Prep\n\n- a point\n")


def test_a_region_belonging_to_another_bundle_is_refused() -> None:
    other = "bundle_019fb000-0000-7000-8000-0000000000ff"
    note = f"{begin_marker(other)}\ngenerated\n{END_MARKER}\n"

    with pytest.raises(AmbiguousOwnedRegionError, match="another bundle"):
        locate_owned_region(note, bundle_id=_BUNDLE_ID)


def test_duplicated_markers_are_refused() -> None:
    note = (
        f"{begin_marker(_BUNDLE_ID)}\na\n{END_MARKER}\n"
        f"{begin_marker(_BUNDLE_ID)}\nb\n{END_MARKER}\n"
    )

    with pytest.raises(AmbiguousOwnedRegionError):
        locate_owned_region(note, bundle_id=_BUNDLE_ID)


def test_legacy_generated_headings_are_adopted_as_the_region() -> None:
    note = (
        "## Prep\n\n- a point\n\n"
        "## Meeting Notes\n\n- old\n\n"
        "## Chapters\n\n- 00:00 old\n\n"
        "## Transcript\n\n**A** old\n"
    )

    region = locate_owned_region(note, bundle_id=_BUNDLE_ID)

    assert region.migrated_legacy_headings
    assert region.prefix == "## Prep\n\n- a point\n\n"
    assert region.suffix == ""


def test_legacy_migration_fails_closed_on_authored_content_between_sections() -> None:
    """One marker pair cannot wrap this without absorbing the author's
    work, so it refuses rather than quietly taking it."""
    note = (
        "## Meeting Notes\n\n- old\n\n"
        "## My own thoughts\n\nauthored, in the middle\n\n"
        "## Transcript\n\n**A** old\n"
    )

    with pytest.raises(LegacyMigrationRefusedError, match="authored heading"):
        locate_owned_region(note, bundle_id=_BUNDLE_ID)


def test_legacy_adoption_keeps_authored_content_below_the_last_section() -> None:
    note = (
        "## Meeting Notes\n\n- old\n\n"
        "## Transcript\n\n**A** old\n\n"
        "## Follow-up\n\nauthored, below\n"
    )

    region = locate_owned_region(note, bundle_id=_BUNDLE_ID)

    assert region.suffix.startswith("## Follow-up")


# -- M13: the write itself ---------------------------------------------------


def _bind_destination(store: BundleStore, target: Path) -> None:
    """Point the document's destination at ``target``'s current bytes, the
    way ingest would have if this file had been the note all along."""
    source_id = store.load_manifest().source_memberships[0].source_id
    snapshot = store.ingest_artefact(
        source_id=source_id,
        content=target.read_bytes(),
        kind="obsidian-note",
        producer="test",
        acquisition_locator=str(target),
    )
    destination = store.add_component(
        DestinationComponentBody(
            note_artefact_id=snapshot.artefact_id,
            vault_relative_path=str(target),
            owned_region_state=OwnedRegionState.NONE,
        )
    )
    document = _head(store)
    run = store.create_run(
        next_action=OperationRef(kind="assemble", rationale="bind destination")
    )
    import os

    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    revision = store.append_revision(
        operation=OperationRef(kind="assemble", rationale="bind destination"),
        parent_revision_ids=(document.revision_id,),
        artefact_ids=(snapshot.artefact_id,),
        component_ids=(destination.component_id,),
        superseded_component_ids=tuple(
            existing.component_id
            for existing in document.components_of(type(destination))
        ),
    )
    store.update_head(run_id=run.run_id, revision_id=revision.revision_id)
    store.release_lease(run_id=run.run_id, new_state=RunState.COMPLETED)


def _accept_product_review(
    store: BundleStore, render_id: str, destination: Path
) -> None:
    pack_path = export_product_review_pack(
        store, render_id=render_id, destination=destination
    )
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    pack["reviewer"] = "Michael Bryan"
    pack["transcript_disposition"] = ProductDisposition.ACCEPTED.value
    pack["minutes_disposition"] = ProductDisposition.ACCEPTED.value
    pack_path.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    record_product_review(store, pack_path=pack_path)


def test_apply_refuses_a_render_without_product_acceptance(tmp_path: Path) -> None:
    store = _reviewed_bundle(tmp_path)
    target = tmp_path / "note.md"
    target.write_text("## Prep\n\n- a point\n", encoding="utf-8")
    _bind_destination(store, target)
    record = render_document(store)

    with pytest.raises(TranscriptError, match="product review"):
        apply_render(store, render_id=record.render_id, target_path=target)

    assert target.read_text(encoding="utf-8") == "## Prep\n\n- a point\n"
    assert tuple(store.iter_applies())[-1].state == ApplyState.NOT_WRITTEN


def test_an_accepted_product_review_allows_apply(tmp_path: Path) -> None:
    store = _reviewed_bundle(tmp_path)
    target = tmp_path / "note.md"
    target.write_text("## Prep\n\n- a point\n", encoding="utf-8")
    _bind_destination(store, target)
    render = render_document(store)
    _accept_product_review(store, render.render_id, tmp_path / "product-review.json")

    outcome = apply_render(store, render_id=render.render_id, target_path=target)

    assert outcome.record.state == ApplyState.VERIFIED
    assert "## Transcript" in target.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rendered_body", "altered candidate", "rendered body"),
        ("policy_version", "obsolete-policy", "policy"),
    ],
)
def test_product_review_refuses_tampered_review_evidence(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    store = _reviewed_bundle(tmp_path)
    target = tmp_path / "note.md"
    target.write_text("## Prep\n", encoding="utf-8")
    _bind_destination(store, target)
    render = render_document(store)
    pack_path = export_product_review_pack(
        store, render_id=render.render_id, destination=tmp_path / "review.json"
    )
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    pack["reviewer"] = "Fixture Reviewer"
    pack["transcript_disposition"] = ProductDisposition.ACCEPTED.value
    pack["minutes_disposition"] = ProductDisposition.ACCEPTED.value
    pack[field] = value
    pack_path.write_text(json.dumps(pack), encoding="utf-8")

    with pytest.raises(InvalidProductReviewPackError, match=message):
        record_product_review(store, pack_path=pack_path)


def test_product_review_refuses_overlapping_chapter_spans(tmp_path: Path) -> None:
    store = _reviewed_bundle(tmp_path)
    target = tmp_path / "note.md"
    target.write_text("## Prep\n", encoding="utf-8")
    _bind_destination(store, target)
    document = project_head(store)
    assert isinstance(document, TranscriptDocumentV1)
    chapters = document.components_of(ChapterSetComponent)[0]
    first, second, *remaining = chapters.chapters
    overlapping = store.add_component(
        ChapterSetComponentBody(
            chapters=(
                first.model_copy(update={"end_ms": second.start_ms + 1}),
                second,
                *remaining,
            )
        )
    )
    run = store.create_run(next_action=OperationRef(kind="chapter-fixture"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    revision = store.append_revision(
        operation=OperationRef(kind="chapter-fixture"),
        parent_revision_ids=(document.revision_id,),
        component_ids=(overlapping.component_id,),
        superseded_component_ids=(chapters.component_id,),
    )
    store.update_head(run_id=run.run_id, revision_id=revision.revision_id)
    store.release_lease(run_id=run.run_id, new_state=RunState.COMPLETED)
    render = render_document(store)

    with pytest.raises(InvalidProductReviewPackError, match="overlaps chapter"):
        export_product_review_pack(
            store, render_id=render.render_id, destination=tmp_path / "review.json"
        )


def test_applying_writes_the_region_and_preserves_the_rest(
    tmp_path: Path,
) -> None:
    store = _reviewed_bundle(tmp_path)
    target = tmp_path / "note.md"
    target.write_text(
        "---\ntitle: Tasking\n---\n\n## Prep\n\n- a point\n\n"
        "## Follow-up\n\nauthored, below\n",
        encoding="utf-8",
    )
    _bind_destination(store, target)
    record = render_document(store)
    _accept_product_review(store, record.render_id, tmp_path / "product-review.json")

    outcome = apply_render(store, render_id=record.render_id, target_path=target)

    written = target.read_text(encoding="utf-8")
    assert outcome.record.state == ApplyState.VERIFIED
    assert written.startswith("---\ntitle: Tasking\n---\n\n## Prep\n\n- a point\n")
    assert "authored, below" in written
    assert "## Discussion Notes" in written
    assert "## Meeting Notes" not in written


def test_re_applying_the_same_render_writes_nothing(tmp_path: Path) -> None:
    """Corpus §9 step 7. A second apply must not grow the file, reorder it,
    or report a write that did not happen."""
    store = _reviewed_bundle(tmp_path)
    target = tmp_path / "note.md"
    target.write_text("## Prep\n\n- a point\n", encoding="utf-8")
    _bind_destination(store, target)
    record = render_document(store)
    _accept_product_review(store, record.render_id, tmp_path / "product-review.json")
    apply_render(store, render_id=record.render_id, target_path=target)
    after_first = target.read_text(encoding="utf-8")

    outcome = apply_render(store, render_id=record.render_id, target_path=target)

    assert outcome.unchanged
    assert not outcome.written
    assert target.read_text(encoding="utf-8") == after_first


def test_a_target_edited_since_the_render_is_refused(tmp_path: Path) -> None:
    """Corpus §9 step 4. Applying anyway would silently discard whatever
    the author changed in the meantime."""
    store = _reviewed_bundle(tmp_path)
    target = tmp_path / "note.md"
    target.write_text("## Prep\n\n- a point\n", encoding="utf-8")
    _bind_destination(store, target)
    record = render_document(store)
    _accept_product_review(store, record.render_id, tmp_path / "product-review.json")
    target.write_text("## Prep\n\n- a point\n\n- and another\n", encoding="utf-8")

    with pytest.raises(StaleTargetError):
        apply_render(store, render_id=record.render_id, target_path=target)

    assert "and another" in target.read_text(encoding="utf-8")


def test_a_refusal_is_still_recorded_as_an_apply(tmp_path: Path) -> None:
    """An apply that did not happen is evidence too."""
    store = _reviewed_bundle(tmp_path)
    target = tmp_path / "note.md"
    target.write_text("## Prep\n", encoding="utf-8")
    _bind_destination(store, target)
    record = render_document(store)
    _accept_product_review(store, record.render_id, tmp_path / "product-review.json")
    target.write_text("## Prep\n\nedited\n", encoding="utf-8")

    with pytest.raises(StaleTargetError):
        apply_render(store, render_id=record.render_id, target_path=target)

    refusals = [
        applied
        for applied in store.iter_applies()
        if applied.state == ApplyState.NOT_WRITTEN
    ]
    assert refusals and refusals[-1].detail


def test_a_later_accepted_render_can_replace_this_bundles_owned_region(
    tmp_path: Path,
) -> None:
    store = _reviewed_bundle(tmp_path)
    target = tmp_path / "note.md"
    target.write_text("## Prep\n\n- authored\n", encoding="utf-8")
    _bind_destination(store, target)
    first = render_document(store)
    _accept_product_review(store, first.render_id, tmp_path / "first-review.json")
    apply_render(store, render_id=first.render_id, target_path=target)

    asyncio.run(
        run_chapter_transform(
            store,
            agent=stage_agent(StagePlan(chapter_starts=(0, 1))),
            model="fixture-model",
        )
    )
    second = render_document(store)
    _accept_product_review(store, second.render_id, tmp_path / "second-review.json")

    outcome = apply_render(store, render_id=second.render_id, target_path=target)

    assert outcome.record.state == ApplyState.VERIFIED
    assert target.read_text(encoding="utf-8").startswith("## Prep\n\n- authored\n")


def test_a_render_of_a_superseded_revision_needs_an_explicit_override(
    tmp_path: Path,
) -> None:
    """M13 §12.22: both freshness gates. The override exists, and taking it
    is recorded."""
    store = _reviewed_bundle(tmp_path)
    target = tmp_path / "note.md"
    target.write_text("## Prep\n", encoding="utf-8")
    _bind_destination(store, target)
    record = render_document(store)
    _accept_product_review(store, record.render_id, tmp_path / "product-review.json")
    asyncio.run(
        run_chapter_transform(
            store,
            agent=stage_agent(StagePlan(chapter_starts=(0, 1))),
            model="fixture-model",
        )
    )

    with pytest.raises(StaleRenderError):
        apply_render(store, render_id=record.render_id, target_path=target)

    outcome = apply_render(
        store,
        render_id=record.render_id,
        target_path=target,
        allow_stale_render=True,
    )
    assert outcome.record.allow_stale_render
    assert outcome.record.state == ApplyState.VERIFIED
