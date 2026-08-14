from __future__ import annotations

from pathlib import Path

import pytest

from jake_tools.transcripts.bundle.ids import mint_id
from jake_tools.transcripts.bundle.records import (
    ComponentBinding,
    ContinuationGraph,
    ContinuationStage,
    StageProgressState,
)
from jake_tools.transcripts.bundle.store import (
    BundleStore,
    ContinuationBindingConflictError,
    StageProgressConflictError,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


def _store(tmp_path: Path) -> BundleStore:
    store = BundleStore(tmp_path / "bundle")
    store.create_bundle()
    return store


def _graph() -> ContinuationGraph:
    return ContinuationGraph(
        schema_version="post-review-continuation-v1",
        stages=(
            ContinuationStage(
                ordinal=0,
                kind="editorial-identity",
                implementation_version="v1",
                config_hash=_SHA_A,
            ),
            ContinuationStage(
                ordinal=1,
                kind="editorial-reflow",
                implementation_version="v1",
                config_hash=_SHA_B,
            ),
        ),
        terminal_checkpoint="product_review_required",
    )


def _activation(store: BundleStore):
    return store.create_continuation_activation(
        originating_run_id=mint_id("run"),
        review_id=mint_id("review"),
        reviewed_revision_id=mint_id("rev"),
        canonical_turns=ComponentBinding(
            component_id=mint_id("component"), content_hash=_SHA_A
        ),
        speaker_review=ComponentBinding(
            component_id=mint_id("component"), content_hash=_SHA_B
        ),
        graph=_graph(),
    )


def test_graph_hash_binds_ordered_stage_identity_and_configuration() -> None:
    graph = _graph()
    reordered = graph.model_copy(update={"stages": tuple(reversed(graph.stages))})
    changed_config = graph.model_copy(
        update={
            "stages": (
                graph.stages[0].model_copy(update={"config_hash": _SHA_C}),
                graph.stages[1],
            )
        }
    )

    assert graph.content_hash == graph.model_copy().content_hash
    assert reordered.content_hash != graph.content_hash
    assert changed_config.content_hash != graph.content_hash


def test_graph_requires_exact_consecutive_ordinals() -> None:
    with pytest.raises(ValueError, match="consecutive ordinals"):
        ContinuationGraph(
            schema_version="post-review-continuation-v1",
            stages=(
                ContinuationStage(
                    ordinal=1,
                    kind="editorial-identity",
                    implementation_version="v1",
                    config_hash=_SHA_A,
                ),
            ),
            terminal_checkpoint="product_review_required",
        )


def test_activation_is_idempotent_only_for_the_same_bound_inputs(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    run_id = mint_id("run")
    review_id = mint_id("review")
    revision_id = mint_id("rev")
    canonical = ComponentBinding(component_id=mint_id("component"), content_hash=_SHA_A)
    speaker_review = ComponentBinding(
        component_id=mint_id("component"), content_hash=_SHA_B
    )
    graph = _graph()

    first = store.create_continuation_activation(
        originating_run_id=run_id,
        review_id=review_id,
        reviewed_revision_id=revision_id,
        canonical_turns=canonical,
        speaker_review=speaker_review,
        graph=graph,
    )
    repeated = store.create_continuation_activation(
        originating_run_id=run_id,
        review_id=review_id,
        reviewed_revision_id=revision_id,
        canonical_turns=canonical,
        speaker_review=speaker_review,
        graph=graph,
    )

    assert repeated == first
    assert store.load_continuation_activation(first.continuation_id) == first

    with pytest.raises(ContinuationBindingConflictError, match="already activates"):
        store.create_continuation_activation(
            originating_run_id=run_id,
            review_id=review_id,
            reviewed_revision_id=revision_id,
            canonical_turns=canonical.model_copy(update={"content_hash": _SHA_C}),
            speaker_review=speaker_review,
            graph=graph,
        )


def test_stage_progress_is_append_only_and_idempotent_for_an_exact_replay(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    activation = _activation(store)
    output = ComponentBinding(component_id=mint_id("component"), content_hash=_SHA_C)

    first = store.append_stage_progress(
        continuation_id=activation.continuation_id,
        stage_ordinal=0,
        state=StageProgressState.COMPLETED,
        input_revision_id=activation.reviewed_revision_id,
        output_revision_id=mint_id("rev"),
        input_components=(activation.canonical_turns, activation.speaker_review),
        output_components=(output,),
        operation_count=0,
    )
    repeated = store.append_stage_progress(
        continuation_id=activation.continuation_id,
        stage_ordinal=0,
        state=StageProgressState.COMPLETED,
        input_revision_id=activation.reviewed_revision_id,
        output_revision_id=first.output_revision_id,
        input_components=(activation.canonical_turns, activation.speaker_review),
        output_components=(output,),
        operation_count=0,
    )

    assert repeated == first
    assert store.load_stage_progress(activation.continuation_id, 0) == first

    with pytest.raises(StageProgressConflictError, match="stage ordinal 0"):
        store.append_stage_progress(
            continuation_id=activation.continuation_id,
            stage_ordinal=0,
            state=StageProgressState.COMPLETED,
            input_revision_id=activation.reviewed_revision_id,
            output_revision_id=first.output_revision_id,
            input_components=(activation.canonical_turns, activation.speaker_review),
            output_components=(output.model_copy(update={"content_hash": _SHA_A}),),
            operation_count=1,
        )


def test_stage_progress_must_match_the_declared_graph_stage(tmp_path: Path) -> None:
    store = _store(tmp_path)
    activation = _activation(store)

    with pytest.raises(StageProgressConflictError, match="does not declare ordinal 2"):
        store.append_stage_progress(
            continuation_id=activation.continuation_id,
            stage_ordinal=2,
            state=StageProgressState.COMPLETED,
            input_revision_id=activation.reviewed_revision_id,
            output_revision_id=None,
            input_components=(activation.canonical_turns,),
            output_components=(activation.canonical_turns,),
            operation_count=0,
        )
