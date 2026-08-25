from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from jake_tools.cli import main
from jake_tools.transcription.asr import build_audio_stage_manifest
from jake_tools.transcription.cache import RunCache, StageManifest, stable_hash
from jake_tools.transcription.chapters import ChapterList
from jake_tools.transcription.integrate import run_integrate
from jake_tools.transcription.minutes import (
    MINUTES_CACHE_NAME,
    MinutesResult,
    MinutesReviewResult,
)
from jake_tools.transcription.models import (
    ChapterSpan,
    PolishedChapter,
    PolishedTurn,
    RawTranscript,
    SourceClip,
    TranscriptProducts,
    Utterance,
)
from jake_tools.transcription.note import human_owned_note_context, parse_note
from jake_tools.transcription.polish import POLISHED_CACHE_NAME, PolishedChapterList
from jake_tools.transcription.product_review import (
    ProductReviewChecks,
    ProductReviewDecision,
    ProductReviewExportError,
    ProductReviewRefusalError,
    ReviewBinding,
    _temporal_geometry_is_valid,
    _validate_raw_stage,
    apply_accepted_product,
    decide_product_review,
    export_product_review,
    review_id_for,
)


def _binding(**updates: object) -> ReviewBinding:
    values: dict[str, object] = {
        "run_id": "run-1",
        "note_path": "/tmp/note.md",
        "source_sha256": "a" * 64,
        "revision_sha256": "b" * 64,
        "review_policy_version": "product-review-v1",
        "policy_sha256": "c" * 64,
        "transcript_sha256": "d" * 64,
        "minutes_sha256": "e" * 64,
        "note_base_sha256": "f" * 64,
        "render_sha256": "1" * 64,
        "pipeline_revision": "revision-1",
        "source_kind": "text",
        "manifest_sha256s": {"polish": "2" * 64, "minutes": "3" * 64},
    }
    values.update(updates)
    return ReviewBinding.model_validate(values)


def test_review_binding_round_trips_as_a_strict_versioned_record() -> None:
    binding = _binding()
    restored = ReviewBinding.model_validate_json(binding.model_dump_json())
    assert restored == binding
    with pytest.raises(ValueError):
        ReviewBinding.model_validate({**binding.model_dump(), "unexpected": True})


def test_review_id_uses_complete_binding_not_run_id_prefix() -> None:
    first = _binding()
    second = _binding(render_sha256="9" * 64)
    assert review_id_for(first) == review_id_for(first)
    assert review_id_for(first) != review_id_for(second)
    assert review_id_for(first) != stable_hash({"run_id": first.run_id})


def test_rejected_decision_requires_a_non_empty_reason() -> None:
    with pytest.raises(ValueError):
        ProductReviewDecision(
            review_id=review_id_for(_binding()),
            binding=_binding(),
            decision="reject",
            reviewer="Michael",
            reason="   ",
            decided_at=datetime.now(UTC),
        )


def test_product_checks_separate_hard_failures_from_semantic_warnings() -> None:
    checks = ProductReviewChecks(
        source_provenance=True,
        polished_provenance=True,
        no_cross_speaker_provenance=True,
        unknown_counts_carried=True,
        chapters_partition=True,
        temporal_geometry=True,
        protected_content=True,
        candidate_shape=True,
        stale_state=True,
        manifests_valid=True,
        semantic_warnings=["Confirm the proper noun against audio."],
    )
    assert checks.passed
    assert checks.semantic_warnings


def test_text_source_temporal_geometry_does_not_claim_unknown_duration() -> None:
    raw = RawTranscript(
        clips=[],
        utterances=[
            Utterance(start=0.0, end=0.0, speaker="Ada", text="first"),
            Utterance(start=0.0, end=0.0, speaker="Ada", text="second"),
        ],
        source_text_sha256="a" * 64,
    )
    spans = [
        ChapterSpan(
            title="Opening", start_utterance=0, end_utterance=0, start_seconds=0.0
        ),
        ChapterSpan(
            title="Closing", start_utterance=1, end_utterance=1, start_seconds=0.0
        ),
    ]

    assert _temporal_geometry_is_valid(raw, spans)


def test_audio_source_temporal_geometry_still_requires_minimum_chapter_duration() -> (
    None
):
    raw = RawTranscript(
        clips=[],
        utterances=[
            Utterance(start=0.0, end=1.0, speaker="Ada", text="first"),
            Utterance(start=2.0, end=3.0, speaker="Ada", text="second"),
        ],
        audio_sha256="a" * 64,
    )
    spans = [
        ChapterSpan(
            title="Opening", start_utterance=0, end_utterance=0, start_seconds=0.0
        ),
        ChapterSpan(
            title="Closing", start_utterance=1, end_utterance=1, start_seconds=2.0
        ),
    ]

    assert not _temporal_geometry_is_valid(raw, spans)


def test_text_raw_review_rejects_manifest_with_wrong_stage_even_when_output_matches(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path / "cache")
    raw = RawTranscript(
        clips=[],
        utterances=[Utterance(start=0.0, end=1.0, speaker="Ada", text="hello")],
        source_text_sha256="4" * 64,
    )
    cache.store("run-1", "raw_transcript", raw)
    cache.store_manifest(
        "run-1",
        StageManifest(
            stage="asr",
            input_hash="1" * 64,
            config_hash="2" * 64,
            input_hashes={"source_text": "3" * 64},
            output_hash=stable_hash(raw.model_dump(mode="json")),
        ),
    )

    assert not _validate_raw_stage(cache, "run-1", raw)


def test_text_raw_review_rejects_manifest_with_mismatched_source_hash(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path / "cache")
    raw = RawTranscript(
        clips=[],
        utterances=[Utterance(start=0.0, end=1.0, speaker="Ada", text="hello")],
        source_text_sha256="4" * 64,
    )
    cache.store("run-1", "raw_transcript", raw)
    cache.store_manifest(
        "run-1",
        StageManifest(
            stage="adapt",
            input_hash="1" * 64,
            config_hash="2" * 64,
            input_hashes={"source_text": "not-a-source-hash"},
            output_hash=stable_hash(raw.model_dump(mode="json")),
        ),
    )

    assert not _validate_raw_stage(cache, "run-1", raw)


def test_audio_raw_review_rejects_manifest_with_mismatched_source_hash(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.m4a"
    source.write_bytes(b"audio")
    cache = RunCache(tmp_path / "cache")
    raw = RawTranscript(
        clips=[SourceClip(path=str(source), offset_seconds=0.0, duration_seconds=1.0)],
        utterances=[],
        audio_sha256="a" * 64,
        asr_model="asr",
        diarisation_model="diarisation",
        diarisation_device="cpu",
        num_speakers=2,
        asr_chunk_duration=120.0,
        asr_chunk_overlap=15.0,
    )
    cache.store("run-1", "raw_transcript", raw)
    cache.store_manifest(
        "run-1",
        build_audio_stage_manifest(raw).model_copy(
            update={"input_hashes": {"audio_sha256": "b" * 64}}
        ),
    )

    assert not _validate_raw_stage(cache, "run-1", raw)


def test_nested_product_review_cli_is_reachable() -> None:
    result = CliRunner().invoke(main, ["transcript", "review", "product", "--help"])
    assert result.exit_code == 0, result.output
    assert "export" in result.output
    assert "decide" in result.output


def test_export_is_read_only_when_product_prerequisites_are_missing(
    tmp_path: Path,
) -> None:
    note = tmp_path / "note.md"
    note.write_text("---\ntags: [note/meeting]\n---\n\n## Meeting Prep\n\nHuman.\n")
    cache = RunCache(tmp_path / "cache")
    before = note.read_bytes()
    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "review",
            "product",
            "export",
            str(note),
            "--run-id",
            "run-1",
            "--cache-root",
            str(cache.root),
        ],
    )
    assert result.exit_code != 0
    assert note.read_bytes() == before
    assert not (cache.root / "run-1" / "product_review.json").exists()


def test_integrate_refuses_before_baseline_mutation_without_an_accepted_review(
    tmp_path: Path,
) -> None:
    note = tmp_path / "note.md"
    note.write_text("---\ntags: [note/meeting]\n---\n\n## Meeting Prep\n")
    cache = RunCache(tmp_path / "cache")
    before = note.read_bytes()
    with pytest.raises(ProductReviewRefusalError):
        run_integrate(
            note,
            TranscriptProducts(
                context="meeting",
                meeting_summary="summary",
                discussion_notes="- topic",
                chapters=[],
            ),
            cache=cache,
            run_id="run-1",
        )
    assert note.read_bytes() == before
    assert not (cache.run_dir("run-1") / "baseline_summary.txt").exists()


def test_export_and_apply_use_the_same_exact_candidate_bytes(tmp_path: Path) -> None:
    note = tmp_path / "note.md"
    note.write_text(
        "---\ntags: [note/meeting]\n---\n\n## Meeting Prep\n\n![[source.txt]]\n"
    )
    source = tmp_path / "source.m4a"
    source.write_bytes(b"source")
    cache = RunCache(tmp_path / "cache")
    run_id = "run-exact"
    raw = RawTranscript(
        clips=[SourceClip(path=str(source), offset_seconds=0.0, duration_seconds=70.0)],
        utterances=[
            Utterance(start=0.0, end=10.0, speaker="Ada", text="first"),
            Utterance(start=60.0, end=70.0, speaker="Ada", text="second"),
        ],
        audio_sha256="a" * 64,
        asr_model="asr-model",
        diarisation_model="diar-model",
        diarisation_device="cpu",
        num_speakers=2,
        asr_chunk_duration=120.0,
        asr_chunk_overlap=15.0,
    )
    chapters = ChapterList(
        chapters=[
            ChapterSpan(
                title="Opening", start_utterance=0, end_utterance=1, start_seconds=0.0
            )
        ]
    )
    polished = PolishedChapterList(
        chapters=[
            PolishedChapter(
                title="Opening",
                start_seconds=0.0,
                summary="Summary",
                turns=[
                    PolishedTurn(speaker="Ada", text="first", source_turn_indices=[0]),
                    PolishedTurn(speaker="Ada", text="second", source_turn_indices=[1]),
                ],
            )
        ]
    )
    minutes = MinutesResult(
        run_id=run_id,
        meeting_summary="Summary",
        discussion_notes="- Topic\n  - Detail\n",
    )
    review = MinutesReviewResult(
        run_id=run_id,
        input_hash="input",
        draft_sha256="draft",
        lexicon_sha256="lexicon",
        polished_sha256=stable_hash(polished.model_dump(mode="json")),
        meeting_summary=minutes.meeting_summary,
        discussion_notes=minutes.discussion_notes,
    )
    note_model = parse_note(note)
    human_hash = stable_hash(human_owned_note_context(note_model))
    cache.store(run_id, "resolved_transcript", raw)
    cache.store(run_id, "raw_transcript", raw)
    cache.store(run_id, "chapters", chapters)
    cache.store(run_id, POLISHED_CACHE_NAME, polished)
    cache.store(run_id, MINUTES_CACHE_NAME, minutes)
    cache.store(run_id, "minutes_review", review)
    cache.store_manifest(
        run_id,
        StageManifest(
            stage="chapterise",
            input_hash="chapter-input",
            config_hash="chapter-config",
            output_hash=stable_hash(chapters.model_dump(mode="json")),
        ),
    )
    cache.store_manifest(
        run_id,
        StageManifest(
            stage="polish",
            input_hash="input",
            config_hash="config",
            input_hashes={
                "resolved_transcript": stable_hash(raw.model_dump(mode="json")),
                "chapters": stable_hash(chapters.model_dump(mode="json")),
                "human_context": human_hash,
                "lexicon": "lexicon",
            },
            output_hash=stable_hash(polished.model_dump(mode="json")),
        ),
    )
    cache.store_manifest(
        run_id,
        StageManifest(
            stage="minutes",
            input_hash="input",
            config_hash="config",
            input_hashes={
                "polished": stable_hash(polished.model_dump(mode="json")),
                "human_context": human_hash,
                "lexicon": "lexicon",
            },
            output_hash=stable_hash(minutes.model_dump(mode="json")),
        ),
    )
    cache.store_manifest(
        run_id,
        StageManifest(
            stage="minutes-review",
            input_hash="review-input",
            config_hash="review-config",
            input_hashes={
                "polished": stable_hash(polished.model_dump(mode="json")),
                "lexicon": "lexicon",
            },
            output_hash=stable_hash(review.model_dump(mode="json")),
        ),
    )

    before = note.read_bytes()
    with pytest.raises(ProductReviewExportError, match="local ASR"):
        export_product_review(note, run_id, cache=cache)

    local_config = {
        "asr_model": raw.asr_model,
        "asr_chunk_duration": raw.asr_chunk_duration,
        "asr_chunk_overlap": raw.asr_chunk_overlap,
        "diarisation_model": raw.diarisation_model,
        "diarisation_device": raw.diarisation_device,
        "num_speakers": raw.num_speakers,
    }
    cache.store_manifest(
        run_id,
        StageManifest(
            stage="asr",
            input_hash=stable_hash({"audio_sha256": raw.audio_sha256, **local_config}),
            config_hash=stable_hash(local_config),
            input_hashes={
                "audio_sha256": raw.audio_sha256 or "",
                "local_config": stable_hash(local_config),
            },
            output_hash=stable_hash(raw.model_dump(mode="json")),
        ),
    )
    cache.store(
        run_id,
        "raw_transcript",
        raw.model_copy(update={"asr_model": "manually-replaced"}),
    )
    with pytest.raises(ProductReviewExportError, match="local ASR"):
        export_product_review(note, run_id, cache=cache)
    cache.store(run_id, "raw_transcript", raw)

    pack = export_product_review(note, run_id, cache=cache)
    assert pack.checks.passed
    assert note.read_bytes() == before
    assert not (cache.run_dir(run_id) / "baseline_summary.txt").exists()
    candidate = Path(pack.candidate_path).read_bytes()
    decide_product_review(
        note,
        run_id,
        review_id=pack.review_id,
        decision="accept",
        reviewer="Michael",
        reason=None,
        cache=cache,
    )
    products = TranscriptProducts(
        context="meeting",
        meeting_summary=minutes.meeting_summary,
        discussion_notes=minutes.discussion_notes,
        chapters=polished.chapters,
    )
    apply_accepted_product(note, products, run_id=run_id, cache=cache)
    assert note.read_bytes() == candidate
