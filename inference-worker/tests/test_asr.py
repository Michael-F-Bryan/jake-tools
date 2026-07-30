"""ASR stage behaviour that doesn't require downloading model weights.

Real `run_asr` call throughout — a near-zero timeout reliably interrupts
it before it gets anywhere near real inference, which is what makes this
fast enough for the default (non-model) suite while still exercising the
real timeout/provenance code path rather than a mock of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from inference_worker.asr import (
    MODEL_ID,
    _choose_cut_point,
    _ChunkDecodeOrderError,
    _ChunkTranscript,
    _cut_point_merge,
    _raw_tokens_from_aligned,
    _RawToken,
    _to_asr_tokens,
    asr_stage_config_hash,
    run_asr,
)
from inference_worker.provenance import (
    best_effort_package_version,
    config_hash,
    local_model_revision,
)


def test_asr_stage_config_hash_folds_in_the_resolved_model_revision():
    """M2: two runs against different cached model revisions must hash
    differently — config_hash must not cover only the compile-time
    MODEL_ID constant. Compares against a hash built from MODEL_ID alone
    to prove the revision (and package versions) are actually mixed in,
    without needing to fake a second real revision on disk."""
    hash_with_revision = asr_stage_config_hash()
    hash_without_revision = config_hash({"model_id": MODEL_ID})

    assert hash_with_revision != hash_without_revision


def test_asr_stage_config_hash_survives_a_missing_package(monkeypatch):
    """N3: asr_stage_config_hash() runs before run_asr's own try block
    even starts — on a platform where a package is genuinely absent (e.g.
    mlx on non-Apple-Silicon Linux), a strict package_version() call here
    would raise PackageNotFoundError and degrade a per-stage failure into
    a whole-run worker-internal-error. Simulates that by making the real
    importlib.metadata.version lookup fail, the actual boundary a missing
    package would hit — not a patch of this module's own code."""
    import importlib.metadata

    def _always_raise(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _always_raise)

    result = asr_stage_config_hash()

    assert len(result) == 64  # a real sha256 hex digest — it didn't raise


def test_asr_stage_config_hash_folds_in_chunk_and_overlap_duration():
    """Chunking finding, point 1: chunk_duration/overlap_duration change
    what actually gets transcribed at chunk boundaries — they're config
    that affects output, so M2 discipline says they must be hashed, not
    silently baked-in constants nobody can see from the hash alone."""
    hash_with_chunk_params = asr_stage_config_hash()
    hash_without_chunk_params = config_hash(
        {
            "model_id": MODEL_ID,
            "model_revision": local_model_revision(MODEL_ID) or "unresolved",
            "parakeet-mlx": best_effort_package_version("parakeet-mlx"),
            "mlx": best_effort_package_version("mlx"),
        }
    )

    assert hash_with_chunk_params != hash_without_chunk_params


def test_run_asr_reports_timeout_and_still_carries_model_provenance():
    result = run_asr(Path("does-not-need-to-exist.wav"), "a" * 64, timeout_s=0.0001)

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.error_class == "timeout"
    assert result.error.retryable is True
    # Public repo metadata resolves without needing the model weights, so
    # provenance should be available even though inference never ran.
    assert result.model_provenance is not None
    assert result.model_provenance.identity.name == "mlx-community/parakeet-tdt-0.6b-v2"


def test_run_asr_populates_input_audio_sha256_on_failure():
    """Chunking finding, point 2: the wav is already hashed (by
    prepare.py) before run_asr is ever called — a failure here is no
    reason to lose that provenance and report input_audio_sha256: null."""
    result = run_asr(Path("does-not-need-to-exist.wav"), "a" * 64, timeout_s=0.0001)

    assert result.status == "failed"
    assert result.input_audio_sha256 == "a" * 64


def test_run_asr_retains_the_wav_on_failure():
    """M5: retained_artefacts is the flagship partial-failure record — a
    failed ASR stage must name the prepared wav that still survives it,
    not leave the field dead/empty. N5: as a bare filename relative to
    the out-dir, consistent with ArtefactRef's filenames — not the
    absolute path the stage was actually called with."""
    wav_path = Path("some/out-dir/prepared.wav")

    result = run_asr(wav_path, "a" * 64, timeout_s=0.0001)

    assert result.error is not None
    assert result.error.retained_artefacts == ["prepared.wav"]


# --- within-chunk decode order (a single chunk's own tokens) --------------


@dataclass
class _FakeAlignedToken:
    text: str
    start: float
    end: float
    confidence: float = 0.9


def test_raw_tokens_from_aligned_offsets_by_chunk_start():
    raw = _raw_tokens_from_aligned(
        [_FakeAlignedToken(text="hi", start=1.0, end=1.5)], offset_ms=120_000
    )

    assert raw == [_RawToken(text="hi", start_ms=121000, end_ms=121500, confidence=0.9)]


def test_raw_tokens_from_aligned_preserves_zero_length_tokens():
    raw = _raw_tokens_from_aligned(
        [_FakeAlignedToken(text="", start=1.234, end=1.234)], offset_ms=0
    )

    assert raw[0].start_ms == raw[0].end_ms == 1234


def test_raw_tokens_from_aligned_raises_on_within_chunk_disorder():
    """A model/library bug distinct from any merge concern: this chunk's
    own decode is internally out of order, before the cross-chunk merge
    ever runs — gets its own error class (asr-chunk-decode-unordered)."""
    tokens = [
        _FakeAlignedToken(text="a", start=5.0, end=5.5),
        _FakeAlignedToken(text="b", start=1.0, end=1.5),
    ]

    with pytest.raises(_ChunkDecodeOrderError):
        _raw_tokens_from_aligned(tokens, offset_ms=0)


# --- M6 span validation on the final typed conversion ----------------------


def test_to_asr_tokens_preserves_a_zero_length_token():
    """Parakeet's duration head structurally emits some zero-length
    tokens (observed 1 in 82 on a real run) — must be preserved as
    evidence, not dropped or rejected."""
    tokens = _to_asr_tokens(
        [_RawToken(text="", start_ms=1234, end_ms=1234, confidence=0.9)]
    )

    assert tokens[0].start_ms == tokens[0].end_ms == 1234


def test_to_asr_tokens_raises_on_a_reversed_span():
    """A token with end before start is a modelling error, not evidence
    worth keeping — run_asr converts this into a clean asr-invalid-output
    stage failure rather than letting it propagate raw."""
    with pytest.raises(ValidationError):
        _to_asr_tokens(
            [_RawToken(text="x", start_ms=2000, end_ms=1000, confidence=0.9)]
        )


# --- cut-point merge: replaces per-token selection entirely ----------------
#
# parakeet-mlx's own transcribe(chunk_duration=..., overlap_duration=...)
# does chunk splitting/decoding AND its own per-token overlap merge
# (merge_longest_contiguous / merge_longest_common_subsequence). Verified
# on a real 28-minute recording, that per-token merge produced first an
# exact duplicate token at a chunk seam, then (after a dedup-only fix) a
# genuine reordering — per-token selection over two independently-decoded
# streams cannot guarantee global ordering. This worker instead owns its
# own chunking and merges by CUT POINT: choose one boundary per adjacent
# pair, take strictly-before-it tokens from the left chunk and
# at-or-after-it tokens from the right chunk. These are real behaviour
# tests of _cut_point_merge/_choose_cut_point against synthetic chunk
# transcripts shaped like what a chunk boundary would produce — no model
# involved. See tests/test_asr_merge_property.py for the randomized,
# many-seeds property test.


def _token(text: str, start_ms: int, end_ms: int | None = None) -> _RawToken:
    return _RawToken(
        text=text,
        start_ms=start_ms,
        end_ms=end_ms if end_ms is not None else start_ms,
        confidence=0.9,
    )


def test_choose_cut_point_prefers_a_real_silence_gap_over_the_plain_midpoint():
    left = _ChunkTranscript(start_ms=0, end_ms=1000, tokens=[_token("a", 860, 870)])
    right = _ChunkTranscript(start_ms=850, end_ms=1850, tokens=[_token("b", 950, 970)])
    # Both start_ms fall inside the overlap [850, 1000); the only gap in
    # the union is a.end(870) -> b.start(950).

    cut = _choose_cut_point(left, right)

    assert cut == round(
        (870 + 950) / 2
    )  # the gap's midpoint (910), not the plain overlap midpoint (925)


def test_choose_cut_point_falls_back_to_midpoint_with_no_tokens_in_overlap():
    left = _ChunkTranscript(start_ms=0, end_ms=1000, tokens=[_token("a", 100, 150)])
    right = _ChunkTranscript(
        start_ms=850, end_ms=1850, tokens=[_token("z", 1500, 1550)]
    )

    cut = _choose_cut_point(left, right)

    assert cut == round(
        (850 + 1000) / 2
    )  # plain overlap midpoint: (right.start + left.end) / 2


def test_choose_cut_point_falls_back_to_midpoint_with_a_single_token_in_overlap():
    left = _ChunkTranscript(start_ms=0, end_ms=1000, tokens=[_token("a", 900, 950)])
    right = _ChunkTranscript(start_ms=850, end_ms=1850, tokens=[])

    cut = _choose_cut_point(left, right)  # one token: no inter-token gap possible

    assert cut == round((850 + 1000) / 2)


def test_cut_point_merge_single_chunk_is_a_no_op():
    chunk = _ChunkTranscript(
        start_ms=0, end_ms=1000, tokens=[_token("a", 0, 50), _token("b", 500, 550)]
    )

    merged, cut_points = _cut_point_merge([chunk])

    assert merged == chunk.tokens
    assert cut_points == []


def test_cut_point_merge_excludes_one_side_of_a_genuine_cross_chunk_duplicate():
    """A hypothetical (not what R1 rerun #1 turned out to be — see
    test_cut_point_merge_preserves_within_chunk_duplicates_as_evidence
    for the confirmed real cause) but still real structural property:
    IF two adjacent chunks each independently decoded an identical token
    sitting in their shared overlap, cut-point merging can't keep both —
    whichever side of the chosen cut point it falls on, only that
    chunk's copy is ever taken. (Whether that's still called a
    "duplicate" doesn't matter after M6/M7 — duplicates are legal raw
    evidence either way — the property worth pinning down is that
    cross-chunk ownership stays exclusive.)"""
    duplicate = _token("0", 911_920, 911_920)
    left = _ChunkTranscript(
        start_ms=800_000,
        end_ms=920_000,
        tokens=[_token("earlier", 900_000, 900_500), duplicate],
    )
    right = _ChunkTranscript(
        start_ms=905_000,
        end_ms=1_025_000,
        tokens=[duplicate, _token("later", 915_000, 915_500)],
    )

    merged, cut_points = _cut_point_merge([left, right])

    zero_length_occurrences = [
        t for t in merged if t.text == "0" and t.start_ms == t.end_ms == 911_920
    ]
    assert len(zero_length_occurrences) == 1
    assert len(cut_points) == 1


def test_cut_point_merge_never_interleaves_streams_the_r1_rerun2_case():
    """The exact failure shape from the real R1 canary rerun #2:
    parakeet-mlx's per-token "closest window centre" merge picked a
    token starting at 1064920ms from one chunk AFTER a token starting at
    1070640ms from the other — a genuine reordering, not a duplicate.
    Cut-point merging can't interleave: every token in the merged result
    comes from exactly one contiguous side of one boundary."""
    # Left chunk runs later in the overlap than right's early tokens —
    # exactly the shape that trips up a per-token nearest-centre choice.
    left = _ChunkTranscript(
        start_ms=950_000,
        end_ms=1_070_000,
        tokens=[_token("x", 1_060_000, 1_060_200), _token("y", 1_064_920, 1_065_100)],
    )
    right = _ChunkTranscript(
        start_ms=1_055_000,
        end_ms=1_175_000,
        tokens=[_token("p", 1_058_000, 1_058_300), _token("q", 1_070_640, 1_070_900)],
    )

    merged, _ = _cut_point_merge([left, right])

    starts = [t.start_ms for t in merged]
    assert starts == sorted(starts)


def test_cut_point_merge_across_three_chunks_keeps_cut_points_increasing():
    # Chunk 1 (the middle chunk) realistically re-decodes both seams it
    # touches — it independently produces its own copy of "b" (which
    # chunk 0 also decoded, in their shared overlap) and its own copy of
    # "d" (which chunk 2 also decoded, in their shared overlap).
    chunks = [
        _ChunkTranscript(
            start_ms=0,
            end_ms=120_000,
            tokens=[_token("a", 10_000, 10_200), _token("b", 110_000, 110_200)],
        ),
        _ChunkTranscript(
            start_ms=105_000,
            end_ms=225_000,
            tokens=[
                _token("b", 110_000, 110_200),
                _token("c", 150_000, 150_200),
                _token("d", 220_000, 220_200),
            ],
        ),
        _ChunkTranscript(
            start_ms=210_000,
            end_ms=300_000,
            tokens=[_token("d", 220_000, 220_200), _token("e", 250_000, 250_200)],
        ),
    ]

    merged, cut_points = _cut_point_merge(chunks)

    assert cut_points == sorted(cut_points)
    assert len(cut_points) == 2
    starts = [t.start_ms for t in merged]
    assert starts == sorted(starts)
    assert [t.text for t in merged] == ["a", "b", "c", "d", "e"]


def test_cut_point_merge_covers_every_non_overlap_token_exactly_once():
    """Completeness: tokens strictly outside any overlap window are
    never contested by a cut-point choice, so they must always survive —
    once, not zero or two times."""
    left = _ChunkTranscript(
        start_ms=0, end_ms=1000, tokens=[_token("before-overlap", 10, 60)]
    )
    right = _ChunkTranscript(
        start_ms=850, end_ms=1850, tokens=[_token("after-overlap", 1500, 1550)]
    )

    merged, _ = _cut_point_merge([left, right])

    assert sum(1 for t in merged if t.text == "before-overlap") == 1
    assert sum(1 for t in merged if t.text == "after-overlap") == 1


def test_cut_point_merge_stable_sorts_the_output():
    """Point 2 of the M6/M7 ruling: the concatenated per-chunk
    contributions are stable-sorted by (start_ms, end_ms) as an explicit
    guarantee, not an inherited assumption about chunk decode order."""
    left = _ChunkTranscript(
        start_ms=0, end_ms=1000, tokens=[_token("b", 200, 200), _token("a", 100, 100)]
    )
    right = _ChunkTranscript(
        start_ms=850, end_ms=1850, tokens=[_token("z", 1500, 1550)]
    )

    merged, _ = _cut_point_merge([left, right])

    keys = [(t.start_ms, t.end_ms) for t in merged]
    assert keys == sorted(keys)


def test_cut_point_merge_preserves_within_chunk_duplicates_as_evidence():
    """M6/M7 ruling, confirmed empirically on the real R1 canary: rerun
    #3 failed on "cut-point merge produced a duplicate token: ('0',
    911920, 911920)" — the same shape as rerun #1. But 911920ms sits
    deep inside chunk 8's own [840000, 960000) window (105s/120s/15s
    chunking), nowhere near either of its overlap boundaries — a
    cut-point merge cannot structurally produce a cross-chunk duplicate
    there. Decoding that one chunk in isolation confirmed it directly:
    parakeet's own decode emitted the identical zero-length token '0'
    at 911920ms THREE times, within its own single-chunk output. This
    was never a merge bug. Per the ruling, raw evidence preserves
    exactly this — canonical normalisation (a later stage) owns
    dedup-with-lineage, not this worker."""
    triple_duplicate = [_token("0", 911_920, 911_920) for _ in range(3)]
    chunk = _ChunkTranscript(
        start_ms=840_000,
        end_ms=960_000,
        tokens=[_token(",", 911_840, 911_920), *triple_duplicate],
    )

    merged, cut_points = _cut_point_merge([chunk])

    assert cut_points == []  # single chunk: no boundary decision at all
    zero_length_occurrences = [
        t for t in merged if t.text == "0" and t.start_ms == t.end_ms == 911_920
    ]
    assert len(zero_length_occurrences) == 3  # all three preserved as raw evidence


def test_to_asr_tokens_preserves_duplicates_as_evidence():
    """The final typed-conversion step must not silently drop a
    within-chunk duplicate either — it's legal raw evidence all the way
    through to the response."""
    tokens = _to_asr_tokens(
        [_token("0", 911_920, 911_920), _token("0", 911_920, 911_920)]
    )

    assert len(tokens) == 2
    assert tokens[0] == tokens[1]
