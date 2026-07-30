"""Property test: cut-point merging is correct by construction across
many randomized, DENSE synthetic chunk-pair scenarios.

The reason the model test in tests/model/test_asr_model.py missed all
three real R1 merge/decode bugs is that its tone/TTS audio yields sparse
tokens, while real speech is dense — a sparse stream rarely lands two
tokens close enough together to exercise an overlap-region merge
decision, or a within-chunk decode quirk, at all. This is a seeded loop
over many seeds (not hypothesis — no new deps), generating dense streams
that deliberately include zero-length tokens, tokens straddling the
overlap boundary, empty overlap regions, identical tokens duplicated
across both chunks (R1 rerun #1's original shape), AND identical tokens
duplicated WITHIN a single chunk's own decode (R1 rerun #3's actual,
confirmed shape — see test_cut_point_merge_preserves_within_chunk_
duplicates_as_evidence in test_asr.py for the empirical investigation).

Per the M6/M7 ruling: duplicates are legal raw evidence. What the merge
must still guarantee structurally is (a) non-strict ordering by
(start_ms, end_ms), (b) every token strictly outside the overlap window
survives exactly once, and (c) a duplicate that *does* survive in the
merged output is explained by a single source chunk already containing
it more than once — never by the same token existing once in EACH chunk
and both copies surviving, which would mean cross-chunk ownership wasn't
exclusive.
"""

from __future__ import annotations

import random
from collections import Counter

from inference_worker.asr import _ChunkTranscript, _cut_point_merge, _RawToken

_SEED_COUNT = 300


def _dense_tokens(
    rng: random.Random, span_start_ms: int, span_end_ms: int, *, prefix: str
) -> list[_RawToken]:
    """A dense, monotonically ordered token stream covering
    [span_start_ms, span_end_ms) with frequent zero-length tokens and
    small-or-no gaps — shaped like real speech, not sparse tone/TTS."""
    tokens: list[_RawToken] = []
    t = span_start_ms
    idx = 0
    while t < span_end_ms:
        duration = 0 if rng.random() < 0.3 else rng.randint(1, 60)
        start_ms = t
        end_ms = start_ms + duration
        tokens.append(
            _RawToken(
                text=f"{prefix}{idx}", start_ms=start_ms, end_ms=end_ms, confidence=0.9
            )
        )
        idx += 1
        gap = rng.choice([0, 0, 0, rng.randint(1, 15)])
        t = end_ms + gap
    return tokens


def _inject_cross_chunk_duplicates(
    rng: random.Random,
    left_tokens: list[_RawToken],
    right_tokens: list[_RawToken],
    overlap_start_ms: int,
    overlap_end_ms: int,
) -> None:
    """R1 rerun #1's original shape: both chunks' independent decodes
    produce an identical (text, start_ms, end_ms) token inside the
    overlap window. Cross-chunk exclusivity means at most one copy can
    survive the merge — see the (c) property in the module docstring."""
    candidates = [
        t for t in left_tokens if overlap_start_ms <= t.start_ms < overlap_end_ms
    ]
    if not candidates:
        return
    chosen = rng.sample(candidates, k=rng.randint(1, min(2, len(candidates))))
    for source in chosen:
        right_tokens.append(
            _RawToken(
                text=source.text,
                start_ms=source.start_ms,
                end_ms=source.end_ms,
                confidence=source.confidence,
            )
        )
    right_tokens.sort(key=lambda t: (t.start_ms, t.end_ms))


def _inject_within_chunk_duplicates(
    rng: random.Random, tokens: list[_RawToken]
) -> None:
    """R1 rerun #3's actual, confirmed shape: a single chunk's own
    parakeet decode repeats the identical token — verified directly by
    decoding the real offending chunk in isolation (three identical
    zero-length tokens, not a merge artefact). These must survive the
    merge as legal raw evidence (M6/M7)."""
    if not tokens:
        return
    source = rng.choice(tokens)
    for _ in range(rng.randint(1, 2)):
        tokens.append(
            _RawToken(
                text=source.text,
                start_ms=source.start_ms,
                end_ms=source.end_ms,
                confidence=0.9,
            )
        )
    tokens.sort(key=lambda t: (t.start_ms, t.end_ms))


def _random_chunk_pair(rng: random.Random) -> tuple[_ChunkTranscript, _ChunkTranscript]:
    chunk_len_ms = rng.randint(400, 2000)
    # Sometimes a genuinely empty overlap region (adjacent, non-
    # overlapping chunks) — otherwise up to roughly a third of the chunk.
    max_overlap = max(1, chunk_len_ms // 3)
    overlap_ms = 0 if rng.random() < 0.15 else rng.randint(1, max_overlap)

    left_start_ms, left_end_ms = 0, chunk_len_ms
    right_start_ms = chunk_len_ms - overlap_ms
    right_end_ms = right_start_ms + chunk_len_ms

    left_tokens = _dense_tokens(rng, left_start_ms, left_end_ms, prefix="L")
    right_tokens = _dense_tokens(rng, right_start_ms, right_end_ms, prefix="R")

    if rng.random() < 0.35:
        _inject_cross_chunk_duplicates(
            rng, left_tokens, right_tokens, right_start_ms, left_end_ms
        )
    if rng.random() < 0.35:
        _inject_within_chunk_duplicates(rng, left_tokens)
    if rng.random() < 0.35:
        _inject_within_chunk_duplicates(rng, right_tokens)

    return (
        _ChunkTranscript(
            start_ms=left_start_ms, end_ms=left_end_ms, tokens=left_tokens
        ),
        _ChunkTranscript(
            start_ms=right_start_ms, end_ms=right_end_ms, tokens=right_tokens
        ),
    )


def _assert_merge_property_holds(
    left: _ChunkTranscript,
    right: _ChunkTranscript,
    merged: list[_RawToken],
    cut_points: list[int],
    *,
    label: str,
) -> None:
    """Shared assertions for both the randomized property loop and the
    fixed R1 regression case below."""
    keys = [(t.start_ms, t.end_ms) for t in merged]
    assert keys == sorted(keys), (
        f"{label}: not ordered (non-strictly) by (start_ms, end_ms): {keys}"
    )

    overlap_start_ms, overlap_end_ms = right.start_ms, left.end_ms

    merged_keys = [(t.text, t.start_ms, t.end_ms) for t in merged]
    merged_counts = Counter(merged_keys)
    left_counts = Counter((t.text, t.start_ms, t.end_ms) for t in left.tokens)
    right_counts = Counter((t.text, t.start_ms, t.end_ms) for t in right.tokens)

    # Completeness: every token strictly outside the overlap window is
    # never contested by the cut-point choice, so its ORIGINAL
    # multiplicity in its own source chunk (1, if it's not itself a
    # within-chunk duplicate; N otherwise) must be preserved exactly —
    # never dropped, never duplicated further by the merge itself.
    non_overlap_left_keys = {
        (t.text, t.start_ms, t.end_ms)
        for t in left.tokens
        if t.start_ms < overlap_start_ms
    }
    non_overlap_right_keys = {
        (t.text, t.start_ms, t.end_ms)
        for t in right.tokens
        if t.start_ms >= overlap_end_ms
    }
    for key in non_overlap_left_keys:
        assert merged_counts[key] == left_counts[key], (
            f"{label}: non-overlap token {key} appeared {merged_counts[key]}x, expected {left_counts[key]}x"
        )
    for key in non_overlap_right_keys:
        assert merged_counts[key] == right_counts[key], (
            f"{label}: non-overlap token {key} appeared {merged_counts[key]}x, expected {right_counts[key]}x"
        )

    # Any surviving duplicate must trace to a single source chunk's own
    # contribution to `merged`, never to copies from BOTH chunks
    # surviving for the same key — that would mean cross-chunk
    # ownership wasn't exclusive. Note this is about which copies
    # actually crossed the cut point, not merely whether the key exists
    # somewhere in both chunks' raw token lists (a key can legitimately
    # exist on both sides — e.g. an injected cross-chunk duplicate — as
    # long as only one side's copies end up owning that instant).
    boundary = cut_points[0] if cut_points else None
    if boundary is not None:
        for key, count in merged_counts.items():
            if count <= 1:
                continue
            left_survivors = sum(
                1
                for t in left.tokens
                if (t.text, t.start_ms, t.end_ms) == key and t.start_ms < boundary
            )
            right_survivors = sum(
                1
                for t in right.tokens
                if (t.text, t.start_ms, t.end_ms) == key and t.start_ms >= boundary
            )
            assert left_survivors == 0 or right_survivors == 0, (
                f"{label}: duplicate {key} (x{count}) has surviving copies from BOTH chunks "
                f"({left_survivors} left, {right_survivors} right) — cross-chunk exclusivity broken"
            )

    if len(cut_points) == 1:
        assert overlap_start_ms <= cut_points[0] <= overlap_end_ms, (
            f"{label}: cut point {cut_points} outside overlap [{overlap_start_ms}, {overlap_end_ms}]"
        )
    else:
        raise AssertionError(
            f"{label}: expected exactly one cut point for a 2-chunk pair, got {cut_points}"
        )


def test_cut_point_merge_property_many_seeds():
    for seed in range(_SEED_COUNT):
        rng = random.Random(seed)
        left, right = _random_chunk_pair(rng)

        merged, cut_points = _cut_point_merge([left, right])

        _assert_merge_property_holds(
            left, right, merged, cut_points, label=f"seed {seed}"
        )


def test_cut_point_merge_property_the_verified_r1_within_chunk_duplicate():
    """The exact fixed regression input from the real R1 canary: chunk 8
    of a real 28-minute recording independently decoded the identical
    zero-length token ('0', 911920, 911920) three times, confirmed by
    decoding that chunk in isolation. 911920ms sits deep inside chunk
    8's own [840000, 960000) window (105s/120s/15s chunking) — nowhere
    near either overlap boundary — so this is paired with a neighbour
    chunk to prove the property still holds with a genuine adjacent
    overlap present, not just in isolation."""
    within_chunk_duplicate = [
        _RawToken(text="0", start_ms=911_920, end_ms=911_920, confidence=0.9)
        for _ in range(3)
    ]
    left = _ChunkTranscript(
        start_ms=840_000,
        end_ms=960_000,
        tokens=[
            _RawToken(text=",", start_ms=911_840, end_ms=911_920, confidence=0.9),
            *within_chunk_duplicate,
            _RawToken(text="next", start_ms=950_000, end_ms=950_400, confidence=0.9),
        ],
    )
    right = _ChunkTranscript(
        start_ms=945_000,
        end_ms=1_065_000,
        tokens=[
            _RawToken(
                text="after", start_ms=1_000_000, end_ms=1_000_400, confidence=0.9
            )
        ],
    )

    merged, cut_points = _cut_point_merge([left, right])

    _assert_merge_property_holds(
        left, right, merged, cut_points, label="R1 within-chunk duplicate"
    )
    zero_length_occurrences = [
        t for t in merged if t.text == "0" and t.start_ms == t.end_ms == 911_920
    ]
    assert len(zero_length_occurrences) == 3  # preserved as evidence, none dropped
