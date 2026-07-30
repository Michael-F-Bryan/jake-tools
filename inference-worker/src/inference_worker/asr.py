"""Stage 2: Parakeet ASR (parakeet-mlx) on the prepared wav.

One pinned model, no backend-selection flags (see models.py docstring).

Long-audio chunking (CHUNK_DURATION_S/OVERLAP_DURATION_S): whole-file
transcription allocates a log-mel buffer proportional to audio duration —
a real 28-minute meeting recording blew Metal's maximum buffer size
(`[metal::malloc] Attempting to allocate 28611722496 bytes which is
greater than the maximum allowed buffer size of 9534832640 bytes`) after
a 10s synthetic smoke test had passed.

parakeet-mlx 0.5.2's own native chunking (`transcribe(chunk_duration=...,
overlap_duration=...)`) fixes the memory problem but its own overlap-
region merge is NOT safe: it picks, per token, whichever chunk's copy it
prefers ("closest window centre") by comparing the two independently-
decoded streams. Verified on the same real 28-minute recording, that
produced first an exact duplicate token at a chunk seam, then — after
fixing the duplicate — a genuine reordering (a token from the "wrong"
side of the overlap chosen out of sequence). Per-token selection over two
independently-decoded streams cannot guarantee global ordering; it's a
structural property of the approach, not a fixable edge case.

This worker instead does its own chunking (reusing parakeet-mlx's own
audio-loading/log-mel/decode primitives — not reimplementing model
internals) and merges adjacent chunks with a CUT-POINT rule instead of
per-token selection: choose exactly one boundary timestamp inside each
pair's overlap region, then take every token before it from the left
chunk and every token at-or-after it from the right chunk. Each instant
in time is then owned by exactly one chunk, so cross-chunk ordering and
exclusivity follow structurally, not from a per-token heuristic.

Cut-point merging does NOT, and is not meant to, deduplicate tokens a
single chunk's own decode repeats. Investigated on the same real
28-minute recording: a merge failure re-appeared after the above fix,
same token and timestamp — decoding just the offending chunk in
isolation confirmed parakeet's own decode emitted the identical
zero-length token three times, deep inside that one chunk's own window
(nowhere near either overlap boundary). A cut-point merge cannot produce
a cross-chunk duplicate by construction, so this was never a merge bug.
See models.py's module docstring for the resulting raw-vs-canonical
contract: this worker's raw ASR output is faithful to what the model
emitted, duplicates and all — strict per-turn ordering/uniqueness is a
canonical-normalisation concern (M6/M7), not this stage's job.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from inference_worker.models import (
    AsrOutput,
    AsrStageResult,
    AsrToken,
    ModelIdentity,
    ModelProvenance,
    StageError,
)
from inference_worker.provenance import (
    best_effort_package_version,
    config_hash,
    local_model_revision,
    stage_observations,
)
from inference_worker.timeouts import StageTimeoutError, enforce_timeout

MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v2"
CHUNK_DURATION_S = 120.0
OVERLAP_DURATION_S = 15.0


@dataclass
class _RawToken:
    """Source-domain-absolute-ms token, before M6 span validation. Kept
    as a plain dataclass (not the pydantic AsrToken) through chunk decode
    and merge so filtering/comparing candidates during the merge doesn't
    pay pydantic construction cost for tokens that get dropped."""

    text: str
    start_ms: int
    end_ms: int
    confidence: float


@dataclass
class _ChunkTranscript:
    start_ms: int
    end_ms: int
    tokens: list[_RawToken]  # already offset to absolute source-domain ms


class _ChunkDecodeOrderError(RuntimeError):
    """A single chunk's own decode was internally out of order — a
    parakeet-mlx/model bug, not a chunk-merge artefact (the merge logic
    never runs on an unordered chunk in the first place)."""


def asr_stage_config_hash() -> str:
    # M2: fold in the resolved model revision + package versions, not just
    # the compile-time MODEL_ID constant — two runs against different
    # cached revisions must hash differently for M11/M12 hash-keyed reuse
    # to be sound. N3: best-effort, not strict package_version() — this
    # runs before run_asr's own try block even starts, so a package
    # missing on some platform (e.g. mlx on non-Apple-Silicon) must not
    # raise here and degrade a per-stage failure into a whole-run
    # worker-internal-error. Chunk/overlap duration affect the actual
    # transcription output at chunk boundaries, so they're config that
    # gets hashed too, not just compile-time constants nobody can see.
    return config_hash(
        {
            "model_id": MODEL_ID,
            "model_revision": local_model_revision(MODEL_ID) or "unresolved",
            "parakeet-mlx": best_effort_package_version("parakeet-mlx"),
            "mlx": best_effort_package_version("mlx"),
            "chunk_duration_s": CHUNK_DURATION_S,
            "overlap_duration_s": OVERLAP_DURATION_S,
        }
    )


def _model_provenance() -> ModelProvenance | None:
    """Best-effort model identity from whatever's locally cached right
    now. None if nothing is cached yet — never a fabricated "unknown"."""
    revision = local_model_revision(MODEL_ID)
    if revision is None:
        return None
    return ModelProvenance(
        identity=ModelIdentity(name=MODEL_ID, version=revision),
        package_versions={
            "parakeet-mlx": best_effort_package_version("parakeet-mlx"),
            "mlx": best_effort_package_version("mlx"),
        },
    )


def _raw_tokens_from_aligned(aligned_tokens, offset_ms: int) -> list[_RawToken]:
    """Convert parakeet-mlx's AlignedToken objects (chunk-relative
    seconds) to _RawToken (absolute ms), verifying the chunk's own
    decode is internally ordered — an out-of-order token WITHIN one
    chunk's decode is a model bug, not a merge artefact, and raises
    _ChunkDecodeOrderError so the caller can give it its own error class
    rather than let a broken chunk poison the cross-chunk merge."""
    raw: list[_RawToken] = []
    previous_start_ms = -1
    for token in aligned_tokens:
        start_ms = round(token.start * 1000) + offset_ms
        end_ms = round(token.end * 1000) + offset_ms
        if start_ms < previous_start_ms:
            raise _ChunkDecodeOrderError(
                f"parakeet decoded an out-of-order token within a single chunk "
                f"(chunk offset {offset_ms}ms): {start_ms}ms follows {previous_start_ms}ms"
            )
        previous_start_ms = start_ms
        raw.append(
            _RawToken(
                text=token.text,
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=token.confidence,
            )
        )
    return raw


def _transcribe_chunks(model, wav_path: Path) -> list[_ChunkTranscript]:
    """Split the wav into overlapping CHUNK_DURATION_S windows — mirroring
    parakeet-mlx's own transcribe(chunk_duration=...) splitting loop —
    and decode each with the model's lower-level generate(), the same
    primitive transcribe() itself calls per chunk. This reuses Parakeet's
    own audio loading/log-mel/decoding; only the cross-chunk *merge* is
    this worker's own (see _cut_point_merge)."""
    import mlx.core as mx
    from parakeet_mlx.audio import get_logmel, load_audio
    from parakeet_mlx.parakeet import DecodingConfig

    decoding_config = DecodingConfig()
    sample_rate = model.preprocessor_config.sample_rate
    audio_data = load_audio(wav_path, sample_rate, mx.bfloat16)
    total_samples = len(audio_data)

    if total_samples / sample_rate <= CHUNK_DURATION_S:
        mel = get_logmel(audio_data, model.preprocessor_config)
        result = model.generate(mel, decoding_config=decoding_config)[0]
        tokens = _raw_tokens_from_aligned(result.tokens, offset_ms=0)
        return [
            _ChunkTranscript(
                start_ms=0,
                end_ms=round(total_samples / sample_rate * 1000),
                tokens=tokens,
            )
        ]

    chunk_samples = int(CHUNK_DURATION_S * sample_rate)
    overlap_samples = int(OVERLAP_DURATION_S * sample_rate)
    step_samples = chunk_samples - overlap_samples

    chunks: list[_ChunkTranscript] = []
    for start in range(0, total_samples, step_samples):
        end = min(start + chunk_samples, total_samples)
        if end - start < model.preprocessor_config.hop_length:
            break  # matches transcribe()'s own guard: prevents a zero-length log-mel
        mel = get_logmel(audio_data[start:end], model.preprocessor_config)
        result = model.generate(mel, decoding_config=decoding_config)[0]
        offset_ms = round(start / sample_rate * 1000)
        tokens = _raw_tokens_from_aligned(result.tokens, offset_ms=offset_ms)
        chunks.append(
            _ChunkTranscript(
                start_ms=offset_ms,
                end_ms=round(end / sample_rate * 1000),
                tokens=tokens,
            )
        )
    return chunks


def _choose_cut_point(left: _ChunkTranscript, right: _ChunkTranscript) -> int:
    """The boundary timestamp for one adjacent chunk pair's overlap
    region [right.start_ms, left.end_ms): the midpoint of the largest
    inter-token silence gap over the union of both chunks' tokens inside
    the overlap, nearest the overlap midpoint as the tie-break — cutting
    inside a silence avoids splitting a word the two chunks decoded
    differently. Falls back to the plain overlap midpoint when there's
    no usable gap (fewer than two tokens land in the overlap, or every
    adjacent pair touches/overlaps in time).

    Known trade-off: the plain-midpoint fallback can orphan a token that
    only ONE side happened to decode near that exact position (e.g. the
    losing chunk had a word there and the winning chunk's own decode was
    silent right at the cut). This is the accepted cost of "ownership by
    threshold" being correct-by-construction for ordering/uniqueness —
    guaranteeing zero information loss in every ambiguous case isn't
    possible while also guaranteeing that, since the two chunks are
    independent decodes that can genuinely disagree in the overlap.
    """
    overlap_start_ms = right.start_ms
    overlap_end_ms = left.end_ms
    overlap_midpoint_ms = round((overlap_start_ms + overlap_end_ms) / 2)

    in_overlap = sorted(
        (
            t
            for t in (*left.tokens, *right.tokens)
            if overlap_start_ms <= t.start_ms < overlap_end_ms
        ),
        key=lambda t: t.start_ms,
    )

    best_gap_size = -1
    best_gap_midpoint = overlap_midpoint_ms
    for previous, current in zip(
        in_overlap, in_overlap[1:], strict=False
    ):  # deliberately unequal lengths (pairs)
        gap_size = current.start_ms - previous.end_ms
        if gap_size <= 0:
            continue  # touching or overlapping spans: not a silence to cut inside
        gap_midpoint = round((previous.end_ms + current.start_ms) / 2)
        is_better = gap_size > best_gap_size or (
            gap_size == best_gap_size
            and abs(gap_midpoint - overlap_midpoint_ms)
            < abs(best_gap_midpoint - overlap_midpoint_ms)
        )
        if is_better:
            best_gap_size = gap_size
            best_gap_midpoint = gap_midpoint

    return best_gap_midpoint


def _cut_point_merge(
    chunks: list[_ChunkTranscript],
) -> tuple[list[_RawToken], list[int]]:
    """Merge adjacent chunk transcripts by cut point, not by per-token
    selection (see module docstring for why the latter is unsound).
    Chunk i and chunk i+1's overlap region never touches chunk i-1 and
    i's (CHUNK_DURATION_S/OVERLAP_DURATION_S guarantee 2*step >
    chunk_duration), so cut points are always strictly increasing and
    every instant in time is claimed by exactly one chunk — cross-chunk
    ordering and exclusivity follow structurally, not from a dedup pass.

    The concatenated per-chunk contributions are then stable-sorted by
    (start_ms, end_ms): within-chunk order is otherwise already correct
    (each chunk's own decode is verified non-strictly ordered — see
    _raw_tokens_from_aligned), but a stable sort makes the overall
    result's ordering an explicit guarantee rather than an inherited
    assumption, and keeps within-chunk ties (including genuine
    duplicates the model itself emitted — legal raw evidence, see
    models.py) in their original decode order rather than reordering
    them arbitrarily.

    Returns (merged_tokens, cut_points_ms); cut_points_ms is recorded on
    the response as evidence of the merge decision (empty if the audio
    fit in a single chunk).
    """
    if len(chunks) == 1:
        return sorted(chunks[0].tokens, key=lambda t: (t.start_ms, t.end_ms)), []

    cut_points_ms = [
        _choose_cut_point(chunks[i], chunks[i + 1]) for i in range(len(chunks) - 1)
    ]
    if cut_points_ms != sorted(cut_points_ms):
        # Structural precondition broken (e.g. CHUNK_DURATION_S/
        # OVERLAP_DURATION_S changed to values where overlaps of
        # non-adjacent chunks touch) — fail loudly rather than emit a
        # result whose ordering the construction argument no longer covers.
        raise RuntimeError(
            f"chunk cut points are not strictly increasing: {cut_points_ms}"
        )

    merged: list[_RawToken] = []
    for i, chunk in enumerate(chunks):
        lower = cut_points_ms[i - 1] if i > 0 else None
        upper = cut_points_ms[i] if i < len(cut_points_ms) else None
        for token in chunk.tokens:
            if lower is not None and token.start_ms < lower:
                continue
            if upper is not None and token.start_ms >= upper:
                continue
            merged.append(token)
    merged.sort(key=lambda t: (t.start_ms, t.end_ms))
    return merged, cut_points_ms


def _to_asr_tokens(raw_tokens: list[_RawToken]) -> list[AsrToken]:
    """Final typed conversion. M6 span validation (end_ms >= start_ms)
    happens here, on the already-merged, already-sorted stream — a
    reversed span is a modelling error in a single token, orthogonal to
    the merge. Duplicates and zero-length tokens pass through untouched:
    they're legal raw evidence (see models.py's raw-vs-canonical
    contract), not something this stage sanitises."""
    return [
        AsrToken(
            text=t.text, start_ms=t.start_ms, end_ms=t.end_ms, confidence=t.confidence
        )
        for t in raw_tokens
    ]


def run_asr(wav_path: Path, wav_sha256: str, timeout_s: float) -> AsrStageResult:
    stage_config_hash = asr_stage_config_hash()
    started = time.monotonic()
    try:
        with enforce_timeout(timeout_s):
            # Imported lazily: importing parakeet_mlx eagerly at module load
            # would pull in mlx/Metal initialisation for every CLI
            # invocation, including ones that never reach this stage.
            from parakeet_mlx import from_pretrained

            model = from_pretrained(MODEL_ID)
            chunks = _transcribe_chunks(model, wav_path)
            merged_raw_tokens, chunk_boundaries_ms = _cut_point_merge(chunks)
    except _ChunkDecodeOrderError as exc:
        return _failed(
            stage_config_hash,
            started,
            wav_sha256,
            "asr-chunk-decode-unordered",
            str(exc),
            retryable=False,
            model_provenance=_model_provenance(),
            retained_artefacts=[wav_path.name],
        )
    except StageTimeoutError as exc:
        return _failed(
            stage_config_hash,
            started,
            wav_sha256,
            "timeout",
            str(exc),
            retryable=True,
            model_provenance=_model_provenance(),
            retained_artefacts=[wav_path.name],
        )
    except Exception as exc:
        return _failed(
            stage_config_hash,
            started,
            wav_sha256,
            "asr-failed",
            f"{type(exc).__name__}: {exc}",
            retryable=False,
            model_provenance=_model_provenance(),
            retained_artefacts=[wav_path.name],
        )

    # A successful decode means `from_pretrained` just loaded (and, if
    # this was the first-ever run, downloaded-then-cached) the model, so a
    # cached revision must now be resolvable. If it somehow isn't, that's
    # a real problem, not a silent "unknown" on a status="completed" stage
    # (M1) — fail the stage instead of lying about its provenance.
    model_provenance = _model_provenance()
    if model_provenance is None:
        return _failed(
            stage_config_hash,
            started,
            wav_sha256,
            "asr-failed",
            f"transcription completed but no cached revision for {MODEL_ID!r} could be resolved locally",
            retryable=False,
            retained_artefacts=[wav_path.name],
        )

    try:
        tokens = _to_asr_tokens(merged_raw_tokens)
    except ValidationError as exc:
        # M6: a reversed span (end_ms < start_ms) is a modelling error in
        # the raw ASR output, not evidence worth keeping — fail the stage
        # rather than silently accepting or discarding it. Zero-length
        # spans (start_ms == end_ms) are valid and pass through untouched.
        return _failed(
            stage_config_hash,
            started,
            wav_sha256,
            "asr-invalid-output",
            f"parakeet returned a token with an invalid span: {exc}",
            retryable=False,
            model_provenance=model_provenance,
            retained_artefacts=[wav_path.name],
        )

    text = "".join(token.text for token in tokens).strip()
    return AsrStageResult(
        status="completed",
        config_hash=stage_config_hash,
        input_audio_sha256=wav_sha256,
        observations=stage_observations(started),
        model_provenance=model_provenance,
        output=AsrOutput(
            text=text, tokens=tokens, chunk_boundaries_ms=chunk_boundaries_ms
        ),
    )


def _failed(
    stage_config_hash: str,
    started: float,
    input_audio_sha256: str,
    error_class: str,
    message: str,
    *,
    retryable: bool,
    model_provenance: ModelProvenance | None = None,
    retained_artefacts: list[str] | None = None,
) -> AsrStageResult:
    return AsrStageResult(
        status="failed",
        config_hash=stage_config_hash,
        # The wav is already hashed by the time run_asr is ever called
        # (prepare.py computed it) — a failure here is no reason to lose
        # that provenance from the record.
        input_audio_sha256=input_audio_sha256,
        model_provenance=model_provenance,
        observations=stage_observations(started),
        error=StageError(
            error_class=error_class,
            message=message,
            retryable=retryable,
            retained_artefacts=retained_artefacts or [],
        ),
    )
