"""Local ASR + diarisation, aligned into the raw transcript.

This is the local half of the transcription pipeline's data boundary: audio
never leaves the machine. :class:`LocalTranscriber` runs Apple-Silicon MLX
ASR (`parakeet-mlx`) and pyannote's speaker-diarisation pipeline
(`pyannote-audio`) against the merged recording produced by plan 003's
`merge-audio`, then aligns the two into a single :class:`RawTranscript` via
the pure :func:`align` function below — the only piece of this module worth
testing without real models.

:func:`align` assigns each ASR segment the speaker whose diarised turn
overlaps it the most. When a segment's overlap spans more than one speaker
(diarisation switched mid-segment, or genuinely overlapping cross-talk —
pyannote's overlap-aware segmentation reports both), the segment is split at
the diarisation boundaries, with its text divided across the split words
proportionally to each side's share of the overlapping duration — an
honest-but-imperfect split, not a word-aligned one; the polish stage (plan
008) carries the residual noise.

**Ordering contract**: :func:`align`'s output is a total order by
non-decreasing `start`, tiebroken by `(start, end, speaker)` — not a
guarantee of disjoint (non-overlapping) spans. Real cross-talk produces
genuinely overlapping diarised turns (e.g. `SPEAKER_00` 0-6s and
`SPEAKER_01` 4-10s both overlapping one ASR segment); the resulting
utterances overlap too, faithfully, because that overlap *is* the
information — collapsing or truncating it would just hide the cross-talk
from the polish stage that's supposed to untangle it. What's guaranteed:
the returned list is sorted under that tiebreak, and identical inputs
always produce identical output.

A diarised turn with no overlapping ASR text produces no utterance at all:
:func:`align` walks the ASR segments (the carriers of text), so
diarisation-only spans are silently dropped rather than emitted as
empty-text utterances — they are far more often diarisation noise (breaths,
cross-talk fragments) than real unheard speech. The same drop applies to an
individual split share that rounds down to zero words when there aren't
enough words in a segment to give every genuine speaker at least one — see
:func:`_apportion`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from parakeet_mlx import from_pretrained as parakeet_from_pretrained
from pyannote.audio import Pipeline

# torchcodec's own `decoders/__init__.py` re-exports `AudioDecoder` from the
# private-looking `_audio_decoder` submodule without an `__all__`, which
# pyright can't see as public — this is still torchcodec's documented,
# stable import path (and the one pyannote's own `io.py` uses internally),
# not an internal implementation detail we're reaching past.
from torchcodec.decoders import (
    AudioDecoder,  # pyright: ignore[reportPrivateImportUsage]
)

from .cache import RunCache, sha256_of
from .models import RawTranscript, SourceClip, Utterance

DEFAULT_ASR_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
DEFAULT_DIARISATION_MODEL = "pyannote/speaker-diarization-3.1"

# parakeet-mlx's Conformer encoder runs full self-attention over the whole
# input in one shot when `chunk_duration` is unset — fine for short clips,
# but its memory cost grows with the *square* of audio length. A real
# ~31-minute meeting blew this up into a 34GB single Metal allocation
# (against a ~9GB ceiling on the machine that hit it). `chunk_duration`/
# `overlap_duration` are parakeet-mlx's own answer: `model.transcribe(...)`
# splits long audio into overlapping windows, runs each separately (bounded
# memory regardless of file length), and stitches the results back into one
# `AlignedResult` with file-relative timestamps and de-duplicated overlap
# text (`parakeet_mlx.alignment.merge_longest_contiguous` /
# `merge_longest_common_subsequence`) — so this is the library's own
# chunking mechanism, not a hand-rolled one. 120s/15s match parakeet-mlx's
# own CLI defaults.
DEFAULT_ASR_CHUNK_DURATION = 120.0
DEFAULT_ASR_CHUNK_OVERLAP = 15.0

UNKNOWN_SPEAKER = "SPEAKER_UNKNOWN"
"""Speaker id for an ASR segment with no overlapping diarised turn at all."""

_RAW_TRANSCRIPT_NAME = "raw_transcript"


class TranscriberError(RuntimeError):
    """Raised when local ASR/diarisation can't proceed."""


class MissingHfTokenError(TranscriberError):
    """Raised when pyannote's gated diarisation model has no token to use."""

    def __init__(self, model: str) -> None:
        super().__init__(
            f"pyannote's diarisation pipeline ({model}) is a gated HuggingFace "
            "model. Set the HF_TOKEN environment variable (see .env) to a "
            "token belonging to an account that has accepted its licence at "
            f"https://huggingface.co/{model}."
        )


@dataclass(frozen=True)
class AsrSegment:
    """One timestamped span of recognised speech, before diarisation."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class DiarisedTurn:
    """One time-labelled speaker turn from the diarisation pipeline."""

    start: float
    end: float
    speaker: str


class Transcriber(Protocol):
    def transcribe(self, audio: Path) -> RawTranscript: ...


class LocalTranscriber:
    """parakeet-mlx ASR + pyannote-audio diarisation, aligned into utterances.

    Construction is cheap and never touches the network or a token — model
    loading and the `HF_TOKEN` check happen lazily inside :meth:`transcribe`,
    so a caller can build one unconditionally (as the CLI handler does) and
    only pay for it — or need a working token — on an actual cache miss.
    """

    def __init__(
        self,
        *,
        hf_token: str | None,
        asr_model: str = DEFAULT_ASR_MODEL,
        diarisation_model: str = DEFAULT_DIARISATION_MODEL,
        asr_chunk_duration: float = DEFAULT_ASR_CHUNK_DURATION,
        asr_chunk_overlap: float = DEFAULT_ASR_CHUNK_OVERLAP,
    ) -> None:
        self._hf_token = hf_token
        self._asr_model = asr_model
        self._diarisation_model = diarisation_model
        self._asr_chunk_duration = asr_chunk_duration
        self._asr_chunk_overlap = asr_chunk_overlap

    def transcribe(self, audio: Path) -> RawTranscript:
        if not self._hf_token:
            raise MissingHfTokenError(self._diarisation_model)

        asr_segments = self._run_asr(audio)
        diarised_turns = self._run_diarisation(audio)
        utterances = align(asr_segments, diarised_turns)

        # `RawTranscript.clips` records provenance for one clip covering the
        # whole merged file (the per-source-clip breakdown from `merge-audio`
        # isn't available here — this command only receives the merged
        # audio path). Duration comes from the latest timestamp either model
        # produced rather than probing the file directly: parakeet-mlx and
        # pyannote already decode the file in whatever format it's in, and
        # duplicating that (e.g. via `soundfile`, which doesn't support M4A)
        # would just be a second, format-fragile way to learn the same fact.
        ends = [item.end for item in (*asr_segments, *diarised_turns)]
        duration = max(ends, default=0.0)
        return RawTranscript(
            clips=[
                SourceClip(
                    path=str(audio), offset_seconds=0.0, duration_seconds=duration
                )
            ],
            utterances=utterances,
            audio_sha256=sha256_of(audio),
        )

    def _run_asr(self, audio: Path) -> list[AsrSegment]:
        model = parakeet_from_pretrained(self._asr_model)
        # Chunked: parakeet-mlx transparently windows long audio (bounded
        # memory) and stitches the result back into file-relative
        # timestamps with de-duplicated overlap text — see the
        # DEFAULT_ASR_CHUNK_DURATION comment above for why this is required
        # rather than optional.
        result = model.transcribe(
            audio,
            chunk_duration=self._asr_chunk_duration,
            overlap_duration=self._asr_chunk_overlap,
        )
        return [
            AsrSegment(
                start=sentence.start, end=sentence.end, text=sentence.text.strip()
            )
            for sentence in result.sentences
        ]

    def _run_diarisation(self, audio: Path) -> list[DiarisedTurn]:
        # Unlike parakeet's raw Conformer encoder, pyannote's segmentation
        # and embedding models don't run one whole-file forward pass:
        # `pyannote.audio.core.inference.Inference` (which the diarisation
        # pipeline uses internally for both) always processes audio through
        # a fixed-duration `SlidingWindow` (`Inference.slide`/`__call__`),
        # so memory is bounded by the window size regardless of file
        # length — confirmed by reading that class, and empirically by
        # running this pipeline end-to-end against the same ~31-minute
        # meeting recording that broke parakeet (see task-004-report.md).
        pipeline = Pipeline.from_pretrained(
            self._diarisation_model, token=self._hf_token
        )
        if pipeline is None:
            raise TranscriberError(
                f"could not load diarisation pipeline {self._diarisation_model!r} "
                "(pyannote returned no pipeline for this checkpoint)"
            )

        # Decode the whole file to an in-memory waveform ourselves rather
        # than handing pyannote a bare path: pyannote's own path-based
        # decoding (via torchcodec) crops the file per-inference-window and
        # asserts each crop returned within 1 sample of the expected count.
        # Real meeting recordings (AAC, via plan 003's ffmpeg merge) can
        # have a few thousand samples' worth of encoder priming delay at
        # the very start, which blew that assertion on the first window of
        # a real ~31-minute meeting (`ValueError: requested chunk [...]
        # resulted in 477184 samples instead of the expected 480000
        # samples`) — a real failure on real audio, not a synthetic one.
        # pyannote's waveform-dict input path pads instead of asserting, so
        # decoding once upfront (linear in file length, ~350MB for 31
        # minutes of mono float32 — nothing like parakeet's O(n^2) blowup)
        # sidesteps the assertion entirely.
        samples = AudioDecoder(audio).get_all_samples()
        output: Any = pipeline(
            {"waveform": samples.data, "sample_rate": samples.sample_rate}
        )
        annotation: Any = output.speaker_diarization
        return [
            DiarisedTurn(start=turn.start, end=turn.end, speaker=str(speaker))
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]


def _overlap(segment: AsrSegment, turn: DiarisedTurn) -> float:
    return max(0.0, min(segment.end, turn.end) - max(segment.start, turn.start))


def _merge_runs(
    turns_by_start: Sequence[DiarisedTurn],
) -> list[tuple[str, float, float]]:
    """Merge consecutive same-speaker turns (already sorted by start) into runs."""
    runs: list[tuple[str, float, float]] = []
    for turn in turns_by_start:
        if runs and runs[-1][0] == turn.speaker:
            speaker, start, end = runs[-1]
            runs[-1] = (speaker, start, max(end, turn.end))
        else:
            runs.append((turn.speaker, turn.start, turn.end))
    return runs


def _apportion(total: int, weights: Sequence[float]) -> list[int]:
    """Split `total` words across `weights` proportionally, guaranteeing every
    genuine speaker at least one word when there are enough to go around.

    A plain largest-remainder split can round a real speaker's share down to
    zero (e.g. 1 word split 9.9s/0.1s rounds the second speaker to nothing),
    which would silently produce an empty-text utterance — the one thing
    this module's docstring promises never happens. So: seed every bucket
    with 1 word first, then apportion the remainder by largest-remainder
    method for proportionality. If there are fewer words than buckets (an
    unavoidable case — you cannot give two speakers a non-empty share of one
    word), the largest-weight buckets get one word each and the rest get
    zero; :func:`_split_segment` drops those zero-count buckets, same as it
    drops a segment with no overlap at all.
    """
    count = len(weights)
    if total <= 0:
        return [0] * count
    if total < count:
        order = sorted(range(count), key=lambda i: (-weights[i], i))
        counts = [0] * count
        for index in order[:total]:
            counts[index] = 1
        return counts

    counts = [1] * count
    remaining = total - count
    weight_sum = sum(weights)
    raw = [remaining * weight / weight_sum for weight in weights]
    extra = [int(share) for share in raw]
    counts = [seed + share for seed, share in zip(counts, extra, strict=True)]
    leftover = remaining - sum(extra)
    order = sorted(range(count), key=lambda i: (-(raw[i] - extra[i]), i))
    for index in order[:leftover]:
        counts[index] += 1
    return counts


def _split_segment(
    segment: AsrSegment, runs: list[tuple[str, float, float]]
) -> list[Utterance]:
    clipped = [
        (speaker, max(start, segment.start), min(end, segment.end))
        for speaker, start, end in runs
    ]
    clipped = [run for run in clipped if run[2] > run[1]]
    if not clipped:
        # Overlap rounded away to nothing (shouldn't happen given callers
        # only reach here with confirmed overlap) — keep the segment whole
        # rather than silently dropping recognised text.
        return [
            Utterance(
                start=segment.start,
                end=segment.end,
                speaker=UNKNOWN_SPEAKER,
                text=segment.text,
            )
        ]

    words = segment.text.split()
    weights = [end - start for _, start, end in clipped]
    word_counts = _apportion(len(words), weights)

    utterances: list[Utterance] = []
    index = 0
    for (speaker, start, end), count in zip(clipped, word_counts, strict=True):
        if count > 0:
            utterances.append(
                Utterance(
                    start=start,
                    end=end,
                    speaker=speaker,
                    text=" ".join(words[index : index + count]),
                )
            )
        index += count
    return utterances


def align(
    asr_segments: Sequence[AsrSegment], diarised_turns: Sequence[DiarisedTurn]
) -> list[Utterance]:
    """Assign each ASR segment a speaker from the diarised turns it overlaps.

    Pure and model-free — every ASR/diarisation call above is thin glue
    around this.

    **Contract**: the returned list is a total order by non-decreasing
    `start`, tiebroken by `(start, end, speaker)`; it is deterministic for
    identical inputs. It is *not* a guarantee of disjoint (non-overlapping)
    spans — see the module docstring. A segment whose overlapping diarised
    turns are all the same speaker is kept whole, with its own timestamps. A
    segment overlapping two or more distinct speakers — a mid-segment
    switch, or genuine cross-talk — is split at the diarised turn boundaries
    (clipped to the segment's own span, consecutive same-speaker turns
    merged first), with its words divided across the split proportionally
    to each side's share of the overlapping duration; when the turns
    genuinely overlap in time, so do the resulting utterances. A segment
    with no diarised overlap at all keeps its own span under
    :data:`UNKNOWN_SPEAKER`, rather than being dropped — ASR still heard
    something, so that's preserved for a human or the polish stage to
    judge.
    """
    utterances: list[Utterance] = []
    for segment in asr_segments:
        overlapping = sorted(
            (turn for turn in diarised_turns if _overlap(segment, turn) > 0),
            key=lambda turn: (turn.start, turn.end, turn.speaker),
        )
        if not overlapping:
            utterances.append(
                Utterance(
                    start=segment.start,
                    end=segment.end,
                    speaker=UNKNOWN_SPEAKER,
                    text=segment.text,
                )
            )
            continue

        speakers = {turn.speaker for turn in overlapping}
        if len(speakers) == 1:
            utterances.append(
                Utterance(
                    start=segment.start,
                    end=segment.end,
                    speaker=next(iter(speakers)),
                    text=segment.text,
                )
            )
            continue

        utterances.extend(_split_segment(segment, _merge_runs(overlapping)))

    # Per-segment processing already yields locally-sorted output, but the
    # ordering contract is a property of the *whole* returned list — sort
    # explicitly so it holds regardless of segment order or cross-segment
    # overlap, with a fully deterministic tiebreak.
    utterances.sort(key=lambda u: (u.start, u.end, u.speaker))
    return utterances


def transcribe_merged_audio(
    audio: Path,
    *,
    run_id: str,
    transcriber: Transcriber,
    cache: RunCache,
) -> RawTranscript:
    """Idempotent ASR+diarisation for one run, cached as `raw_transcript.json`.

    Consults the cache first: a hit is returned as-is without ever calling
    `transcriber` — so a re-run of an already-transcribed run doesn't need a
    working `HF_TOKEN` or a model download. The cache is a convenience; the
    audio stays the authority, so a cache miss always transcribes fresh.
    """
    cached = cache.load(run_id, _RAW_TRANSCRIPT_NAME, RawTranscript)
    if cached is not None:
        return cached

    result = transcriber.transcribe(audio)
    cache.store(run_id, _RAW_TRANSCRIPT_NAME, result)
    return result
