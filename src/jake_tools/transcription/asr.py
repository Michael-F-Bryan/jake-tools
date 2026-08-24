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

import gc
import inspect
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import mlx.core as mx
import torch
from parakeet_mlx import from_pretrained as parakeet_from_pretrained
from pyannote.audio import Pipeline
from pydantic import BaseModel

# torchcodec's own `decoders/__init__.py` re-exports `AudioDecoder` from the
# private-looking `_audio_decoder` submodule without an `__all__`, which
# pyright can't see as public — this is still torchcodec's documented,
# stable import path (and the one pyannote's own `io.py` uses internally),
# not an internal implementation detail we're reaching past.
from torchcodec.decoders import (
    AudioDecoder,  # pyright: ignore[reportPrivateImportUsage]
)

from ..cache_models import CacheEnvelope
from .cache import RunCache, StageManifest, sha256_of, stable_hash
from .memory_watchdog import MemoryWatchdog, default_memory_budget_bytes
from .models import RawTranscript, SourceClip, StageTiming, Utterance

DEFAULT_ASR_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
DEFAULT_DIARISATION_MODEL = "pyannote/speaker-diarization-community-1"
DiarisationDevice = Literal["auto", "cpu", "mps"]

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

# Chunking bounds parakeet's memory growth but doesn't cap the size of any
# single allocation — an oversized chunk (or pyannote's own inference
# window, below) should still fail with a catchable exception rather than
# grow unbounded. `mx.set_memory_limit` and `torch.mps.set_per_process_
# memory_fraction` are each library's fail-fast form: MLX raises once an
# allocation would exceed the limit and RAM+swap are exhausted; torch's
# MPS allocator raises an out-of-memory error the same way. Both are set
# to the same ~60% policy the memory watchdog uses (see
# `memory_watchdog.DEFAULT_MEMORY_BUDGET_FRACTION`) — belt-and-braces, not
# a substitute for the watchdog: these only catch the accelerator-side
# share of an allocation, and yesterday's crash showed at least part of a
# runaway grows as ordinary swappable CPU memory these caps never see.
DEFAULT_MPS_MEMORY_FRACTION = 0.6

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


class AcceleratorOutOfMemoryError(TranscriberError):
    """Raised when MLX/torch raise while running ASR or diarisation, most
    likely because the fail-fast caps (`mx.set_memory_limit`/`torch.mps.
    set_per_process_memory_fraction`, see `DEFAULT_MPS_MEMORY_FRACTION`)
    tripped on an oversized allocation.

    Neither library exposes a distinct exception type for "the cap
    tripped" versus any other `RuntimeError` the call could raise, so this
    wraps the underlying error rather than asserting with certainty it was
    an out-of-memory condition — "possibly" in the message is deliberate,
    not hedging for its own sake. Without this, a cap trip would surface
    as a raw traceback through Click instead of a clean `ClickException`
    the way `MissingHfTokenError` and the pipeline-load failure already
    do (`cli/transcript.py`'s `except TranscriberError` already covers
    this subclass — no CLI change needed to wire it up)."""

    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(f"{stage} failed, possibly out of accelerator memory: {cause}")


class AsrSegment(BaseModel):
    """One timestamped span of recognised speech, before diarisation."""

    start: float
    end: float
    text: str


class DiarisedTurn(BaseModel):
    """One time-labelled speaker turn from the diarisation pipeline."""

    start: float
    end: float
    speaker: str


class AsrCheckpoint(CacheEnvelope):
    """Content/config-bound ASR output that can be resumed independently."""

    audio_sha256: str
    model: str
    chunk_duration: float
    chunk_overlap: float
    segments: list[AsrSegment]


class DiarisationCheckpoint(CacheEnvelope):
    """Content/config-bound diarisation output that can be resumed independently."""

    audio_sha256: str
    model: str
    device: Literal["cpu", "mps"]
    num_speakers: int | None
    turns: list[DiarisedTurn]


def build_audio_stage_manifest(raw: RawTranscript) -> StageManifest:
    """Describe the exact local model inputs and raw output for review binding."""
    local_config = {
        "asr_model": raw.asr_model,
        "asr_chunk_duration": raw.asr_chunk_duration,
        "asr_chunk_overlap": raw.asr_chunk_overlap,
        "diarisation_model": raw.diarisation_model,
        "diarisation_device": raw.diarisation_device,
        "num_speakers": raw.num_speakers,
    }
    return StageManifest(
        stage="asr",
        input_hash=stable_hash({"audio_sha256": raw.audio_sha256, **local_config}),
        config_hash=stable_hash(local_config),
        input_hashes={
            "audio_sha256": raw.audio_sha256 or "",
            "local_config": stable_hash(local_config),
        },
        output_hash=stable_hash(raw.model_dump(mode="json")),
    )


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
        diarisation_device: DiarisationDevice = "auto",
        num_speakers: int | None = None,
        memory_budget_bytes: int | None = None,
    ) -> None:
        self._hf_token = hf_token
        self._asr_model = asr_model
        self._diarisation_model = diarisation_model
        self._asr_chunk_duration = asr_chunk_duration
        self._asr_chunk_overlap = asr_chunk_overlap
        self._diarisation_device: DiarisationDevice = diarisation_device
        self._num_speakers = num_speakers
        # None means "use the shared ~60%-of-RAM default" — resolved lazily
        # via _resolve_memory_budget_bytes() rather than here, so the
        # default always reflects the machine actually running
        # `transcribe()`, and both the watchdog and the MLX cap below
        # agree on one number instead of each computing their own.
        self._memory_budget_bytes = memory_budget_bytes

    def _resolve_memory_budget_bytes(self) -> int:
        """The byte budget the watchdog and `mx.set_memory_limit` both
        enforce: an explicit constructor override if one was given, else
        `default_memory_budget_bytes()`. A single method so the two
        callers can't silently disagree the way they once did (the MLX
        cap used to call the module default directly, ignoring a caller-
        supplied override the watchdog already respected)."""
        return (
            self._memory_budget_bytes
            if self._memory_budget_bytes is not None
            else default_memory_budget_bytes()
        )

    @property
    def diarisation_device(self) -> DiarisationDevice:
        return self._diarisation_device

    @property
    def num_speakers(self) -> int | None:
        return self._num_speakers

    def transcribe(
        self,
        audio: Path,
        *,
        num_speakers: int | None = None,
        media_duration_seconds: float | None = None,
    ) -> RawTranscript:
        if not self._hf_token:
            raise MissingHfTokenError(self._diarisation_model)

        # Must start before model loading: model loading is exactly where
        # the real acceptance run's runaway allocation grew. This is a
        # best-effort in-process backstop, not a guarantee — see the
        # memory_watchdog module docstring for what it can't promise under
        # GIL/scheduler starvation, and why the mx/torch caps below (and,
        # until an out-of-process supervisor exists, external supervision
        # of a first real run) are this watchdog's real partners in
        # safety, not redundant with it. Stopped in `finally` so a
        # normal-or-failed run doesn't leave the poll thread running.
        speaker_count = self._num_speakers if num_speakers is None else num_speakers
        selected_device = select_diarisation_device(self._diarisation_device)
        watchdog = MemoryWatchdog(budget_bytes=self._resolve_memory_budget_bytes())
        watchdog.start()
        try:
            asr_segments = self._run_asr(audio)
            diarised_turns = self._run_diarisation(
                audio, device=selected_device, num_speakers=speaker_count
            )
        finally:
            watchdog.stop()
        utterances = align(asr_segments, diarised_turns)

        # `RawTranscript.clips` records provenance for one clip covering the
        # whole merged file. The composed pipeline passes the authoritative
        # duration from `merge-audio`'s source clips; the timestamp fallback is
        # retained only for direct legacy callers that have no source probe.
        ends = [item.end for item in (*asr_segments, *diarised_turns)]
        duration = (
            media_duration_seconds
            if media_duration_seconds is not None
            else max(ends, default=0.0)
        )
        return RawTranscript(
            clips=[
                SourceClip(
                    path=str(audio), offset_seconds=0.0, duration_seconds=duration
                )
            ],
            utterances=utterances,
            audio_sha256=sha256_of(audio),
            asr_model=self._asr_model,
            diarisation_model=self._diarisation_model,
            diarisation_device=selected_device,
            num_speakers=speaker_count,
            asr_chunk_duration=self._asr_chunk_duration,
            asr_chunk_overlap=self._asr_chunk_overlap,
        )

    def transcribe_cached(
        self,
        audio: Path,
        *,
        run_id: str,
        cache: RunCache,
        num_speakers: int | None = None,
        media_duration_seconds: float | None = None,
    ) -> RawTranscript:
        """Run or resume ASR and diarisation from independent checkpoints."""
        audio_sha256 = sha256_of(audio)
        speaker_count = self._num_speakers if num_speakers is None else num_speakers
        device = select_diarisation_device(self._diarisation_device)

        diarisation_checkpoint = cache.load_resumable(
            run_id, "diarisation_checkpoint", DiarisationCheckpoint
        )
        diarisation_valid = diarisation_checkpoint is not None and (
            diarisation_checkpoint.audio_sha256 == audio_sha256
            and diarisation_checkpoint.model == self._diarisation_model
            and diarisation_checkpoint.device == device
            and diarisation_checkpoint.num_speakers == speaker_count
        )
        diarisation_recomputed = not diarisation_valid
        if diarisation_recomputed and not self._hf_token:
            raise MissingHfTokenError(self._diarisation_model)

        asr_checkpoint = cache.load_resumable(run_id, "asr_checkpoint", AsrCheckpoint)
        asr_valid = asr_checkpoint is not None and (
            asr_checkpoint.audio_sha256 == audio_sha256
            and asr_checkpoint.model == self._asr_model
            and asr_checkpoint.chunk_duration == self._asr_chunk_duration
            and asr_checkpoint.chunk_overlap == self._asr_chunk_overlap
        )
        asr_recomputed = not asr_valid
        if asr_recomputed:
            asr_checkpoint = self._run_cached_asr(
                audio,
                run_id,
                cache,
                audio_sha256,
                media_duration_seconds=media_duration_seconds,
            )
        else:
            assert asr_checkpoint is not None
            _record_timing(
                cache,
                run_id,
                _timing(
                    "ASR",
                    started=time.monotonic(),
                    cache_hit=True,
                    model=self._asr_model,
                    config={
                        "chunk_duration": self._asr_chunk_duration,
                        "chunk_overlap": self._asr_chunk_overlap,
                    },
                    media_duration=media_duration_seconds,
                ),
            )
        assert asr_checkpoint is not None

        if diarisation_recomputed:
            diarisation_checkpoint = self._run_cached_diarisation(
                audio,
                run_id,
                cache,
                audio_sha256,
                device=device,
                num_speakers=speaker_count,
                media_duration_seconds=media_duration_seconds,
            )
        else:
            assert diarisation_checkpoint is not None
            _record_timing(
                cache,
                run_id,
                _timing(
                    "diarisation",
                    started=time.monotonic(),
                    cache_hit=True,
                    device=device,
                    model=self._diarisation_model,
                    config={"num_speakers": speaker_count},
                    media_duration=media_duration_seconds,
                ),
            )
        assert diarisation_checkpoint is not None

        cached = cache.load_resumable(run_id, _RAW_TRANSCRIPT_NAME, RawTranscript)
        raw_valid = cached is not None and (
            cached.audio_sha256 == audio_sha256
            and cached.asr_model == self._asr_model
            and cached.diarisation_model == self._diarisation_model
            and cached.diarisation_device == device
            and cached.num_speakers == speaker_count
            and cached.asr_chunk_duration == self._asr_chunk_duration
            and cached.asr_chunk_overlap == self._asr_chunk_overlap
        )
        if raw_valid and not (asr_recomputed or diarisation_recomputed):
            assert cached is not None
            return cached.model_copy(update={"timings": cache.load_timings(run_id)})

        started = time.monotonic()
        started_at = datetime.now(UTC)
        raw = RawTranscript(
            clips=[
                SourceClip(
                    path=str(audio),
                    offset_seconds=0.0,
                    duration_seconds=(
                        media_duration_seconds
                        if media_duration_seconds is not None
                        else max(
                            _duration(asr_checkpoint.segments),
                            _duration(diarisation_checkpoint.turns),
                        )
                    ),
                )
            ],
            utterances=align(asr_checkpoint.segments, diarisation_checkpoint.turns),
            audio_sha256=audio_sha256,
            asr_model=self._asr_model,
            diarisation_model=self._diarisation_model,
            diarisation_device=device,
            num_speakers=speaker_count,
            asr_chunk_duration=self._asr_chunk_duration,
            asr_chunk_overlap=self._asr_chunk_overlap,
        )
        _record_timing(
            cache,
            run_id,
            _timing(
                "alignment/checkpoint promotion",
                started=started,
                started_at=started_at,
                model=self._diarisation_model,
                device=device,
                config={"num_speakers": speaker_count},
                media_duration=raw.clips[0].duration_seconds,
            ),
        )
        raw = raw.model_copy(update={"timings": cache.load_timings(run_id)})
        cache.store(run_id, _RAW_TRANSCRIPT_NAME, raw)
        cache.store_manifest(run_id, build_audio_stage_manifest(raw))
        return raw

    def _run_cached_asr(
        self,
        audio: Path,
        run_id: str,
        cache: RunCache,
        audio_sha256: str,
        *,
        media_duration_seconds: float | None,
    ) -> AsrCheckpoint:
        started = time.monotonic()
        started_at = datetime.now(UTC)
        try:
            segments = self._run_asr(audio)
            checkpoint = AsrCheckpoint(
                audio_sha256=audio_sha256,
                model=self._asr_model,
                chunk_duration=self._asr_chunk_duration,
                chunk_overlap=self._asr_chunk_overlap,
                segments=segments,
            )
            cache.store(run_id, "asr_checkpoint", checkpoint)
            return checkpoint
        finally:
            _record_timing(
                cache,
                run_id,
                _timing(
                    "ASR",
                    started=started,
                    started_at=started_at,
                    model=self._asr_model,
                    config={
                        "chunk_duration": self._asr_chunk_duration,
                        "chunk_overlap": self._asr_chunk_overlap,
                    },
                    media_duration=media_duration_seconds,
                ),
            )

    def _run_cached_diarisation(
        self,
        audio: Path,
        run_id: str,
        cache: RunCache,
        audio_sha256: str,
        *,
        device: Literal["cpu", "mps"],
        num_speakers: int | None,
        media_duration_seconds: float | None,
    ) -> DiarisationCheckpoint:
        started = time.monotonic()
        started_at = datetime.now(UTC)
        try:
            turns = self._run_diarisation(
                audio, device=device, num_speakers=num_speakers
            )
            checkpoint = DiarisationCheckpoint(
                audio_sha256=audio_sha256,
                model=self._diarisation_model,
                device=device,
                num_speakers=num_speakers,
                turns=turns,
            )
            cache.store(run_id, "diarisation_checkpoint", checkpoint)
            return checkpoint
        finally:
            _record_timing(
                cache,
                run_id,
                _timing(
                    "diarisation",
                    started=started,
                    started_at=started_at,
                    device=device,
                    model=self._diarisation_model,
                    config={"num_speakers": num_speakers},
                    media_duration=media_duration_seconds,
                ),
            )

    def _run_asr(self, audio: Path) -> list[AsrSegment]:
        # Fail-fast rather than grow unbounded — see the
        # DEFAULT_MPS_MEMORY_FRACTION comment for why this and the MPS cap
        # below exist alongside chunking and the process-level watchdog.
        # Uses the same resolved budget the watchdog does (see
        # _resolve_memory_budget_bytes) rather than the raw module
        # default, so a caller-supplied override governs both.
        mx.set_memory_limit(self._resolve_memory_budget_bytes())

        _log_stage("ASR starting")
        started = time.monotonic()
        model = parakeet_from_pretrained(self._asr_model)
        sample_rate = model.preprocessor_config.sample_rate

        def _log_chunk_progress(current_samples: int, total_samples: int) -> None:
            _log_stage(
                f"ASR progress: {current_samples / sample_rate:.0f}s / "
                f"{total_samples / sample_rate:.0f}s"
            )

        # Chunked: parakeet-mlx transparently windows long audio (bounded
        # memory) and stitches the result back into file-relative
        # timestamps with de-duplicated overlap text — see the
        # DEFAULT_ASR_CHUNK_DURATION comment above for why this is required
        # rather than optional. `chunk_callback` is parakeet-mlx's own
        # per-chunk progress hook (called with sample offsets before each
        # chunk is processed) — cheap, so wired straight to stage logging
        # rather than built up into anything more elaborate.
        try:
            result = model.transcribe(
                audio,
                chunk_duration=self._asr_chunk_duration,
                overlap_duration=self._asr_chunk_overlap,
                chunk_callback=_log_chunk_progress,
            )
        except RuntimeError as exc:
            raise AcceleratorOutOfMemoryError("ASR", exc) from exc
        segments = [
            AsrSegment(
                start=sentence.start, end=sentence.end, text=sentence.text.strip()
            )
            for sentence in result.sentences
        ]

        # Sequential peaks: parakeet's model + decoded audio and pyannote's
        # models + decoded waveform are each large enough on their own;
        # never needing both live at once is what keeps the process's peak
        # memory to one stage's cost instead of the sum. `del` drops our
        # only references before the diarisation pipeline loads;
        # `mx.clear_cache()` releases MLX's own buffer cache on top of
        # that (Python's refcounting alone doesn't touch it).
        del model, result
        gc.collect()
        mx.clear_cache()
        _log_stage(f"ASR complete in {time.monotonic() - started:.1f}s")
        return segments

    def _run_diarisation(
        self,
        audio: Path,
        *,
        device: Literal["cpu", "mps"],
        num_speakers: int | None,
    ) -> list[DiarisedTurn]:
        # See DEFAULT_MPS_MEMORY_FRACTION: torch's MPS allocator's own
        # fail-fast cap, set immediately before the stage that uses it.
        if device == "mps":
            try:
                torch.mps.set_per_process_memory_fraction(DEFAULT_MPS_MEMORY_FRACTION)
            except RuntimeError as exc:
                raise AcceleratorOutOfMemoryError("diarisation", exc) from exc

        _log_stage("diarisation starting")
        started = time.monotonic()
        # Unlike parakeet's raw Conformer encoder, pyannote's segmentation
        # and embedding models don't run one whole-file forward pass:
        # `pyannote.audio.core.inference.Inference` (which the diarisation
        # pipeline uses internally for both) always processes audio through
        # a fixed-duration `SlidingWindow` (`Inference.slide`/`__call__`),
        # so memory is bounded by the window size regardless of file
        # length — confirmed by reading that class, and empirically by
        # running this pipeline end-to-end against the same ~31-minute
        # meeting recording that broke parakeet (see task-004-report.md).
        try:
            pipeline: Any = Pipeline.from_pretrained(
                self._diarisation_model, token=self._hf_token
            )
            if pipeline is None:
                raise TranscriberError(
                    f"could not load diarisation pipeline {self._diarisation_model!r} "
                    "(pyannote returned no pipeline for this checkpoint)"
                )
            pipeline.to(torch.device(device))
        except RuntimeError as exc:
            raise AcceleratorOutOfMemoryError("diarisation", exc) from exc

        # A plain ProgressHook-compatible callable is intentionally used here:
        # pyannote invokes it for segmentation, embeddings, and clustering.
        progress_hook = _DiarisationProgressHook()
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
        kwargs: dict[str, object] = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if _accepts_hook(pipeline):
            kwargs["hook"] = progress_hook
        try:
            output: Any = pipeline(
                {"waveform": samples.data, "sample_rate": samples.sample_rate},
                **kwargs,
            )
        except RuntimeError as exc:
            raise AcceleratorOutOfMemoryError("diarisation", exc) from exc
        annotation: Any = output.speaker_diarization
        turns = [
            DiarisedTurn(start=turn.start, end=turn.end, speaker=str(speaker))
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        _log_stage(f"diarisation complete in {time.monotonic() - started:.1f}s")
        return turns


class _DiarisationProgressHook:
    """Rate-limited pyannote hook that keeps stderr useful and quiet."""

    def __init__(self) -> None:
        self._last_logged = 0.0
        self._last_step: str | None = None

    def __call__(
        self,
        step_name: str,
        step_artifact: Any,
        file: Any = None,
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        now = time.monotonic()
        finished = completed is not None and total is not None and completed >= total
        changed = step_name != self._last_step
        if not changed and not finished and now - self._last_logged < 1.0:
            return
        progress = (
            f" {completed}/{total}"
            if completed is not None and total is not None
            else ""
        )
        _log_stage(f"diarisation progress: {step_name}{progress}")
        self._last_logged = now
        self._last_step = step_name


def _accepts_hook(pipeline: Any) -> bool:
    try:
        parameters = inspect.signature(pipeline.__call__).parameters.values()
    except TypeError, ValueError:
        return True
    return any(
        parameter.name == "hook" or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def select_diarisation_device(requested: DiarisationDevice) -> Literal["cpu", "mps"]:
    """Resolve a requested device without silently falling back."""
    if requested == "cpu":
        return "cpu"
    mps_available = bool(torch.backends.mps.is_available())
    if requested == "mps" and not mps_available:
        raise TranscriberError(
            "diarisation device 'mps' was requested but MPS is unavailable"
        )
    if requested == "auto":
        return "mps" if mps_available else "cpu"
    return requested


def _duration(items: Sequence[Any]) -> float:
    return max((float(item.end) for item in items), default=0.0)


def _timing(
    stage: str,
    *,
    started: float,
    cache_hit: bool = False,
    device: str | None = None,
    model: str | None = None,
    config: dict[str, str | int | float | bool | None] | None = None,
    media_duration: float | None = None,
    started_at: datetime | None = None,
) -> StageTiming:
    ended = time.monotonic()
    elapsed = max(0.0, ended - started)
    return StageTiming(
        stage=stage,
        started_at=started_at or datetime.now(UTC),
        ended_at=datetime.now(UTC),
        elapsed_seconds=elapsed,
        cache_hit=cache_hit,
        device=device,
        model=model,
        config=config or {},
        media_duration_seconds=media_duration,
        rtf=(elapsed / media_duration if media_duration else None),
        api_rate_cost=0.0,
    )


def _record_timing(cache: RunCache, run_id: str, timing: StageTiming) -> None:
    cache.record_timing(run_id, timing)


def _log_stage(message: str) -> None:
    """Flushed stderr line for one pipeline-stage transition.

    A run that dies mid-stage (the original incident's stderr log was
    completely empty) should still leave a trail on disk showing which
    stage it reached — hence the unconditional flush rather than relying
    on Python's default buffering to get there eventually.
    """
    print(f"[transcribe] {message}", file=sys.stderr, flush=True)


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
    num_speakers: int | None = None,
    media_duration_seconds: float | None = None,
) -> RawTranscript:
    """Run local stages from independent checkpoints, with safe legacy handling."""
    if isinstance(transcriber, LocalTranscriber):
        return transcriber.transcribe_cached(
            audio,
            run_id=run_id,
            cache=cache,
            num_speakers=num_speakers,
            media_duration_seconds=media_duration_seconds,
        )

    audio_sha256 = sha256_of(audio)
    cached = cache.load_resumable(run_id, _RAW_TRANSCRIPT_NAME, RawTranscript)
    if cached is not None and (
        cached.audio_sha256 == audio_sha256
        and cached.asr_model is not None
        and cached.diarisation_model is not None
        and cached.diarisation_device is not None
    ):
        return cached.model_copy(update={"timings": cache.load_timings(run_id)})

    transcribe: Any = transcriber.transcribe
    parameters = inspect.signature(transcribe).parameters.values()
    accepts_speaker_count = any(
        parameter.name == "num_speakers"
        or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    result = (
        transcribe(audio, num_speakers=num_speakers)
        if accepts_speaker_count
        else transcribe(audio)
    )
    cache.store(run_id, _RAW_TRANSCRIPT_NAME, result)
    if result.audio_sha256 and all(
        value is not None
        for value in (
            result.asr_model,
            result.diarisation_model,
            result.diarisation_device,
            result.asr_chunk_duration,
            result.asr_chunk_overlap,
        )
    ):
        cache.store_manifest(run_id, build_audio_stage_manifest(result))
    return result
