"""The raw transcript as stable, numbered words, and how it is chunked.

A word's ID is its position in the transcript's segments, flattened in
order: the same source bytes always give the same IDs, whatever window is
polished. Chunks own disjoint, contiguous ID ranges that together cover the
window exactly once; neighbouring words are shown to a model only as
read-only context.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from ..transcription import Transcript, TranscriptionResult
from .lexicon import ends_sentence, is_filler


class SourceError(ValueError):
    """The transcript or the requested window cannot be polished."""

    def __init__(self, argument: str, message: str) -> None:
        super().__init__(message)
        self.argument = argument


class SourceWord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    text: str
    start_seconds: float
    end_seconds: float
    confidence: float
    speaker: str | None
    overlap: bool
    segment: int
    sentence_initial: bool
    """First word of its segment, or after a word ending a sentence."""


class LoadedSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    sha256: str
    bytes: int
    transcript: Transcript
    words: tuple[SourceWord, ...]


def load_source(path: Path) -> LoadedSource:
    """Read ``path`` once: a ``transcribe`` result, or its bare ``transcript``."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SourceError(
            "transcript_path", f"cannot read {path}: {exc.strerror}"
        ) from exc
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise SourceError("transcript_path", f"{path} is not JSON") from exc
    try:
        if isinstance(payload, dict) and "transcript" in payload:
            transcript = TranscriptionResult.model_validate(payload).transcript
        else:
            transcript = Transcript.model_validate(payload)
    except ValidationError as exc:
        raise SourceError(
            "transcript_path",
            f"{path} is not a transcribe result: {exc.errors()[0]['msg']}",
        ) from exc
    words: list[SourceWord] = []
    previous: str | None = None
    for index, segment in enumerate(transcript.segments):
        for position, word in enumerate(segment.words):
            words.append(
                SourceWord(
                    id=len(words),
                    text=word.text,
                    start_seconds=word.start_seconds,
                    end_seconds=word.end_seconds,
                    confidence=word.confidence,
                    speaker=word.speaker,
                    overlap=word.overlap,
                    segment=index,
                    sentence_initial=position == 0
                    or previous is None
                    or ends_sentence(previous),
                )
            )
            previous = word.text
    return LoadedSource(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        bytes=len(raw),
        transcript=transcript,
        words=tuple(words),
    )


class Window(BaseModel):
    """The contiguous word range being polished."""

    model_config = ConfigDict(frozen=True)

    first_word: int
    last_word: int
    start_seconds: float | None = None
    end_seconds: float | None = None

    def ids(self) -> range:
        return range(self.first_word, self.last_word + 1)


def select_window(
    words: tuple[SourceWord, ...],
    start_seconds: float | None,
    end_seconds: float | None,
) -> Window:
    """From the first word starting at or after ``start_seconds`` to the last
    starting before ``end_seconds``; everything when both are ``None``."""
    if not words:
        raise SourceError("transcript_path", "the transcript has no words")
    if (
        start_seconds is not None
        and end_seconds is not None
        and end_seconds <= start_seconds
    ):
        raise SourceError("end_seconds", "end_seconds must be after start_seconds")
    inside = [
        word.id
        for word in words
        if (start_seconds is None or word.start_seconds >= start_seconds)
        and (end_seconds is None or word.start_seconds < end_seconds)
    ]
    if not inside:
        raise SourceError("start_seconds", "no words start inside the window")
    return Window(
        first_word=min(inside),
        last_word=max(inside),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


class Chunk(BaseModel):
    """One model call's editable words, plus read-only context around them."""

    model_config = ConfigDict(frozen=True)

    index: int
    first_word: int
    last_word: int
    context_first: int
    context_last: int

    def ids(self) -> range:
        return range(self.first_word, self.last_word + 1)

    def owns(self, word_id: int) -> bool:
        return self.first_word <= word_id <= self.last_word

    @property
    def size(self) -> int:
        return self.last_word - self.first_word + 1


TARGET_CHUNK_WORDS = 900
MAX_CHUNK_WORDS = 1_200
CONTEXT_BEFORE_WORDS = 40
CONTEXT_AFTER_WORDS = 25


def turns(words: tuple[SourceWord, ...], window: Window) -> list[range]:
    """Maximal runs of one speaker label (``None`` included) in ``window``."""
    runs: list[range] = []
    start = window.first_word
    for word_id in range(window.first_word + 1, window.last_word + 2):
        if (
            word_id > window.last_word
            or words[word_id].speaker != words[word_id - 1].speaker
        ):
            runs.append(range(start, word_id))
            start = word_id
    return runs


def _split_long(words: tuple[SourceWord, ...], run: range) -> list[range]:
    """Cut a turn longer than a chunk at sentence ends (else at the limit)."""
    pieces: list[range] = []
    start = run.start
    while run.stop - start > MAX_CHUNK_WORDS:
        limit = start + MAX_CHUNK_WORDS
        cut = next(
            (
                i + 1
                for i in range(limit - 1, start + TARGET_CHUNK_WORDS // 2, -1)
                if ends_sentence(words[i].text) and not is_filler(words[i].text)
            ),
            limit,
        )
        pieces.append(range(start, cut))
        start = cut
    pieces.append(range(start, run.stop))
    return pieces


def plan_chunks(words: tuple[SourceWord, ...], window: Window) -> list[Chunk]:
    """Pack whole speaker turns into chunks of about ``TARGET_CHUNK_WORDS``."""
    pieces = [
        piece for run in turns(words, window) for piece in _split_long(words, run)
    ]
    groups: list[range] = []
    start = pieces[0].start
    stop = start
    for piece in pieces:
        if stop - start >= TARGET_CHUNK_WORDS or (
            stop > start and piece.stop - start > MAX_CHUNK_WORDS
        ):
            groups.append(range(start, stop))
            start = piece.start
        stop = piece.stop
    groups.append(range(start, stop))
    if len(groups) > 1 and len(groups[-1]) < TARGET_CHUNK_WORDS // 4:
        # A short tail rides with its predecessor rather than paying for a call.
        tail = groups.pop()
        if tail.stop - groups[-1].start <= MAX_CHUNK_WORDS:
            groups[-1] = range(groups[-1].start, tail.stop)
        else:
            groups.append(tail)
    chunks = [
        Chunk(
            index=index,
            first_word=group.start,
            last_word=group.stop - 1,
            context_first=max(window.first_word, group.start - CONTEXT_BEFORE_WORDS),
            context_last=min(window.last_word, group.stop - 1 + CONTEXT_AFTER_WORDS),
        )
        for index, group in enumerate(groups, start=1)
    ]
    owned = [word_id for chunk in chunks for word_id in chunk.ids()]
    assert owned == list(window.ids()), "chunks must own every window word once"
    return chunks
