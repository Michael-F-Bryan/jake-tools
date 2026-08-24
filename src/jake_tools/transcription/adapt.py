"""Adapt a pre-diarised text transcript (Gemini/Teams export) into a raw transcript.

Some meetings arrive as text, not audio: Google Meet can hand over a
Gemini-produced diarised transcript document; Teams exports a `.vtt` (or
`.docx`, out of scope here — the coordinating agent hands this module a
plain-text/VTT file) with speaker cues. Either way, the pipeline must enter
its normal flow at the same seam ASR+diarisation feeds:
:class:`~.models.RawTranscript`.

Two deterministic parsers are tried first, in order:

* :func:`parse_vtt` — WebVTT cue timings plus `<v Name>` voice tags.
* :func:`parse_named_lines` — `Name: text` / `Name (00:12:34): text` plain
  text.

Neither one rewrites a single word of the source; they only recognise
structure that is already there. Anything neither parser accepts falls back
to :func:`adapt_transcript`'s LLM path, which asks the model to *restructure*
the document into utterances — preserving wording exactly, only segmenting
and labelling. That instruction is load-bearing: this adapter feeds the
factual record the rest of the pipeline (polish, minutes, integrate) treats
as ground truth, and a paraphrasing adapter would poison it silently. A
reviewer should scrutinise the fallback prompt for exactly this.

Speaker labels pass through verbatim in every path: a real name stays a real
name, "Speaker 1" stays "Speaker 1" for plan 006's speaker resolution to
deal with. Timestamps absent from the source become zero (plain text with no
per-line stamps) or the model's best effort (LLM fallback) — polish and
chapterise tolerate coarse timestamps from text sources.

Every parser (deterministic or LLM) returns utterances sorted by
non-decreasing `start` (Python's stable sort keeps original document order
for ties), matching the ordering contract `RawTranscript.utterances`
documents in `models.py`.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

from pydantic import BaseModel

from ..claude import ClaudeAgent
from ..prompting import StructuredPrompt
from .cache import RunCache
from .models import RawTranscript, Utterance

_UNKNOWN_SPEAKER = "Unknown"

# --- WebVTT -----------------------------------------------------------------

_VTT_TIMESTAMP = r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})"
_CUE_TIMING_RE = re.compile(rf"^\s*{_VTT_TIMESTAMP}\s*-->\s*{_VTT_TIMESTAMP}")
_VOICE_TAG_RE = re.compile(r"<v(?:\.[\w-]+)*\s+([^>]+)>")
_TAG_RE = re.compile(r"<[^>]+>")


def _vtt_seconds(hours: str | None, minutes: str, seconds: str, millis: str) -> float:
    return (
        (int(hours) if hours else 0) * 3600
        + int(minutes) * 60
        + int(seconds)
        + (int(millis) / 1000)
    )


def _cue_speaker_and_text(payload: str) -> tuple[str, str]:
    """Pull the `<v Name>` speaker (if any) and the cue's verbatim text.

    Only the first voice tag in a cue is honoured — real Teams/Gemini VTT
    exports put one speaker per cue. Any tag is stripped from the text; the
    words themselves are never altered.
    """
    match = _VOICE_TAG_RE.search(payload)
    if match is None:
        return _UNKNOWN_SPEAKER, _TAG_RE.sub("", payload).strip()
    speaker = match.group(1).strip()
    remainder = payload[match.end() :]
    remainder = re.sub(r"</v>\s*$", "", remainder.strip())
    return speaker, _TAG_RE.sub("", remainder).strip()


def parse_vtt(text: str) -> list[Utterance] | None:
    """Parse a WebVTT transcript into utterances, or `None` if `text` isn't VTT.

    Deterministic: cue timings and `<v Name>` voice tags are read verbatim,
    nothing is inferred or rewritten. A cue with no voice tag keeps its
    recognised text under `_UNKNOWN_SPEAKER` rather than being dropped —
    the same "never lose recognised words" choice `asr.align` makes for
    diarisation-less ASR segments.
    """
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalised.lstrip("﻿").lstrip().startswith("WEBVTT"):
        return None

    utterances: list[Utterance] = []
    for block in re.split(r"\n{2,}", normalised):
        lines = [line for line in block.splitlines() if line.strip()]
        timing_index = next(
            (i for i, line in enumerate(lines) if _CUE_TIMING_RE.match(line)), None
        )
        if timing_index is None:
            continue  # the WEBVTT header, or a NOTE/STYLE block, or blank

        match = _CUE_TIMING_RE.match(lines[timing_index])
        assert match is not None
        start = _vtt_seconds(*match.group(1, 2, 3, 4))
        end = _vtt_seconds(*match.group(5, 6, 7, 8))
        payload = " ".join(lines[timing_index + 1 :]).strip()
        if not payload:
            continue
        speaker, cue_text = _cue_speaker_and_text(payload)
        if not cue_text:
            continue
        utterances.append(
            Utterance(start=start, end=end, speaker=speaker, text=cue_text)
        )

    utterances.sort(key=lambda u: (u.start, u.end))
    return utterances


# --- name-prefixed plain text ------------------------------------------------

# "Jane Doe: text" or "Jane Doe (00:12:34): text". The speaker group is
# deliberately restrictive (letters/space/apostrophe/period/hyphen only, and
# must start with a letter) so ordinary "Key: value" lines in an unrelated
# document don't masquerade as a speaker turn.
_NAME_LINE_RE = re.compile(
    r"^(?P<speaker>[A-Za-z][A-Za-z'.\- ]{0,49}?)"
    r"(?:\s*\((?P<timestamp>\d{1,2}(?::\d{2}){1,2})\))?"
    r":\s+(?P<text>\S.*)$"
)


def _clock_to_seconds(raw: str) -> float:
    parts = [int(part) for part in raw.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes * 60 + seconds)
    hours, minutes, seconds = parts
    return float(hours * 3600 + minutes * 60 + seconds)


def parse_named_lines(text: str) -> list[Utterance] | None:
    """Parse `Name: text` / `Name (00:12:34): text` plain text into utterances.

    A line that doesn't match the speaker-prefix pattern is treated as a
    continuation of the previous utterance (wrapped paragraphs), *unless* it
    is the very first non-blank line, in which case the whole document is
    rejected (`None`) — this parser only claims documents that clearly open
    in this shape, leaving anything ambiguous to the LLM fallback. A missing
    per-line timestamp becomes `0.0` rather than a guess.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    utterances: list[Utterance] = []
    current: Utterance | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        match = _NAME_LINE_RE.match(line)
        if match is None:
            if current is None:
                return None
            current.text = f"{current.text} {line}".strip()
            continue

        timestamp = match.group("timestamp")
        start = _clock_to_seconds(timestamp) if timestamp else 0.0
        current = Utterance(
            start=start,
            end=start,
            speaker=match.group("speaker").strip(),
            text=match.group("text").strip(),
        )
        utterances.append(current)

    if not utterances:
        return None
    utterances.sort(key=lambda u: (u.start, u.end))
    return utterances


# --- LLM fallback -------------------------------------------------------------


class AdaptedUtterance(BaseModel):
    """One utterance as the LLM fallback reports it — mirrors `Utterance`."""

    start: float
    end: float
    speaker: str
    text: str


class AdaptedTranscript(BaseModel):
    """The LLM fallback's structured reply: the whole document, segmented."""

    utterances: list[AdaptedUtterance]


class AdaptTranscriptPrompt(StructuredPrompt[AdaptedTranscript]):
    template = textwrap.dedent("""\
        You are converting a pre-diarised meeting transcript into structured
        utterances. The document below already has speakers, and maybe
        timestamps, laid out in some ad-hoc textual shape that could not be
        parsed with a fixed pattern.

        Your job is to RESTRUCTURE the document, not rewrite it:

        - Preserve every speaker's wording EXACTLY as written in the source.
          Do not paraphrase, summarise, correct grammar or spelling, or drop
          filler words — copy each turn's words verbatim into its `text`
          field.
        - Split the document into utterances: one per contiguous turn of
          speech from a single speaker.
        - Use the speaker label already present in the source, verbatim
          (e.g. "Jane Doe" stays "Jane Doe"; "Speaker 1" stays "Speaker 1").
          Never invent a name or resolve one speaker label to another.
        - If the source gives timestamps, convert them to seconds, relative
          to the start of the meeting (the first moment is 0.0). If a turn
          has no timestamp, estimate one from the surrounding cues if you
          reasonably can, or use 0.0 if there is no timing information
          anywhere in the document.
        - List utterances in the order they occur in the document, with
          non-decreasing `start` times.

        Document:
        {{ document }}
        """)
    response_model = AdaptedTranscript

    document: str


class AdaptError(RuntimeError):
    """Raised when `adapt_transcript` cannot produce a `RawTranscript`."""


class NoFallbackAgentError(AdaptError):
    """Raised when neither deterministic parser matched and no agent was given."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"{str(path)!r} looks like neither WebVTT nor name-prefixed plain "
            "text, and no agent was supplied for the LLM fallback."
        )


async def _adapt_via_llm(document: str, agent: ClaudeAgent) -> list[Utterance]:
    adapted, _reply = await agent.run_structured(
        AdaptTranscriptPrompt(document=document), stage="adapt"
    )
    utterances = [
        Utterance(start=item.start, end=item.end, speaker=item.speaker, text=item.text)
        for item in adapted.utterances
    ]
    utterances.sort(key=lambda u: (u.start, u.end))
    return utterances


# --- entry point --------------------------------------------------------------


async def adapt_transcript(
    path: Path, *, agent: ClaudeAgent | None = None
) -> RawTranscript:
    """Adapt a pre-diarised transcript file into a `RawTranscript`.

    Tries :func:`parse_vtt`, then :func:`parse_named_lines`, in that order —
    both deterministic, both preferred over the LLM fallback whenever they
    match. Only when neither recognises the document's shape does this fall
    back to an LLM call via `agent` (raising :class:`NoFallbackAgentError` if
    none was supplied). `clips` is always empty and `audio_sha256` always
    `None`: this is a text source, not an audio one (see `models.py`).
    """
    text = path.read_text(encoding="utf-8")

    utterances = parse_vtt(text)
    if utterances is None:
        utterances = parse_named_lines(text)
    if utterances is None:
        if agent is None:
            raise NoFallbackAgentError(path)
        utterances = await _adapt_via_llm(text, agent)

    return RawTranscript(clips=[], utterances=utterances, audio_sha256=None)


async def run_adapt(
    path: Path, *, run_id: str, agent: ClaudeAgent, cache: RunCache
) -> RawTranscript:
    """Adapt and cache a text ramp only when its content/config manifest matches."""
    document = path.read_text(encoding="utf-8")
    stage_agent = agent.for_stage("adapt").with_telemetry(cache.telemetry_sink(run_id))
    manifest = cache.stage_manifest(
        "adapt",
        inputs={"source_text": document},
        config={
            "agent": stage_agent.defaults.model_dump(mode="json"),
            "prompt": AdaptTranscriptPrompt.template,
            "response_schema": AdaptedTranscript.model_json_schema(),
        },
    )
    cached = cache.load_resumable(run_id, "raw_transcript", RawTranscript)
    if cached is not None and cache.load_manifest(run_id, "adapt") == manifest:
        stage_agent.record_cache_hit()
        return cached
    if cached is not None or cache.load_manifest(run_id, "adapt") is not None:
        cache.invalidate_downstream(
            run_id, reason="adapt input or configuration changed"
        )
    adapted = await adapt_transcript(path, agent=stage_agent)
    cache.store(run_id, "raw_transcript", adapted)
    cache.store_manifest(run_id, manifest)
    return adapted
