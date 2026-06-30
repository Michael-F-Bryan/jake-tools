from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .models import (
    SourceArtifact,
    TranscriptArtifact,
    TranscriptSourceRef,
    TranscriptTurn,
)

_TIMESTAMP_LINE_RE = re.compile(
    r"^\s*(?:\[(?P<bracketed>\d{1,2}:\d{2}(?::\d{2})?)\]|(?P<bare>\d{1,2}:\d{2}(?::\d{2})?))\s*(?:[-–—]\s*)?(?P<rest>.+?)\s*$"
)
_STANDALONE_TIMESTAMP_RE = re.compile(
    r"^\s*(?P<timestamp>\d{1,2}:\d{2}(?::\d{2})?)\s*$"
)
_SPEAKER_TEXT_RE = re.compile(r"^(?P<speaker>[^:]{1,120}):\s*(?P<text>.+)$")
_TRANSCRIPT_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+transcript\b", re.IGNORECASE)
_TRANSCRIPT_MARKER_RE = re.compile(r"\btranscript\b", re.IGNORECASE)
_TRANSCRIPTION_ENDED_RE = re.compile(r"^\s*Transcription ended after\b", re.IGNORECASE)
_ANY_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")


class ParsePrimitiveError(ValueError):
    pass


@dataclass
class _GeminiTurnBuilder:
    start_seconds: float
    speaker: str
    text_lines: list[str]
    line_start: int
    line_end: int

    def finish(self, *, end_seconds: float | None = None) -> TranscriptTurn:
        end = (
            self.start_seconds
            if end_seconds is None
            else max(end_seconds, self.start_seconds)
        )
        text = " ".join(
            part.strip() for part in self.text_lines if part.strip()
        ).strip()
        return TranscriptTurn(
            start=self.start_seconds,
            end=end,
            speaker=self.speaker,
            text=text,
        )


def _timestamp_to_seconds(timestamp: str) -> float:
    parts = [int(part) for part in timestamp.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes * 60 + seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours * 3600 + minutes * 60 + seconds)
    raise ParsePrimitiveError(f"Unsupported timestamp format: {timestamp!r}")


def _parse_timestamp_line(line: str) -> tuple[float, str, str] | None:
    timestamp_match = _TIMESTAMP_LINE_RE.match(line)
    if timestamp_match is None:
        return None

    rest = timestamp_match.group("rest").strip()
    speaker_match = _SPEAKER_TEXT_RE.match(rest)
    if speaker_match is None:
        return None

    timestamp = timestamp_match.group("bracketed") or timestamp_match.group("bare")
    if timestamp is None:
        return None
    return (
        _timestamp_to_seconds(timestamp),
        speaker_match.group("speaker").strip(),
        speaker_match.group("text").strip(),
    )


def _parse_standalone_timestamp(line: str) -> float | None:
    match = _STANDALONE_TIMESTAMP_RE.match(line)
    if match is None:
        return None
    return _timestamp_to_seconds(match.group("timestamp"))


def _transcript_start_index(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        if _TRANSCRIPT_HEADING_RE.match(line) or _TRANSCRIPT_MARKER_RE.search(line):
            return index + 1
    return 0


def _resolve_source_text_path(source: SourceArtifact) -> Path:
    if source.raw_text_path is not None:
        return source.raw_text_path
    if source.source_path is not None:
        return source.source_path
    raise ParsePrimitiveError(
        "SourceArtifact is missing both raw_text_path and source_path for Gemini parse."
    )


def parse_gemini_transcript(source: SourceArtifact) -> TranscriptArtifact:
    source_text_path = _resolve_source_text_path(source)
    if not source_text_path.exists():
        raise ParsePrimitiveError(
            f"Source text path does not exist: {source_text_path}"
        )

    lines = source_text_path.read_text(encoding="utf-8").splitlines()
    start_index = _transcript_start_index(lines)

    inline_artifact = _parse_inline_gemini_transcript(
        lines,
        source_text_path=source_text_path,
        start_index=start_index,
    )
    if inline_artifact.turns:
        return inline_artifact

    standalone_artifact = _parse_standalone_gemini_transcript(
        lines,
        source_text_path=source_text_path,
        start_index=start_index,
    )
    if standalone_artifact.turns:
        return standalone_artifact

    raise ParsePrimitiveError(
        f"No Gemini transcript timestamp lines found in {source_text_path}"
    )


def _parse_inline_gemini_transcript(
    lines: list[str], *, source_text_path: Path, start_index: int
) -> TranscriptArtifact:
    first_turn_index: int | None = None
    for index in range(start_index, len(lines)):
        if _parse_timestamp_line(lines[index]) is not None:
            first_turn_index = index
            break
    if first_turn_index is None:
        return TranscriptArtifact(turns=[], source_refs=[])

    turns: list[TranscriptTurn] = []
    source_refs: list[TranscriptSourceRef] = []
    current: _GeminiTurnBuilder | None = None

    def flush_current(next_start: float | None = None) -> None:
        nonlocal current
        if current is None:
            return
        turn = current.finish(end_seconds=next_start)
        if turn.text:
            turns.append(turn)
            source_refs.append(
                TranscriptSourceRef(
                    turn_index=len(turns) - 1,
                    source_ref=f"{source_text_path}:lines {current.line_start}-{current.line_end}",
                )
            )
        current = None

    for index in range(first_turn_index, len(lines)):
        line_number = index + 1
        line = lines[index]

        if _ANY_HEADING_RE.match(line) and current is not None:
            flush_current()
            break

        parsed = _parse_timestamp_line(line)
        if parsed is not None:
            start_seconds, speaker, text = parsed
            flush_current(next_start=start_seconds)
            current = _GeminiTurnBuilder(
                start_seconds=start_seconds,
                speaker=speaker,
                text_lines=[text],
                line_start=line_number,
                line_end=line_number,
            )
            continue

        if current is None:
            continue
        continuation = line.strip()
        if not continuation:
            continue
        current.text_lines.append(continuation)
        current.line_end = line_number

    flush_current()
    return TranscriptArtifact(turns=turns, source_refs=source_refs)


@dataclass
class _StandaloneBlock:
    start_seconds: float
    lines: list[tuple[int, str]]


def _parse_standalone_gemini_transcript(
    lines: list[str], *, source_text_path: Path, start_index: int
) -> TranscriptArtifact:
    blocks: list[_StandaloneBlock] = []
    current: _StandaloneBlock | None = None

    for index in range(start_index, len(lines)):
        line_number = index + 1
        line = lines[index]
        if _TRANSCRIPTION_ENDED_RE.match(line):
            if current is not None:
                blocks.append(current)
            break

        timestamp = _parse_standalone_timestamp(line)
        if timestamp is not None:
            if current is not None:
                blocks.append(current)
            current = _StandaloneBlock(start_seconds=timestamp, lines=[])
            continue

        if current is not None:
            current.lines.append((line_number, line.rstrip()))
    else:
        if current is not None:
            blocks.append(current)

    turns: list[TranscriptTurn] = []
    source_refs: list[TranscriptSourceRef] = []
    for block_index, block in enumerate(blocks):
        next_start = (
            blocks[block_index + 1].start_seconds
            if block_index + 1 < len(blocks)
            else block.start_seconds
        )
        entries = _speaker_entries_from_block(block.lines)
        if not entries:
            continue
        span = max(0.0, next_start - block.start_seconds)
        for entry_index, entry in enumerate(entries):
            start = block.start_seconds + (span * entry_index / len(entries))
            end = block.start_seconds + (span * (entry_index + 1) / len(entries))
            if end <= start:
                end = start
            text = " ".join(part.strip() for part in entry.text_lines).strip()
            turns.append(
                TranscriptTurn(
                    start=round(start, 3),
                    end=round(end, 3),
                    speaker=entry.speaker,
                    text=text,
                )
            )
            source_refs.append(
                TranscriptSourceRef(
                    turn_index=len(turns) - 1,
                    source_ref=f"{source_text_path}:lines {entry.line_start}-{entry.line_end}",
                )
            )

    warnings = []
    if turns:
        warnings.append(
            "Parsed Gemini transcript from standalone timestamp blocks; turn times inside each block are interpolated."
        )
    return TranscriptArtifact(turns=turns, source_refs=source_refs, warnings=warnings)


def _speaker_entries_from_block(
    lines: list[tuple[int, str]],
) -> list[_GeminiTurnBuilder]:
    entries: list[_GeminiTurnBuilder] = []
    current: _GeminiTurnBuilder | None = None
    for line_number, line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        speaker_match = _SPEAKER_TEXT_RE.match(stripped)
        if speaker_match is not None:
            if current is not None:
                entries.append(current)
            current = _GeminiTurnBuilder(
                start_seconds=0.0,
                speaker=speaker_match.group("speaker").strip(),
                text_lines=[speaker_match.group("text").strip()],
                line_start=line_number,
                line_end=line_number,
            )
            continue
        if current is not None:
            current.text_lines.append(stripped)
            current.line_end = line_number
    if current is not None:
        entries.append(current)
    return entries


def parse_scribe_transcript(scribe_json_path: Path) -> TranscriptArtifact:
    payload = json.loads(scribe_json_path.read_text(encoding="utf-8"))
    segments = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(segments, list):
        raise ParsePrimitiveError("Transcript JSON does not contain a segments list")

    turns: list[TranscriptTurn] = []
    source_refs: list[TranscriptSourceRef] = []
    for segment_index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        speaker = (
            segment.get("speaker")
            or segment.get("speaker_label")
            or segment.get("speaker_name")
            or f"Speaker {segment_index}"
        )
        turns.append(
            TranscriptTurn(
                start=float(segment.get("start", 0.0) or 0.0),
                end=float(segment.get("end", segment.get("start", 0.0)) or 0.0),
                speaker=str(speaker),
                text=text,
            )
        )
        source_refs.append(
            TranscriptSourceRef(
                turn_index=len(turns) - 1,
                source_ref=f"{scribe_json_path}:segments[{segment_index - 1}]",
            )
        )

    return TranscriptArtifact(turns=turns, source_refs=source_refs)
