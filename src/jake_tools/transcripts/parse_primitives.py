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
_SPEAKER_TEXT_RE = re.compile(r"^(?P<speaker>[^:]{1,120}):\s*(?P<text>.+)$")
_TRANSCRIPT_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+transcript\b", re.IGNORECASE)
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
    start_index = 0
    for index, line in enumerate(lines):
        if _TRANSCRIPT_HEADING_RE.match(line):
            start_index = index + 1
            break

    first_turn_index: int | None = None
    for index in range(start_index, len(lines)):
        if _parse_timestamp_line(lines[index]) is not None:
            first_turn_index = index
            break
    if first_turn_index is None:
        for index, line in enumerate(lines):
            if _parse_timestamp_line(line) is not None:
                first_turn_index = index
                break
    if first_turn_index is None:
        raise ParsePrimitiveError(
            f"No Gemini transcript timestamp lines found in {source_text_path}"
        )

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
                    source_ref=(
                        f"{source_text_path}:lines {current.line_start}-{current.line_end}"
                    ),
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
