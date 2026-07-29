from __future__ import annotations

import json
import re
from pathlib import Path

from .errors import TranscriptError
from .models import (
    SourceArtifact,
    TranscriptArtifact,
    TranscriptSourceRef,
    TranscriptTurn,
)

_VTT_TIMESTAMP_RE = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}(?::\d{2})?\.\d{3})\s+-->\s+(?P<end>\d{1,2}:\d{2}(?::\d{2})?\.\d{3})"
)
_VTT_VOICE_RE = re.compile(
    r"<v\s+(?P<speaker>[^>]+)>\s*(?P<text>.*?)\s*</v>",
    re.IGNORECASE | re.DOTALL,
)
_VTT_TAG_RE = re.compile(r"<[^>]+>")
_NON_SPEECH_RE = re.compile(r"^(?:\[[^\]]+\]|\([^)]+\))$", re.IGNORECASE)

UNKNOWN_SPEAKER = "Unknown speaker"


class ParsePrimitiveError(TranscriptError):
    pass


def _timestamp_to_seconds(timestamp: str) -> float:
    parts = [int(part) for part in timestamp.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes * 60 + seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours * 3600 + minutes * 60 + seconds)
    raise ParsePrimitiveError(f"Unsupported timestamp format: {timestamp!r}")


def _vtt_timestamp_to_seconds(timestamp: str) -> float:
    seconds_part, milliseconds_part = timestamp.rsplit(".", maxsplit=1)
    return _timestamp_to_seconds(seconds_part) + (int(milliseconds_part) / 1000)


def _strip_vtt_tags(text: str) -> str:
    return _VTT_TAG_RE.sub("", text).replace("\n", " ").strip()


def _parse_vtt_voice_text(lines: list[str]) -> tuple[str, str]:
    body = "\n".join(lines).strip()
    voice_match = _VTT_VOICE_RE.search(body)
    if voice_match is None:
        return UNKNOWN_SPEAKER, _strip_vtt_tags(body)
    return (
        voice_match.group("speaker").strip(),
        _strip_vtt_tags(voice_match.group("text")),
    )


def _resolve_source_text_path(source: SourceArtifact) -> Path:
    if source.raw_text_path is not None:
        return source.raw_text_path
    if source.source_path is not None:
        return source.source_path
    raise ParsePrimitiveError(
        "SourceArtifact is missing both raw_text_path and source_path."
    )


def parse_teams_vtt(source: SourceArtifact) -> TranscriptArtifact:
    vtt_path = _resolve_source_text_path(source)
    if not vtt_path.exists():
        raise ParsePrimitiveError(f"Teams VTT path does not exist: {vtt_path}")

    lines = vtt_path.read_text(encoding="utf-8-sig").splitlines()
    turns: list[TranscriptTurn] = []
    source_refs: list[TranscriptSourceRef] = []
    warnings: list[str] = []

    index = 0
    while index < len(lines):
        timestamp_match = _VTT_TIMESTAMP_RE.match(lines[index])
        if timestamp_match is None:
            index += 1
            continue

        start_line = index + 1
        start = _vtt_timestamp_to_seconds(timestamp_match.group("start"))
        end = _vtt_timestamp_to_seconds(timestamp_match.group("end"))
        cue_lines: list[str] = []
        index += 1
        while index < len(lines) and lines[index].strip():
            cue_lines.append(lines[index])
            index += 1

        speaker, text = _parse_vtt_voice_text(cue_lines)
        if not text:
            warnings.append(f"Skipped empty Teams VTT cue at line {start_line}.")
            continue
        turns.append(
            TranscriptTurn(
                start=round(start, 3),
                end=round(max(end, start), 3),
                speaker=speaker,
                text=text,
            )
        )
        source_refs.append(
            TranscriptSourceRef(
                turn_index=len(turns) - 1,
                source_ref=f"{vtt_path}:cue line {start_line}",
            )
        )

    if not turns:
        raise ParsePrimitiveError(f"No Teams VTT cues found in {vtt_path}")
    return TranscriptArtifact(turns=turns, source_refs=source_refs, warnings=warnings)


def parse_youtube_json3(source: SourceArtifact) -> TranscriptArtifact:
    caption_path = _resolve_source_text_path(source)
    if not caption_path.exists():
        raise ParsePrimitiveError(
            f"YouTube caption path does not exist: {caption_path}"
        )
    try:
        payload = json.loads(caption_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ParsePrimitiveError(
            f"YouTube captions are not valid JSON3: {caption_path}"
        ) from exc
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        raise ParsePrimitiveError(
            "YouTube JSON3 captions do not contain an events list"
        )

    turns: list[TranscriptTurn] = []
    source_refs: list[TranscriptSourceRef] = []
    malformed_events = 0
    # YouTube JSON3 captions do not distinguish speakers; every turn from a
    # single caption track is attributed to this placeholder label.
    speaker = "Speaker"
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            malformed_events += 1
            continue
        segments = event.get("segs")
        if not isinstance(segments, list):
            continue
        text = "".join(
            str(segment.get("utf8", ""))
            for segment in segments
            if isinstance(segment, dict)
        )
        text = re.sub(r"\s+", " ", text).strip()
        if not text or _NON_SPEECH_RE.match(text):
            continue
        try:
            start = float(event.get("tStartMs", 0)) / 1000.0
            duration = float(event.get("dDurationMs", 0)) / 1000.0
        except TypeError, ValueError:
            malformed_events += 1
            continue
        if (
            turns
            and turns[-1].text.casefold() == text.casefold()
            and start < turns[-1].end
        ):
            continue
        turns.append(
            TranscriptTurn(
                start=round(start, 3),
                end=round(max(start, start + duration), 3),
                speaker=speaker,
                text=text,
            )
        )
        source_refs.append(
            TranscriptSourceRef(
                turn_index=len(turns) - 1,
                source_ref=f"{caption_path}:events[{event_index}]",
            )
        )

    if not turns:
        raise ParsePrimitiveError(
            f"No spoken YouTube caption events found in {caption_path}"
        )
    warnings = []
    if malformed_events:
        warnings.append(
            f"Skipped {malformed_events} malformed YouTube caption event(s)."
        )
    return TranscriptArtifact(turns=turns, source_refs=source_refs, warnings=warnings)


def parse_scribe_transcript(scribe_json_path: Path) -> TranscriptArtifact:
    try:
        payload = json.loads(scribe_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ParsePrimitiveError(
            f"Scribe transcript is not valid JSON: {scribe_json_path}"
        ) from exc
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
            or UNKNOWN_SPEAKER
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

    if not turns:
        raise ParsePrimitiveError(
            f"No spoken transcript turns found in {scribe_json_path}"
        )
    return TranscriptArtifact(turns=turns, source_refs=source_refs)
