from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from .models import RecordingRef, SourceNote

EMBED_RE = re.compile(r"!\[\[([^\]]+)\]\]")
MARKDOWN_LINK_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".mp4", ".webm", ".ogg", ".flac"}


class RecordingResolutionError(FileNotFoundError):
    pass


def _normalise_link_target(target: str) -> str:
    return target.split("|", 1)[0].strip()


def _created_at(path: Path) -> datetime:
    stat = path.stat()
    ts = stat.st_birthtime if hasattr(stat, "st_birthtime") else stat.st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _candidate_paths(note_path: Path, target: str) -> list[Path]:
    note_dir = note_path.parent
    target_path = Path(target)
    candidates = [note_dir / target_path]

    for parent in [note_dir, *note_dir.parents]:
        candidates.append(parent / target_path)
        candidates.append(parent / "Attachments" / target_path.name)
        if (parent / ".obsidian").exists():
            break

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)

    return deduped


def resolve_recording_path(note_path: Path, raw_link: str) -> Path:
    target = _normalise_link_target(raw_link)

    for candidate in _candidate_paths(note_path, target):
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    raise RecordingResolutionError(
        f"Unable to resolve recording {raw_link!r} referenced by {note_path}"
    )


def extract_recording_links(note_body: str) -> list[str]:
    links = [*EMBED_RE.findall(note_body), *MARKDOWN_LINK_RE.findall(note_body)]
    return [link for link in links if Path(_normalise_link_target(link)).suffix.lower() in AUDIO_EXTENSIONS]


def load_source_note(note_path: Path) -> SourceNote:
    body = note_path.read_text(encoding="utf-8")
    recordings = [
        RecordingRef(
            raw_link=link,
            resolved_path=resolve_recording_path(note_path, link),
            created_at=_created_at(resolve_recording_path(note_path, link)),
        )
        for link in extract_recording_links(body)
    ]
    recordings.sort(key=lambda item: item.created_at)
    return SourceNote(path=note_path, body=body, recordings=recordings)
