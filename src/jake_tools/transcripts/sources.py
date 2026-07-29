from __future__ import annotations

import contextlib
import datetime as dt
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from .errors import TranscriptError
from .models import SourceArtifact, YoutubeCaptureMetadata
from .obsidian import load_source_note

CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


class SourcePrimitiveError(TranscriptError):
    pass


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def source_from_obsidian_note(note_path: Path) -> SourceArtifact:
    source_note = load_source_note(note_path)
    artifact = source_note.to_source_artifact()
    artifact.source_path = source_note.path.resolve()
    artifact.attachments = [
        recording.resolved_path.resolve() for recording in source_note.recordings
    ]
    return artifact


def _caption_tracks(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    return {str(language): formats for language, formats in payload.items()}


def _pick_caption_language(
    tracks: dict[str, object], *, language: str, prefer_original: bool
) -> str | None:
    requested = language.casefold()
    by_casefold = {track.casefold(): track for track in tracks}
    candidates = [f"{requested}-orig", requested] if prefer_original else [requested]
    for candidate in candidates:
        if candidate in by_casefold:
            return by_casefold[candidate]

    regional = sorted(
        track for track in tracks if track.casefold().startswith(f"{requested}-")
    )
    return regional[0] if regional else None


def _select_youtube_caption(
    info: dict[str, object], *, language: str
) -> tuple[str, Literal["manual", "automatic"]]:
    manual = _caption_tracks(info.get("subtitles"))
    selected = _pick_caption_language(manual, language=language, prefer_original=False)
    if selected is not None:
        return selected, "manual"

    automatic = _caption_tracks(info.get("automatic_captions"))
    selected = _pick_caption_language(
        automatic, language=language, prefer_original=True
    )
    if selected is not None:
        return selected, "automatic"

    display_language = "English" if language.casefold() == "en" else language
    raise SourcePrimitiveError(
        f"No {display_language} captions are available for this YouTube video."
    )


def _run_ytdlp(
    command: list[str], *, run_command: CommandRunner
) -> subprocess.CompletedProcess[str]:
    try:
        result = run_command(command)
    except FileNotFoundError as exc:
        raise SourcePrimitiveError(
            "yt-dlp is required for YouTube transcript capture but was not found."
        ) from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise SourcePrimitiveError(f"yt-dlp failed: {message}") from exc
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise SourcePrimitiveError(f"yt-dlp failed: {message}")
    return result


def _youtube_public_metadata(info: dict[str, object]) -> dict[str, object]:
    keys = (
        "id",
        "title",
        "channel",
        "channel_id",
        "uploader",
        "upload_date",
        "duration",
        "webpage_url",
        "language",
        "availability",
    )
    return {key: info[key] for key in keys if info.get(key) is not None}


def source_from_youtube(
    url: str,
    *,
    out_dir: Path,
    language: str = "en",
    run_command: CommandRunner = _run,
) -> SourceArtifact:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_result = _run_ytdlp(
        [
            "yt-dlp",
            "--no-update",
            "--skip-download",
            "--dump-single-json",
            url,
        ],
        run_command=run_command,
    )
    try:
        raw_info = json.loads(metadata_result.stdout)
    except json.JSONDecodeError as exc:
        raise SourcePrimitiveError(
            "yt-dlp returned invalid video metadata JSON."
        ) from exc
    if not isinstance(raw_info, dict):
        raise SourcePrimitiveError("yt-dlp video metadata was not a JSON object.")
    info: dict[str, object] = raw_info

    video_id = str(info.get("id") or "").strip()
    title = str(info.get("title") or "").strip()
    if not video_id or not title:
        raise SourcePrimitiveError("YouTube metadata is missing the video ID or title.")

    channel = str(info.get("channel") or info.get("uploader") or "").strip()
    upload_date = str(info.get("upload_date") or "")
    published: dt.date | None = None
    if upload_date:
        with contextlib.suppress(ValueError):
            published = dt.datetime.strptime(upload_date, "%Y%m%d").date()
    duration = info.get("duration")
    duration_seconds = (
        int(duration) if isinstance(duration, int | float) and duration > 0 else 0
    )
    missing = [
        label
        for label, present in (
            ("channel", bool(channel)),
            ("publication date", published is not None),
            ("duration", duration_seconds > 0),
        )
        if not present
    ]
    if missing:
        raise SourcePrimitiveError(
            "YouTube metadata is missing required source-note provenance: "
            + ", ".join(missing)
        )

    subtitle_track, subtitle_kind = _select_youtube_caption(info, language=language)
    caption_template = out_dir / "captions.%(ext)s"
    caption_path = out_dir / f"captions.{subtitle_track}.json3"
    caption_path.unlink(missing_ok=True)
    write_flag = "--write-subs" if subtitle_kind == "manual" else "--write-auto-subs"
    _run_ytdlp(
        [
            "yt-dlp",
            "--no-update",
            "--skip-download",
            write_flag,
            "--sub-langs",
            subtitle_track,
            "--sub-format",
            "json3",
            "--output",
            str(caption_template),
            url,
        ],
        run_command=run_command,
    )
    if not caption_path.exists():
        raise SourcePrimitiveError(
            f"yt-dlp completed but did not create JSON3 captions at {caption_path}"
        )

    public_info = _youtube_public_metadata(info)
    info_path = out_dir / "video-info.json"
    info_path.write_text(
        json.dumps(public_info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    source_url = str(info.get("webpage_url") or url)
    warnings = (
        ["YouTube automatic captions were used; technical terms may need correction."]
        if subtitle_kind == "automatic"
        else []
    )
    capture_metadata = YoutubeCaptureMetadata(
        video_id=video_id,
        channel=channel,
        channel_id=str(info.get("channel_id") or ""),
        duration_seconds=duration_seconds,
        language=str(info.get("language") or language),
        subtitle_track=subtitle_track,
        subtitle_kind=subtitle_kind,
        capture_method="yt-dlp",
    )
    return SourceArtifact(
        kind="youtube",
        source_path=info_path,
        source_url=source_url,
        message_id=f"youtube:{video_id}",
        title=title,
        date=published,
        organisation=channel or None,
        raw_text_path=caption_path,
        metadata=capture_metadata.model_dump(mode="json"),
        warnings=warnings,
    )
