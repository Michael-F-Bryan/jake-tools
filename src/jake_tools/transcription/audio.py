"""ffmpeg/ffprobe wrapper for the transcription pipeline's audio stage.

Multi-clip meetings must be concatenated into a single recording before
ASR/diarisation runs, so the diariser hears the same voices throughout and
speaker clusters stay consistent across the meeting. :class:`AudioTool` is
the seam: :class:`FfmpegAudioTool` is the real, subprocess-backed
implementation; tests inject a fake instead of shelling out to ffmpeg.

:meth:`FfmpegAudioTool.merge` tries the concat demuxer with stream copy
first (fast, lossless, but requires matching codec parameters across
clips); Obsidian clips usually match, but "usually" is not a design
guarantee, so a non-zero exit falls back to a re-encode concat via
``filter_complex``. :meth:`FfmpegAudioTool.cut` re-encodes a ``[start,
end)`` span to a small mono AAC file, sized for sending over Discord (plan
006's snippet-request flow).

:func:`merge_note_audio` is the orchestration this module exists to serve:
given a parsed note, a :class:`~.obsidian.VaultClient`, and an
:class:`AudioTool`, it resolves the note's audio embeds, orders them, merges
them, and stashes the result in the run cache keyed by :func:`run_id_for`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from .cache import RunCache, sha256_of
from .models import SourceClip
from .note import ParsedNote
from .obsidian import VaultClient

_STAMP_RE = re.compile(r"Recording (\d{14})\.m4a$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_AUDIO_SUFFIXES = (".m4a",)
_STDERR_TAIL_LINES = 20
_CUT_BITRATE = "64k"


class AudioToolError(RuntimeError):
    """Raised when ffmpeg/ffprobe fails and no fallback resolves the operation."""


class NoAudioEmbedsError(RuntimeError):
    """Raised when a note has no resolvable audio embeds to merge."""


class AudioTool(Protocol):
    def duration_seconds(self, path: Path) -> float: ...

    def merge(self, clips: Sequence[Path], out: Path) -> list[SourceClip]: ...

    def cut(self, source: Path, start: float, end: float, out: Path) -> Path: ...


class FfmpegAudioTool:
    """Shells out to ``ffmpeg``/``ffprobe``. Every subprocess call captures
    stderr; on failure the raised :class:`AudioToolError` includes its tail
    so operators can debug from the message alone.
    """

    def __init__(self, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
        self._ffmpeg = ffmpeg
        self._ffprobe = ffprobe

    def ensure_available(self) -> None:
        """Raise a clear "install ffmpeg" message if either binary is missing."""
        missing = [
            name for name in (self._ffmpeg, self._ffprobe) if shutil.which(name) is None
        ]
        if missing:
            raise AudioToolError(
                f"required tool(s) not found on PATH: {', '.join(missing)}. "
                "Install ffmpeg (e.g. `brew install ffmpeg`)."
            )

    def duration_seconds(self, path: Path) -> float:
        result = self._run(
            [
                self._ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
        )
        try:
            return float(result.stdout.strip())
        except ValueError as exc:
            raise AudioToolError(
                f"could not parse ffprobe duration for {path}: {result.stdout!r}"
            ) from exc

    def merge(self, clips: Sequence[Path], out: Path) -> list[SourceClip]:
        self.ensure_available()
        if not clips:
            raise AudioToolError("merge requires at least one clip")
        out.parent.mkdir(parents=True, exist_ok=True)
        durations = [self.duration_seconds(clip) for clip in clips]

        if len(clips) == 1:
            shutil.copyfile(clips[0], out)
        else:
            try:
                self._merge_copy_concat(clips, out)
            except AudioToolError:
                self._merge_reencode_concat(clips, out)

        offsets: list[float] = []
        cumulative = 0.0
        for duration in durations:
            offsets.append(cumulative)
            cumulative += duration

        return [
            SourceClip(path=str(clip), offset_seconds=offset, duration_seconds=duration)
            for clip, offset, duration in zip(clips, offsets, durations, strict=True)
        ]

    def cut(self, source: Path, start: float, end: float, out: Path) -> Path:
        self.ensure_available()
        out.parent.mkdir(parents=True, exist_ok=True)
        self._run(
            [
                self._ffmpeg,
                "-y",
                "-i",
                str(source),
                "-ss",
                str(start),
                "-t",
                str(end - start),
                "-ac",
                "1",
                "-c:a",
                "aac",
                "-b:a",
                _CUT_BITRATE,
                str(out),
            ]
        )
        return out

    def _merge_copy_concat(self, clips: Sequence[Path], out: Path) -> None:
        fd, list_path_str = tempfile.mkstemp(suffix=".txt", prefix="jake-tools-concat-")
        list_path = Path(list_path_str)
        try:
            with os.fdopen(fd, "w") as handle:
                for clip in clips:
                    handle.write(f"file '{_escape_concat_path(clip)}'\n")
            self._run(
                [
                    self._ffmpeg,
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(list_path),
                    "-c",
                    "copy",
                    str(out),
                ]
            )
        finally:
            list_path.unlink(missing_ok=True)

    def _merge_reencode_concat(self, clips: Sequence[Path], out: Path) -> None:
        inputs: list[str] = []
        for clip in clips:
            inputs += ["-i", str(clip)]
        count = len(clips)
        concat_inputs = "".join(f"[{index}:a]" for index in range(count))
        filter_str = f"{concat_inputs}concat=n={count}:v=0:a=1[outa]"
        self._run(
            [
                self._ffmpeg,
                "-y",
                *inputs,
                "-filter_complex",
                filter_str,
                "-map",
                "[outa]",
                str(out),
            ]
        )

    def _run(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(argv, capture_output=True, text=True, check=False)
        except OSError as exc:
            raise AudioToolError(f"failed to run {argv[0]}: {exc}") from exc
        if result.returncode != 0:
            raise AudioToolError(_error_message(argv[0], result))
        return result


def _escape_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def _error_message(tool: str, result: subprocess.CompletedProcess[str]) -> str:
    stderr_lines = result.stderr.strip().splitlines()
    tail = "\n".join(stderr_lines[-_STDERR_TAIL_LINES:])
    return f"{tool} exited {result.returncode}: {tail}"


def clip_sort_key(path: Path, note_order_index: int) -> tuple[int, str, int]:
    """Sort key for merge ordering.

    Clips whose filename embeds a ``Recording YYYYMMDDHHMMSS.m4a`` stamp sort
    chronologically ahead of anything without one; within either group, the
    note's embed order is the tiebreak (zero-padded so it also sorts as a
    string, keeping the return type uniform).
    """
    match = _STAMP_RE.search(path.name)
    if match is not None:
        return (0, match.group(1), note_order_index)
    return (1, f"{note_order_index:010d}", note_order_index)


def run_id_for(note_path: Path, audio_sha256: str) -> str:
    """The cross-plan run-id convention: ``<note-stem-slugified>-<hash prefix>``."""
    slug = _SLUG_RE.sub("-", note_path.stem.lower()).strip("-")
    return f"{slug}-{audio_sha256[:12]}"


class MergeAudioResult(BaseModel):
    """The JSON document ``jake-tools transcript merge-audio`` prints."""

    run_id: str
    merged_path: str
    audio_sha256: str
    clips: list[SourceClip]


def merge_note_audio(
    note: ParsedNote,
    *,
    vault: VaultClient,
    audio_tool: AudioTool,
    cache: RunCache,
) -> MergeAudioResult:
    """Merge every audio embed in ``note`` into one recording, cached by run id.

    Embed targets are filtered to those that look like audio files (Obsidian
    voice memos are ``.m4a``) — a note may embed other things too. A
    ``target|alias`` embed (the note parser captures ``![[target|alias]]``
    verbatim) has its alias stripped before resolution.

    The merged file's hash isn't known until after merging, so the merge
    happens in a scratch directory first; the result is then copied into its
    final, hash-keyed run directory.
    """
    audio_embeds = [
        (index, embed.split("|", 1)[0])
        for index, embed in enumerate(note.embeds)
        if embed.split("|", 1)[0].lower().endswith(_AUDIO_SUFFIXES)
    ]
    if not audio_embeds:
        raise NoAudioEmbedsError(f"note {note.path!r} has no audio embeds to merge")

    resolved = [(vault.resolve_embed(target), index) for index, target in audio_embeds]
    ordered = sorted(resolved, key=lambda pair: clip_sort_key(pair[0], pair[1]))
    clip_paths = [path for path, _ in ordered]

    with tempfile.TemporaryDirectory(prefix="jake-tools-merge-") as tmp:
        staged = Path(tmp) / "merged.m4a"
        clips = audio_tool.merge(clip_paths, staged)
        audio_sha256 = sha256_of(staged)
        run_id = run_id_for(Path(note.path), audio_sha256)
        merged_path = cache.run_dir(run_id) / "merged.m4a"
        shutil.copyfile(staged, merged_path)

    return MergeAudioResult(
        run_id=run_id,
        merged_path=str(merged_path),
        audio_sha256=audio_sha256,
        clips=clips,
    )
