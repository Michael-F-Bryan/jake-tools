from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from jake_tools.transcription.audio import (
    AudioEmbedResolutionError,
    AudioToolError,
    FfmpegAudioTool,
    NoAudioEmbedsError,
    clip_sort_key,
    merge_note_audio,
    run_id_for,
)
from jake_tools.transcription.cache import RunCache
from jake_tools.transcription.models import SourceClip
from jake_tools.transcription.note import parse_note
from jake_tools.transcription.obsidian import ObsidianCliError

FIXTURES = Path(__file__).parent / "fixtures"


# --- clip_sort_key -----------------------------------------------------


def test_clip_sort_key_orders_stamped_clips_chronologically() -> None:
    early = Path("Recording 20260801090000.m4a")
    late = Path("Recording 20260802090000.m4a")

    assert clip_sort_key(early, note_order_index=1) < clip_sort_key(
        late, note_order_index=0
    )


def test_clip_sort_key_falls_back_to_note_order_for_unstamped_clips() -> None:
    first = Path("intro.m4a")
    second = Path("outro.m4a")

    assert clip_sort_key(first, note_order_index=0) < clip_sort_key(
        second, note_order_index=1
    )


def test_clip_sort_key_sorts_stamped_clips_before_unstamped_clips() -> None:
    stamped = Path("Recording 20260801090000.m4a")
    unstamped = Path("bonus-clip.m4a")

    assert clip_sort_key(stamped, note_order_index=5) < clip_sort_key(
        unstamped, note_order_index=0
    )


# --- run_id_for ----------------------------------------------------------


def test_run_id_for_combines_slugified_stem_and_hash_prefix() -> None:
    run_id = run_id_for(Path("/vault/2026-08-11 Team Sync.md"), "abcdef0123456789")

    assert run_id == "2026-08-11-team-sync-abcdef012345"


def test_run_id_for_is_stable_for_the_same_inputs() -> None:
    note_path = Path("note.md")
    audio_sha256 = "0" * 64

    assert run_id_for(note_path, audio_sha256) == run_id_for(note_path, audio_sha256)


# --- merge_note_audio orchestration --------------------------------------


class FakeVaultClient:
    """Fake VaultClient: maps embed targets to pre-registered paths."""

    def __init__(self, root: Path, files: dict[str, Path]) -> None:
        self._root = root
        self._files = files

    def vault_root(self) -> Path:
        return self._root

    def resolve_embed(self, target: str) -> Path:
        try:
            return self._files[target]
        except KeyError:
            raise ObsidianCliError(f"no fake mapping for {target!r}") from None


class FakeAudioTool:
    """Fake AudioTool that records calls instead of shelling out to ffmpeg."""

    def __init__(self) -> None:
        self.merge_calls: list[tuple[list[Path], Path]] = []
        self.durations: dict[Path, float] = {}

    def duration_seconds(self, path: Path) -> float:
        return self.durations.get(path, 1.0)

    def merge(self, clips: Sequence[Path], out: Path) -> list[SourceClip]:
        self.merge_calls.append((list(clips), out))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake merged audio")
        offset = 0.0
        result: list[SourceClip] = []
        for clip in clips:
            duration = self.duration_seconds(clip)
            result.append(
                SourceClip(
                    path=str(clip), offset_seconds=offset, duration_seconds=duration
                )
            )
            offset += duration
        return result

    def cut(self, source: Path, start: float, end: float, out: Path) -> Path:
        out.write_bytes(b"fake cut audio")
        return out


def test_merge_note_audio_resolves_sorts_merges_and_caches(tmp_path: Path) -> None:
    note = parse_note(FIXTURES / "meeting_note.md")
    clip_a = tmp_path / "clip_a.m4a"
    clip_b = tmp_path / "clip_b.m4a"
    clip_a.write_bytes(b"a")
    clip_b.write_bytes(b"b")
    vault = FakeVaultClient(
        tmp_path,
        {
            "Recording 20260803090000.m4a": clip_a,
            "Recording 20260803094500.m4a": clip_b,
        },
    )
    audio_tool = FakeAudioTool()
    cache = RunCache(tmp_path / "cache")

    result = merge_note_audio(note, vault=vault, audio_tool=audio_tool, cache=cache)

    assert audio_tool.merge_calls[0][0] == [clip_a, clip_b]
    assert result.clips == [
        SourceClip(path=str(clip_a), offset_seconds=0.0, duration_seconds=1.0),
        SourceClip(path=str(clip_b), offset_seconds=1.0, duration_seconds=1.0),
    ]
    assert result.run_id.startswith("meeting-note-")
    merged_path = Path(result.merged_path)
    assert merged_path.exists()
    assert merged_path.read_bytes() == b"fake merged audio"
    assert result.audio_sha256


def test_merge_note_audio_sorts_out_of_order_embeds_chronologically(
    tmp_path: Path,
) -> None:
    note_text = (
        "---\ntags:\n  - note/meeting\n---\n\n"
        "![[Recording 20260803094500.m4a]]\n\n"
        "![[Recording 20260803090000.m4a]]\n"
    )
    note_path = tmp_path / "note.md"
    note_path.write_text(note_text)
    note = parse_note(note_path)

    resolved_dir = tmp_path / "Attachments"
    resolved_dir.mkdir()
    clip_late = resolved_dir / "Recording 20260803094500.m4a"
    clip_early = resolved_dir / "Recording 20260803090000.m4a"
    clip_late.write_bytes(b"late")
    clip_early.write_bytes(b"early")
    vault = FakeVaultClient(
        tmp_path,
        {
            "Recording 20260803094500.m4a": clip_late,
            "Recording 20260803090000.m4a": clip_early,
        },
    )
    audio_tool = FakeAudioTool()
    cache = RunCache(tmp_path / "cache")

    merge_note_audio(note, vault=vault, audio_tool=audio_tool, cache=cache)

    assert audio_tool.merge_calls[0][0] == [clip_early, clip_late]


def test_merge_note_audio_strips_alias_suffix_before_resolving(tmp_path: Path) -> None:
    note_text = (
        "---\ntags:\n  - note/meeting\n---\n\n"
        "![[Recording 20260803090000.m4a|recording]]\n"
    )
    note_path = tmp_path / "note.md"
    note_path.write_text(note_text)
    note = parse_note(note_path)

    clip = tmp_path / "clip.m4a"
    clip.write_bytes(b"a")
    vault = FakeVaultClient(tmp_path, {"Recording 20260803090000.m4a": clip})
    audio_tool = FakeAudioTool()
    cache = RunCache(tmp_path / "cache")

    result = merge_note_audio(note, vault=vault, audio_tool=audio_tool, cache=cache)

    assert audio_tool.merge_calls[0][0] == [clip]
    assert result.clips[0].path == str(clip)


def test_merge_note_audio_raises_when_note_has_no_audio_embeds(tmp_path: Path) -> None:
    note_path = tmp_path / "no_embeds.md"
    note_path.write_text("---\ntags:\n  - note/meeting\n---\n\nJust some text.\n")
    note = parse_note(note_path)
    audio_tool = FakeAudioTool()
    cache = RunCache(tmp_path / "cache")

    with pytest.raises(NoAudioEmbedsError, match=str(note_path)):
        merge_note_audio(
            note,
            vault=FakeVaultClient(tmp_path, {}),
            audio_tool=audio_tool,
            cache=cache,
        )


def test_merge_note_audio_names_the_note_when_an_embed_fails_to_resolve(
    tmp_path: Path,
) -> None:
    """An embed that exists but can't be resolved is a different failure mode
    from having no embeds at all, but the plan's requirement is the same:
    the error must name the note, not just the unresolvable embed target.
    """
    note_path = tmp_path / "unresolvable.md"
    note_path.write_text(
        "---\ntags:\n  - note/meeting\n---\n\n![[Recording 20260803090000.m4a]]\n"
    )
    note = parse_note(note_path)
    audio_tool = FakeAudioTool()
    cache = RunCache(tmp_path / "cache")
    vault = FakeVaultClient(tmp_path, {})  # no mapping -> resolve_embed raises

    with pytest.raises(AudioEmbedResolutionError) as excinfo:
        merge_note_audio(note, vault=vault, audio_tool=audio_tool, cache=cache)

    message = str(excinfo.value)
    assert str(note_path) in message
    assert "Recording 20260803090000.m4a" in message
    assert not audio_tool.merge_calls  # failed before any merge was attempted


# --- FfmpegAudioTool command assembly ------------------------------------


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> object:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_duration_seconds_parses_ffprobe_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> object:
        calls.append(argv)
        return _completed(stdout="12.5\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    tool = FfmpegAudioTool()

    duration = tool.duration_seconds(Path("clip.m4a"))

    assert duration == 12.5
    assert calls[0][0] == "ffprobe"
    assert "clip.m4a" in calls[0]


def test_merge_tries_copy_concat_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> object:
        calls.append(argv)
        if argv[0] == "ffprobe":
            return _completed(stdout="2.0\n")
        # ffmpeg concat: pretend the output file was written.
        Path(argv[-1]).write_bytes(b"merged")
        return _completed()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/" + name)
    tool = FfmpegAudioTool()

    clip_a = tmp_path / "a.m4a"
    clip_b = tmp_path / "b.m4a"
    clip_a.write_bytes(b"a")
    clip_b.write_bytes(b"b")
    out = tmp_path / "out.m4a"

    tool.merge([clip_a, clip_b], out)

    ffmpeg_calls = [call for call in calls if call[0] == "ffmpeg"]
    assert len(ffmpeg_calls) == 1
    assembled = ffmpeg_calls[0]
    assert "-c" in assembled and "copy" in assembled
    assert assembled[assembled.index("-c") + 1] == "copy"
    assert "-f" in assembled and assembled[assembled.index("-f") + 1] == "concat"


def test_merge_falls_back_to_reencode_when_copy_concat_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> object:
        calls.append(argv)
        if argv[0] == "ffprobe":
            return _completed(stdout="2.0\n")
        if "-filter_complex" in argv:
            Path(argv[-1]).write_bytes(b"merged")
            return _completed()
        # The copy-concat attempt fails (mismatched codec params).
        return _completed(returncode=1, stderr="codec parameters mismatch")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/" + name)
    tool = FfmpegAudioTool()

    clip_a = tmp_path / "a.m4a"
    clip_b = tmp_path / "b.m4a"
    clip_a.write_bytes(b"a")
    clip_b.write_bytes(b"b")
    out = tmp_path / "out.m4a"

    clips = tool.merge([clip_a, clip_b], out)

    ffmpeg_calls = [call for call in calls if call[0] == "ffmpeg"]
    assert len(ffmpeg_calls) == 2
    assert "-c" in ffmpeg_calls[0] and "copy" in ffmpeg_calls[0]
    assert "-filter_complex" in ffmpeg_calls[1]
    assert clips[0].duration_seconds == 2.0
    assert clips[1].offset_seconds == 2.0


def test_merge_raises_when_both_paths_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(argv: list[str], **kwargs: object) -> object:
        if argv[0] == "ffprobe":
            return _completed(stdout="2.0\n")
        return _completed(returncode=1, stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/" + name)
    tool = FfmpegAudioTool()

    clip_a = tmp_path / "a.m4a"
    clip_b = tmp_path / "b.m4a"
    clip_a.write_bytes(b"a")
    clip_b.write_bytes(b"b")

    with pytest.raises(AudioToolError, match="boom"):
        tool.merge([clip_a, clip_b], tmp_path / "out.m4a")


def test_cut_assembles_a_mono_reencode_command_for_the_given_span(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> object:
        calls.append(argv)
        Path(argv[-1]).write_bytes(b"cut")
        return _completed()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/" + name)
    tool = FfmpegAudioTool()
    source = tmp_path / "source.m4a"
    source.write_bytes(b"source")
    out = tmp_path / "snippet.m4a"

    result = tool.cut(source, 12.0, 17.5, out)

    assert result == out
    assert out.read_bytes() == b"cut"
    assert len(calls) == 1
    argv = calls[0]
    assert argv[0] == "ffmpeg"
    assert argv[argv.index("-ss") + 1] == "12.0"
    assert argv[argv.index("-t") + 1] == "5.5"
    assert argv[argv.index("-ac") + 1] == "1"
    assert "-c:a" in argv and argv[argv.index("-c:a") + 1] == "aac"


def test_ensure_available_raises_clearly_when_ffmpeg_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    tool = FfmpegAudioTool()

    with pytest.raises(AudioToolError, match="Install ffmpeg"):
        tool.ensure_available()


# --- live: real ffmpeg ----------------------------------------------------


@pytest.mark.live
def test_ffmpeg_merge_of_two_generated_clips_sums_their_durations(
    tmp_path: Path,
) -> None:
    tool = FfmpegAudioTool()
    tool.ensure_available()

    clip_a = tmp_path / "clip_a.m4a"
    clip_b = tmp_path / "clip_b.m4a"
    for clip, freq in ((clip_a, "440"), (clip_b, "220")):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={freq}:duration=1",
                str(clip),
            ],
            capture_output=True,
            check=True,
        )

    out = tmp_path / "merged.m4a"
    clips = tool.merge([clip_a, clip_b], out)

    assert out.exists()
    expected_total = sum(clip.duration_seconds for clip in clips)
    merged_duration = tool.duration_seconds(out)
    assert merged_duration == pytest.approx(expected_total, abs=0.3)
