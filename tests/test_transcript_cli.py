"""Behaviour of the ``jake-tools transcript`` CLI group.

Per the CLI-options memo (`memo-cli-options.md`, rule 3), these tests stay
thin: they verify flag parsing -> options object (`transcript_options.py`)
and, for `merge-audio`, that the built dependencies get delegated to the
library seam — `merge_note_audio` is monkeypatched with a recording fake, so
no real Obsidian/ffmpeg process runs here. Library behaviour itself is
covered at the `transcription/audio.py` seam in
`tests/test_transcription_audio.py`.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from jake_tools.cli import main
from jake_tools.cli.transcript_options import (
    AudioOptions,
    CacheOptions,
    ObsidianOptions,
    audio_options,
    cache_options,
    obsidian_options,
)
from jake_tools.transcription.audio import MergeAudioResult, NoAudioEmbedsError
from jake_tools.transcription.models import SourceClip
from jake_tools.transcription.note import ParsedNote

FIXTURES = Path(__file__).parent / "fixtures"

# `jake_tools.cli`'s __init__ rebinds the name `transcript` to the Click
# group (`from .transcript import transcript`), shadowing the submodule —
# so `jake_tools.cli.transcript` is the group, not the module. Fetch the
# actual module via importlib to monkeypatch its `merge_note_audio` binding.
transcript_cli = importlib.import_module("jake_tools.cli.transcript")


# --- options decorators: flag parsing -> options object --------------------


@click.command()
@obsidian_options
def _obsidian_probe(obsidian_options: ObsidianOptions) -> None:
    click.echo(f"vault={obsidian_options.vault} binary={obsidian_options.binary}")


def test_obsidian_options_decorator_builds_options_from_flags() -> None:
    result = CliRunner().invoke(
        _obsidian_probe, ["--vault", "MyVault", "--obsidian-binary", "custom-obsidian"]
    )

    assert result.exit_code == 0
    assert result.output == "vault=MyVault binary=custom-obsidian\n"


def test_obsidian_options_decorator_defaults_binary_to_obsidian() -> None:
    result = CliRunner().invoke(_obsidian_probe, [])

    assert result.exit_code == 0
    assert result.output == "vault=None binary=obsidian\n"


@click.command()
@audio_options
def _audio_probe(audio_options: AudioOptions) -> None:
    click.echo(f"ffmpeg={audio_options.ffmpeg} ffprobe={audio_options.ffprobe}")


def test_audio_options_decorator_builds_options_from_flags() -> None:
    result = CliRunner().invoke(
        _audio_probe, ["--ffmpeg", "custom-ffmpeg", "--ffprobe", "custom-ffprobe"]
    )

    assert result.exit_code == 0
    assert result.output == "ffmpeg=custom-ffmpeg ffprobe=custom-ffprobe\n"


def test_audio_options_decorator_defaults_to_bare_binary_names() -> None:
    result = CliRunner().invoke(_audio_probe, [])

    assert result.exit_code == 0
    assert result.output == "ffmpeg=ffmpeg ffprobe=ffprobe\n"


@click.command()
@cache_options
def _cache_probe(cache_options: CacheOptions) -> None:
    click.echo(f"root={cache_options.root}")


def test_cache_options_decorator_builds_options_from_flags(tmp_path: Path) -> None:
    result = CliRunner().invoke(_cache_probe, ["--cache-root", str(tmp_path)])

    assert result.exit_code == 0
    assert result.output == f"root={tmp_path}\n"


def test_cache_options_decorator_defaults_root_to_none() -> None:
    result = CliRunner().invoke(_cache_probe, [])

    assert result.exit_code == 0
    assert result.output == "root=None\n"


# --- merge-audio: delegation -------------------------------------------------


def test_merge_audio_delegates_to_merge_note_audio_and_prints_its_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    fake_result = MergeAudioResult(
        run_id="meeting-note-abcdef012345",
        merged_path="/cache/meeting-note-abcdef012345/merged.m4a",
        audio_sha256="abcdef0123456789",
        clips=[
            SourceClip(path="/vault/clip.m4a", offset_seconds=0.0, duration_seconds=5.0)
        ],
    )

    def fake_merge_note_audio(
        note: ParsedNote, *, vault: object, audio_tool: object, cache: object
    ) -> MergeAudioResult:
        captured["note_path"] = note.path
        captured["vault"] = vault
        captured["audio_tool"] = audio_tool
        captured["cache"] = cache
        return fake_result

    monkeypatch.setattr(transcript_cli, "merge_note_audio", fake_merge_note_audio)
    runner = CliRunner()
    note_path = FIXTURES / "meeting_note.md"

    result = runner.invoke(
        main,
        [
            "transcript",
            "merge-audio",
            str(note_path),
            "--vault",
            "MyVault",
            "--ffmpeg",
            "custom-ffmpeg",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == json.loads(fake_result.model_dump_json())
    assert captured["note_path"] == str(note_path)
    # The flags reached the dependencies handed to the library seam.
    assert captured["vault"]._vault == "MyVault"  # type: ignore[attr-defined]
    assert captured["audio_tool"]._ffmpeg == "custom-ffmpeg"  # type: ignore[attr-defined]


def test_merge_audio_exits_nonzero_with_a_clear_error_for_a_note_with_no_embeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    note_path = tmp_path / "no_embeds.md"
    note_path.write_text("---\ntags:\n  - note/meeting\n---\n\nJust some text.\n")

    def raising_merge_note_audio(
        note: ParsedNote, *, vault: object, audio_tool: object, cache: object
    ) -> MergeAudioResult:
        raise NoAudioEmbedsError(f"note {note.path!r} has no audio embeds to merge")

    monkeypatch.setattr(transcript_cli, "merge_note_audio", raising_merge_note_audio)
    runner = CliRunner()

    result = runner.invoke(main, ["transcript", "merge-audio", str(note_path)])

    assert result.exit_code != 0
    assert str(note_path) in result.output


def test_transcript_merge_audio_help_exits_zero() -> None:
    result = CliRunner().invoke(main, ["transcript", "merge-audio", "--help"])

    assert result.exit_code == 0
    assert "merge" in result.output.lower()
