"""Behaviour of ``jake-tools transcript merge-audio`` (cli/transcript.py)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from click.testing import CliRunner

from jake_tools.cli import main
from jake_tools.cli.context import AppContext
from jake_tools.transcription.cache import RunCache
from jake_tools.transcription.models import SourceClip

FIXTURES = Path(__file__).parent / "fixtures"


class FakeVaultClient:
    def __init__(self, root: Path, files: dict[str, Path]) -> None:
        self._root = root
        self._files = files

    def vault_root(self) -> Path:
        return self._root

    def resolve_embed(self, target: str) -> Path:
        return self._files[target]


class FakeAudioTool:
    def duration_seconds(self, path: Path) -> float:
        return 1.0

    def merge(self, clips: Sequence[Path], out: Path) -> list[SourceClip]:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake merged audio")
        offset = 0.0
        result: list[SourceClip] = []
        for clip in clips:
            result.append(
                SourceClip(path=str(clip), offset_seconds=offset, duration_seconds=1.0)
            )
            offset += 1.0
        return result

    def cut(self, source: Path, start: float, end: float, out: Path) -> Path:
        out.write_bytes(b"fake cut audio")
        return out


def _app_context(tmp_path: Path, files: dict[str, Path]) -> AppContext:
    return AppContext(
        vault_client_factory=lambda: FakeVaultClient(tmp_path, files),
        audio_tool_factory=FakeAudioTool,
        run_cache_factory=lambda: RunCache(tmp_path / "cache"),
    )


def test_merge_audio_prints_json_with_run_id_and_clips(tmp_path: Path) -> None:
    clip_a = tmp_path / "clip_a.m4a"
    clip_b = tmp_path / "clip_b.m4a"
    clip_a.write_bytes(b"a")
    clip_b.write_bytes(b"b")
    app = _app_context(
        tmp_path,
        {
            "Recording 20260803090000.m4a": clip_a,
            "Recording 20260803094500.m4a": clip_b,
        },
    )
    runner = CliRunner()

    result = runner.invoke(
        main,
        ["transcript", "merge-audio", str(FIXTURES / "meeting_note.md")],
        obj=app,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["run_id"].startswith("meeting-note-")
    assert payload["audio_sha256"]
    assert Path(payload["merged_path"]).exists()
    assert [clip["path"] for clip in payload["clips"]] == [str(clip_a), str(clip_b)]


def test_merge_audio_exits_nonzero_with_a_clear_error_for_a_note_with_no_embeds(
    tmp_path: Path,
) -> None:
    note_path = tmp_path / "no_embeds.md"
    note_path.write_text("---\ntags:\n  - note/meeting\n---\n\nJust some text.\n")
    app = _app_context(tmp_path, {})
    runner = CliRunner()

    result = runner.invoke(main, ["transcript", "merge-audio", str(note_path)], obj=app)

    assert result.exit_code != 0
    assert str(note_path) in result.output


def test_transcript_merge_audio_help_exits_zero() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["transcript", "merge-audio", "--help"])

    assert result.exit_code == 0
    assert "merge" in result.output.lower()
