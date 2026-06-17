from pathlib import Path
import importlib
from click.testing import CliRunner

from jake_tools.cli.transcribe import transcribe


class DummyResult:
    def __init__(self) -> None:
        self.mode = "transcript"
        self.note_path = Path("/tmp/Meeting.md")
        self.updated = False

    def model_dump(self, mode: str = "python") -> dict:
        return {
            "mode": self.mode,
            "note_path": str(self.note_path),
            "updated": self.updated,
        }


def test_obsidian_recording_cli_human_output(monkeypatch, tmp_path) -> None:
    note = tmp_path / "Meeting.md"
    note.write_text("stub", encoding="utf-8")

    cli_module = importlib.import_module("jake_tools.cli.transcribe")
    monkeypatch.setattr(
        cli_module,
        "process_obsidian_recording",
        lambda hermes, obsidian_note, mode, dry_run: DummyResult(),
    )

    runner = CliRunner()
    result = runner.invoke(transcribe, ["obsidian-recording", "--mode", "transcript", "--dry-run", str(note)])

    assert result.exit_code == 0
    assert "mode: transcript" in result.output
    assert "updated: False" in result.output


def test_obsidian_recording_cli_json_output(monkeypatch, tmp_path) -> None:
    note = tmp_path / "Meeting.md"
    note.write_text("stub", encoding="utf-8")

    cli_module = importlib.import_module("jake_tools.cli.transcribe")
    monkeypatch.setattr(
        cli_module,
        "process_obsidian_recording",
        lambda hermes, obsidian_note, mode, dry_run: DummyResult(),
    )

    runner = CliRunner()
    result = runner.invoke(transcribe, ["obsidian-recording", "--json", str(note)])

    assert result.exit_code == 0
    assert '"mode": "transcript"' in result.output
