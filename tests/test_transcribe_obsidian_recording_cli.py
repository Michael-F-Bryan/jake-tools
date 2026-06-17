from pathlib import Path

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


def fake_process_obsidian_recording(hermes, obsidian_note, mode, dry_run):
    return DummyResult()


def test_obsidian_recording_cli_human_output(tmp_path) -> None:
    note = tmp_path / "Meeting.md"
    note.write_text("stub", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        transcribe,
        ["obsidian-recording", "--mode", "transcript", "--dry-run", str(note)],
        obj={"process_obsidian_recording": fake_process_obsidian_recording},
    )

    assert result.exit_code == 0
    assert "mode: transcript" in result.output
    assert "updated: False" in result.output


def test_obsidian_recording_cli_json_output(tmp_path) -> None:
    note = tmp_path / "Meeting.md"
    note.write_text("stub", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        transcribe,
        ["obsidian-recording", "--json", str(note)],
        obj={"process_obsidian_recording": fake_process_obsidian_recording},
    )

    assert result.exit_code == 0
    assert '"mode": "transcript"' in result.output
