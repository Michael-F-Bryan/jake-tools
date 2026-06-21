import importlib
from pathlib import Path

from click.testing import CliRunner

from jake_tools.cli.transcribe import transcribe

transcribe_cli = importlib.import_module("jake_tools.cli.transcribe")


class DummyResult:
    def __init__(self) -> None:
        self.note_path = Path("/tmp/Meeting.md")
        self.updated = False

    def json_summary(self) -> dict:
        return {
            "chapter_summaries": [{"title": "Kickoff", "start_timestamp": "00:00", "end_timestamp": "00:30"}],
            "ai_stage_stats": [{"stage": "chaptering", "total_tokens": 42, "estimated_cost_usd": 0.01}],
            "ai_totals": {"stage_count": 1, "total_tokens": 42, "estimated_cost_usd": 0.01},
            "speaker_message_counts": [{"speaker": "Vet West", "messages": 3}],
        }


def fake_process_obsidian_recording(hermes, obsidian_note, dry_run):
    return DummyResult()


def test_obsidian_recording_cli_human_output(tmp_path, monkeypatch) -> None:
    note = tmp_path / "Meeting.md"
    note.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        transcribe_cli,
        "process_obsidian_recording",
        fake_process_obsidian_recording,
    )

    runner = CliRunner()
    result = runner.invoke(
        transcribe,
        ["obsidian-recording", "--dry-run", str(note)],
    )

    assert result.exit_code == 0
    assert "note: /tmp/Meeting.md" in result.output
    assert "updated: False" in result.output


def test_obsidian_recording_cli_json_output(tmp_path, monkeypatch) -> None:
    note = tmp_path / "Meeting.md"
    note.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        transcribe_cli,
        "process_obsidian_recording",
        fake_process_obsidian_recording,
    )

    runner = CliRunner()
    result = runner.invoke(
        transcribe,
        ["obsidian-recording", "--json", str(note)],
    )

    assert result.exit_code == 0
    assert '"chapter_summaries"' in result.output
    assert '"speaker_message_counts"' in result.output
    assert '"note_path"' not in result.output
    assert '"updated"' not in result.output
