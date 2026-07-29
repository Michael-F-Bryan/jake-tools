import importlib
import json
from pathlib import Path

from click.testing import CliRunner

from jake_tools.cli.transcribe import transcribe
from jake_tools.cli.transcript import transcript
from jake_tools.transcripts.models import (
    ChapterSummary,
    CoordinatorResult,
    SpeakerMessageCount,
)

transcript_cli = importlib.import_module("jake_tools.cli.transcript")


def _result() -> CoordinatorResult:
    return CoordinatorResult(
        note_path=Path("/tmp/Meeting.md"),
        updated=False,
        chapter_summaries=[
            ChapterSummary(
                title="Kickoff", start_timestamp="00:00", end_timestamp="00:30"
            )
        ],
        speaker_message_counts=[SpeakerMessageCount(speaker="Vet West", messages=3)],
    )


async def fake_run_obsidian_recording_recipe(agent, obsidian_note, *, dry_run, workdir):
    return _result()


def test_obsidian_recording_cli_human_output(tmp_path, monkeypatch) -> None:
    note = tmp_path / "Meeting.md"
    note.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        transcript_cli,
        "run_obsidian_recording_recipe",
        fake_run_obsidian_recording_recipe,
    )

    runner = CliRunner()
    result = runner.invoke(
        transcript,
        ["obsidian-recording", "--dry-run", str(note)],
    )

    assert result.exit_code == 0
    assert "note: /tmp/Meeting.md" in result.output
    assert "updated: False" in result.output


def test_obsidian_recording_cli_json_output(tmp_path, monkeypatch) -> None:
    note = tmp_path / "Meeting.md"
    note.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        transcript_cli,
        "run_obsidian_recording_recipe",
        fake_run_obsidian_recording_recipe,
    )

    runner = CliRunner()
    result = runner.invoke(
        transcript,
        ["obsidian-recording", "--json", str(note)],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == _result().json_summary()
    assert "note_path" not in payload
    assert "updated" not in payload


def test_deprecated_transcribe_alias_forwards_to_the_same_command(
    tmp_path, monkeypatch
) -> None:
    note = tmp_path / "Meeting.md"
    note.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        transcript_cli,
        "run_obsidian_recording_recipe",
        fake_run_obsidian_recording_recipe,
    )

    runner = CliRunner()
    via_alias = runner.invoke(
        transcribe, ["obsidian-recording", "--dry-run", str(note)]
    )
    via_transcript = runner.invoke(
        transcript, ["obsidian-recording", "--dry-run", str(note)]
    )

    assert via_alias.exit_code == via_transcript.exit_code == 0
    assert via_alias.output == via_transcript.output
