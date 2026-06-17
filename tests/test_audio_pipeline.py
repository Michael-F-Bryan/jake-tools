from datetime import datetime, timezone
import json
import subprocess

from jake_tools.transcripts.audio import AudioPipelineError, build_concat_plan, concatenate_recordings, run_scribe
from jake_tools.transcripts.models import RecordingRef


class DummyCompletedProcess:
    def __init__(self, stdout: str = "", stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr


def test_build_concat_plan_orders_inputs_by_creation_time(tmp_path) -> None:
    later = RecordingRef(
        raw_link="later.m4a",
        resolved_path=tmp_path / "later.m4a",
        created_at=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
    )
    earlier = RecordingRef(
        raw_link="earlier.m4a",
        resolved_path=tmp_path / "earlier.m4a",
        created_at=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
    )

    plan = build_concat_plan([later, earlier], tmp_path / "merged.mp3")

    assert plan.inputs_in_creation_order == [earlier.resolved_path, later.resolved_path]


def test_concatenate_recordings_writes_concat_manifest_and_invokes_ffmpeg(monkeypatch, tmp_path) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], check: bool, capture_output: bool, text: bool):
        commands.append(command)
        return DummyCompletedProcess()

    monkeypatch.setattr(subprocess, "run", fake_run)

    plan = build_concat_plan(
        [
            RecordingRef(
                raw_link="meeting.m4a",
                resolved_path=tmp_path / "meeting.m4a",
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        ],
        tmp_path / "merged.mp3",
    )

    concat_file = tmp_path / "inputs.txt"
    concatenate_recordings(plan, concat_file)

    assert concat_file.read_text(encoding="utf-8") == f"file '{tmp_path / 'meeting.m4a'}'"
    assert commands == [[
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-vn",
        "-c:a",
        "libmp3lame",
        str(tmp_path / "merged.mp3"),
    ]]


def test_run_scribe_uses_json_output_and_parses_segments(monkeypatch, tmp_path) -> None:
    commands: list[list[str]] = []
    output_json = tmp_path / "merged.json"

    def fake_run(command: list[str], check: bool, capture_output: bool, text: bool):
        commands.append(command)
        output_json.write_text('{"segments": [{}, {}], "warnings": ["warn"]}', encoding="utf-8")
        return DummyCompletedProcess(stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    report = run_scribe(tmp_path / "merged.mp3", output_json)

    assert commands == [["scribe", "-o", str(output_json), "--format", "json", str(tmp_path / "merged.mp3")]]
    assert report.segment_count == 2
    assert report.warnings == ["warn"]


def test_audio_pipeline_wraps_subprocess_failure(monkeypatch, tmp_path) -> None:
    def fake_run(command: list[str], check: bool, capture_output: bool, text: bool):
        raise subprocess.CalledProcessError(1, command, stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    try:
        run_scribe(tmp_path / "merged.mp3", tmp_path / "merged.json")
    except AudioPipelineError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected AudioPipelineError")
