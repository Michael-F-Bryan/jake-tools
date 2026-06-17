import subprocess
from datetime import datetime, timezone
from pathlib import Path
import json

from jake_tools.transcripts.audio import AudioPipelineError, build_concat_plan, concatenate_recordings, run_scribe
from jake_tools.transcripts.models import RecordingRef


def completed_process(*, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=stderr)


def recording(path: Path, *, raw_link: str = "meeting.m4a") -> RecordingRef:
    return RecordingRef(
        raw_link=raw_link,
        resolved_path=path,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


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


def test_concatenate_recordings_writes_concat_manifest_and_invokes_ffmpeg(tmp_path) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return completed_process()

    plan = build_concat_plan(
        [recording(tmp_path / "meeting.m4a")],
        tmp_path / "merged.mp3",
    )

    concat_file = tmp_path / "inputs.txt"
    concatenate_recordings(plan, concat_file, run_command=fake_run)

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


def test_run_scribe_uses_json_output_and_parses_segments(tmp_path) -> None:
    commands: list[list[str]] = []
    output_json = tmp_path / "merged.json"

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        output_json.write_text('{"segments": [{}, {}], "warnings": ["warn"]}', encoding="utf-8")
        return completed_process(stdout="ok", stderr="")

    report = run_scribe(tmp_path / "merged.mp3", output_json, run_command=fake_run)

    assert commands == [["scribe", "-o", str(output_json), "--format", "json", str(tmp_path / "merged.mp3")]]
    assert report.segment_count == 2
    assert report.warnings == ["warn"]


def test_audio_pipeline_wraps_subprocess_failure(tmp_path) -> None:
    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, command, stderr="boom")

    try:
        run_scribe(tmp_path / "merged.mp3", tmp_path / "merged.json", run_command=fake_run)
    except AudioPipelineError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected AudioPipelineError")
