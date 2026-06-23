from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

from .models import ConcatPlan, RecordingRef, ScribeRunReport


class AudioPipelineError(RuntimeError):
    pass


def build_concat_plan(recordings: list[RecordingRef], output_path: Path) -> ConcatPlan:
    ordered = [
        recording.resolved_path
        for recording in sorted(recordings, key=lambda item: item.created_at)
    ]
    return ConcatPlan(inputs_in_creation_order=ordered, output_merged_audio=output_path)


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def _concat_manifest_entry(path: Path) -> str:
    escaped = str(path).replace("'", r"'\\''")
    return f"file '{escaped}'"


def concatenate_recordings(
    plan: ConcatPlan,
    concat_file: Path,
    *,
    run_command: CommandRunner = _run,
) -> None:
    concat_file.write_text(
        "\n".join(
            _concat_manifest_entry(path) for path in plan.inputs_in_creation_order
        ),
        encoding="utf-8",
    )

    try:
        run_command(
            [
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
                str(plan.output_merged_audio),
            ]
        )
    except subprocess.CalledProcessError as exc:
        raise AudioPipelineError(exc.stderr or exc.stdout or str(exc)) from exc


def run_scribe(
    input_audio: Path,
    output_json: Path,
    *,
    run_command: CommandRunner = _run,
) -> ScribeRunReport:
    try:
        result = run_command(
            [
                "scribe",
                "-o",
                str(output_json),
                "--format",
                "json",
                str(input_audio),
            ]
        )
    except subprocess.CalledProcessError as exc:
        raise AudioPipelineError(exc.stderr or exc.stdout or str(exc)) from exc

    segment_count = None
    warnings: list[str] = []
    if output_json.exists():
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            segments = payload.get("segments")
            if isinstance(segments, list):
                segment_count = len(segments)
            raw_warnings = payload.get("warnings")
            if isinstance(raw_warnings, list):
                warnings = [str(item) for item in raw_warnings]

    return ScribeRunReport(
        input_audio=input_audio,
        output_json=output_json,
        warnings=warnings,
        segment_count=segment_count,
        stdout=result.stdout,
        stderr=result.stderr,
    )
