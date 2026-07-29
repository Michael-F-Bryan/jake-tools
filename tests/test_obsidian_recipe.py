from __future__ import annotations

import json
from pathlib import Path

from agent_fakes import fake_agent, structured

from jake_tools.transcripts import obsidian_recipe
from jake_tools.transcripts.models import ConcatPlan, ScribeRunReport


def _recipe_responses() -> list[dict]:
    return [
        {
            "mapping": {
                "SPEAKER_01": {
                    "name": "Michael Bryan",
                    "confidence": 0.9,
                    "reason": "matches attendee list",
                }
            },
        },
        {
            "turns": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "speaker": "SPEAKER_01",
                    "text": "Hello team",
                }
            ],
        },
        {
            "chapters": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "title": "Kickoff",
                    "summary": "Opened the meeting.",
                }
            ],
            "boundary_source": "llm",
        },
        {
            "summary": "Kickoff sync.",
            "key_points": ["Opened the meeting."],
        },
    ]


def _write_note(tmp_path: Path) -> tuple[Path, Path]:
    recording = tmp_path / "meeting.m4a"
    recording.write_bytes(b"")
    note = tmp_path / "Meeting.md"
    note.write_text("# Meeting\n\n![[meeting.m4a]]\n", encoding="utf-8")
    return note, recording


def _patch_audio_pipeline(monkeypatch, *, warnings: list[str]) -> None:
    def fake_concatenate(plan: ConcatPlan, concat_file: Path) -> None:
        concat_file.write_text("stub", encoding="utf-8")

    def fake_run_scribe(input_audio: Path, output_json: Path) -> ScribeRunReport:
        output_json.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "start": 0,
                            "end": 2,
                            "speaker": "SPEAKER_01",
                            "text": "Hello team",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return ScribeRunReport(
            input_audio=input_audio,
            output_json=output_json,
            warnings=warnings,
            segment_count=1,
        )

    monkeypatch.setattr(obsidian_recipe, "concatenate_recordings", fake_concatenate)
    monkeypatch.setattr(obsidian_recipe, "run_scribe", fake_run_scribe)


async def test_obsidian_recipe_surfaces_scribe_warnings_in_result_and_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    note, _recording = _write_note(tmp_path)
    _patch_audio_pipeline(
        monkeypatch, warnings=["Low microphone gain detected during capture."]
    )
    agent = fake_agent(*(structured(payload) for payload in _recipe_responses()))
    workdir = tmp_path / "work"

    result = await obsidian_recipe.run_obsidian_recording_recipe(
        agent,
        note,
        dry_run=True,
        workdir=workdir,
    )

    assert result.warnings == ["Low microphone gain detected during capture."]
    assert result.json_summary()["warnings"] == [
        "Low microphone gain detected during capture."
    ]
    manifest = json.loads((workdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["warnings"] == ["Low microphone gain detected during capture."]


async def test_obsidian_recipe_dry_run_preserves_note_and_reports_no_warnings(
    tmp_path: Path, monkeypatch
) -> None:
    note, _recording = _write_note(tmp_path)
    _patch_audio_pipeline(monkeypatch, warnings=[])
    agent = fake_agent(*(structured(payload) for payload in _recipe_responses()))

    result = await obsidian_recipe.run_obsidian_recording_recipe(
        agent,
        note,
        dry_run=True,
    )

    assert result.updated is False
    assert result.warnings == []
    assert note.read_text(encoding="utf-8") == "# Meeting\n\n![[meeting.m4a]]\n"
    assert "## Meeting Notes" not in note.read_text(encoding="utf-8")


async def test_obsidian_recipe_ephemeral_run_writes_no_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    """Without a workdir, artefacts live in a TemporaryDirectory that is gone
    by the time this call returns, so the recipe must not claim a manifest
    exists anywhere the caller could find it."""
    note, recording = _write_note(tmp_path)
    _patch_audio_pipeline(monkeypatch, warnings=[])
    agent = fake_agent(*(structured(payload) for payload in _recipe_responses()))

    result = await obsidian_recipe.run_obsidian_recording_recipe(
        agent,
        note,
        dry_run=True,
    )

    assert result.updated is False
    assert set(tmp_path.iterdir()) == {note, recording}
