from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner
from pydantic import BaseModel

from jake_tools.cli import main
from jake_tools.hermes import Reply


class FakeHermes:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.prompt_turn_counts: list[int] = []

    def run_structured(self, prompt: Any) -> tuple[BaseModel, Reply]:
        if not self._responses:
            raise AssertionError("No fake response configured for run_structured call.")
        self.prompt_turn_counts.append(len(getattr(prompt, "turns", [])))
        payload = self._responses.pop(0)
        model = prompt.response_model
        return model.model_validate(payload), Reply(text=json.dumps(payload))


def _write_transcript(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "turns": [
                    {
                        "start": 0.0,
                        "end": 5.0,
                        "speaker": "Speaker 1",
                        "text": "hello there",
                    },
                    {
                        "start": 5.0,
                        "end": 9.0,
                        "speaker": "Speaker 2",
                        "text": "general kenobi",
                    },
                ],
                "source_refs": [],
                "speakers": {},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )


def test_transcript_stage_polish_happy_path_with_structured_output(
    tmp_path: Path,
) -> None:
    turns_path = tmp_path / "turns.json"
    out_path = tmp_path / "turns.polished.json"
    ledger_out = tmp_path / "turns.polished.ledger.json"
    _write_transcript(turns_path)

    fake_hermes = FakeHermes(
        responses=[
            {
                "turns": [
                    {
                        "start": 0.0,
                        "end": 5.0,
                        "speaker": "Speaker 1",
                        "text": "Hello there.",
                    },
                    {
                        "start": 5.0,
                        "end": 9.0,
                        "speaker": "Speaker 2",
                        "text": "General Kenobi.",
                    },
                ],
                "ledger": {"merge_allowed": False, "entries": [], "notes": ""},
            }
        ]
    )

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "stage",
            "polish",
            str(turns_path),
            "--out",
            str(out_path),
            "--ledger-out",
            str(ledger_out),
            "--json",
        ],
        obj={"hermes": fake_hermes},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["turns"][0]["text"] == "Hello there."
    ledger_payload = json.loads(ledger_out.read_text(encoding="utf-8"))
    assert ledger_payload["merge_allowed"] is False


def test_transcript_stage_polish_flags_possible_over_simplification(
    tmp_path: Path,
) -> None:
    turns_path = tmp_path / "turns.json"
    out_path = tmp_path / "turns.polished.json"
    _write_transcript(turns_path)

    fake_hermes = FakeHermes(
        responses=[
            {
                "turns": [
                    {
                        "start": 0.0,
                        "end": 5.0,
                        "speaker": "Speaker 1",
                        "text": "Hello.",
                    },
                    {
                        "start": 5.0,
                        "end": 9.0,
                        "speaker": "Speaker 2",
                        "text": "General Kenobi.",
                    },
                ],
                "ledger": {"merge_allowed": False, "entries": [], "notes": ""},
            }
        ]
    )

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "stage",
            "polish",
            str(turns_path),
            "--out",
            str(out_path),
        ],
        obj={"hermes": fake_hermes},
    )

    assert result.exit_code != 0
    assert "polish.content-retention" in result.output
    assert "possible over-simplification" in result.output


def test_transcript_stage_polish_rejects_operational_chatter_regression(
    tmp_path: Path,
) -> None:
    turns_path = tmp_path / "turns.json"
    out_path = tmp_path / "turns.polished.json"
    _write_transcript(turns_path)

    fake_hermes = FakeHermes(
        responses=[
            {
                "turns": [
                    {
                        "start": 0.0,
                        "end": 5.0,
                        "speaker": "Speaker 1",
                        "text": "Loading the transcript-polisher skill.",
                    },
                    {
                        "start": 5.0,
                        "end": 9.0,
                        "speaker": "Speaker 2",
                        "text": "General Kenobi.",
                    },
                ],
                "ledger": {"merge_allowed": False, "entries": [], "notes": ""},
            }
        ]
    )

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "stage",
            "polish",
            str(turns_path),
            "--out",
            str(out_path),
        ],
        obj={"hermes": fake_hermes},
    )

    assert result.exit_code != 0
    assert "boilerplate.no-operational-chatter" in result.output


def test_transcript_stage_polish_accepts_manifest_input(tmp_path: Path) -> None:
    chunk_path = tmp_path / "chunk-001.json"
    out_path = tmp_path / "turns.polished.json"
    _write_transcript(chunk_path)
    manifest_path = tmp_path / "chunks.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "split-manifest",
                "stages": [
                    {
                        "stage": "chunk-001",
                        "status": "pass",
                        "artefacts": [str(chunk_path.name)],
                    }
                ],
                "artefact_paths": {"chunk-001": str(chunk_path.name)},
                "command_metadata": [],
                "ai_totals": None,
            }
        ),
        encoding="utf-8",
    )

    fake_hermes = FakeHermes(
        responses=[
            {
                "turns": [
                    {
                        "start": 0.0,
                        "end": 5.0,
                        "speaker": "Speaker 1",
                        "text": "Hello there.",
                    },
                    {
                        "start": 5.0,
                        "end": 9.0,
                        "speaker": "Speaker 2",
                        "text": "General Kenobi.",
                    },
                ],
                "ledger": {"merge_allowed": False, "entries": [], "notes": ""},
            }
        ]
    )

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "stage",
            "polish",
            str(manifest_path),
            "--out",
            str(out_path),
            "--json",
        ],
        obj={"hermes": fake_hermes},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert len(payload["turns"]) == 2


def test_transcript_stage_polish_processes_manifest_chunks_individually(
    tmp_path: Path,
) -> None:
    chunk_one = tmp_path / "chunk-001.json"
    chunk_two = tmp_path / "chunk-002.json"
    chunk_one.write_text(
        json.dumps(
            {
                "turns": [
                    {
                        "start": 0.0,
                        "end": 5.0,
                        "speaker": "Speaker 1",
                        "text": "hello there",
                    }
                ],
                "source_refs": [],
                "speakers": {},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    chunk_two.write_text(
        json.dumps(
            {
                "turns": [
                    {
                        "start": 5.0,
                        "end": 9.0,
                        "speaker": "Speaker 2",
                        "text": "general kenobi",
                    }
                ],
                "source_refs": [],
                "speakers": {},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "chunks.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "split-manifest",
                "stages": [
                    {
                        "stage": "chunk-001",
                        "status": "pass",
                        "artefacts": ["chunk-001.json"],
                    },
                    {
                        "stage": "chunk-002",
                        "status": "pass",
                        "artefacts": ["chunk-002.json"],
                    },
                ],
                "artefact_paths": {
                    "chunk-001": "chunk-001.json",
                    "chunk-002": "chunk-002.json",
                },
                "command_metadata": [],
                "ai_totals": None,
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "turns.polished.json"
    ledger_out = tmp_path / "turns.polished.ledger.json"
    fake_hermes = FakeHermes(
        responses=[
            {
                "turns": [
                    {
                        "start": 0.0,
                        "end": 5.0,
                        "speaker": "Speaker 1",
                        "text": "Hello there.",
                    }
                ],
                "ledger": {"merge_allowed": False, "entries": [], "notes": "chunk 1"},
            },
            {
                "turns": [
                    {
                        "start": 5.0,
                        "end": 9.0,
                        "speaker": "Speaker 2",
                        "text": "General Kenobi.",
                    }
                ],
                "ledger": {"merge_allowed": False, "entries": [], "notes": "chunk 2"},
            },
        ]
    )

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "stage",
            "polish",
            str(manifest_path),
            "--out",
            str(out_path),
            "--ledger-out",
            str(ledger_out),
            "--json",
        ],
        obj={"hermes": fake_hermes},
    )

    assert result.exit_code == 0
    assert fake_hermes.prompt_turn_counts == [1, 1]
    payload = json.loads(result.output)
    assert [turn["text"] for turn in payload["turns"]] == [
        "Hello there.",
        "General Kenobi.",
    ]
    ledger_payload = json.loads(ledger_out.read_text(encoding="utf-8"))
    assert ledger_payload["notes"] == "chunk-001: chunk 1\nchunk-002: chunk 2"


def test_transcript_stage_map_speakers_title_chapters_and_minutes(
    tmp_path: Path,
) -> None:
    turns_path = tmp_path / "turns.json"
    mapping_out = tmp_path / "mapping.json"
    chapters_out = tmp_path / "chapters.json"
    minutes_out = tmp_path / "minutes.json"
    _write_transcript(turns_path)

    fake_hermes = FakeHermes(
        responses=[
            {
                "mapping": {
                    "Speaker 1": {
                        "name": "Alex",
                        "confidence": 0.8,
                        "reason": "Matched agenda opening style.",
                    }
                },
                "unresolved": ["Speaker 2"],
                "notes": "Conservative mapping.",
            },
            {
                "chapters": [
                    {
                        "start": 0.0,
                        "end": 9.0,
                        "title": "Introductions",
                        "summary": "Participants greet and start discussion.",
                    }
                ],
                "boundary_source": "deterministic",
            },
            {
                "summary": "Meeting opened with introductions.",
                "key_points": ["Kickoff and context alignment."],
            },
        ]
    )

    runner = CliRunner()
    map_result = runner.invoke(
        main,
        [
            "transcript",
            "stage",
            "map-speakers",
            str(turns_path),
            "--attendee",
            "Alex",
            "--out",
            str(mapping_out),
            "--json",
        ],
        obj={"hermes": fake_hermes},
    )
    assert map_result.exit_code == 0
    map_payload = json.loads(map_result.output)
    assert "Speaker 1" in map_payload["mapping"]

    chapters_result = runner.invoke(
        main,
        [
            "transcript",
            "stage",
            "title-chapters",
            str(turns_path),
            "--out",
            str(chapters_out),
            "--json",
        ],
        obj={"hermes": fake_hermes},
    )
    assert chapters_result.exit_code == 0
    chapters_payload = json.loads(chapters_result.output)
    assert chapters_payload["boundary_source"] == "llm"

    minutes_result = runner.invoke(
        main,
        [
            "transcript",
            "stage",
            "minutes",
            str(turns_path),
            "--out",
            str(minutes_out),
            "--json",
        ],
        obj={"hermes": fake_hermes},
    )
    assert minutes_result.exit_code == 0
    minutes_payload = json.loads(minutes_result.output)
    assert minutes_payload["summary"].startswith("Meeting opened")
