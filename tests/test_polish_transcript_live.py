"""Real-model acceptance for ``polish_transcript`` (``--slow``; costs money).

These run the production operation through the real Claude Agent SDK and
CLI, with the credentials of whoever runs them. They assert stable
invariants (source accounting, protected meaning kept, decisions well
formed enough to land, the review reporting a planted garble), never exact
prose. A provider or credential failure fails the test: report it as
blocked, not as covered.

``test_a_real_recording`` needs ``JAKE_POLISH_TRANSCRIPT``: the path of a
saved ``transcribe`` result of a real conversation (optionally
``JAKE_POLISH_START``/``JAKE_POLISH_END`` seconds). Without it the test is
skipped as blocked. Never point it at a reference transcript: the input is
what the model sees.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from polish_fixtures import write_transcript

from jake_tools.config import load_config
from jake_tools.operations.catalogue import OPERATIONS
from jake_tools.polishing.lexicon import lexical
from jake_tools.polishing.pipeline import PolishResult

pytestmark = pytest.mark.slow

CONVERSATION: list[tuple[str | None, str]] = [
    (
        "speaker_1",
        "Okay so um the the main thing is the training calendar for next term.",
    ),
    ("speaker_2", "Right. I I I think we should run the radio course first."),
    ("speaker_1", "Yeah. So we meet on Tuesday, no, Wednesday the fourth to plan it."),
    ("speaker_2", "Wednesday works. But we did not agree on the budget yet, did we?"),
    ("speaker_1", "No. The cap was twelve fifty euro per person, I believe."),
    (
        "speaker_2",
        "And the uh the venue purple sideways the lantern of committee forms.",
    ),
    ("speaker_1", "Sorry? You you broke up there."),
    (
        "speaker_2",
        "The venue needs the committee forms signed. I'll, I'll send them tonight.",
    ),
]


async def _polish(**arguments: object) -> PolishResult:
    operation = next(op for op in OPERATIONS if op.name == "polish_transcript")
    handler = operation.build(load_config())
    result = await handler(**arguments)
    assert isinstance(result, PolishResult)
    return result


def _retained(out: Path) -> list[str]:
    ledger = json.loads((out / "ledger.json").read_text())
    return [lexical(w["text"]) for w in ledger["words"] if w["status"] != "removed"]


async def test_a_short_conversation_keeps_its_meaning(tmp_path: Path) -> None:
    source = write_transcript(tmp_path / "raw.json", CONVERSATION)
    raw = source.read_bytes()
    out = tmp_path / "polished"
    result = await _polish(transcript_path=str(source), output_directory=str(out))

    assert result.status == "complete", result.chunks
    assert source.read_bytes() == raw and result.source_unchanged
    ledger = json.loads((out / "ledger.json").read_text())
    assert [w["id"] for w in ledger["words"]] == list(range(result.counts.window_words))
    retained = _retained(out)
    # The self-correction, the negation, the hedge and the figures were said.
    for word in (
        "tuesday",
        "no",
        "wednesday",
        "fourth",
        "not",
        "twelve",
        "fifty",
        "believe",
    ):
        assert word in retained, word
    model_edits = [e for e in ledger["edits"] if e["origin"] == "edit_model"]
    assert model_edits, "the edit role made no accepted decision on obvious stutters"
    misquoted = [
        r for r in ledger["rejected"] if r["reason"].startswith("text does not match")
    ]
    proposals = len(model_edits) + len(ledger["rejected"])
    assert len(misquoted) * 3 <= proposals, ledger["rejected"]
    review = json.loads((out / "review.json").read_text())
    assert review["complete"] is True
    garble_ids = range(
        [w["id"] for w in ledger["words"] if w["text"] == "purple"][0] - 3,
        [w["id"] for w in ledger["words"] if w["text"] == "forms."][0] + 1,
    )
    assert any(
        item["first_word"] in garble_ids or item["last_word"] in garble_ids
        for item in review["items"]
    ), review["items"]
    assert result.estimated_cost_usd <= 5.0


async def test_a_real_recording(tmp_path: Path) -> None:
    path = os.environ.get("JAKE_POLISH_TRANSCRIPT")
    if not path:
        pytest.skip("blocked: set JAKE_POLISH_TRANSCRIPT to a saved transcribe result")
    start = os.environ.get("JAKE_POLISH_START")
    end = os.environ.get("JAKE_POLISH_END")
    source = Path(path).resolve()
    raw = source.read_bytes()
    out = tmp_path / "polished"
    result = await _polish(
        transcript_path=str(source),
        output_directory=str(out),
        start_seconds=float(start) if start else None,
        end_seconds=float(end) if end else None,
    )
    assert source.read_bytes() == raw
    assert result.status == "complete", [
        c
        for c in result.chunks
        if c.edit.status != "completed" or c.review.status != "completed"
    ]
    ledger = json.loads((out / "ledger.json").read_text())
    assert [w["id"] for w in ledger["words"]] == list(
        range(result.window.first_word, result.window.last_word + 1)
    )
    misquoted = [
        r for r in ledger["rejected"] if r["reason"].startswith("text does not match")
    ]
    model_edits = [e for e in ledger["edits"] if e["origin"] == "edit_model"]
    assert len(misquoted) * 4 <= len(model_edits) + len(ledger["rejected"]) + 4
    assert result.elapsed_seconds < 300
