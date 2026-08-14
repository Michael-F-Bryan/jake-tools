"""Semantic oracle for the deliberately broken transcript repair canary.

The clean reference is test evidence, not model context.  The live test feeds only
``meeting_context``, participants, and ``broken_text`` to the production pipeline,
then evaluates the resulting editorial dialogue with these helpers.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from jake_tools.transcripts.bundle.ids import (
    ComponentId,
    OverlapGroupId,
    ParticipantId,
    TurnId,
)

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "transcripts"
    / "deliberately-broken-repair-canary.json"
)


def _load() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _dialogue(fixture: dict[str, Any], field: str) -> str:
    return "\n".join(
        str(turn[field])
        for turn in fixture["turns"]
        if turn.get(field) is not None and str(turn[field]).strip()
    )


def _semantic_failures(fixture: dict[str, Any], candidate: str) -> tuple[str, ...]:
    failures: list[str] = []
    for contract in fixture["required_semantics"]:
        for pattern in contract["required_regexes"]:
            if re.search(pattern, candidate) is None:
                failures.append(f"{contract['id']}: missing /{pattern}/")
        for pattern in contract["forbidden_regexes"]:
            if re.search(pattern, candidate) is not None:
                failures.append(f"{contract['id']}: contains forbidden /{pattern}/")
    return tuple(failures)


def test_fixture_source_accounting_contract_is_total_and_unambiguous() -> None:
    fixture = _load()
    turns = fixture["turns"]
    keys = [turn["key"] for turn in turns]
    expected = fixture["structural_expectations"]

    assert len(keys) == len(set(keys))
    assert [turn["start_ms"] for turn in turns] == sorted(
        turn["start_ms"] for turn in turns
    )
    assert fixture["source_span_ms"] == [turns[0]["start_ms"], turns[-1]["end_ms"]]

    accounted = Counter(expected["required_dropped_keys"])
    for group in expected["required_merge_groups"]:
        accounted.update(group)
    accounted.update(key for key in keys if key not in accounted)
    assert accounted == Counter(dict.fromkeys(keys, 1))

    output_keys = [
        turn["key"]
        for turn in turns
        if turn.get("clean_text") is not None and str(turn["clean_text"]).strip()
    ]
    assert expected["required_display_order"] == output_keys
    assert all(
        not next(turn for turn in turns if turn["key"] == key)["clean_text"]
        for key in expected["required_dropped_keys"]
    )


def test_fixture_carries_exact_editorial_lineage_and_reviewed_overlap_authority() -> (
    None
):
    fixture = _load()
    turns = fixture["turns"]
    lineage = fixture["lineage"]
    canonical = lineage["canonical_turn_set"]
    review = lineage["speaker_review"]

    TypeAdapter(ComponentId).validate_python(canonical["component_id"])
    TypeAdapter(ComponentId).validate_python(review["component_id"])
    for participant_id in lineage["participant_ids"].values():
        TypeAdapter(ParticipantId).validate_python(participant_id)

    canonical_payload = []
    review_payload = []
    for turn in turns:
        TypeAdapter(TurnId).validate_python(turn["canonical_turn_id"])
        span = turn["canonical_source_span"]
        interval = turn["constituent_interval"]
        attribution = turn["effective_attribution"]
        assert span == {
            "canonical_component_id": canonical["component_id"],
            "turn_id": turn["canonical_turn_id"],
            "start_char": 0,
            "end_char": len(turn["broken_text"]),
        }
        assert interval == {
            "turn_id": turn["canonical_turn_id"],
            "start_ms": turn["start_ms"],
            "end_ms": turn["end_ms"],
        }
        assert attribution["speaker_review_component_id"] == review["component_id"]
        assert (
            attribution["participant_id"] == lineage["participant_ids"][turn["speaker"]]
        )
        assert turn["derivation_mode"] == "verbatim"
        assert turn["operation_ancestry"] == [f"identity:{turn['canonical_turn_id']}"]
        canonical_payload.append(
            {
                "turn_id": turn["canonical_turn_id"],
                "speaker": turn["speaker"],
                "text": turn["broken_text"],
                "start_ms": turn["start_ms"],
                "end_ms": turn["end_ms"],
            }
        )
        review_payload.append(
            {
                "turn_id": turn["canonical_turn_id"],
                "participant_id": attribution["participant_id"],
            }
        )

    assert (
        canonical["content_hash"]
        == hashlib.sha256(
            json.dumps(
                canonical_payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    )
    assert (
        review["content_hash"]
        == hashlib.sha256(
            json.dumps(review_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )

    overlap = lineage["reviewed_overlap_groups"]
    assert len(overlap) == 1
    group = overlap[0]
    TypeAdapter(OverlapGroupId).validate_python(group["overlap_group_id"])
    assert group["speaker_review_component_id"] == review["component_id"]
    by_key = {turn["key"]: turn for turn in turns}
    assert group["turn_ids"] == [
        by_key[key]["canonical_turn_id"] for key in group["turn_keys"]
    ]
    assert group["turn_keys"] == [
        "tentative-proposal-a",
        "objection-overlap",
        "tentative-proposal-b",
    ]

    risk = fixture["expected_risk_findings"]
    assert risk == [
        {
            "id": "context-disambiguated-can-bus",
            "finding_type": "context-disambiguated-technical-term",
            "severity": "human-disposition-required",
            "source_turn_keys": ["lpc-split-a", "lpc-split-b"],
            "broken_form": "can buzz",
            "proposed_form": "CAN bus",
            "basis": "meeting_context, not mechanically recoverable from source bytes",
            "required_disposition": "pending-human-review",
        }
    ]


def test_hidden_clean_oracle_passes_every_semantic_canary() -> None:
    fixture = _load()

    assert _semantic_failures(fixture, _dialogue(fixture, "clean_text")) == ()


def test_deliberately_broken_input_fails_the_semantic_canary() -> None:
    fixture = _load()
    failures = _semantic_failures(fixture, _dialogue(fixture, "broken_text"))

    assert failures
    assert any(failure.startswith("technical-terms:") for failure in failures)
