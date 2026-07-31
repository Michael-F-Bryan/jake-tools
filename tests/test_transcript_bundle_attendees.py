"""M22: inferring attendees when a note does not declare them.

The interesting tests here are the refusals. Inference is admissible only
because it cannot become an attribution -- so what has to hold is that it
never invents a name, never outranks an explicit declaration, and never
records a guess as the operator's word.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions, Message, ResultMessage
from fixtures_bundle import registered_source_and_artefact

from jake_tools.claude import AgentSpec, ClaudeAgent
from jake_tools.transcripts.bundle.adapters import AdapterError, adapt_obsidian_note
from jake_tools.transcripts.bundle.attendees import (
    ExternalParticipant,
    infer_attendees,
    name_is_supported,
    operator_participants,
)
from jake_tools.transcripts.bundle.components import ParticipantDeclarationSource
from jake_tools.transcripts.bundle.store import BundleStore

_OPS_LOG = """---
tags:
  - ops-log
date: "[[July 29, 2026]]"
---

- Chatting with [[Steven Crawford]] and [[Matt Lavender]] about training
- Referring to the [[SES Unit Operational Profile]]
- [[Michael Bryan]] can get data to help correlate attendance
- [[Steven Crawford]] and [[Rob Crawford]] are available to run [[On-road Driving]]
"""


def _agent_returning(payload: dict[str, Any]) -> ClaudeAgent:
    def _run_query(
        *, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        async def stream() -> AsyncIterator[Message]:
            yield ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="fixture",
                total_cost_usd=None,
                usage={"input_tokens": 0, "output_tokens": 0},
                result=None,
                structured_output=payload,
            )

        return stream()

    return ClaudeAgent(defaults=AgentSpec(model="fixture-model"), run_query=_run_query)


def _store(tmp_path: Path) -> BundleStore:
    store = BundleStore(tmp_path / "bundle")
    store.create_bundle()
    return store


# -- the fabrication guard ---------------------------------------------------


def test_a_name_the_note_does_not_contain_is_rejected(tmp_path: Path) -> None:
    """The guard that makes inference admissible at all: a name from
    nowhere is dropped, not stored with a confident-looking provenance."""
    note = tmp_path / "ops-log.md"
    note.write_text(_OPS_LOG, encoding="utf-8")
    agent = _agent_returning(
        {
            "attendees": [
                {"display_name": "Steven Crawford", "evidence": "Chatting with"},
                {"display_name": "Wilhelmina Fortesque", "evidence": "invented"},
            ]
        }
    )

    resolved = asyncio.run(infer_attendees(agent, note_path=note, note_text=_OPS_LOG))

    assert [p.display_name for p in resolved] == ["Steven Crawford"]


def test_a_partial_surname_match_is_not_support() -> None:
    """The note names Rob Crawford, so a bare surname match would wave
    through "Sarah Crawford" -- every word has to appear."""
    haystack = "chatting with steven crawford and matt lavender about training"

    assert name_is_supported("Steven Crawford", haystack=haystack)
    assert not name_is_supported("Sarah Crawford", haystack=haystack)


def test_a_one_letter_name_is_not_support() -> None:
    """Without a floor, a single character matches nearly any document."""
    assert not name_is_supported("A", haystack="a note about a meeting")


def test_inference_returning_nothing_usable_is_empty_not_an_error(
    tmp_path: Path,
) -> None:
    """A note with no discernible people is a real situation, and the
    caller's own refusal names the fix better than one from in here."""
    note = tmp_path / "ops-log.md"
    note.write_text("- some notes about a policy\n", encoding="utf-8")

    resolved = asyncio.run(
        infer_attendees(
            agent=_agent_returning({"attendees": []}),
            note_path=note,
            note_text="- some notes about a policy\n",
        )
    )

    assert resolved == ()


# -- provenance --------------------------------------------------------------


def test_an_inferred_name_is_never_recorded_as_the_operator_s_word(
    tmp_path: Path,
) -> None:
    """M22: recording a guess as an operator assertion would be a lie
    about exactly the field M19 exists to protect."""
    note = tmp_path / "ops-log.md"
    note.write_text(_OPS_LOG, encoding="utf-8")
    agent = _agent_returning(
        {"attendees": [{"display_name": "Matt Lavender", "evidence": "Chatting with"}]}
    )

    resolved = asyncio.run(infer_attendees(agent, note_path=note, note_text=_OPS_LOG))

    assert resolved[0].declaration_source == ParticipantDeclarationSource.NOTE_INFERRED
    assert "inferred from" in resolved[0].declaration_evidence


def test_the_evidence_quote_is_carried_into_the_record(tmp_path: Path) -> None:
    """A reviewer choosing between names should be able to see what the
    guess was based on."""
    note = tmp_path / "ops-log.md"
    note.write_text(_OPS_LOG, encoding="utf-8")
    agent = _agent_returning(
        {
            "attendees": [
                {
                    "display_name": "Steven Crawford",
                    "evidence": "Chatting with Steven Crawford about training",
                }
            ]
        }
    )

    resolved = asyncio.run(infer_attendees(agent, note_path=note, note_text=_OPS_LOG))

    assert "Chatting with Steven Crawford" in resolved[0].declaration_evidence


def test_operator_declared_names_keep_the_operator_source() -> None:
    resolved = operator_participants(("Michael Bryan", "Michael Bryan", ""))

    assert [p.display_name for p in resolved] == ["Michael Bryan"]
    assert resolved[0].declaration_source == ParticipantDeclarationSource.OPERATOR


# -- precedence, at the adapter --------------------------------------------


def test_frontmatter_attendees_win_over_an_inferred_name(tmp_path: Path) -> None:
    """M22's resolution order. An explicit list is a declaration and beats
    a guess, so the record keeps frontmatter provenance."""
    note = tmp_path / "meeting.md"
    note.write_text(
        '---\nAttendees:\n  - "[[Michael Bryan]]"\n---\n\n- notes\n', encoding="utf-8"
    )
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=note.read_bytes())

    adaptation = adapt_obsidian_note(
        store,
        note_artefact_id=artefact.artefact_id,
        note_path=note,
        extra_participants=(
            ExternalParticipant(
                display_name="Michael Bryan",
                declaration_source=ParticipantDeclarationSource.NOTE_INFERRED,
                declaration_evidence="inferred",
            ),
        ),
    )

    participant = adaptation.participants.participants[0]
    assert (
        participant.declaration_source == ParticipantDeclarationSource.NOTE_FRONTMATTER
    )


def test_an_inferred_participant_is_declared_never_speaking_evidenced(
    tmp_path: Path,
) -> None:
    """M19: declaration is presence evidence, not speech evidence -- and an
    inferred declaration is weaker still. Nothing about being in the room
    says you said any particular thing."""
    note = tmp_path / "ops-log.md"
    note.write_text(_OPS_LOG, encoding="utf-8")
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=note.read_bytes())

    adaptation = adapt_obsidian_note(
        store,
        note_artefact_id=artefact.artefact_id,
        note_path=note,
        extra_participants=(
            ExternalParticipant(
                display_name="Steven Crawford",
                declaration_source=ParticipantDeclarationSource.NOTE_INFERRED,
                declaration_evidence="inferred",
            ),
        ),
    )

    assert all(
        participant.status.value == "declared"
        for participant in adaptation.participants.participants
    )


def test_a_note_with_no_participants_at_all_is_still_refused(
    tmp_path: Path,
) -> None:
    """Inference is a fallback, not a guarantee. When it finds nobody the
    note supports, ingest still refuses rather than assembling a document
    whose review step has an empty menu."""
    note = tmp_path / "ops-log.md"
    note.write_text("- notes about a policy\n", encoding="utf-8")
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=note.read_bytes())

    with pytest.raises(AdapterError, match="nothing to declare"):
        adapt_obsidian_note(
            store, note_artefact_id=artefact.artefact_id, note_path=note
        )
