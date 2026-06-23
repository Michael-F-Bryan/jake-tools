from typing import Any

import pytest

from jake_tools.hermes import Hermes
from jake_tools.transcripts.polish import polish_transcript


class StubAgent:
    """Fake LLM agent injected at the `AgentConversation` seam."""

    def __init__(self, final_response: str | None) -> None:
        self.final_response = final_response
        self.prompts: list[str] = []

    def run_conversation(self, user_message: str) -> dict[str, Any]:
        self.prompts.append(user_message)
        return {"final_response": self.final_response}


def _hermes_returning(final_response: str | None) -> tuple[Hermes, StubAgent]:
    agent = StubAgent(final_response)
    hermes = Hermes(agent_factory=lambda _spec: agent)
    return hermes, agent


def test_polish_transcript_returns_agent_text() -> None:
    hermes, agent = _hermes_returning("Polished transcript")

    polished = polish_transcript(hermes, "raw transcript text")

    assert polished == "Polished transcript"
    assert "raw transcript text" in agent.prompts[0]


def test_polish_transcript_raises_when_reply_has_no_text() -> None:
    hermes, _ = _hermes_returning(None)

    with pytest.raises(ValueError, match="No response from Hermes"):
        polish_transcript(hermes, "raw transcript text")
