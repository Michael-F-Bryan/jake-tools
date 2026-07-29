import pytest
from agent_fakes import AgentTurn, fake_agent, scripted_query_of

from jake_tools.transcripts.polish import polish_transcript


async def test_polish_transcript_returns_agent_text() -> None:
    agent = fake_agent(AgentTurn(text="Polished transcript"))

    polished = await polish_transcript(agent, "raw transcript text")

    assert polished == "Polished transcript"
    assert "raw transcript text" in scripted_query_of(agent).prompts[0]


async def test_polish_transcript_raises_when_reply_has_no_text() -> None:
    agent = fake_agent(AgentTurn(text=None))

    with pytest.raises(ValueError, match="No response from the agent"):
        await polish_transcript(agent, "raw transcript text")
