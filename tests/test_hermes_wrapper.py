from pydantic import BaseModel

from jake_tools.hermes import Hermes, HermesResult


class Payload(BaseModel):
    answer: int


class FakeAgent:
    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def run_conversation(self, user_message: str) -> dict:
        self.prompts.append(user_message)
        return self._responses.pop(0)


def hermes_with(agent: FakeAgent) -> Hermes:
    return Hermes(agent_factory=lambda model, provider: agent)


def test_hermes_result_aliases_final_response() -> None:
    result = HermesResult.model_validate({"final_response": "ok", "completed": True})
    assert result.response == "ok"


def test_oneshot_structured_parses_valid_json_without_repair() -> None:
    agent = FakeAgent([
        {"final_response": '{"answer": 42}', "completed": True},
    ])
    hermes = hermes_with(agent)

    result = hermes.oneshot_structured("Return the answer.", Payload)

    assert result == Payload(answer=42)
    assert len(agent.prompts) == 1


def test_oneshot_structured_repairs_invalid_json_once() -> None:
    agent = FakeAgent([
        {"final_response": 'nope', "completed": True},
        {"final_response": '{"answer": 7}', "completed": True},
    ])
    hermes = hermes_with(agent)

    result = hermes.oneshot_structured("Return the answer.", Payload)

    assert result == Payload(answer=7)
    assert len(agent.prompts) == 2
    assert "Validation error:" in agent.prompts[1]
    assert "Previous response:" in agent.prompts[1]


def test_oneshot_structured_rejects_empty_response() -> None:
    agent = FakeAgent([
        {"final_response": None, "completed": True},
        {"final_response": None, "completed": True},
    ])
    hermes = hermes_with(agent)

    try:
        hermes.oneshot_structured("Return the answer.", Payload)
    except ValueError as exc:
        assert str(exc) == "No response from Hermes"
    else:
        raise AssertionError("expected ValueError")
