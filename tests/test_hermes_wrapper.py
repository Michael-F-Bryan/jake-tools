from pydantic import BaseModel

from jake_tools.hermes import Hermes, HermesResult


class Payload(BaseModel):
    answer: int


class FakeAgent:
    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def run_conversation(self, prompt: str) -> dict:
        self.prompts.append(prompt)
        return self._responses.pop(0)


def test_hermes_result_aliases_final_response() -> None:
    result = HermesResult.model_validate({"final_response": "ok", "completed": True})
    assert result.response == "ok"


def test_oneshot_structured_parses_valid_json_without_repair(monkeypatch) -> None:
    agent = FakeAgent([
        {"final_response": '{"answer": 42}', "completed": True},
    ])
    hermes = Hermes()
    monkeypatch.setattr(Hermes, "new_agent", lambda self, model=None, provider=None: agent)

    result = hermes.oneshot_structured("Return the answer.", Payload)

    assert result == Payload(answer=42)
    assert len(agent.prompts) == 1


def test_oneshot_structured_repairs_invalid_json_once(monkeypatch) -> None:
    agent = FakeAgent([
        {"final_response": 'nope', "completed": True},
        {"final_response": '{"answer": 7}', "completed": True},
    ])
    hermes = Hermes()
    monkeypatch.setattr(Hermes, "new_agent", lambda self, model=None, provider=None: agent)

    result = hermes.oneshot_structured("Return the answer.", Payload)

    assert result == Payload(answer=7)
    assert len(agent.prompts) == 2
    assert "Validation error:" in agent.prompts[1]
    assert "Previous response:" in agent.prompts[1]


def test_oneshot_structured_rejects_empty_response(monkeypatch) -> None:
    agent = FakeAgent([
        {"final_response": None, "completed": True},
        {"final_response": None, "completed": True},
    ])
    hermes = Hermes()
    monkeypatch.setattr(Hermes, "new_agent", lambda self, model=None, provider=None: agent)

    try:
        hermes.oneshot_structured("Return the answer.", Payload)
    except ValueError as exc:
        assert str(exc) == "No response from Hermes"
    else:
        raise AssertionError("expected ValueError")
