from typing import ClassVar

from pydantic import BaseModel

from jake_tools.hermes import Hermes, HermesResult
from jake_tools.prompting import StructuredPrompt


class Payload(BaseModel):
    answer: int


class PayloadPrompt(StructuredPrompt[Payload]):
    response_model: ClassVar[type[BaseModel]] = Payload
    template: ClassVar[str] = "{{ instruction }}"

    instruction: str


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


def test_run_structured_with_result_parses_valid_json_without_repair() -> None:
    agent = FakeAgent(
        [
            {
                "final_response": '{"answer": 42}',
                "completed": True,
                "api_calls": 1,
                "total_tokens": 12,
                "estimated_cost_usd": 0.03,
            },
        ]
    )
    hermes = hermes_with(agent)

    payload, result = hermes.run_structured_with_result(
        PayloadPrompt(instruction="Return the answer.")
    )

    assert payload == Payload(answer=42)
    assert len(agent.prompts) == 1
    assert result == HermesResult(
        final_response='{"answer": 42}',
        completed=True,
        api_calls=1,
        total_tokens=12,
        estimated_cost_usd=0.03,
    )


def test_run_structured_with_result_repairs_invalid_json_once() -> None:
    agent = FakeAgent(
        [
            {
                "final_response": "nope",
                "completed": True,
                "api_calls": 1,
                "input_tokens": 11,
                "output_tokens": 2,
                "total_tokens": 13,
                "estimated_cost_usd": 0.01,
            },
            {
                "final_response": '{"answer": 7}',
                "completed": True,
                "api_calls": 1,
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 10,
                "estimated_cost_usd": 0.02,
            },
        ]
    )
    hermes = hermes_with(agent)

    payload, result = hermes.run_structured_with_result(
        PayloadPrompt(instruction="Return the answer.")
    )

    assert payload == Payload(answer=7)
    assert len(agent.prompts) == 2
    assert "Validation error:" in agent.prompts[1]
    assert "Previous response:" in agent.prompts[1]
    assert result.api_calls == 2
    assert result.input_tokens == 18
    assert result.output_tokens == 5
    assert result.total_tokens == 23
    assert result.estimated_cost_usd == 0.03


def test_run_structured_with_result_returns_payload_and_usage() -> None:
    agent = FakeAgent(
        [
            {
                "final_response": '{"answer": 42}',
                "completed": True,
                "api_calls": 1,
                "total_tokens": 9,
            },
        ]
    )
    hermes = hermes_with(agent)

    payload, result = hermes.run_structured_with_result(
        PayloadPrompt(instruction="Return the answer.")
    )

    assert payload == Payload(answer=42)
    assert result.total_tokens == 9


def test_oneshot_structured_rejects_empty_response() -> None:
    agent = FakeAgent(
        [
            {"final_response": None, "completed": True},
            {"final_response": None, "completed": True},
        ]
    )
    hermes = hermes_with(agent)

    try:
        hermes._oneshot_structured("Return the answer.", Payload)
    except ValueError as exc:
        assert str(exc) == "No response from Hermes"
    else:
        raise AssertionError("expected ValueError")
