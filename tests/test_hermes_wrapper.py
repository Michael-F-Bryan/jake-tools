from typing import ClassVar

from pydantic import BaseModel

from jake_tools.hermes import AgentSpec, Hermes, HermesResult, _default_agent_factory
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
    return Hermes(agent_factory=lambda spec: agent)


class RecordingFactory:
    def __init__(self, agent: FakeAgent):
        self.agent = agent
        self.specs: list[AgentSpec] = []

    def __call__(self, spec: AgentSpec) -> FakeAgent:
        self.specs.append(spec)
        return self.agent


class FakeAIAgent:
    kwargs: dict[str, object] | None = None

    def __init__(self, **kwargs: object):
        type(self).kwargs = kwargs

    def run_conversation(self, user_message: str) -> dict[str, object]:
        return {"final_response": "ok", "completed": True}


def test_hermes_result_aliases_final_response() -> None:
    result = HermesResult.model_validate({"final_response": "ok", "completed": True})
    assert result.response == "ok"


def test_new_agent_builds_agent_spec_for_backwards_compatible_call() -> None:
    agent = FakeAgent([])
    factory = RecordingFactory(agent)
    hermes = Hermes(default_model="default-model", agent_factory=factory)

    assert hermes.new_agent("model-a", "provider-a") is agent

    assert factory.specs == [AgentSpec(model="model-a", provider="provider-a")]


def test_new_agent_uses_default_model_and_empty_provider() -> None:
    agent = FakeAgent([])
    factory = RecordingFactory(agent)
    hermes = Hermes(default_model="default-model", agent_factory=factory)

    assert hermes.new_agent() is agent

    assert factory.specs == [AgentSpec(model="default-model", provider="")]


def test_default_agent_factory_maps_scoped_spec_to_ai_agent(monkeypatch) -> None:
    import jake_tools.hermes as hermes_module

    monkeypatch.setattr(hermes_module, "AIAgent", FakeAIAgent)

    agent = _default_agent_factory(
        AgentSpec(
            model="model-a",
            provider="provider-a",
            enabled_toolsets=["session_search"],
            system_prompt="system",
            parent_session_id="parent-1",
            max_iterations=3,
            session_db="/tmp/state.db",
        )
    )

    assert isinstance(agent, FakeAIAgent)
    assert FakeAIAgent.kwargs == {
        "model": "model-a",
        "provider": "provider-a",
        "quiet_mode": True,
        "enabled_toolsets": ["session_search"],
        "ephemeral_system_prompt": "system",
        "parent_session_id": "parent-1",
        "session_db": "/tmp/state.db",
        "max_iterations": 3,
    }


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


def test_run_structured_with_result_uses_requested_model_and_provider() -> None:
    agent = FakeAgent(
        [
            {
                "final_response": '{"answer": 42}',
                "completed": True,
            },
        ]
    )
    factory = RecordingFactory(agent)
    hermes = Hermes(default_model="default-model", agent_factory=factory)

    payload, _ = hermes.run_structured_with_result(
        PayloadPrompt(instruction="Return the answer."),
        model="model-a",
        provider="provider-a",
    )

    assert payload == Payload(answer=42)
    assert factory.specs == [AgentSpec(model="model-a", provider="provider-a")]


def test_run_structured_with_result_parses_fenced_json_without_repair() -> None:
    agent = FakeAgent(
        [
            {
                "final_response": '```json\n{"answer": 42}\n```',
                "completed": True,
                "api_calls": 1,
            },
        ]
    )
    hermes = hermes_with(agent)

    payload, result = hermes.run_structured_with_result(
        PayloadPrompt(instruction="Return the answer.")
    )

    assert payload == Payload(answer=42)
    assert result.api_calls == 1
    assert len(agent.prompts) == 1


def test_run_agent_structured_parses_fenced_json_without_repair() -> None:
    agent = FakeAgent(
        [
            {
                "final_response": '```\n{"answer": 12}\n```',
                "completed": True,
                "api_calls": 1,
            },
        ]
    )
    factory = RecordingFactory(agent)
    hermes = Hermes(agent_factory=factory)

    payload, result = hermes.run_agent_structured(
        AgentSpec(model="model-a", provider="provider-a", enabled_toolsets=["file"]),
        PayloadPrompt(instruction="Return the answer."),
    )

    assert payload == Payload(answer=12)
    assert result.api_calls == 1
    assert len(agent.prompts) == 1


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


def test_run_agent_structured_passes_scoped_agent_spec_to_factory() -> None:
    agent = FakeAgent(
        [
            {
                "final_response": '{"answer": 11}',
                "completed": True,
                "api_calls": 1,
            },
        ]
    )
    factory = RecordingFactory(agent)
    hermes = Hermes(agent_factory=factory)
    spec = AgentSpec(
        model="judgement-model",
        provider="openrouter",
        enabled_toolsets=["session_search", "file"],
        system_prompt="stay scoped",
        parent_session_id="parent-123",
        max_iterations=4,
        session_db="/tmp/hermes.db",
    )

    payload, result = hermes.run_agent_structured(
        spec,
        PayloadPrompt(instruction="Return the answer."),
    )

    assert payload == Payload(answer=11)
    assert result.api_calls == 1
    assert factory.specs == [spec]
    assert len(agent.prompts) == 1


def test_run_agent_structured_repairs_invalid_json_once() -> None:
    agent = FakeAgent(
        [
            {
                "final_response": "nope",
                "completed": True,
                "api_calls": 1,
                "input_tokens": 5,
                "output_tokens": 1,
                "total_tokens": 6,
                "estimated_cost_usd": 0.01,
            },
            {
                "final_response": '{"answer": 9}',
                "completed": True,
                "api_calls": 1,
                "input_tokens": 4,
                "output_tokens": 2,
                "total_tokens": 6,
                "estimated_cost_usd": 0.02,
            },
        ]
    )
    factory = RecordingFactory(agent)
    hermes = Hermes(agent_factory=factory)

    payload, result = hermes.run_agent_structured(
        AgentSpec(model="model-a", provider="provider-a", enabled_toolsets=["file"]),
        PayloadPrompt(instruction="Return the answer."),
    )

    assert payload == Payload(answer=9)
    assert len(factory.specs) == 1
    assert len(agent.prompts) == 2
    assert "Validation error:" in agent.prompts[1]
    assert result.api_calls == 2
    assert result.input_tokens == 9
    assert result.output_tokens == 3
    assert result.total_tokens == 12
    assert result.estimated_cost_usd == 0.03


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
