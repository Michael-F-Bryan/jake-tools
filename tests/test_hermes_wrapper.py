from typing import ClassVar

from pydantic import BaseModel

from jake_tools.hermes import AgentSpec, Hermes, Reply, _default_agent_factory
from jake_tools.prompting import Prompt, StructuredPrompt


class EchoPrompt(Prompt):
    template: ClassVar[str] = "Echo: {{ instruction }}"
    instruction: str


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
        return {"final_response": "ok"}


def test_reply_from_run_extracts_text_error_and_usage() -> None:
    reply = Reply.from_run(
        {
            "final_response": "ok",
            "error": "boom",
            "model": "model-a",
            "provider": "provider-a",
            "api_calls": 2,
            "total_tokens": 9,
            "estimated_cost_usd": 0.03,
        }
    )

    assert reply.text == "ok"
    assert reply.error == "boom"
    assert reply.usage.model == "model-a"
    assert reply.usage.provider == "provider-a"
    assert reply.usage.api_calls == 2
    assert reply.usage.total_tokens == 9
    assert reply.usage.estimated_cost_usd == 0.03


def test_agent_spec_merge_overrides_latest_values() -> None:
    base = AgentSpec(model="base", provider="openrouter", enabled_toolsets=["file"])
    override = AgentSpec(
        model="override", provider="", enabled_toolsets=["session_search"]
    )

    merged = base.merge(override)

    assert merged == AgentSpec(
        model="override",
        provider="",
        enabled_toolsets=["session_search"],
        system_prompt=None,
        parent_session_id=None,
        max_iterations=None,
        session_db=None,
    )


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


def test_run_accepts_prompt_model_and_passes_merged_spec() -> None:
    agent = FakeAgent([{"final_response": "ok", "model": "override"}])
    factory = RecordingFactory(agent)
    hermes = Hermes(
        defaults=AgentSpec(model="default-model", provider="default-provider"),
        agent_factory=factory,
    )

    reply = hermes.run(
        EchoPrompt(instruction="hello"),
        spec=AgentSpec(model="override-model", provider="", enabled_toolsets=["file"]),
    )

    assert reply.text == "ok"
    assert factory.specs == [
        AgentSpec(
            model="override-model",
            provider="",
            enabled_toolsets=["file"],
            system_prompt=None,
            parent_session_id=None,
            max_iterations=None,
            session_db=None,
        )
    ]
    assert agent.prompts == ["Echo: hello"]


def test_run_structured_parses_valid_json_without_repair() -> None:
    agent = FakeAgent(
        [
            {
                "final_response": '{"answer": 42}',
                "api_calls": 1,
                "total_tokens": 12,
                "estimated_cost_usd": 0.03,
            }
        ]
    )
    hermes = Hermes(agent_factory=lambda spec: agent)

    payload, reply = hermes.run_structured(
        PayloadPrompt(instruction="Return the answer.")
    )

    assert payload == Payload(answer=42)
    assert reply.usage.api_calls == 1
    assert reply.usage.total_tokens == 12
    assert reply.usage.estimated_cost_usd == 0.03
    assert len(agent.prompts) == 1


def test_run_structured_parses_fenced_json_without_repair() -> None:
    agent = FakeAgent(
        [
            {
                "final_response": '```json\n{"answer": 42}\n```',
                "api_calls": 1,
            }
        ]
    )
    hermes = Hermes(agent_factory=lambda spec: agent)

    payload, reply = hermes.run_structured(
        PayloadPrompt(instruction="Return the answer.")
    )

    assert payload == Payload(answer=42)
    assert reply.usage.api_calls == 1
    assert len(agent.prompts) == 1


def test_run_structured_repairs_invalid_json_once_and_accumulates_usage() -> None:
    agent = FakeAgent(
        [
            {
                "final_response": "nope",
                "model": "model-a",
                "provider": "provider-a",
                "api_calls": 1,
                "input_tokens": 11,
                "output_tokens": 2,
                "total_tokens": 13,
                "estimated_cost_usd": 0.01,
            },
            {
                "final_response": '{"answer": 7}',
                "model": "model-b",
                "provider": "provider-b",
                "api_calls": 1,
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 10,
                "estimated_cost_usd": 0.02,
            },
        ]
    )
    hermes = Hermes(agent_factory=lambda spec: agent)

    payload, reply = hermes.run_structured(
        PayloadPrompt(instruction="Return the answer.")
    )

    assert payload == Payload(answer=7)
    assert len(agent.prompts) == 2
    assert "Validation error:" in agent.prompts[1]
    assert "Previous response:" in agent.prompts[1]
    assert reply.usage.api_calls == 2
    assert reply.usage.input_tokens == 18
    assert reply.usage.output_tokens == 5
    assert reply.usage.total_tokens == 23
    assert reply.usage.estimated_cost_usd == 0.03
    assert reply.usage.model == "model-b"
    assert reply.usage.provider == "provider-b"


def test_run_structured_raises_on_missing_response_after_repair() -> None:
    agent = FakeAgent(
        [
            {"final_response": None},
            {"final_response": None},
        ]
    )
    hermes = Hermes(agent_factory=lambda spec: agent)

    try:
        hermes.run_structured(PayloadPrompt(instruction="Return the answer."))
    except ValueError as exc:
        assert str(exc) == "No response from Hermes"
    else:
        raise AssertionError("expected ValueError")
