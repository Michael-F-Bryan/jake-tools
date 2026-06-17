from collections.abc import Callable
from typing import Any, Literal, Protocol, TypeVar

from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from run_agent import AIAgent


class HermesMessage(BaseModel):
    """One message in the agent conversation history (OpenAI-compatible shape)."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[Any] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    reasoning: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class HermesGuardrailSignature(BaseModel):
    tool_name: str
    args_hash: str


class HermesGuardrail(BaseModel):
    """Metadata from ``ToolGuardrailDecision.to_metadata()`` when a guardrail halts the turn."""

    model_config = ConfigDict(extra="allow")

    action: Literal["allow", "warn", "block", "halt"] | str = "allow"
    code: str = "allow"
    message: str = ""
    tool_name: str = ""
    count: int = 0
    signature: HermesGuardrailSignature | None = None


class HermesResult(BaseModel):
    """Result dict returned by ``AIAgent.run_conversation()``."""

    final_response: str | None = None
    last_reasoning: str | None = None
    messages: list[HermesMessage] = Field(default_factory=list)
    api_calls: int = 0
    completed: bool = False
    turn_exit_reason: str | None = None
    failed: bool = False
    partial: bool = False
    interrupted: bool = False
    response_transformed: bool = False
    response_previewed: bool = False
    model: str | None = None
    provider: str | None = None
    base_url: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    last_prompt_tokens: int = 0
    estimated_cost_usd: float = 0.0
    cost_status: str | None = None
    cost_source: str | None = None
    session_id: str | None = None
    error: str | None = None
    failure_reason: str | None = None
    guardrail: HermesGuardrail | None = None
    pending_steer: str | None = None
    interrupt_message: str | None = None

    @property
    def response(self) -> str | None:
        return self.final_response


ONESHOT_STRUCTURED_PROMPT = Template(
    """
You are a helpful assistant that returns a structured response.

Return a JSON object matching this schema:
{{ schema_json }}

The input is:
{{ input }}
""".strip()
)

ONESHOT_STRUCTURED_INVALID_JSON_PROMPT = Template(
    """
The previous response did not validate against the required schema.

Validation error:
{{ last_error }}

Previous response:
{{ previous_response }}

Return only corrected JSON that matches the schema. No markdown fences and no commentary.
""".strip()
)

T = TypeVar("T", bound=BaseModel)


class AgentConversation(Protocol):
    def run_conversation(self, user_message: str, *args: Any, **kwargs: Any) -> dict[str, Any]: ...


AgentFactory = Callable[[str, str], AgentConversation]


def _default_agent_factory(model: str, provider: str) -> AgentConversation:
    return AIAgent(model=model, provider=provider, quiet_mode=True)


class Hermes(BaseModel):
    """A high-level wrapper around the Hermes agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    default_model: str = "gpt-5.4-mini"
    agent_factory: AgentFactory = Field(default=_default_agent_factory)

    def new_agent(
        self, model: str | None = None, provider: str | None = None
    ) -> AgentConversation:
        return self.agent_factory(model or self.default_model, provider or "")

    def oneshot(self, prompt: str) -> HermesResult:
        agent = self.new_agent()
        result = agent.run_conversation(prompt)
        return HermesResult.model_validate(result)

    def _parse_structured_response(self, response: Any, model_type: type[T]) -> T:
        if response is None:
            raise ValueError("No response from Hermes")

        if isinstance(response, str):
            return model_type.model_validate_json(response)

        return model_type.model_validate(response)

    def oneshot_structured(self, prompt: str, model_type: type[T]) -> T:
        """Run Hermes and parse the reply into ``model_type``.

        Retries once with an explicit repair prompt when the initial response does
        not validate as the requested structured payload.
        """
        prompt = ONESHOT_STRUCTURED_PROMPT.render(
            schema_json=model_type.model_json_schema(),
            input=prompt,
        )
        agent = self.new_agent()
        result = HermesResult.model_validate(agent.run_conversation(prompt))

        try:
            return self._parse_structured_response(result.response, model_type)
        except (ValidationError, ValueError) as exc:
            repair_prompt = ONESHOT_STRUCTURED_INVALID_JSON_PROMPT.render(
                last_error=str(exc),
                previous_response=result.response,
            )

        repair_result = HermesResult.model_validate(agent.run_conversation(repair_prompt))
        return self._parse_structured_response(repair_result.response, model_type)
