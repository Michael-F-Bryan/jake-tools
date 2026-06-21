from collections.abc import Callable
from typing import Any, Literal, Protocol, TypeVar, cast

from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from run_agent import AIAgent

from .prompting import StructuredPrompt


DEFAULT_MODEL = "gpt-5.4-mini"


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


def _sum_int(results: list[HermesResult], field_name: str) -> int:
    return sum(getattr(result, field_name) for result in results)


def _sum_float(results: list[HermesResult], field_name: str) -> float:
    return sum(getattr(result, field_name) for result in results)


def _combine_results(results: list[HermesResult]) -> HermesResult:
    if not results:
        raise ValueError("cannot combine zero Hermes results")

    first = results[0]
    last = results[-1]
    return HermesResult(
        final_response=last.final_response,
        last_reasoning=last.last_reasoning,
        messages=[message for result in results for message in result.messages],
        api_calls=_sum_int(results, "api_calls"),
        completed=last.completed,
        turn_exit_reason=last.turn_exit_reason,
        failed=any(result.failed for result in results),
        partial=any(result.partial for result in results),
        interrupted=any(result.interrupted for result in results),
        response_transformed=any(result.response_transformed for result in results),
        response_previewed=any(result.response_previewed for result in results),
        model=last.model or first.model,
        provider=last.provider or first.provider,
        base_url=last.base_url or first.base_url,
        input_tokens=_sum_int(results, "input_tokens"),
        output_tokens=_sum_int(results, "output_tokens"),
        cache_read_tokens=_sum_int(results, "cache_read_tokens"),
        cache_write_tokens=_sum_int(results, "cache_write_tokens"),
        reasoning_tokens=_sum_int(results, "reasoning_tokens"),
        prompt_tokens=_sum_int(results, "prompt_tokens"),
        completion_tokens=_sum_int(results, "completion_tokens"),
        total_tokens=_sum_int(results, "total_tokens"),
        last_prompt_tokens=last.last_prompt_tokens,
        estimated_cost_usd=_sum_float(results, "estimated_cost_usd"),
        cost_status=last.cost_status or first.cost_status,
        cost_source=last.cost_source or first.cost_source,
        session_id=last.session_id or first.session_id,
        error=last.error
        or next((result.error for result in results if result.error), None),
        failure_reason=last.failure_reason
        or next(
            (result.failure_reason for result in results if result.failure_reason), None
        ),
        guardrail=last.guardrail
        or next((result.guardrail for result in results if result.guardrail), None),
        pending_steer=last.pending_steer,
        interrupt_message=last.interrupt_message,
    )


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


class AgentConversation(Protocol):
    def run_conversation(
        self, user_message: str, *args: Any, **kwargs: Any
    ) -> dict[str, Any]: ...


class AgentSpec(BaseModel):
    """Configuration for constructing a Hermes worker agent."""

    model: str
    provider: str = ""
    enabled_toolsets: list[str] = Field(default_factory=list)
    system_prompt: str | None = None
    parent_session_id: str | None = None
    max_iterations: int | None = None
    session_db: Any | None = None


AgentFactory = Callable[[AgentSpec], AgentConversation]


def _default_agent_factory(spec: AgentSpec) -> AgentConversation:
    kwargs: dict[str, Any] = {
        "model": spec.model,
        "provider": spec.provider,
        "quiet_mode": True,
        "enabled_toolsets": spec.enabled_toolsets,
    }
    if spec.system_prompt is not None:
        kwargs["ephemeral_system_prompt"] = spec.system_prompt
    if spec.parent_session_id is not None:
        kwargs["parent_session_id"] = spec.parent_session_id
    if spec.session_db is not None:
        kwargs["session_db"] = spec.session_db
    if spec.max_iterations is not None:
        kwargs["max_iterations"] = spec.max_iterations
    return AIAgent(**kwargs)


class Hermes(BaseModel):
    """A high-level wrapper around the Hermes agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Pin a cheap/fast tool-workflow model unless the CLI caller overrides it.
    default_model: str = DEFAULT_MODEL
    agent_factory: AgentFactory = Field(default=_default_agent_factory)

    def new_agent(
        self, model: str | None = None, provider: str | None = None
    ) -> AgentConversation:
        spec = AgentSpec(model=model or self.default_model, provider=provider or "")
        return self.agent_factory(spec)

    def oneshot(self, prompt: str) -> HermesResult:
        agent = self.new_agent()
        result = agent.run_conversation(prompt)
        return HermesResult.model_validate(result)

    def _parse_structured_response[T: BaseModel](self, response: Any, model_type: type[T]) -> T:
        if response is None:
            raise ValueError("No response from Hermes")

        if isinstance(response, str):
            return model_type.model_validate_json(response)

        return model_type.model_validate(response)

    def run_structured[T: BaseModel](self, prompt: StructuredPrompt[T]) -> T:
        """Render a typed prompt and parse the reply into its response model."""
        payload, _ = self.run_structured_with_result(prompt)
        return payload

    def run_structured_with_result[T: BaseModel](
        self, prompt: StructuredPrompt[T]
    ) -> tuple[T, HermesResult]:
        """Render a typed prompt and return both parsed payload and Hermes usage."""
        response_model = cast(type[T], prompt.response_model)
        return self._oneshot_structured_with_result(
            prompt.render(), response_model
        )

    def run_agent_structured[T: BaseModel](
        self,
        spec: AgentSpec,
        prompt: StructuredPrompt[T],
    ) -> tuple[T, HermesResult]:
        """Run a structured prompt through a scoped, possibly tool-enabled agent."""
        response_model = cast(type[T], prompt.response_model)
        return self._structured_with_agent(self.agent_factory(spec), prompt.render(), response_model)

    def _oneshot_structured[T: BaseModel](self, prompt: str, model_type: type[T]) -> T:
        payload, _ = self._oneshot_structured_with_result(prompt, model_type)
        return payload

    def _oneshot_structured_with_result[T: BaseModel](
        self, prompt: str, model_type: type[T]
    ) -> tuple[T, HermesResult]:
        """Run Hermes and parse the reply into ``model_type``.

        Retries once with an explicit repair prompt when the initial response does
        not validate as the requested structured payload.
        """
        return self._structured_with_agent(self.new_agent(), prompt, model_type)

    def _structured_with_agent[T: BaseModel](
        self,
        agent: AgentConversation,
        prompt: str,
        model_type: type[T],
    ) -> tuple[T, HermesResult]:
        prompt = ONESHOT_STRUCTURED_PROMPT.render(
            schema_json=model_type.model_json_schema(),
            input=prompt,
        )
        result = HermesResult.model_validate(agent.run_conversation(prompt))

        try:
            payload = self._parse_structured_response(result.response, model_type)
            return payload, result
        except (ValidationError, ValueError) as exc:
            repair_prompt = ONESHOT_STRUCTURED_INVALID_JSON_PROMPT.render(
                last_error=str(exc),
                previous_response=result.response,
            )

        repair_result = HermesResult.model_validate(
            agent.run_conversation(repair_prompt)
        )
        payload = self._parse_structured_response(repair_result.response, model_type)
        combined = _combine_results([result, repair_result])
        return payload, combined
