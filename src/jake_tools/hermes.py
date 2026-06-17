from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
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
    """Result dict returned by ``AIAgent.run_conversation()``.

    Hermes returns a plain dict whose keys vary slightly by exit path (success,
    interrupt, API failure, truncation, guardrail halt, etc.). The full success
    path in ``agent/conversation_loop.py`` always includes usage and session
    fields; early returns may omit some of the optional metadata below.
    """

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
        """Alias for ``final_response``."""
        return self.final_response


class Hermes(BaseModel):
    """
    A high-level wrapper around the Hermes agent.
    """

    default_model: str = "gpt-5.4-mini"

    def new_agent(
        self, model: str | None = None, provider: str | None = None
    ) -> AIAgent:
        return AIAgent(
            model=model or self.default_model,
            provider=provider or "",
        )

    def oneshot(self, prompt: str) -> HermesResult:
        agent = self.new_agent()
        result = agent.run_conversation(prompt)
        return HermesResult.model_validate(result)
