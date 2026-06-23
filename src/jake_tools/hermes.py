from collections.abc import Callable
from typing import Any, Protocol, TypeVar, cast

from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from run_agent import AIAgent

from .ai_usage import Usage
from .prompting import Prompt, StructuredPrompt

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_PROVIDER = "openai-code"
TModel = TypeVar("TModel", bound=BaseModel)


class Reply(BaseModel):
    text: str | None = None
    error: str | None = None
    usage: Usage = Field(default_factory=Usage)

    @classmethod
    def from_run(cls, raw: dict[str, Any]) -> Reply:
        return cls(
            text=raw.get("final_response"),
            error=raw.get("error"),
            usage=Usage.from_run(raw),
        )


def _strip_markdown_json_fence(response: str) -> str:
    """Return bare JSON when a model wraps structured output in a Markdown fence."""

    stripped = response.strip()
    if not stripped.startswith("```"):
        return response

    lines = stripped.splitlines()
    if len(lines) < 2 or not lines[0].startswith("```"):
        return response

    closing_index = None
    for index in range(len(lines) - 1, 0, -1):
        if lines[index].strip() == "```":
            closing_index = index
            break

    if closing_index is None:
        return response

    return "\n".join(lines[1:closing_index]).strip()


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

Return only corrected JSON that matches the schema and includes all required fields. Populate all fields with relevant content based on the previous response and the schema. Preserve content from the previous response where possible. Do not include markdown fences or commentary.

Validation error:
{{ last_error }}

Previous response:
{{ previous_response }}
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

    def merge(self, override: AgentSpec | None) -> AgentSpec:
        if override is None:
            return self.model_copy(deep=True)
        merged = self.model_dump()
        merged.update(override.model_dump())
        return AgentSpec.model_validate(merged)


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

    defaults: AgentSpec = Field(
        default_factory=lambda: AgentSpec(
            model=DEFAULT_MODEL,
            provider=DEFAULT_PROVIDER,
        )
    )
    agent_factory: AgentFactory = Field(default=_default_agent_factory)

    def run(self, prompt: str | Prompt, spec: AgentSpec | None = None) -> Reply:
        rendered_prompt = prompt if isinstance(prompt, str) else prompt.render()
        resolved = self.defaults.merge(spec)
        agent = self.agent_factory(resolved)
        return Reply.from_run(agent.run_conversation(rendered_prompt))

    def run_structured(
        self,
        prompt: StructuredPrompt[TModel],
        spec: AgentSpec | None = None,
    ) -> tuple[TModel, Reply]:
        response_model = cast(type[TModel], prompt.response_model)
        wrapped_prompt = _render_schema_wrapped_prompt(prompt.render(), response_model)
        resolved = self.defaults.merge(spec)
        agent = self.agent_factory(resolved)

        first = Reply.from_run(agent.run_conversation(wrapped_prompt))
        try:
            payload = _parse_structured_response(first.text, response_model)
            return payload, first
        except (ValidationError, ValueError) as exc:
            repair_prompt = _render_repair_prompt(str(exc), first.text)

        repair = Reply.from_run(agent.run_conversation(repair_prompt))
        payload = _parse_structured_response(repair.text, response_model)
        return payload, Reply(
            text=repair.text,
            error=repair.error or first.error,
            usage=first.usage + repair.usage,
        )


def _render_schema_wrapped_prompt(prompt: str, model_type: type[BaseModel]) -> str:
    return ONESHOT_STRUCTURED_PROMPT.render(
        schema_json=model_type.model_json_schema(),
        input=prompt,
    )


def _render_repair_prompt(last_error: str, previous_response: str | None) -> str:
    return ONESHOT_STRUCTURED_INVALID_JSON_PROMPT.render(
        last_error=last_error,
        previous_response=previous_response,
    )


def _parse_structured_response[TModel: BaseModel](
    response: Any, model_type: type[TModel]
) -> TModel:
    if response is None:
        raise ValueError("No response from Hermes")
    if isinstance(response, str):
        return model_type.model_validate_json(_strip_markdown_json_fence(response))
    return model_type.model_validate(response)
