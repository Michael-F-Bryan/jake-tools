from typing import ClassVar

import pytest
from jinja2 import UndefinedError

from jake_tools.hermes import AgentSpec, Hermes
from jake_tools.prompting import Prompt
from jake_tools.transcripts.models import (
    Chapter,
    ChaptersPayload,
    MeetingMinutes,
    SpeakerMapping,
    TranscriptTurn,
    TranscriptTurnsPayload,
)
from jake_tools.transcripts.stages import (
    ChapteringPrompt,
    MeetingMinutesPrompt,
    SpeakerMappingPrompt,
    TranscriptPolishPrompt,
)


class FakeAgent:
    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def run_conversation(self, user_message: str) -> dict:
        self.prompts.append(user_message)
        return self._responses.pop(0)


def hermes_with(agent: FakeAgent) -> Hermes:
    def factory(_spec: AgentSpec) -> FakeAgent:
        return agent

    return Hermes(agent_factory=factory)


def _turns() -> list[TranscriptTurn]:
    return [TranscriptTurn(start=0.0, end=2.0, speaker="SPEAKER_01", text="Hello team")]


def test_template_referencing_unknown_variable_is_rejected_on_definition() -> None:
    with pytest.raises(TypeError, match="not declared as fields"):

        class _Bad(Prompt):
            template: ClassVar[str] = "Hello {{ name }}"


def test_field_unused_by_template_is_rejected_on_definition() -> None:
    with pytest.raises(TypeError, match="never uses"):

        class _Bad(Prompt):
            template: ClassVar[str] = "Hello there"

            name: str


def test_strict_undefined_catches_mistyped_nested_access_at_render() -> None:
    class Risky(Prompt):
        template: ClassVar[str] = "Name: {{ person.naem }}"

        person: dict[str, str]

    with pytest.raises(UndefinedError):
        Risky(person={"name": "Ada"}).render()


def test_speaker_mapping_prompt_embeds_typed_inputs() -> None:
    rendered = SpeakerMappingPrompt(
        title="Vet West call",
        attendees=["Michael Bryan", "Vet West"],
        turns=_turns(),
    ).render()

    assert "Vet West call" in rendered
    assert "Michael Bryan" in rendered
    assert "Hello team" in rendered


def test_meeting_minutes_prompt_omits_chapter_block_when_absent() -> None:
    without = MeetingMinutesPrompt(turns=_turns()).render()
    assert "Chapter plan:" not in without

    with_chapters = MeetingMinutesPrompt(
        turns=_turns(),
        chapters=[Chapter(title="Kickoff", start=0, end=30, summary="Start")],
    ).render()
    assert "Chapter plan:" in with_chapters
    assert "Kickoff" in with_chapters


def test_every_structured_prompt_binds_a_response_model() -> None:
    assert SpeakerMappingPrompt.__dict__["response_model"] is SpeakerMapping
    assert ChapteringPrompt.__dict__["response_model"] is ChaptersPayload
    assert MeetingMinutesPrompt.__dict__["response_model"] is MeetingMinutes
    assert TranscriptPolishPrompt.__dict__["response_model"] is TranscriptTurnsPayload


def test_run_structured_renders_prompt_and_parses_response_model() -> None:
    agent = FakeAgent(
        [
            {
                "final_response": '{"summary": "Quick sync", "key_points": ["Shipped it"]}',
                "completed": True,
                "api_calls": 1,
            },
        ]
    )
    hermes = hermes_with(agent)

    minutes, result = hermes.run_structured_with_result(
        MeetingMinutesPrompt(turns=_turns())
    )

    assert minutes == MeetingMinutes(summary="Quick sync", key_points=["Shipped it"])
    assert "Hello team" in agent.prompts[0]
    assert result.api_calls == 1


def test_run_structured_response_model_is_reflected_in_the_request_schema() -> None:
    agent = FakeAgent(
        [
            {"final_response": '{"summary": "s", "key_points": []}', "completed": True},
        ]
    )
    hermes = hermes_with(agent)

    hermes.run_structured(MeetingMinutesPrompt(turns=_turns()))

    assert "key_points" in agent.prompts[0]
