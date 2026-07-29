from typing import ClassVar

import pytest
from agent_fakes import fake_agent, scripted_query_of, structured
from jinja2 import UndefinedError

from jake_tools.prompting import Prompt
from jake_tools.transcripts.models import (
    ChapterPlan,
    MeetingMinutes,
    SourceNotePlan,
    SpeakerMapping,
    TranscriptTurn,
)
from jake_tools.transcripts.stages import (
    ChapterTitlePrompt,
    MinutesPrompt,
    PolishStagePrompt,
    SourceNotePlanPrompt,
    SpeakerMapPrompt,
)


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


def test_speaker_map_prompt_embeds_typed_inputs() -> None:
    rendered = SpeakerMapPrompt(
        attendees=["Michael Bryan", "Vet West"],
        turns=_turns(),
    ).render()

    assert "Michael Bryan" in rendered
    assert "Hello team" in rendered


def test_prompt_with_correction_field_surfaces_the_previous_error() -> None:
    without = SpeakerMapPrompt(attendees=["Michael Bryan"], turns=_turns()).render()
    assert "Your previous response was invalid" not in without

    with_correction = SpeakerMapPrompt(
        attendees=["Michael Bryan"],
        turns=_turns(),
        correction="'mapping' is a required property",
    ).render()
    assert "Your previous response was invalid" in with_correction
    assert "'mapping' is a required property" in with_correction


def test_every_structured_prompt_binds_a_response_model() -> None:
    assert SpeakerMapPrompt.__dict__["response_model"] is SpeakerMapping
    assert ChapterTitlePrompt.__dict__["response_model"] is ChapterPlan
    assert MinutesPrompt.__dict__["response_model"] is MeetingMinutes
    assert PolishStagePrompt.__dict__["response_model"].__name__ == "PolishStagePayload"
    assert SourceNotePlanPrompt.__dict__["response_model"] is SourceNotePlan


async def test_run_structured_renders_prompt_and_parses_response_model() -> None:
    agent = fake_agent(
        structured({"summary": "Quick sync", "key_points": ["Shipped it"]})
    )

    minutes, result = await agent.run_structured(
        MinutesPrompt(turns=_turns(), chapters=[])
    )

    assert minutes == MeetingMinutes(summary="Quick sync", key_points=["Shipped it"])
    assert "Hello team" in scripted_query_of(agent).prompts[0]
    assert result.usage.api_calls == 1


async def test_run_structured_response_model_is_reflected_in_the_request_schema() -> (
    None
):
    agent = fake_agent(structured({"summary": "s", "key_points": []}))

    await agent.run_structured(MinutesPrompt(turns=_turns(), chapters=[]))

    # The schema travels in output_format rather than inside the prompt.
    schema = scripted_query_of(agent).options[0].output_format
    assert schema is not None
    assert "key_points" in schema["schema"]["properties"]
