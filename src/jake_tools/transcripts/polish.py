from __future__ import annotations

from typing import ClassVar

from ..hermes import Hermes
from ..prompting import Prompt


class TranscriptSkillPolishPrompt(Prompt):
    template: ClassVar[
        str
    ] = """
You are a helpful assistant that polishes transcripts.

Polish the following transcript using the `transcript-polisher` skill:

````md
{{ transcript }}
````

Respond with just the polished transcript, no other text, additional commentary, or the surrounding markdown code block.
"""

    transcript: str


def polish_transcript(hermes: Hermes, transcript: str) -> str:
    result = hermes.oneshot(TranscriptSkillPolishPrompt(transcript=transcript).render())
    if not result.response:
        raise ValueError("No response from Hermes")

    return result.response
