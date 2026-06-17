from jinja2 import Template

from ..hermes import Hermes

PROMPT = Template(
    """
You are a helpful assistant that polishes transcripts.

Polish the following transcript using the `transcript-polisher` skill:

````md
{{ transcript }}
````

Respond with just the polished transcript, no other text, additional commentary, or the surrounding markdown code block.
"""
)


def polish_transcript(hermes: Hermes, transcript: str) -> str:
    result = hermes.oneshot(PROMPT.render(transcript=transcript))
    if not result.response:
        raise ValueError("No response from Hermes")

    print(result.model_dump_json(indent=2))
    return result.response
