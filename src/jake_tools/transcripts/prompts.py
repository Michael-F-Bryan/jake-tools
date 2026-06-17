from __future__ import annotations

import json
from jinja2 import Template

SPEAKER_MAPPING_PROMPT = Template(
    """
You are the speaker-mapping specialist for an Obsidian meeting recording workflow.

Infer speaker names conservatively from the transcript turns below. If you cannot justify a real name from the transcript, leave the speaker unresolved.

Transcript turns:
{{ turns_json }}

Return JSON with:
- `mapping`: object keyed by speaker label with `{name, confidence, reason}`
- `unresolved`: list of unresolved speaker labels
- `notes`: short note on what was inferred automatically
""".strip()
)

CHAPTERING_PROMPT = Template(
    """
You are the chaptering specialist for an Obsidian meeting recording workflow.

Create broad thematic chapters for the transcript. Prefer topic shifts and agenda changes over rigid time slices. Chapters must be contiguous, in order, and cover the whole transcript.

Transcript turns:
{{ turns_json }}

Return JSON with one top-level key, `chapters`, whose value is a list of objects containing `title`, `start`, `end`, and `summary`.
""".strip()
)

MEETING_MINUTES_PROMPT = Template(
    """
You are the meeting-minutes specialist for an Obsidian meeting recording workflow.

Write faithful high-level meeting notes from the transcript. Preserve uncertainty when the transcript is unclear.
Prefer concrete outcomes, instructions, appointments, and next steps over generic summary prose.

Transcript turns:
{{ turns_json }}

{% if chapters_json %}
Chapter plan:
{{ chapters_json }}
{% endif %}

Return JSON with:
- `summary`: one-sentence overview of the meeting
- `key_points`: concise high-level meeting notes written as dot-point-sized statements
""".strip()
)


def render_speaker_mapping_prompt(turns: list[dict]) -> str:
    return SPEAKER_MAPPING_PROMPT.render(turns_json=json.dumps(turns, ensure_ascii=False, indent=2))


def render_chaptering_prompt(turns: list[dict]) -> str:
    return CHAPTERING_PROMPT.render(turns_json=json.dumps(turns, ensure_ascii=False, indent=2))


def render_minutes_prompt(turns: list[dict], chapters: list[dict] | None = None) -> str:
    return MEETING_MINUTES_PROMPT.render(
        turns_json=json.dumps(turns, ensure_ascii=False, indent=2),
        chapters_json=json.dumps(chapters, ensure_ascii=False, indent=2) if chapters else "",
    )
