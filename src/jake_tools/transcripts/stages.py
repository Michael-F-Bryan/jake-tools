from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import ClassVar, Protocol

from pydantic import BaseModel, Field, ValidationError

from ..claude import Reply
from ..prompting import StructuredPrompt
from .errors import TranscriptError
from .merge import OPERATIONAL_CHATTER_RE
from .models import (
    ChapterPlan,
    MeetingMinutes,
    SourceArtifact,
    SourceNotePlan,
    SpeakerMapping,
    TranscriptArtifact,
    TranscriptTurn,
)
from .verify import verify_turns

_MARKDOWN_FENCE_RE = re.compile(r"^\s*```{3,}", re.MULTILINE)
_SOURCE_BOILERPLATE_RE = re.compile(
    r"^\s*##\s*(Meeting Notes|Chapters|Transcript)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_CONTENT_WORD_RE = re.compile(r"\b[\w']+\b")
MAX_POLISH_CONTENT_CUT_RATIO = 0.15


class StructuredAgent(Protocol):
    async def run_structured[TModel: BaseModel](
        self, prompt: StructuredPrompt[TModel]
    ) -> tuple[TModel, Reply]: ...


class StagePrimitiveError(TranscriptError):
    pass


class PolishStagePayload(BaseModel):
    turns: list[TranscriptTurn] = Field(default_factory=list)


class PolishStagePrompt(StructuredPrompt[PolishStagePayload]):
    response_model: ClassVar[type[BaseModel]] = PolishStagePayload
    template: ClassVar[str] = """
Polish these transcript turns conservatively for readability while preserving meaning.

Rules:
- Return JSON only.
- Keep turn order and turn count unchanged.
- Keep every turn's `start`, `end`, and `speaker` unchanged.
- Never include operational chatter.
- Never include markdown fences.
- Never include source boilerplate headings such as "## Transcript".

Source context:
{{ context }}

Transcript turns:
{{ turns | json }}
{% if correction %}

Your previous response was invalid: {{ correction }}
Return corrected JSON that fixes this issue.
{% endif %}
"""

    context: str = ""
    turns: list[TranscriptTurn]
    correction: str = ""


class SpeakerMapPrompt(StructuredPrompt[SpeakerMapping]):
    response_model: ClassVar[type[BaseModel]] = SpeakerMapping
    template: ClassVar[str] = """
Known attendees:
{{ attendees | json }}

Infer speaker identities conservatively from these transcript turns.
If uncertain, keep labels unresolved.

Transcript turns:
{{ turns | json }}
{% if correction %}

Your previous response was invalid: {{ correction }}
Return corrected JSON that fixes this issue.
{% endif %}
"""

    attendees: list[str]
    turns: list[TranscriptTurn]
    correction: str = ""


class ChapterTitlePrompt(StructuredPrompt[ChapterPlan]):
    response_model: ClassVar[type[BaseModel]] = ChapterPlan
    template: ClassVar[str] = """
Return a ChapterPlan with concise, descriptive chapter titles and summaries.
When draft chapters are provided, keep chapter boundaries unchanged.

Source context:
{{ context }}

Transcript turns:
{{ turns | json }}

Draft chapters:
{{ draft_chapters | json }}
{% if correction %}

Your previous response was invalid: {{ correction }}
Return corrected JSON that fixes this issue.
{% endif %}
"""

    context: str = ""
    turns: list[TranscriptTurn]
    draft_chapters: list[dict[str, object]]
    correction: str = ""


class MinutesPrompt(StructuredPrompt[MeetingMinutes]):
    response_model: ClassVar[type[BaseModel]] = MeetingMinutes
    template: ClassVar[str] = """
Write faithful minutes in structured JSON.
Prioritize outcomes, decisions, and concrete actions over generic summary prose.

Transcript turns:
{{ turns | json }}

Chapter plan:
{{ chapters | json }}
{% if correction %}

Your previous response was invalid: {{ correction }}
Return corrected JSON that fixes this issue.
{% endif %}
"""

    turns: list[TranscriptTurn]
    chapters: list[dict[str, object]]
    correction: str = ""


class SourceNotePlanPrompt(StructuredPrompt[SourceNotePlan]):
    response_model: ClassVar[type[BaseModel]] = SourceNotePlan
    template: ClassVar[str] = """
Create a faithful source-note plan for this recorded talk.

Return:
- a concise summary and 3-7 high-signal key points grounded only in the source
- concise chapter titles and summaries, keeping every draft chapter boundary unchanged

Preserve technical names and terminology. Do not add advice, credibility judgements,
or claims that are not present in the transcript.

Source metadata:
{{ source | json }}

Transcript turns:
{{ turns | json }}

Draft chapters:
{{ draft_chapters | json }}
{% if correction %}

Your previous response was invalid: {{ correction }}
Return corrected JSON that fixes this issue.
{% endif %}
"""

    source: dict[str, object]
    turns: list[TranscriptTurn]
    draft_chapters: list[dict[str, object]]
    correction: str = ""


def _with_correction[TModel: BaseModel](
    prompt: StructuredPrompt[TModel], correction: str
) -> StructuredPrompt[TModel]:
    """Feed the previous validation failure back into the retry, if the
    concrete prompt has a `correction` field to receive it."""
    if "correction" not in type(prompt).model_fields:
        return prompt
    return prompt.model_copy(update={"correction": correction})


async def run_structured_with_retries[TModel: BaseModel](
    agent: StructuredAgent, prompt: StructuredPrompt[TModel], *, max_attempts: int
) -> tuple[TModel, Reply]:
    if max_attempts < 1:
        raise StagePrimitiveError("max_attempts must be at least 1.")

    errors: list[str] = []
    attempt = prompt
    for _ in range(max_attempts):
        try:
            return await agent.run_structured(attempt)
        except (ValidationError, ValueError) as exc:
            errors.append(str(exc))
            attempt = _with_correction(prompt, errors[-1])

    raise StagePrimitiveError(
        "Structured response failed schema validation after retries: "
        + " | ".join(errors)
    )


def _contains_forbidden_boilerplate(text: str) -> bool:
    return bool(
        OPERATIONAL_CHATTER_RE.search(text)
        or _MARKDOWN_FENCE_RE.search(text)
        or _SOURCE_BOILERPLATE_RE.search(text)
    )


def _ensure_no_forbidden_output(turns: list[TranscriptTurn]) -> None:
    for turn in turns:
        if _contains_forbidden_boilerplate(turn.text):
            raise StagePrimitiveError(
                "Polished transcript failed output gate: "
                "boilerplate.no-operational-chatter/no-markdown-fences/no-source-boilerplate"
            )


def _turns_word_count(turns: list[TranscriptTurn]) -> int:
    return sum(len(_CONTENT_WORD_RE.findall(turn.text)) for turn in turns)


def _ensure_polish_preserves_content(
    source_turns: list[TranscriptTurn], polished_turns: list[TranscriptTurn]
) -> None:
    source_words = _turns_word_count(source_turns)
    if source_words == 0:
        return

    polished_words = _turns_word_count(polished_turns)
    cut_ratio = (source_words - polished_words) / source_words
    if cut_ratio <= MAX_POLISH_CONTENT_CUT_RATIO:
        return

    raise StagePrimitiveError(
        "Polished transcript failed audit gate: polish.content-retention "
        f"possible over-simplification; source_words={source_words}, "
        f"polished_words={polished_words}, cut_ratio={cut_ratio:.1%}, "
        f"max_allowed={MAX_POLISH_CONTENT_CUT_RATIO:.1%}"
    )


async def run_polish_stage(
    agent: StructuredAgent,
    transcript: TranscriptArtifact,
    *,
    context: str = "",
    max_attempts: int,
) -> tuple[TranscriptArtifact, Reply]:
    payload, reply = await run_structured_with_retries(
        agent,
        PolishStagePrompt(context=context, turns=transcript.turns),
        max_attempts=max_attempts,
    )
    _ensure_no_forbidden_output(payload.turns)
    _ensure_polish_preserves_content(transcript.turns, payload.turns)

    same_structure = len(payload.turns) == len(transcript.turns) and all(
        (source.start, source.end, source.speaker)
        == (polished.start, polished.end, polished.speaker)
        for source, polished in zip(transcript.turns, payload.turns, strict=True)
    )
    if not same_structure:
        raise StagePrimitiveError(
            "Polished transcript changed turn count, timestamps, or speakers."
        )

    polished = transcript.model_copy(update={"turns": payload.turns})
    verification = verify_turns(transcript, polished, affected_paths=[])
    if verification.failed_gate_ids:
        raise StagePrimitiveError(
            "Polished transcript failed verification gates: "
            + ", ".join(verification.failed_gate_ids)
        )
    return polished, reply


def _ordered_word_coverage(source: str, candidate: str) -> float:
    source_words = re.findall(r"[\w']+", source.casefold())
    if not source_words:
        return 1.0
    candidate_words = re.findall(r"[\w']+", candidate.casefold())
    retained = sum(
        block.size
        for block in SequenceMatcher(
            a=source_words, b=candidate_words, autojunk=False
        ).get_matching_blocks()
    )
    return retained / len(source_words)


async def run_youtube_polish_stage(
    agent: StructuredAgent,
    transcript: TranscriptArtifact,
    *,
    context: str,
    max_attempts: int,
    target_minutes: float = 5.0,
) -> tuple[TranscriptArtifact, list[tuple[str, Reply]]]:
    """Polish a long-form transcript in fixed-duration chunks.

    Long YouTube transcripts exceed a single structured-prompt budget, so this
    windows the transcript by ``target_minutes`` and runs `run_polish_stage`
    per window, gating each chunk's rewrite for the same source-fidelity
    invariants a single-shot polish would need.
    """
    if target_minutes <= 0:
        raise StagePrimitiveError("target_minutes must be greater than zero.")
    if not transcript.turns:
        raise StagePrimitiveError("Transcript has no turns to polish.")

    chunk_seconds = target_minutes * 60.0
    ranges: list[tuple[int, int]] = []
    chunk_start = 0
    window_start = transcript.turns[0].start
    for turn_index, turn in enumerate(transcript.turns):
        if turn_index > chunk_start and turn.start - window_start >= chunk_seconds:
            ranges.append((chunk_start, turn_index))
            chunk_start = turn_index
            window_start = turn.start
    ranges.append((chunk_start, len(transcript.turns)))

    polished_turns: list[TranscriptTurn] = []
    replies: list[tuple[str, Reply]] = []
    for chunk_number, (start_index, end_index) in enumerate(ranges, start=1):
        source_turns = transcript.turns[start_index:end_index]
        chunk = TranscriptArtifact(
            turns=source_turns,
            source_refs=[
                source_ref.model_copy(
                    update={"turn_index": source_ref.turn_index - start_index}
                )
                for source_ref in transcript.source_refs
                if start_index <= source_ref.turn_index < end_index
            ],
            speakers=transcript.speakers,
            warnings=transcript.warnings,
        )
        polished, reply = await run_polish_stage(
            agent,
            chunk,
            context=context,
            max_attempts=max_attempts,
        )
        for source_turn, polished_turn in zip(
            source_turns, polished.turns, strict=True
        ):
            same_boundary = (
                source_turn.start == polished_turn.start
                and source_turn.end == polished_turn.end
                and source_turn.speaker == polished_turn.speaker
            )
            faithful = (
                len(re.findall(r"[\w']+", source_turn.text)) < 5
                or _ordered_word_coverage(source_turn.text, polished_turn.text) >= 0.5
            )
            if not same_boundary or not faithful:
                raise StagePrimitiveError(
                    f"Polish chunk {chunk_number} failed source fidelity."
                )
        polished_turns.extend(polished.turns)
        replies.append((f"transcript_polish.chunk-{chunk_number:03d}", reply))

    return transcript.model_copy(update={"turns": polished_turns}), replies


async def run_map_speakers_stage(
    agent: StructuredAgent,
    transcript: TranscriptArtifact,
    *,
    attendees: list[str],
    max_attempts: int,
) -> tuple[SpeakerMapping, Reply]:
    return await run_structured_with_retries(
        agent,
        SpeakerMapPrompt(attendees=attendees, turns=transcript.turns),
        max_attempts=max_attempts,
    )


async def run_title_chapters_stage(
    agent: StructuredAgent,
    transcript: TranscriptArtifact,
    *,
    draft_plan: ChapterPlan | None,
    context: str = "",
    max_attempts: int,
) -> tuple[ChapterPlan, Reply]:
    draft = (
        [chapter.model_dump(mode="json") for chapter in draft_plan.chapters]
        if draft_plan is not None
        else []
    )
    chapter_plan, reply = await run_structured_with_retries(
        agent,
        ChapterTitlePrompt(
            context=context,
            turns=transcript.turns,
            draft_chapters=draft,
        ),
        max_attempts=max_attempts,
    )
    if draft_plan is None:
        return chapter_plan.model_copy(update={"boundary_source": "llm"}), reply
    if len(chapter_plan.chapters) != len(draft_plan.chapters):
        raise StagePrimitiveError(
            "Chapter-title stage changed deterministic chapter count."
        )
    titled_chapters = [
        generated.model_copy(update={"start": draft.start, "end": draft.end})
        for generated, draft in zip(
            chapter_plan.chapters, draft_plan.chapters, strict=True
        )
    ]
    return ChapterPlan(
        chapters=titled_chapters,
        boundary_source=draft_plan.boundary_source,
    ), reply


async def run_minutes_stage(
    agent: StructuredAgent,
    transcript: TranscriptArtifact,
    *,
    chapters: ChapterPlan | None,
    max_attempts: int,
) -> tuple[MeetingMinutes, Reply]:
    chapter_payload = (
        [chapter.model_dump(mode="json") for chapter in chapters.chapters]
        if chapters is not None
        else []
    )
    return await run_structured_with_retries(
        agent,
        MinutesPrompt(turns=transcript.turns, chapters=chapter_payload),
        max_attempts=max_attempts,
    )


async def run_source_note_plan_stage(
    agent: StructuredAgent,
    source: SourceArtifact,
    transcript: TranscriptArtifact,
    *,
    draft_plan: ChapterPlan,
    max_attempts: int,
) -> tuple[SourceNotePlan, Reply]:
    plan, reply = await run_structured_with_retries(
        agent,
        SourceNotePlanPrompt(
            source=source.model_dump(mode="json"),
            turns=transcript.turns,
            draft_chapters=[
                chapter.model_dump(mode="json") for chapter in draft_plan.chapters
            ],
        ),
        max_attempts=max_attempts,
    )
    expected_boundaries = [
        (chapter.start, chapter.end) for chapter in draft_plan.chapters
    ]
    actual_boundaries = [
        (chapter.start, chapter.end) for chapter in plan.chapters.chapters
    ]
    if actual_boundaries != expected_boundaries:
        raise StagePrimitiveError(
            "Source-note plan changed deterministic chapter boundaries."
        )
    return plan, reply
