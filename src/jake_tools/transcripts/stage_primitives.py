from __future__ import annotations

import re
from typing import ClassVar, Protocol

from pydantic import BaseModel, Field, ValidationError

from ..hermes import Reply
from ..prompting import StructuredPrompt
from .models import (
    ChapterPlan,
    MeetingMinutes,
    SourceArtifact,
    SourceNotePlan,
    SpeakerMapping,
    TranscriptArtifact,
    TranscriptTurn,
)
from .verify_primitives import verify_turns

_OPERATIONAL_CHATTER_RE = re.compile(
    r"^\s*Loading the transcript-polisher skill\.?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_MARKDOWN_FENCE_RE = re.compile(r"^\s*```{3,}", re.MULTILINE)
_SOURCE_BOILERPLATE_RE = re.compile(
    r"^\s*##\s*(Meeting Notes|Chapters|Transcript)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_CONTENT_WORD_RE = re.compile(r"\b[\w']+\b")
MAX_POLISH_CONTENT_CUT_RATIO = 0.15


class StructuredHermes(Protocol):
    def run_structured[TModel: BaseModel](
        self, prompt: StructuredPrompt[TModel]
    ) -> tuple[TModel, Reply]: ...


class StagePrimitiveError(RuntimeError):
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
"""

    context: str = ""
    turns: list[TranscriptTurn]


class SpeakerMapPrompt(StructuredPrompt[SpeakerMapping]):
    response_model: ClassVar[type[BaseModel]] = SpeakerMapping
    template: ClassVar[str] = """
Known attendees:
{{ attendees | json }}

Infer speaker identities conservatively from these transcript turns.
If uncertain, keep labels unresolved.

Transcript turns:
{{ turns | json }}
"""

    attendees: list[str]
    turns: list[TranscriptTurn]


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
"""

    context: str = ""
    turns: list[TranscriptTurn]
    draft_chapters: list[dict[str, object]]


class MinutesPrompt(StructuredPrompt[MeetingMinutes]):
    response_model: ClassVar[type[BaseModel]] = MeetingMinutes
    template: ClassVar[str] = """
Write faithful minutes in structured JSON.
Prioritize outcomes, decisions, and concrete actions over generic summary prose.

Transcript turns:
{{ turns | json }}

Chapter plan:
{{ chapters | json }}
"""

    turns: list[TranscriptTurn]
    chapters: list[dict[str, object]]


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
"""

    source: dict[str, object]
    turns: list[TranscriptTurn]
    draft_chapters: list[dict[str, object]]


def run_structured_with_retries[TModel: BaseModel](
    hermes: StructuredHermes, prompt: StructuredPrompt[TModel], *, max_attempts: int
) -> tuple[TModel, Reply]:
    if max_attempts < 1:
        raise StagePrimitiveError("max_attempts must be at least 1.")

    errors: list[str] = []
    for _ in range(max_attempts):
        try:
            return hermes.run_structured(prompt)
        except (ValidationError, ValueError) as exc:
            errors.append(str(exc))

    raise StagePrimitiveError(
        "Structured response failed schema validation after retries: "
        + " | ".join(errors[-2:])
    )


def _contains_forbidden_boilerplate(text: str) -> bool:
    return bool(
        _OPERATIONAL_CHATTER_RE.search(text)
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


def run_polish_stage(
    hermes: StructuredHermes,
    transcript: TranscriptArtifact,
    *,
    context: str = "",
    max_attempts: int,
) -> tuple[TranscriptArtifact, Reply]:
    payload, reply = run_structured_with_retries(
        hermes,
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


def run_map_speakers_stage(
    hermes: StructuredHermes,
    transcript: TranscriptArtifact,
    *,
    attendees: list[str],
    max_attempts: int,
) -> tuple[SpeakerMapping, Reply]:
    return run_structured_with_retries(
        hermes,
        SpeakerMapPrompt(attendees=attendees, turns=transcript.turns),
        max_attempts=max_attempts,
    )


def run_title_chapters_stage(
    hermes: StructuredHermes,
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
    chapter_plan, reply = run_structured_with_retries(
        hermes,
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


def run_minutes_stage(
    hermes: StructuredHermes,
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
    return run_structured_with_retries(
        hermes,
        MinutesPrompt(turns=transcript.turns, chapters=chapter_payload),
        max_attempts=max_attempts,
    )


def run_source_note_plan_stage(
    hermes: StructuredHermes,
    source: SourceArtifact,
    transcript: TranscriptArtifact,
    *,
    draft_plan: ChapterPlan,
    max_attempts: int,
) -> tuple[SourceNotePlan, Reply]:
    plan, reply = run_structured_with_retries(
        hermes,
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
