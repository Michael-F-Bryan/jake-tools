from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field, ValidationError

from ..hermes import Hermes, Reply
from ..prompting import StructuredPrompt
from .models import (
    ChapterPlan,
    MeetingMinutes,
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


class StagePrimitiveError(RuntimeError):
    pass


class PolishLedgerEntry(BaseModel):
    action: str
    source_turn_indices: list[int] = Field(default_factory=list)
    output_turn_indices: list[int] = Field(default_factory=list)
    reason: str = ""


class PolishLedger(BaseModel):
    merge_allowed: bool = False
    entries: list[PolishLedgerEntry] = Field(default_factory=list)
    notes: str = ""


class PolishStagePayload(BaseModel):
    turns: list[TranscriptTurn] = Field(default_factory=list)
    ledger: PolishLedger = Field(default_factory=PolishLedger)


class PolishStagePrompt(StructuredPrompt[PolishStagePayload]):
    response_model: ClassVar[type[BaseModel]] = PolishStagePayload
    template: ClassVar[str] = """
You are the transcript-polishing specialist for transcript stage primitives.

Rewrite the transcript turns conservatively for readability while preserving meaning.

Rules:
- Return JSON only.
- Keep turn order unchanged.
- Keep every turn's `start`, `end`, and `speaker` unchanged unless `ledger.merge_allowed` is true.
- Never include operational chatter.
- Never include markdown fences.
- Never include source boilerplate headings such as "## Transcript".
- If any merge/split/drop happened, set `ledger.merge_allowed=true` and provide detailed ledger entries.

Transcript turns:
{{ turns | json }}
"""

    turns: list[TranscriptTurn]


class SpeakerMapPrompt(StructuredPrompt[SpeakerMapping]):
    response_model: ClassVar[type[BaseModel]] = SpeakerMapping
    template: ClassVar[str] = """
You are the speaker-mapping specialist for transcript stage primitives.

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
You are the chapter-title specialist for transcript stage primitives.

Return a ChapterPlan with concise, descriptive chapter titles and summaries.
When draft chapters are provided, keep chapter boundaries unchanged.

Transcript turns:
{{ turns | json }}

Draft chapters:
{{ draft_chapters | json }}
"""

    turns: list[TranscriptTurn]
    draft_chapters: list[dict[str, object]]


class MinutesPrompt(StructuredPrompt[MeetingMinutes]):
    response_model: ClassVar[type[BaseModel]] = MeetingMinutes
    template: ClassVar[str] = """
You are the meeting-minutes specialist for transcript stage primitives.

Write faithful minutes in structured JSON.
Prioritize outcomes, decisions, and concrete actions over generic summary prose.

Transcript turns:
{{ turns | json }}

Chapter plan:
{{ chapters | json }}
"""

    turns: list[TranscriptTurn]
    chapters: list[dict[str, object]]


def run_structured_with_retries[TModel: BaseModel](
    hermes: Hermes, prompt: StructuredPrompt[TModel], *, max_attempts: int
) -> tuple[TModel, Reply]:
    if max_attempts < 1:
        raise StagePrimitiveError("--max-attempts must be at least 1.")

    errors: list[str] = []
    for _ in range(max_attempts):
        try:
            return hermes.run_structured(prompt)
        except (ValidationError, ValueError) as exc:
            errors.append(str(exc))

    raise StagePrimitiveError(
        "Structured stage failed schema validation after retries: "
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


def _merge_ledger_is_explicit(ledger: PolishLedger) -> bool:
    return ledger.merge_allowed and bool(ledger.entries)


def run_polish_stage(
    hermes: Hermes,
    transcript: TranscriptArtifact,
    *,
    max_attempts: int,
) -> tuple[TranscriptArtifact, PolishLedger, Reply]:
    payload, reply = run_structured_with_retries(
        hermes,
        PolishStagePrompt(turns=transcript.turns),
        max_attempts=max_attempts,
    )
    _ensure_no_forbidden_output(payload.turns)

    polished = transcript.model_copy(update={"turns": payload.turns})
    verification = verify_turns(transcript, polished, affected_paths=[])
    if verification.failed_gate_ids:
        if verification.failed_gate_ids == [
            "turns.coverage-preserved"
        ] and _merge_ledger_is_explicit(payload.ledger):
            return polished, payload.ledger, reply
        raise StagePrimitiveError(
            "Polished transcript failed verification gates: "
            + ", ".join(verification.failed_gate_ids)
        )

    if len(payload.turns) != len(transcript.turns) and not _merge_ledger_is_explicit(
        payload.ledger
    ):
        raise StagePrimitiveError(
            "Turn coverage changed without explicit merge ledger entries."
        )

    return polished, payload.ledger, reply


def run_map_speakers_stage(
    hermes: Hermes,
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
    hermes: Hermes,
    transcript: TranscriptArtifact,
    *,
    draft_plan: ChapterPlan | None,
    max_attempts: int,
) -> tuple[ChapterPlan, Reply]:
    draft = (
        [chapter.model_dump(mode="json") for chapter in draft_plan.chapters]
        if draft_plan is not None
        else []
    )
    chapter_plan, reply = run_structured_with_retries(
        hermes,
        ChapterTitlePrompt(turns=transcript.turns, draft_chapters=draft),
        max_attempts=max_attempts,
    )
    return chapter_plan.model_copy(update={"boundary_source": "llm"}), reply


def run_minutes_stage(
    hermes: Hermes,
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


def load_transcript_or_manifest(input_path: Path) -> TranscriptArtifact:
    raw = input_path.read_text(encoding="utf-8")
    try:
        return TranscriptArtifact.model_validate_json(raw)
    except ValidationError:
        pass

    from .models import RunManifest

    manifest = RunManifest.model_validate_json(raw)
    base_dir = input_path.parent
    merged_turns: list[TranscriptTurn] = []
    merged_source_refs = []
    merged_warnings: list[str] = []

    for stage in manifest.stages:
        for artefact in stage.artefacts:
            chunk_path = artefact if artefact.is_absolute() else (base_dir / artefact)
            if not chunk_path.exists():
                raise StagePrimitiveError(
                    f"Chunk artefact referenced by manifest does not exist: {chunk_path}"
                )
            chunk = TranscriptArtifact.model_validate_json(
                chunk_path.read_text(encoding="utf-8")
            )
            merged_turns.extend(chunk.turns)
            merged_source_refs.extend(chunk.source_refs)
            merged_warnings.extend(chunk.warnings)

    if not merged_turns:
        raise StagePrimitiveError(
            "Manifest did not resolve to any transcript turns for stage input."
        )

    return TranscriptArtifact(
        turns=merged_turns,
        source_refs=merged_source_refs,
        speakers={},
        warnings=merged_warnings,
    )
