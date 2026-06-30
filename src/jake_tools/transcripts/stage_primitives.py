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
    RunManifest,
    SpeakerMapping,
    TranscriptArtifact,
    TranscriptSourceRef,
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


def _resolve_manifest_chunks(
    manifest: RunManifest, *, base_dir: Path
) -> list[tuple[str, TranscriptArtifact]]:
    chunks: list[tuple[str, TranscriptArtifact]] = []
    for stage in manifest.stages:
        for artefact in stage.artefacts:
            chunk_path = artefact if artefact.is_absolute() else (base_dir / artefact)
            if not chunk_path.exists():
                raise StagePrimitiveError(
                    f"Chunk artefact referenced by manifest does not exist: {chunk_path}"
                )
            chunks.append(
                (
                    stage.stage,
                    TranscriptArtifact.model_validate_json(
                        chunk_path.read_text(encoding="utf-8")
                    ),
                )
            )
    if not chunks:
        raise StagePrimitiveError(
            "Manifest did not resolve to any transcript turns for stage input."
        )
    return chunks


def _merge_transcript_chunks(chunks: list[TranscriptArtifact]) -> TranscriptArtifact:
    merged_turns: list[TranscriptTurn] = []
    merged_source_refs: list[TranscriptSourceRef] = []
    merged_warnings: list[str] = []
    for chunk in chunks:
        turn_offset = len(merged_turns)
        merged_turns.extend(chunk.turns)
        merged_source_refs.extend(
            source_ref.model_copy(
                update={"turn_index": source_ref.turn_index + turn_offset}
            )
            for source_ref in chunk.source_refs
        )
        merged_warnings.extend(chunk.warnings)
    return TranscriptArtifact(
        turns=merged_turns,
        source_refs=merged_source_refs,
        speakers={},
        warnings=merged_warnings,
    )


def polish_manifest_chunks(
    hermes: Hermes,
    manifest_path: Path,
    *,
    max_attempts: int,
) -> tuple[TranscriptArtifact, PolishLedger, Reply | None]:
    manifest = RunManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    chunks = _resolve_manifest_chunks(manifest, base_dir=manifest_path.parent)
    polished_chunks: list[TranscriptArtifact] = []
    ledger_entries: list[PolishLedgerEntry] = []
    ledger_notes: list[str] = []
    merge_allowed = False
    last_reply: Reply | None = None

    for stage_name, chunk in chunks:
        polished, ledger, reply = run_polish_stage(
            hermes,
            chunk,
            max_attempts=max_attempts,
        )
        turn_offset = sum(len(previous.turns) for previous in polished_chunks)
        polished_chunks.append(polished)
        last_reply = reply
        merge_allowed = merge_allowed or ledger.merge_allowed
        ledger_entries.extend(
            entry.model_copy(
                update={
                    "source_turn_indices": [
                        index + turn_offset for index in entry.source_turn_indices
                    ],
                    "output_turn_indices": [
                        index + turn_offset for index in entry.output_turn_indices
                    ],
                }
            )
            for entry in ledger.entries
        )
        if ledger.notes:
            ledger_notes.append(f"{stage_name}: {ledger.notes}")

    return (
        _merge_transcript_chunks(polished_chunks),
        PolishLedger(
            merge_allowed=merge_allowed,
            entries=ledger_entries,
            notes="\n".join(ledger_notes),
        ),
        last_reply,
    )


def load_transcript_or_manifest(input_path: Path) -> TranscriptArtifact:
    raw = input_path.read_text(encoding="utf-8")
    try:
        return TranscriptArtifact.model_validate_json(raw)
    except ValidationError:
        pass

    manifest = RunManifest.model_validate_json(raw)
    chunks = [
        chunk
        for _stage_name, chunk in _resolve_manifest_chunks(
            manifest, base_dir=input_path.parent
        )
    ]
    return _merge_transcript_chunks(chunks)
