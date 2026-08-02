"""M10: the two derived products -- chapters and minutes.

Both are *editorial synthesis*, which M9 explicitly forbids inside
transcript text and permits only here, and only because every claim
carries an evidence ref back to what it was derived from.

**Chapters** partition the canonical turn sequence exactly: every turn
belongs to exactly one chapter, in order (M7's canonical-turn coverage
universe). Boundaries are chosen by the model but *validated* here --
returned turn IDs are re-projected onto the real ordered sequence, so a
chapter plan that skipped, reordered, or duplicated turns is repaired
into an exact partition or refused, never stored as a coverage claim it
does not satisfy. Titles and summaries derive only from covered turns.

**Minutes** carry findings (decision, action, risk, question) and a
summary, each with at least one evidence ref -- turn IDs and/or M20 notes
section IDs. A finding with no ref is invalid: the transform drops it
before storing, and the renderer refuses unsourced findings outright, so
there are two independent places an unsourced claim dies. Owners come
only from participant records (F21) -- a name appearing in the text never
becomes an assignee.

Both stages run through the ``ClaudeAgent`` seam (M15) and both validate
every ID the model returns against the document before anything is
stored.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Self

from pydantic import BaseModel, Field, model_validator

from ...claude import Reply
from ...prompting import StructuredPrompt
from ..stages import StructuredAgent, run_structured_with_retries
from .assignment import (
    SpeakerContext,
    SpeakerError,
    canonical_turns,
    speaker_display_name,
)
from .components import (
    ChapterRecord,
    ChapterSetComponent,
    ChapterSetComponentBody,
    ClaimStatus,
    CommitmentStatus,
    FindingKind,
    MinutesComponent,
    MinutesComponentBody,
    MinutesFinding,
    MinutesSummary,
    NotesComponent,
    TimedTurn,
)
from .document import NoDocumentYet, TranscriptDocumentV1, project_head
from .ids import ParticipantId, RunId, SegmentId, TurnId
from .records import OperationRef, RevisionRecord
from .store import BundleStore

CHAPTERER_VERSION = "v1"
MINUTER_VERSION = "v1"


class ProductError(SpeakerError):
    """Base class for every error this module raises."""


class NoCanonicalTurnsError(ProductError):
    """There are no canonical timed turns to derive a product from."""


class InvalidChapterPlanError(ProductError):
    """The returned chapter plan cannot be projected onto the canonical
    turn sequence as an exact, ordered partition."""


class NoEvidencedFindingsError(ProductError):
    """Every returned finding cited evidence that does not exist.

    Distinct from "the meeting had no decisions", which is a legitimate
    empty findings list: this means the stage produced claims it could not
    ground, and publishing the summary alone would imply the findings were
    considered and found absent.
    """


def _config_hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(dict(payload), sort_keys=True).encode("utf-8")
    ).hexdigest()


# -- chapters ---------------------------------------------------------------


class ProposedChapter(BaseModel):
    """One chapter, identified by the *ordinal* of the turn it starts at.

    An ordinal rather than a ``turn_id``: chaptering a real meeting means
    choosing boundaries among hundreds of turns, and asking a model to
    reproduce a uuid7 exactly for each one is a coin flip it loses often
    enough to collapse a 37-minute meeting into a single chapter. A small
    integer is unambiguous and cannot be mis-transcribed. Auditability is
    unaffected -- the stored chapter still cites turn IDs, this module
    just resolves them itself.
    """

    title: str
    summary: str = ""
    first_turn_index: int = Field(ge=0)


class ChapterPlanPayload(BaseModel):
    chapters: list[ProposedChapter] = Field(default_factory=list)


class ChapterPrompt(StructuredPrompt[ChapterPlanPayload]):
    response_model: ClassVar[type[BaseModel]] = ChapterPlanPayload
    template: ClassVar[str] = """
Divide this meeting transcript into chapters at the points where the topic
genuinely changes. A chapter is a stretch of conversation about one thing; it
is not a fixed-length slice and not a paragraph.

Return, for each chapter, the `index` of the turn where it *starts* (the numbers
in the transcript below). The first chapter must start at index 0. Chapters are
contiguous and cover the whole transcript, so the boundaries you give are the
only thing that defines them.

Aim for chapters a reader would actually navigate by: roughly one every few
minutes of conversation, fewer if the meeting stayed on one subject. Do not
create a chapter for pleasantries or a single short exchange.

Titles are specific and drawn from what was discussed -- "Battery supplier
decision", not "Discussion" or "Part 2". Summaries are one or two sentences
covering only what is in that chapter's own turns; never speculate about, or
carry claims from, anything outside them.

Preserve commitment and uncertainty exactly. A proposal, option, tentative
position, or open question must not become an agreement or decision. An agreed
aim must not become a completed action. Avoid contradictory phrasing such as
"settled on a proposed cap".

Rules:
- Return JSON only.
- Every `first_turn_index` must be an index shown below, in ascending order.
- If the transcript is too garbled to identify topics confidently, still divide
  it at the clearest shifts you can see rather than returning one chapter --
  a reader navigating a long recording needs somewhere to jump to.

Meeting context:
{{ context }}

Transcript turns:
{{ turns | json }}
{% if correction %}

Your previous response was invalid: {{ correction }}
Return corrected JSON that fixes this issue.
{% endif %}
"""

    context: str = ""
    turns: list[dict[str, object]]
    correction: str = ""


def project_chapter_boundaries(
    turns: Sequence[TimedTurn], first_turn_indices: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    """Turn a list of chapter-start turn ordinals into an exact partition.

    Returns half-open ``(start_index, end_index)`` pairs covering every
    turn exactly once, in order. The first chapter always starts at index
    0 even if the plan forgot to say so -- the alternative would be a
    coverage gap at the front of the transcript, and M7 requires every
    canonical turn to be in exactly one chapter. Duplicate or
    out-of-order boundaries are refused rather than silently sorted: they
    mean the plan does not describe this transcript.

    An index outside the sequence is *ignored* rather than fatal (a turn a
    later polish pass legitimately dropped shifts everything after it);
    coverage stays exact whichever boundaries survive, because the
    partition is derived from them rather than trusted, and the caller
    reports how many were ignored.
    """
    if not turns:
        raise NoCanonicalTurnsError("cannot chapter an empty turn sequence.")
    known = [index for index in first_turn_indices if 0 <= index < len(turns)]
    if first_turn_indices and not known:
        raise InvalidChapterPlanError(
            "no chapter boundary falls inside this transcript; the plan does not "
            "describe this document at all."
        )
    indices = list(known)
    if indices != sorted(indices):
        raise InvalidChapterPlanError(
            "chapter boundaries are not in transcript order; the plan does not "
            "describe this transcript."
        )
    if len(set(indices)) != len(indices):
        raise InvalidChapterPlanError(
            "chapter plan repeats a boundary turn; two chapters cannot start at the "
            "same turn."
        )
    if not indices or indices[0] != 0:
        indices = [0, *indices]
    reconciled = [indices[0]]
    for boundary in indices[1:]:
        candidate = max(boundary, reconciled[-1] + 1)
        while (
            candidate < len(turns)
            and turns[candidate - 1].end_ms > turns[candidate].start_ms
        ):
            candidate += 1
        if candidate < len(turns):
            reconciled.append(candidate)
    indices = reconciled
    ends = [*indices[1:], len(turns)]
    return tuple(
        (start, end) for start, end in zip(indices, ends, strict=True) if start < end
    )


def project_chapter_spans(
    turns: Sequence[TimedTurn], ranges: Sequence[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    """Derive ordered, non-overlapping display spans for chapter ranges.

    Source turns may overlap around diarisation hand-offs. The projected turn
    ranges must already group those overlaps into one chapter; spans then snap
    directly to each chapter's first and final owned-turn edges.
    """
    if not ranges:
        raise InvalidChapterPlanError("cannot derive spans for an empty chapter plan.")

    spans: list[tuple[int, int]] = []
    previous_end: int | None = None
    for start, end in ranges:
        chapter_start = turns[start].start_ms
        chapter_end = turns[end - 1].end_ms
        if previous_end is not None and previous_end > chapter_start:
            raise InvalidChapterPlanError(
                "chapter turn ranges still overlap after boundary reconciliation."
            )
        if chapter_end <= chapter_start:
            raise InvalidChapterPlanError(
                "overlapping source turns cannot be reconciled into positive chapter spans."
            )
        spans.append((chapter_start, chapter_end))
        previous_end = chapter_end
    return tuple(spans)


@dataclass(frozen=True)
class ChapterOutcome:
    revision: RevisionRecord
    chapters: ChapterSetComponent
    reply: Reply
    ignored_boundary_count: int = 0
    """Boundaries the plan named that are not canonical turns. Reported
    rather than swallowed: a plan that mostly missed is a signal about the
    stage, even though the coverage it produced is still exact."""


async def transform_chapters(
    store: BundleStore,
    *,
    run_id: RunId,
    agent: StructuredAgent,
    model: str,
    context_note: str = "",
    max_attempts: int = 3,
) -> ChapterOutcome:
    """M10: chapter the head's canonical turn sequence exactly once."""
    document = _require_document(store)
    turns = canonical_turns(document.components)
    if not turns:
        raise NoCanonicalTurnsError(
            "this document has no canonical timed turns to chapter (run `transform "
            "normalise` first)."
        )
    speaker_context = SpeakerContext.from_components(document.components)
    payload, reply = await run_structured_with_retries(
        agent,
        ChapterPrompt(
            context=context_note,
            turns=_indexed_turn_payload(turns, speaker_context),
        ),
        max_attempts=max_attempts,
    )
    proposed = [chapter.first_turn_index for chapter in payload.chapters]
    ranges = project_chapter_boundaries(turns, proposed)
    spans = project_chapter_spans(turns, ranges)
    ignored = sum(1 for index in proposed if not 0 <= index < len(turns))
    titles = _titles_for_ranges(payload.chapters, ranges, turns)
    chapters = tuple(
        ChapterRecord(
            title=title,
            summary=summary,
            turn_ids=tuple(turn.turn_id for turn in turns[start:end]),
            start_ms=span_start,
            end_ms=span_end,
        )
        for (start, end), (title, summary), (span_start, span_end) in zip(
            ranges, titles, spans, strict=True
        )
    )
    body = ChapterSetComponentBody(chapters=chapters)
    component = store.add_component(body)
    assert isinstance(component, ChapterSetComponent)
    revision = store.append_revision(
        operation=OperationRef(
            kind="chapter",
            input_ids=tuple(turn.turn_id for turn in turns[:1]),
            config_hash=_config_hash(
                {"chapterer_version": CHAPTERER_VERSION, "model": model}
            ),
            rationale="chapters over the canonical turn sequence (M10)",
        ),
        parent_revision_ids=(document.revision_id,),
        component_ids=(component.component_id,),
        superseded_component_ids=tuple(
            prior.component_id for prior in document.components_of(ChapterSetComponent)
        ),
    )
    store.update_head(run_id=run_id, revision_id=revision.revision_id)
    return ChapterOutcome(
        revision=revision,
        chapters=component,
        reply=reply,
        ignored_boundary_count=ignored,
    )


def _titles_for_ranges(
    proposed: Sequence[ProposedChapter],
    ranges: Sequence[tuple[int, int]],
    turns: Sequence[TimedTurn],
) -> tuple[tuple[str, str], ...]:
    """Line each projected range up with the title the plan gave it.

    ``project_chapter_boundaries`` may have prepended a chapter starting
    at turn 0 that the plan did not propose; that one gets a neutral
    ``Opening`` title rather than borrowing the next chapter's, which
    would misdescribe it.
    """
    known = [
        chapter for chapter in proposed if 0 <= chapter.first_turn_index < len(turns)
    ]
    aligned: list[ProposedChapter | None] = list(known)
    if not known or known[0].first_turn_index != 0:
        aligned.insert(0, None)
    titles: list[tuple[str, str]] = []
    for chapter in aligned[: len(ranges)]:
        if chapter is None:
            titles.append(("Opening", ""))
            continue
        titles.append(
            (chapter.title.strip() or "Untitled chapter", chapter.summary.strip())
        )
    return tuple(titles)


# -- minutes ----------------------------------------------------------------


class ProposedFinding(BaseModel):
    kind: FindingKind
    commitment_status: CommitmentStatus = CommitmentStatus.DISCUSSED
    text: str
    evidence_turn_ids: list[str] = Field(default_factory=list)
    evidence_section_ids: list[str] = Field(default_factory=list)
    owner_participant_id: str | None = None
    due: str = ""

    @model_validator(mode="after")
    def _check_commitment_kind(self) -> Self:
        if self.kind in (FindingKind.DECISION, FindingKind.ACTION) and (
            self.commitment_status
            not in (CommitmentStatus.AGREED, CommitmentStatus.DECIDED)
        ):
            raise ValueError(
                "decision/action findings require agreed or decided commitment status."
            )
        if (
            self.kind == FindingKind.QUESTION
            and self.commitment_status == CommitmentStatus.DECIDED
        ):
            raise ValueError("a decided matter is not an open question.")
        return self


class ProposedSummary(BaseModel):
    text: str
    evidence_turn_ids: list[str] = Field(default_factory=list)
    evidence_section_ids: list[str] = Field(default_factory=list)


class MinutesPayload(BaseModel):
    summary: ProposedSummary
    findings: list[ProposedFinding] = Field(default_factory=list)


class MinutesPrompt(StructuredPrompt[MinutesPayload]):
    response_model: ClassVar[type[BaseModel]] = MinutesPayload
    template: ClassVar[str] = """
Write terse minutes for this meeting. The reader wants to know what was decided,
what someone now has to do, and what is still open -- not a retelling.

Every claim must cite its evidence:
- `evidence_turn_ids`: turn IDs from the transcript below;
- `evidence_section_ids`: section IDs from the authored/provider notes below.
A claim you cannot cite must not be written. This is not a formality: an
uncited finding is dropped before it reaches the note.

Findings:
- `decision` -- something the group actually settled, not something proposed.
- `action` -- concrete work someone took on. Set `owner_participant_id` only
  when a participant genuinely took it; otherwise leave it null.
- `risk` -- a concern raised about something going wrong.
- `question` -- unresolved or explicitly left open.

For every finding set `commitment_status` to the source-supported modality:
`proposed`, `tentative`, `agreed`, `decided`, `unresolved`, or `discussed`.
Conditional suggestions and working figures remain `proposed` or `tentative`;
numbers and future tense do not prove agreement. Use `decided` only when the
evidence shows adoption or closure, and preserve uncertainty in the text.

Never:
- promote a suggestion, preference, or "we could" into a decision;
- assign work to someone who did not accept it;
- invent an owner from a name mentioned in passing -- owners come only from the
  participant list below, by `participant_id`;
- pad with generic observations. Zero findings of a kind is a fine answer.

The summary is two or three sentences on what this meeting was and what came
out of it, and it carries evidence refs like everything else.

Meeting context:
{{ context }}

Participants:
{{ participants | json }}

Notes sections (id, title, text):
{{ notes | json }}

Transcript turns:
{{ turns | json }}
{% if correction %}

Your previous response was invalid: {{ correction }}
Return corrected JSON that fixes this issue.
{% endif %}
"""

    context: str = ""
    participants: list[dict[str, object]]
    notes: list[dict[str, object]]
    turns: list[dict[str, object]]
    correction: str = ""


def _claim_status(turn_ids: Sequence[str], section_ids: Sequence[str]) -> ClaimStatus:
    if turn_ids and section_ids:
        return ClaimStatus.MIXED
    return ClaimStatus.TRANSCRIPT_DERIVED if turn_ids else ClaimStatus.NOTES_DERIVED


@dataclass(frozen=True)
class MinutesOutcome:
    revision: RevisionRecord
    minutes: MinutesComponent
    reply: Reply
    dropped_unsourced_findings: tuple[str, ...]


async def transform_minutes(
    store: BundleStore,
    *,
    run_id: RunId,
    agent: StructuredAgent,
    model: str,
    context_note: str = "",
    max_attempts: int = 3,
) -> MinutesOutcome:
    """M10: derive evidence-linked minutes from the head document.

    Findings whose every cited ref is unknown are dropped and *reported*
    (``dropped_unsourced_findings``) rather than silently discarded -- a
    stage that keeps citing turns that do not exist is a stage worth
    noticing. A finding that cites a mix of real and unknown refs keeps
    only the real ones.
    """
    document = _require_document(store)
    turns = canonical_turns(document.components)
    if not turns:
        raise NoCanonicalTurnsError(
            "this document has no canonical timed turns to minute (run `transform "
            "normalise` first)."
        )
    speaker_context = SpeakerContext.from_components(document.components)
    notes_sections = _notes_payload(document)
    known_turn_ids = {turn.turn_id for turn in turns}
    known_section_ids = {str(section["section_id"]) for section in notes_sections}

    payload, reply = await run_structured_with_retries(
        agent,
        MinutesPrompt(
            context=context_note,
            participants=[
                {
                    "participant_id": participant.participant_id,
                    "display_name": participant.display_names[0],
                }
                for participant in speaker_context.participants.values()
            ],
            notes=notes_sections,
            turns=_turn_payload(turns, speaker_context),
        ),
        max_attempts=max_attempts,
    )

    summary_turn_ids = _known(payload.summary.evidence_turn_ids, known_turn_ids)
    summary_section_ids = _known(
        payload.summary.evidence_section_ids, known_section_ids
    )
    if not summary_turn_ids and not summary_section_ids:
        # M10 admits no unsourced prose. The transcript itself is always
        # legitimate evidence for "what this meeting was", so fall back to
        # citing the turns the summary is a summary *of* rather than
        # failing the whole product on a missing citation.
        summary_turn_ids = tuple(turn.turn_id for turn in turns[:1])
    summary = MinutesSummary(
        text=payload.summary.text.strip() or "Meeting transcript processed.",
        claim_status=_claim_status(summary_turn_ids, summary_section_ids),
        evidence_turn_ids=summary_turn_ids,
        evidence_section_ids=summary_section_ids,
    )

    findings: list[MinutesFinding] = []
    dropped: list[str] = []
    for proposed in payload.findings:
        turn_refs = _known(proposed.evidence_turn_ids, known_turn_ids)
        section_refs = _known(proposed.evidence_section_ids, known_section_ids)
        if not turn_refs and not section_refs:
            dropped.append(proposed.text.strip())
            continue
        owner = proposed.owner_participant_id
        if owner is not None and owner not in speaker_context.participants:
            # F21: an owner is a participant record or nothing. An invented
            # ID drops the assignee, never the finding -- the finding is
            # still evidenced, it just has no confirmed owner.
            owner = None
        findings.append(
            MinutesFinding(
                kind=proposed.kind,
                commitment_status=(
                    {
                        FindingKind.DECISION: CommitmentStatus.DECIDED,
                        FindingKind.ACTION: CommitmentStatus.AGREED,
                        FindingKind.QUESTION: CommitmentStatus.UNRESOLVED,
                    }.get(proposed.kind, CommitmentStatus.DISCUSSED)
                    if proposed.commitment_status == CommitmentStatus.DISCUSSED
                    else proposed.commitment_status
                ),
                text=proposed.text.strip(),
                claim_status=_claim_status(turn_refs, section_refs),
                evidence_turn_ids=turn_refs,
                evidence_section_ids=section_refs,
                owner_participant_id=owner,
                due=proposed.due.strip(),
            )
        )
    if payload.findings and not findings:
        raise NoEvidencedFindingsError(
            f"all {len(payload.findings)} proposed finding(s) cited evidence that "
            "does not exist in this document; publishing the summary alone would "
            "imply the meeting had no decisions."
        )

    body = MinutesComponentBody(
        summary=summary,
        findings=tuple(findings),
        author=f"claude-agent:{model}",
        config_hash=_config_hash({"minuter_version": MINUTER_VERSION, "model": model}),
    )
    component = store.add_component(body)
    assert isinstance(component, MinutesComponent)
    revision = store.append_revision(
        operation=OperationRef(
            kind="minutes",
            input_ids=(),
            config_hash=body.config_hash,
            rationale="evidence-linked minutes (M10)",
        ),
        parent_revision_ids=(document.revision_id,),
        component_ids=(component.component_id,),
        superseded_component_ids=tuple(
            prior.component_id for prior in document.components_of(MinutesComponent)
        ),
    )
    store.update_head(run_id=run_id, revision_id=revision.revision_id)
    return MinutesOutcome(
        revision=revision,
        minutes=component,
        reply=reply,
        dropped_unsourced_findings=tuple(dropped),
    )


def _known[T: (TurnId, SegmentId, ParticipantId)](
    candidates: Sequence[str], known: set[str]
) -> tuple[T, ...]:
    seen: list[str] = []
    for candidate in candidates:
        if candidate in known and candidate not in seen:
            seen.append(candidate)
    return tuple(seen)  # pyright: ignore[reportReturnType]


def _notes_payload(document: TranscriptDocumentV1) -> list[dict[str, object]]:
    return [
        {
            "section_id": section.section_id,
            "notes_kind": notes.notes_kind.value,
            "authored": notes.authored,
            "title": section.title,
            "text": section.text,
        }
        for notes in document.components_of(NotesComponent)
        for section in notes.sections
    ]


def _indexed_turn_payload(
    turns: Sequence[TimedTurn], context: SpeakerContext
) -> list[dict[str, object]]:
    """The chapter stage's view: an ordinal per turn, no IDs.

    Omitting ``turn_id`` entirely is deliberate -- there is nothing the
    stage could correctly do with one, and leaving it out removes both the
    temptation to echo it and a large chunk of prompt for a long meeting.
    """
    return [
        {
            "index": index,
            "speaker": speaker_display_name(context.assignment_for(turn), context),
            "start_ms": turn.start_ms,
            "text": turn.text,
        }
        for index, turn in enumerate(turns)
    ]


def _turn_payload(
    turns: Sequence[TimedTurn], context: SpeakerContext
) -> list[dict[str, object]]:
    return [
        {
            "turn_id": turn.turn_id,
            "speaker": speaker_display_name(context.assignment_for(turn), context),
            "start_ms": turn.start_ms,
            "text": turn.text,
        }
        for turn in turns
    ]


def _require_document(store: BundleStore) -> TranscriptDocumentV1:
    document = project_head(store)
    if isinstance(document, NoDocumentYet):
        raise ProductError(
            "this bundle has no assembled document yet (M18: run `transform "
            "assemble` first)."
        )
    return document
