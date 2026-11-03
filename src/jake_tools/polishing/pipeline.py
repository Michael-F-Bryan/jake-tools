"""Polish one transcript into a fresh directory, chunk by chunk.

Per chunk there are at most two tool-less ``ClaudeAgent.run_structured``
calls: the edit role, then the detection-only review role (which also runs
when the edit call failed, over the unedited words). Calls are bounded by
a per-call budget and timeout, a whole-operation budget (reserved before a
call starts, so it is never exceeded by concurrent calls) and a deadline
inside Hermes' 300 s tool-call limit. A call that fails, times out or is
not started for lack of budget or time is reported per chunk, shown in the
transcript where it applies, and makes the result ``partial``; it never
looks complete.

The source file is read once and never written; its hash is checked again
at the end. The output directory must not exist. Prompts, parsed replies
and errors are saved under ``diagnostics/`` as each call runs, and
``metadata.json`` is written last.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import anyio
from pydantic import BaseModel, ValidationError

from ..claude import AgentSpec, ClaudeAgent, ClaudeAgentError, EffortLevel
from ..prompting import StructuredPrompt
from .edits import (
    Edit,
    EditState,
    Rejection,
    ReviewItem,
    apply_edit_decisions,
    apply_review,
    chunk_model_edits,
)
from .lexicon import is_filler
from .prompts import (
    EDIT_SYSTEM_PROMPT,
    REVIEW_SYSTEM_PROMPT,
    EditDecisions,
    EditPrompt,
    GlossaryEntry,
    PromptEdit,
    PromptLine,
    PromptWord,
    ReviewFindings,
    ReviewPrompt,
)
from .render import (
    ChunkNotice,
    LedgerWord,
    RenderInput,
    ledger_words,
    render_markdown,
    timestamp,
)
from .source import (
    Chunk,
    LoadedSource,
    SourceError,
    SourceWord,
    Window,
    load_source,
    plan_chunks,
    select_window,
)

POLICY_VERSION = "polish-transcript/1"
EDIT_MODEL = "sonnet"
REVIEW_MODEL = "opus"
"""Opus flagged material garbles with fewer unsupported readings than
Sonnet in the prototype comparison (``_working/transcript-polishing``)."""
EFFORT: EffortLevel = "medium"
MAX_TURNS = 3
EDIT_CALL_BUDGET_USD = 0.75
REVIEW_CALL_BUDGET_USD = 1.00
DEFAULT_BUDGET_USD = 5.0
MAX_BUDGET_USD = 25.0
CALL_TIMEOUT_SECONDS = 150.0
DEADLINE_SECONDS = 270.0
"""Inside Hermes' default 300 s MCP tool-call limit, with room to write."""
MIN_CALL_SECONDS = 30.0
CONCURRENCY = 4
MAX_CHUNKS = 16
MAX_GLOSSARY_TERMS = 50
MAX_TERM_CHARS = 60
MAX_NOTE_CHARS = 300
LOW_CONFIDENCE = 0.5

Role = Literal["edit", "review"]


class PolishFailedError(RuntimeError):
    """No chunk could be edited; the directory holds the diagnostics."""

    def __init__(self, message: str, output_directory: Path) -> None:
        super().__init__(message)
        self.output_directory = output_directory


# --- the result ---------------------------------------------------------------------


class StageReport(BaseModel):
    status: Literal["completed", "failed", "skipped", "pending"] = "pending"
    models: list[str] = []
    """The models the provider reported for the call (the CLI may add a
    small model of its own alongside the one requested)."""
    elapsed_seconds: float | None = None
    estimated_cost_usd: float | None = None
    error: str | None = None
    accepted: int = 0
    rejected: int = 0
    restored: int = 0


class ChunkReport(BaseModel):
    index: int
    first_word: int
    last_word: int
    start_seconds: float
    end_seconds: float
    edit: StageReport = StageReport()
    review: StageReport = StageReport()


class PolishCounts(BaseModel):
    window_words: int
    retained_words: int
    removed_words: int
    words_in_glossary_candidates: int
    removed_by_kind: dict[str, int]
    formats: int
    glossary_candidates: int
    restored_after_review: int
    rejected_decisions: int
    review_items: int


class PolishFiles(BaseModel):
    transcript: str
    ledger: str
    review: str
    metadata: str
    diagnostics: str


class PolishResult(BaseModel):
    """What ``polish_transcript`` returns; ``metadata.json`` holds the same."""

    status: Literal["complete", "partial"]
    output_directory: str
    files: PolishFiles
    source_path: str
    source_sha256: str
    source_unchanged: bool
    window: Window
    counts: PolishCounts
    chunks: list[ChunkReport]
    estimated_cost_usd: float
    cost_is_estimate_only: bool
    """True when some call reported no cost and was charged at its cap."""
    elapsed_seconds: float
    notes: list[str]


class PolishPolicy(BaseModel):
    version: str = POLICY_VERSION
    edit_model: str = EDIT_MODEL
    review_model: str = REVIEW_MODEL
    effort: str = EFFORT
    max_turns: int = MAX_TURNS
    edit_call_budget_usd: float = EDIT_CALL_BUDGET_USD
    review_call_budget_usd: float = REVIEW_CALL_BUDGET_USD
    budget_usd: float = DEFAULT_BUDGET_USD
    call_timeout_seconds: float = CALL_TIMEOUT_SECONDS
    deadline_seconds: float = DEADLINE_SECONDS
    concurrency: int = CONCURRENCY
    tools: tuple[str, ...] = ()
    glossary: tuple[GlossaryEntry, ...] = ()


class PolishMetadata(BaseModel):
    started_at: datetime
    policy: PolishPolicy
    engine: dict[str, str]
    result: PolishResult


# --- arguments ----------------------------------------------------------------------


def parse_glossary(entries: list[str] | None) -> tuple[GlossaryEntry, ...]:
    """``"TERM"`` or ``"TERM: context for the editor"``, each term once."""
    parsed: list[GlossaryEntry] = []
    seen: set[str] = set()
    for raw in entries or []:
        term, _, note = raw.partition(":")
        term, note = term.strip(), note.strip()
        if not term or len(term) > MAX_TERM_CHARS or any(c in term for c in "[]\n\r"):
            raise SourceError(
                "glossary",
                f"each glossary term needs 1-{MAX_TERM_CHARS} characters and no "
                "brackets or line breaks",
            )
        if len(note) > MAX_NOTE_CHARS or "\n" in note:
            raise SourceError(
                "glossary",
                f"a glossary note is one line of at most {MAX_NOTE_CHARS} characters",
            )
        if term.casefold() in seen:
            raise SourceError("glossary", f"glossary term {term!r} is listed twice")
        seen.add(term.casefold())
        parsed.append(GlossaryEntry(term=term, note=note or None))
    if len(parsed) > MAX_GLOSSARY_TERMS:
        raise SourceError("glossary", f"at most {MAX_GLOSSARY_TERMS} glossary terms")
    return tuple(parsed)


def check_paths(transcript_path: Path, output_directory: Path) -> None:
    for argument, path in (
        ("transcript_path", transcript_path),
        ("output_directory", output_directory),
    ):
        if not path.is_absolute():
            raise SourceError(argument, f"{argument} must be an absolute path")
    if not transcript_path.is_file():
        raise SourceError("transcript_path", f"{transcript_path} is not a file")
    if output_directory.exists():
        raise SourceError(
            "output_directory",
            f"{output_directory} already exists; polishing writes a fresh directory",
        )
    if not output_directory.parent.is_dir():
        raise SourceError(
            "output_directory", f"{output_directory.parent} is not a directory"
        )


# --- prompts' view of the source ----------------------------------------------------


def prompt_lines(words: tuple[SourceWord, ...], chunk: Chunk) -> list[PromptLine]:
    """One line per source segment and editability, context included."""
    lines: list[PromptLine] = []
    for word_id in range(chunk.context_first, chunk.context_last + 1):
        word = words[word_id]
        editable = chunk.owns(word_id)
        text = f"({word.text})" if is_filler(word.text) else word.text
        marks = ("~" if word.confidence < LOW_CONFIDENCE else "") + (
            "^" if word.overlap else ""
        )
        previous = words[word_id - 1] if word_id > chunk.context_first else None
        if (
            previous is None
            or previous.segment != word.segment
            or chunk.owns(previous.id) != editable
        ):
            lines.append(
                PromptLine(
                    editable=editable,
                    time=_prompt_time(word.start_seconds),
                    speaker=word.speaker or "unassigned",
                    words=[],
                )
            )
        lines[-1].words.append(PromptWord(id=word_id, text=text, marks=marks))
    return lines


def _prompt_time(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    clock = f"{minutes:02}:{secs:04.1f}"
    return f"{hours}:{clock}" if hours else clock


def prompt_edits(state: EditState, chunk: Chunk) -> list[PromptEdit]:
    return [
        PromptEdit(
            id=edit.id,
            kind=edit.kind,
            words=f"{edit.first_word}-{edit.last_word}"
            if edit.last_word != edit.first_word
            else str(edit.first_word),
            before=edit.before,
            after=edit.after,
            reason=edit.reason or "",
        )
        for edit in chunk_model_edits(state, chunk)
    ]


# --- running ------------------------------------------------------------------------


@dataclass
class _Budget:
    limit: float
    spent: float = 0.0
    reserved: float = 0.0
    estimated: bool = False

    def reserve(self, amount: float) -> bool:
        if self.spent + self.reserved + amount > self.limit + 1e-9:
            return False
        self.reserved += amount
        return True

    def settle(self, amount: float, cost: float | None) -> float:
        """Release a reservation, charging ``cost`` (or the cap, if unknown)."""
        self.reserved -= amount
        if cost is None:
            self.estimated = True
            cost = amount
        self.spent += cost
        return cost


@dataclass
class _Run:
    agent: ClaudeAgent
    source: LoadedSource
    glossary: tuple[GlossaryEntry, ...]
    diagnostics: Path
    budget: _Budget
    deadline: float
    state: EditState
    reports: dict[int, ChunkReport]
    review: dict[int, list[ReviewItem]] = field(default_factory=dict)
    rejections: list[Rejection] = field(default_factory=list)
    reservations: dict[tuple[int, Role], float] = field(default_factory=dict)

    def spec(self, role: Role) -> AgentSpec:
        return AgentSpec(
            model=EDIT_MODEL if role == "edit" else REVIEW_MODEL,
            tools=(),
            allowed_tools=(),
            mcp_servers={},
            setting_sources=(),
            permission_mode="dontAsk",
            cwd=self.diagnostics,
            max_turns=MAX_TURNS,
            max_budget_usd=EDIT_CALL_BUDGET_USD
            if role == "edit"
            else REVIEW_CALL_BUDGET_USD,
            effort=EFFORT,
            system_prompt=EDIT_SYSTEM_PROMPT
            if role == "edit"
            else REVIEW_SYSTEM_PROMPT,
        )

    async def call[T: BaseModel](
        self, chunk: Chunk, role: Role, prompt: StructuredPrompt[T]
    ) -> T | None:
        report = (
            self.reports[chunk.index].edit
            if role == "edit"
            else self.reports[chunk.index].review
        )
        name = f"chunk-{chunk.index:02}-{role}"
        (self.diagnostics / f"{name}-prompt.txt").write_text(prompt.render())
        remaining = self.deadline - time.monotonic()
        cap = EDIT_CALL_BUDGET_USD if role == "edit" else REVIEW_CALL_BUDGET_USD
        if remaining < MIN_CALL_SECONDS:
            report.status, report.error = "skipped", "not started: operation deadline"
            return None
        if not self.budget.reserve(cap):
            report.status, report.error = "skipped", "not started: budget exhausted"
            return None
        self.reservations[(chunk.index, role)] = cap
        started = time.monotonic()
        cost: float | None = None
        try:
            with anyio.fail_after(min(CALL_TIMEOUT_SECONDS, remaining)):
                parsed, reply = await self.agent.run_structured(
                    prompt,
                    self.spec(role),
                    stage=f"polish-transcript-{role}",
                    scope=name,
                )
        except TimeoutError:
            report.status = "failed"
            report.error = f"timed out after {time.monotonic() - started:.0f} s"
        except ClaudeAgentError as exc:
            cost = exc.usage.estimated_cost_usd
            report.status, report.error = "failed", _clip(str(exc))
        except ValidationError as exc:
            report.status = "failed"
            report.error = _clip(
                f"reply did not match {prompt.response_model.__name__}: {exc}"
            )
        except Exception as exc:
            report.status = "failed"
            report.error = _clip(f"{type(exc).__name__}: {exc}")
        else:
            cost = reply.usage.estimated_cost_usd
            report.status = "completed"
            report.models = sorted(reply.usage.model_usage) or (
                [reply.usage.model] if reply.usage.model else []
            )
            (self.diagnostics / f"{name}-response.json").write_text(
                _ResponseRecord(
                    parsed=parsed.model_dump(mode="json"),
                    usage=reply.usage.model_dump(mode="json"),
                ).model_dump_json(indent=2)
            )
            report.elapsed_seconds = round(time.monotonic() - started, 3)
            report.estimated_cost_usd = self.budget.settle(
                self.reservations.pop((chunk.index, role)), cost
            )
            return parsed
        report.elapsed_seconds = round(time.monotonic() - started, 3)
        report.estimated_cost_usd = self.budget.settle(
            self.reservations.pop((chunk.index, role)), cost
        )
        (self.diagnostics / f"{name}-error.txt").write_text((report.error or "") + "\n")
        return None

    async def chunk(self, chunk: Chunk, limiter: anyio.CapacityLimiter) -> None:
        async with limiter:
            report = self.reports[chunk.index]
            lines = prompt_lines(self.source.words, chunk)
            decisions = await self.call(
                chunk,
                "edit",
                EditPrompt(
                    first_word=chunk.first_word,
                    last_word=chunk.last_word,
                    glossary=list(self.glossary),
                    lines=lines,
                ),
            )
            if isinstance(decisions, EditDecisions):
                rejected = apply_edit_decisions(
                    self.state, chunk, decisions, self.glossary
                )
                self.rejections += rejected
                report.edit.accepted = len(chunk_model_edits(self.state, chunk))
                report.edit.rejected = len(rejected)
            findings = await self.call(
                chunk,
                "review",
                ReviewPrompt(
                    first_word=chunk.first_word,
                    last_word=chunk.last_word,
                    lines=lines,
                    edits=prompt_edits(self.state, chunk),
                ),
            )
            if isinstance(findings, ReviewFindings):
                before = len(chunk_model_edits(self.state, chunk))
                items, rejected = apply_review(self.state, chunk, findings)
                self.review[chunk.index] = items
                self.rejections += rejected
                report.review.accepted = len(items)
                report.review.rejected = len(rejected)
                report.review.restored = before - len(
                    chunk_model_edits(self.state, chunk)
                )


class _ResponseRecord(BaseModel):
    parsed: dict[str, object]
    usage: dict[str, object]


def _clip(text: str, limit: int = 600) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def polish(
    *,
    transcript_path: Path,
    output_directory: Path,
    glossary: tuple[GlossaryEntry, ...] = (),
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    budget_usd: float = DEFAULT_BUDGET_USD,
    agent: ClaudeAgent | None = None,
) -> PolishResult:
    """Polish ``transcript_path`` into the new ``output_directory``.

    Raises :class:`SourceError` before anything is written for a bad
    argument or source, and :class:`PolishFailedError` (after writing the
    diagnostics) when not one chunk could be edited.
    """
    started_at = datetime.now(UTC)
    started = time.monotonic()
    if not 0 < budget_usd <= MAX_BUDGET_USD:
        raise SourceError(
            "max_budget_usd", f"max_budget_usd must be in (0, {MAX_BUDGET_USD:g}]"
        )
    check_paths(transcript_path, output_directory)
    source = load_source(transcript_path)
    window = select_window(source.words, start_seconds, end_seconds)
    chunks = plan_chunks(source.words, window)
    if len(chunks) > MAX_CHUNKS:
        raise SourceError(
            "end_seconds",
            f"the window needs {len(chunks)} chunks; at most {MAX_CHUNKS} fit one "
            "call's time limit. Polish it in parts with start_seconds/end_seconds.",
        )
    output_directory.mkdir()
    diagnostics = output_directory / "diagnostics"
    diagnostics.mkdir()
    words = source.words
    run = _Run(
        agent=agent or ClaudeAgent(),
        source=source,
        glossary=glossary,
        diagnostics=diagnostics,
        budget=_Budget(limit=budget_usd),
        deadline=started + DEADLINE_SECONDS,
        state=EditState.start(words, window),
        reports={
            c.index: ChunkReport(
                index=c.index,
                first_word=c.first_word,
                last_word=c.last_word,
                start_seconds=words[c.first_word].start_seconds,
                end_seconds=words[c.last_word].end_seconds,
            )
            for c in chunks
        },
    )
    limiter = anyio.CapacityLimiter(CONCURRENCY)
    with anyio.move_on_after(DEADLINE_SECONDS + 10):
        async with anyio.create_task_group() as group:
            for chunk in chunks:
                group.start_soon(run.chunk, chunk, limiter)
    for (index, role), amount in list(run.reservations.items()):
        # Cancelled at the deadline: the provider never reported a cost.
        stage = run.reports[index].edit if role == "edit" else run.reports[index].review
        stage.estimated_cost_usd = run.budget.settle(amount, None)
    for report in run.reports.values():
        for stage in (report.edit, report.review):
            if stage.status == "pending":
                stage.status, stage.error = (
                    "failed",
                    "stopped at the operation deadline",
                )
    return _finish(run, chunks, window, output_directory, started_at, started)


def _finish(
    run: _Run,
    chunks: list[Chunk],
    window: Window,
    output_directory: Path,
    started_at: datetime,
    started: float,
) -> PolishResult:
    state, source, words = run.state, run.source, run.source.words
    ledger = ledger_words(state)
    _check_accounting(state, ledger)
    review = [
        (f"R{n}", item)
        for n, item in enumerate(
            sorted(
                (i for items in run.review.values() for i in items),
                key=lambda i: i.first_word,
            ),
            start=1,
        )
    ]
    notices: list[ChunkNotice] = []
    for chunk in chunks:
        report = run.reports[chunk.index]
        span = (
            f"{timestamp(report.start_seconds)}–{timestamp(report.end_seconds)} "
            f"(words {chunk.first_word}–{chunk.last_word})"
        )
        problems = []
        if report.edit.status != "completed":
            problems.append(
                f"not polished ({report.edit.error}); only standalone fillers were removed"
            )
        if report.review.status != "completed":
            problems.append(f"not reviewed ({report.review.error})")
        if problems:
            notices.append(
                ChunkNotice(
                    chunk=chunk.index,
                    first_word=chunk.first_word,
                    last_word=chunk.last_word,
                    message=f"{span}: " + "; ".join(problems) + ".",
                )
            )
    if not any(r.edit.status == "completed" for r in run.reports.values()):
        status: Literal["complete", "partial"] = "partial"
    else:
        status = "partial" if notices else "complete"
    unchanged = _sha256(source.path) == source.sha256
    applied = [e for e in state.edits.values() if e.status == "applied"]
    removed_kinds = Counter(state.edits[e].kind for e in state.removed.values())
    counts = PolishCounts(
        window_words=len(ledger),
        retained_words=sum(w.status == "retained" for w in ledger),
        removed_words=sum(w.status == "removed" for w in ledger),
        words_in_glossary_candidates=len(state.candidates),
        removed_by_kind=dict(sorted(removed_kinds.items())),
        formats=sum(e.kind == "format" for e in applied),
        glossary_candidates=sum(e.kind == "glossary_candidate" for e in applied),
        restored_after_review=sum(
            e.status == "restored_after_review" for e in state.edits.values()
        ),
        rejected_decisions=len(run.rejections),
        review_items=len(review),
    )
    notes = [n.message for n in notices]
    if not unchanged:
        notes.append(
            "The source file changed while it was being polished; outputs reflect the bytes read at the start."
        )
    if run.budget.estimated:
        notes.append("Some calls reported no cost; each was counted at its cap.")
    files = PolishFiles(
        transcript=str(output_directory / "transcript.md"),
        ledger=str(output_directory / "ledger.json"),
        review=str(output_directory / "review.json"),
        metadata=str(output_directory / "metadata.json"),
        diagnostics=str(run.diagnostics),
    )
    result = PolishResult(
        status=status,
        output_directory=str(output_directory),
        files=files,
        source_path=str(source.path),
        source_sha256=source.sha256,
        source_unchanged=unchanged,
        window=window,
        counts=counts,
        chunks=[run.reports[c.index] for c in chunks],
        estimated_cost_usd=round(run.budget.spent, 6),
        cost_is_estimate_only=run.budget.estimated,
        elapsed_seconds=round(time.monotonic() - started, 3),
        notes=notes,
    )
    markdown = render_markdown(
        RenderInput(
            state=state,
            ledger=ledger,
            review=review,
            notices=notices,
            source_name=source.path.name,
            source_sha256=source.sha256,
            status=status,
        )
    )
    _write(Path(files.transcript), markdown)
    _write(
        Path(files.ledger),
        _Ledger(
            policy_version=POLICY_VERSION,
            source_path=str(source.path),
            source_sha256=source.sha256,
            window=window,
            words=ledger,
            edits=sorted(state.edits.values(), key=lambda e: (e.first_word, e.id)),
            rejected=run.rejections,
        ).model_dump_json(),
    )
    _write(
        Path(files.review),
        _ReviewFile(
            complete=all(r.review.status == "completed" for r in run.reports.values()),
            items=[
                _ReviewEntry(
                    id=item_id,
                    chunk=item.chunk,
                    first_word=item.first_word,
                    last_word=item.last_word,
                    start_seconds=words[item.first_word].start_seconds,
                    end_seconds=words[item.last_word].end_seconds,
                    speaker=words[item.first_word].speaker,
                    categories=list(item.categories),
                    text=state.text(item.first_word, item.last_word),
                    note=item.note,
                )
                for item_id, item in review
            ],
        ).model_dump_json(indent=2),
    )
    _write(
        Path(files.metadata),
        PolishMetadata(
            started_at=started_at,
            policy=PolishPolicy(budget_usd=run.budget.limit, glossary=run.glossary),
            engine=source.transcript.engine.model_dump(),
            result=result,
        ).model_dump_json(indent=2),
    )
    if not any(r.edit.status == "completed" for r in run.reports.values()):
        first_error = next(
            (r.edit.error for r in run.reports.values() if r.edit.error), "unknown"
        )
        raise PolishFailedError(
            f"no chunk could be edited (first error: {first_error})", output_directory
        )
    return result


class _Ledger(BaseModel):
    policy_version: str
    source_path: str
    source_sha256: str
    window: Window
    words: list[LedgerWord]
    edits: list[Edit]
    rejected: list[Rejection]


class _ReviewEntry(BaseModel):
    id: str
    chunk: int
    first_word: int
    last_word: int
    start_seconds: float
    end_seconds: float
    speaker: str | None
    categories: list[str]
    text: str
    note: str


class _ReviewFile(BaseModel):
    complete: bool
    """Whether every chunk's review ran. Even then the list is not exhaustive."""
    items: list[_ReviewEntry]


def _check_accounting(state: EditState, ledger: list[LedgerWord]) -> None:
    """Every window word exactly once, in order, each change an applied edit."""
    assert [w.id for w in ledger] == list(state.window.ids())
    for word_id, edit in [*state.removed.items(), *state.candidates.items()]:
        assert state.edits[edit].status == "applied", (word_id, edit)
    assert not set(state.removed) & set(state.candidates)


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _write(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
