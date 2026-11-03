"""What code accepts from the two model roles, and the resulting word state.

Every source word in the window ends up exactly one of: retained (possibly
with a punctuation/case display), removed by one accepted deletion, or part
of one tentative glossary candidate. There is no free replacement: a
deletion, a same-letters format and a glossary term are the only changes.

Guards are mechanical and judge only what can be checked without audio:

- every decision is inside the chunk that proposed it, and quotes its span;
- a repetition leaves an adjacent identical copy by the same speaker (and
  that copy is then pinned), and a lone negation/intensifier repeated may be
  emphasis, so it stays;
- a restart must be said again word for word straight after it (its last
  word may be cut off: "go-" before "going"), otherwise it may be a
  self-correction or a different claim;
- a filler is only a set-off "you know" or "I mean";
- no repetition or restart removes a number (it may be a year or digits);
- nothing spans a speaker change, a ``null`` speaker or overlapping speech
  (except a short filler), and deletions are budgeted per chunk;
- formats change only trailing punctuation and a first letter's case, never
  a question mark, a possible name's capital or an all-caps word;
- a glossary candidate names a supplied term, is spelled plausibly like the
  span it replaces, and replaces no protected word (negation, number,
  modality, scope, pronoun) the term does not itself contain.

What these guards cannot judge (a repetition that was meant, punctuation
that shifts a claim) the review role may question: each edit it names, and
every model edit inside a passage it flags, is restored to the source words.
The review is not a safety guarantee.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Literal

from pydantic import BaseModel

from .lexicon import (
    EMPHASIS,
    is_filler,
    is_number,
    lexical,
    looks_like_name,
    protected_kind,
    split_token,
)
from .prompts import (
    DeletionDecision,
    EditDecisions,
    FormatDecision,
    GlossaryDecision,
    GlossaryEntry,
    IssueCategory,
    ReviewFindings,
    ReviewIssue,
)
from .source import Chunk, SourceWord, Window

EditKind = Literal["filler", "repetition", "restart", "format", "glossary_candidate"]

MAX_REPETITION_WORDS = 4
MAX_RESTART_WORDS = 6
FILLER_PHRASES = frozenset({("you", "know"), ("i", "mean")})
MAX_GLOSSARY_WORDS = 4
MAX_COPY_GAP_SECONDS = 1.5
MAX_RESTART_GAP_SECONDS = 2.0
DELETION_SHARE = 0.20
MIN_DELETION_BUDGET = 10
GLOSSARY_SIMILARITY = 0.5
MAX_ISSUE_WORDS = 60
_TRAILING = re.compile(r"[.,;:!?…—]{0,3}")


class Edit(BaseModel):
    """One change to the reader text, with the source words it covers."""

    id: str
    origin: Literal["deterministic", "edit_model"]
    chunk: int | None
    kind: EditKind
    first_word: int
    last_word: int
    before: str
    after: str
    reason: str | None = None
    cited: tuple[int, int] | None = None
    """The span the model cited, when its quoted text sat a word or two away."""
    status: Literal["applied", "restored_after_review"] = "applied"
    review_note: str | None = None


class Rejection(BaseModel):
    """A model decision code refused, kept for diagnosis."""

    chunk: int
    role: Literal["edit", "review"]
    kind: str
    proposal: dict[str, Any]
    reason: str


class ReviewItem(BaseModel):
    """A place a reader should check; never a correction."""

    chunk: int
    first_word: int
    last_word: int
    categories: list[IssueCategory]
    note: str


@dataclass
class EditState:
    """The whole window's word state, mutated chunk by chunk."""

    words: tuple[SourceWord, ...]
    window: Window
    removed: dict[int, str] = field(default_factory=dict)
    formats: dict[int, tuple[str, str]] = field(default_factory=dict)
    """word id -> (edit id, display)"""
    candidates: dict[int, str] = field(default_factory=dict)
    """every word id a glossary candidate covers -> edit id"""
    edits: dict[str, Edit] = field(default_factory=dict)
    paragraphs: set[int] = field(default_factory=set)

    @classmethod
    def start(cls, words: tuple[SourceWord, ...], window: Window) -> EditState:
        """Remove standalone fillers; nothing else is deterministic."""
        state = cls(words=words, window=window)
        for word_id in window.ids():
            if is_filler(words[word_id].text):
                edit = Edit(
                    id=f"f{word_id}",
                    origin="deterministic",
                    chunk=None,
                    kind="filler",
                    first_word=word_id,
                    last_word=word_id,
                    before=words[word_id].text,
                    after="",
                )
                state.edits[edit.id] = edit
                state.removed[word_id] = edit.id
        return state

    def text(self, first: int, last: int) -> str:
        return " ".join(w.text for w in self.words[first : last + 1])

    def is_model_removed(self, word_id: int) -> bool:
        edit = self.removed.get(word_id)
        return edit is not None and self.edits[edit].origin == "edit_model"

    def restore(self, edit: Edit, note: str) -> None:
        for mapping in (self.removed, self.candidates):
            for word_id in [k for k, v in mapping.items() if v == edit.id]:
                del mapping[word_id]
        for word_id in [k for k, v in self.formats.items() if v[0] == edit.id]:
            del self.formats[word_id]
        self.edits[edit.id] = edit.model_copy(
            update={"status": "restored_after_review", "review_note": note}
        )


def quote_matches(
    words: tuple[SourceWord, ...], first: int, last: int, text: str
) -> bool:
    """Whether ``text`` names the span's words (case, punctuation and the
    parenthesised fillers aside). Catches a model citing the wrong IDs."""
    span = [lexical(w.text) for w in words[first : last + 1] if not is_filler(w.text)]
    return span == _quoted(text)


def _quoted(text: str) -> list[str]:
    return [
        lexical(t)
        for t in text.replace("(", " ").replace(")", " ").split()
        if not is_filler(t) and lexical(t)
    ]


QUOTE_SLACK = 2
"""How far a cited span may sit from the words it quotes and still be found."""


def locate_span(
    words: tuple[SourceWord, ...],
    chunk: Chunk,
    first: int,
    last: int,
    text: str,
    *,
    interchangeable: bool = False,
) -> tuple[int, int] | str:
    """The chunk span ``text`` quotes: the cited one, or else the nearest
    span starting within ``QUOTE_SLACK`` words of it that matches (models do
    cite one word off). A tie is refused, unless the candidates are
    ``interchangeable`` copies of one repetition: then the earlier is taken,
    and the repetition guard still needs the other to stay."""
    if first > last:
        return "first_word is after last_word"
    if not (chunk.owns(first) and chunk.owns(last)):
        return "outside this chunk's editable words"
    if quote_matches(words, first, last, text):
        return first, last
    wanted = len(_quoted(text))
    found: list[tuple[int, int]] = []
    for start in range(first - QUOTE_SLACK, first + QUOTE_SLACK + 1):
        if not chunk.owns(start) or is_filler(words[start].text) or wanted == 0:
            continue
        end, seen = start, 0
        while chunk.owns(end):
            seen += not is_filler(words[end].text)
            if seen == wanted:
                break
            end += 1
        if seen == wanted and quote_matches(words, start, end, text):
            found.append((start, end))
    if found:
        distance = min(abs(a - first) + abs(b - last) for a, b in found)
        nearest = [
            (a, b) for a, b in found if abs(a - first) + abs(b - last) == distance
        ]
        if len(nearest) == 1 or interchangeable:
            return nearest[0]
    return "text does not match the source words of the span"


@dataclass
class _ChunkEditor:
    state: EditState
    chunk: Chunk
    glossary: dict[str, GlossaryEntry]
    rejections: list[Rejection] = field(default_factory=list)
    pinned: set[int] = field(default_factory=set)
    serial: int = 0
    deleted: int = 0

    @property
    def words(self) -> tuple[SourceWord, ...]:
        return self.state.words

    def reject(self, kind: str, proposal: BaseModel, reason: str) -> None:
        self.rejections.append(
            Rejection(
                chunk=self.chunk.index,
                role="edit",
                kind=kind,
                proposal=proposal.model_dump(mode="json"),
                reason=reason,
            )
        )

    def add(self, **fields: Any) -> Edit:
        self.serial += 1
        edit = Edit(
            id=f"c{self.chunk.index}-e{self.serial}",
            origin="edit_model",
            chunk=self.chunk.index,
            **fields,
        )
        self.state.edits[edit.id] = edit
        return edit

    def resolve[D: DeletionDecision | GlossaryDecision](
        self, kind: str, decision: D
    ) -> D | None:
        """``decision`` on the span its text quotes, or ``None`` (rejected)."""
        span = locate_span(
            self.words,
            self.chunk,
            decision.first_word,
            decision.last_word,
            decision.text,
            interchangeable=getattr(decision, "kind", None) == "repetition",
        )
        if isinstance(span, str):
            self.reject(kind, decision, span)
            return None
        if span == (decision.first_word, decision.last_word):
            return decision
        return decision.model_copy(update={"first_word": span[0], "last_word": span[1]})

    def retained_after(self, word_id: int, limit: int) -> list[int]:
        """Up to ``limit`` retained IDs after ``word_id`` in this chunk, same
        speaker, stopping at a speaker change."""
        speaker = self.words[word_id].speaker
        found: list[int] = []
        for i in range(word_id + 1, self.chunk.last_word + 1):
            if self.words[i].speaker != speaker or len(found) == limit:
                break
            if i not in self.state.removed:
                found.append(i)
        return found

    def retained_before(self, word_id: int, limit: int) -> list[int]:
        speaker = self.words[word_id].speaker
        found: list[int] = []
        for i in range(word_id - 1, self.chunk.first_word - 1, -1):
            if self.words[i].speaker != speaker or len(found) == limit:
                break
            if i not in self.state.removed:
                found.append(i)
        return found[::-1]

    def turn_keeps_words(self, first: int, last: int) -> bool:
        """Whether the speaker run around ``first..last`` keeps a word."""
        speaker = self.words[first].speaker
        start, stop = first, last
        while start > 0 and self.words[start - 1].speaker == speaker:
            start -= 1
        while stop + 1 < len(self.words) and self.words[stop + 1].speaker == speaker:
            stop += 1
        return any(
            i not in self.state.removed
            for i in range(start, stop + 1)
            if not first <= i <= last
        )

    def protected(self, word_id: int) -> str | None:
        word = self.words[word_id]
        return protected_kind(
            lexical(word.text),
            name=looks_like_name(word.text, sentence_initial=word.sentence_initial),
        )

    # --- deletions ----------------------------------------------------------------

    def delete(self, proposed: DeletionDecision) -> None:
        decision = self.resolve("deletion", proposed)
        if decision is None:
            return
        first, last = decision.first_word, decision.last_word
        problem = self.deletion_problem(decision)
        if problem is not None:
            self.reject("deletion", proposed, problem)
            return
        core = [i for i in range(first, last + 1) if i not in self.state.removed]
        edit = self.add(
            kind=decision.kind,
            first_word=first,
            last_word=last,
            before=self.state.text(first, last),
            after="",
            reason=decision.reason,
            cited=_cited(proposed, decision),
        )
        for i in core:
            self.state.removed[i] = edit.id
        self.deleted += len(core)

    def deletion_problem(self, decision: DeletionDecision) -> str | None:
        first, last = decision.first_word, decision.last_word
        span = range(first, last + 1)
        if any(self.state.is_model_removed(i) for i in span):
            return "overlaps an accepted deletion"
        core = [i for i in span if i not in self.state.removed]
        if not core:
            return "only fillers code already removed"
        if pinned := [i for i in core if i in self.pinned]:
            return f"an accepted edit relies on word {pinned[0]} staying"
        speakers = {self.words[i].speaker for i in span}
        if len(speakers) != 1:
            return "spans a speaker change"
        if not self.turn_keeps_words(first, last):
            return "would remove a whole speaker turn"
        budget = max(MIN_DELETION_BUDGET, int(self.chunk.size * DELETION_SHARE))
        if self.deleted + len(core) > budget:
            return f"exceeds this chunk's deletion budget of {budget} words"
        match decision.kind:
            case "filler":
                return self.filler_problem(first, core)
            case "repetition":
                return self.repetition_problem(first, last, core, speakers)
            case "restart":
                return self.restart_problem(last, core, speakers)

    def filler_problem(self, first: int, core: list[int]) -> str | None:
        """Only a set-off "you know" / "I mean": anything else may carry
        meaning ("I like it", "I mean it", "you know the answer")."""
        if tuple(lexical(self.words[i].text) for i in core) not in FILLER_PHRASES:
            return (
                "only a set-off 'you know' or 'I mean' is a filler (code already "
                "removes um/uh/erm)"
            )
        before = self.retained_before(first, 1)
        opens = (
            not before
            or self.words[core[0]].sentence_initial
            or "," in split_token(self.words[before[0]].text)[2]
        )
        closes = bool(set(split_token(self.words[core[-1]].text)[2]) & set(",.!"))
        if not (opens and closes):
            return "not set off by punctuation, so it may carry meaning"
        return None

    def repetition_problem(
        self, first: int, last: int, core: list[int], speakers: set[str | None]
    ) -> str | None:
        if len(core) > MAX_REPETITION_WORDS:
            return f"a repetition is at most {MAX_REPETITION_WORDS} words"
        if None in speakers or any(self.words[i].overlap for i in core):
            return "unattributed or overlapping speech may be another speaker's copy"
        tokens = [lexical(self.words[i].text) for i in core]
        unit = _repeated_unit(tokens)
        if len(unit) == 1 and unit[0] in EMPHASIS:
            return f"a repeated {unit[0]!r} may be emphasis"
        if any(is_number(t) for t in unit):
            return "a repeated number may be a year or dictated digits"
        for copy in (
            self.retained_after(last, len(unit)),
            self.retained_before(first, len(unit)),
        ):
            if (
                len(copy) == len(unit)
                and [lexical(self.words[i].text) for i in copy] == unit
                and not any(self.words[i].overlap for i in copy)
                and self.gap(core, copy) <= MAX_COPY_GAP_SECONDS
            ):
                self.pinned.update(copy)
                return None
        return "no identical copy by the same speaker stays next to it"

    def gap(self, span: list[int], copy: list[int]) -> float:
        earlier, later = (span, copy) if span[0] < copy[0] else (copy, span)
        return self.words[later[0]].start_seconds - self.words[earlier[-1]].end_seconds

    def restart_problem(
        self, last: int, core: list[int], speakers: set[str | None]
    ) -> str | None:
        if len(core) > MAX_RESTART_WORDS:
            return f"a restart is at most {MAX_RESTART_WORDS} words"
        if None in speakers or any(self.words[i].overlap for i in core):
            return "unattributed or overlapping speech cannot be a restart"
        said = [lexical(self.words[i].text) for i in core]
        if any(is_number(t) for t in said):
            return "a number may be a year or dictated digits, not a restart"
        follow = self.retained_after(last, len(core))
        if (
            not follow
            or self.words[follow[0]].start_seconds - self.words[last].end_seconds
            > MAX_RESTART_GAP_SECONDS
        ):
            return "the speaker does not continue straight after it"
        again = [lexical(self.words[i].text) for i in follow]
        *head, tail = said
        echoed = len(again) == len(said) and again[: len(head)] == head
        if echoed and tail != again[-1]:
            # Only a cut-off last word ("go-" before "going"), never a
            # different word, and never a protected one ("no" before "not").
            echoed = (
                again[-1].startswith(tail)
                and len(tail) < len(again[-1])
                and self.protected(core[-1]) is None
            )
        if not echoed:
            return (
                "what follows does not say it again word for word (or finish a "
                "cut-off word): it may be a different claim, so it stays"
            )
        self.pinned.update(follow)
        return None

    # --- formats ------------------------------------------------------------------

    def format(self, decision: FormatDecision) -> None:
        problem = self.format_problem(decision)
        if problem is not None:
            self.reject("format", decision, problem)
            return
        i = decision.word
        edit = self.add(
            kind="format",
            first_word=i,
            last_word=i,
            before=self.words[i].text,
            after=decision.display,
        )
        self.state.formats[i] = (edit.id, decision.display)

    def format_problem(self, decision: FormatDecision) -> str | None:
        i = decision.word
        if not self.chunk.owns(i):
            return "outside this chunk's editable words"
        if i in self.state.removed:
            return "the word was removed"
        if i in self.state.formats or i in self.state.candidates:
            return "the word already has an accepted edit"
        source = self.words[i].text
        lead, core, trail = split_token(source)
        new_lead, new_core, new_trail = split_token(decision.display)
        if new_lead != lead or new_core.casefold() != core.casefold() or not new_core:
            return "changes more than case and trailing punctuation"
        if new_core[1:] != core[1:]:
            return "changes case beyond the first letter"
        if not _TRAILING.fullmatch(new_trail):
            return "unsupported trailing punctuation"
        if ("?" in trail) != ("?" in new_trail):
            return "adds or removes a question mark"
        word = self.words[i]
        if (
            core[0].isupper()
            and new_core[0].islower()
            and looks_like_name(source, sentence_initial=word.sentence_initial)
        ):
            return "lowercases a possible name"
        if core[0].isupper() and new_core[0].islower() and len(core) > 1 and core.isupper():
            return "lowercases an all-caps word"
        if decision.display == source:
            return "no change"
        return None

    # --- glossary -----------------------------------------------------------------

    def candidate(self, proposed: GlossaryDecision) -> None:
        decision = self.resolve("glossary", proposed)
        if decision is None:
            return
        problem = self.glossary_problem(decision)
        if problem is not None:
            self.reject("glossary", proposed, problem)
            return
        first, last = decision.first_word, decision.last_word
        entry = self.glossary[decision.term.casefold()]
        lead = split_token(self.words[first].text)[0]
        trail = split_token(self.words[last].text)[2]
        edit = self.add(
            kind="glossary_candidate",
            first_word=first,
            last_word=last,
            before=self.state.text(first, last),
            after=f"{lead}[{entry.term}?]{trail}",
            reason=decision.reason,
            cited=_cited(proposed, decision),
        )
        for i in range(first, last + 1):
            self.state.candidates[i] = edit.id

    def glossary_problem(self, decision: GlossaryDecision) -> str | None:
        entry = self.glossary.get(decision.term.casefold())
        if entry is None:
            return "term is not in the supplied glossary"
        span = range(decision.first_word, decision.last_word + 1)
        if len(span) > MAX_GLOSSARY_WORDS:
            return f"a glossary span is at most {MAX_GLOSSARY_WORDS} words"
        if any(
            i in self.state.removed
            or i in self.state.formats
            or i in self.state.candidates
            for i in span
        ):
            return "a word in the span already has an accepted edit"
        if len({self.words[i].speaker for i in span}) != 1:
            return "spans a speaker change"
        heard = "".join(
            c for i in span for c in lexical(self.words[i].text) if c.isalnum()
        )
        term = "".join(c for c in entry.term.casefold() if c.isalnum())
        if SequenceMatcher(None, heard, term).ratio() < GLOSSARY_SIMILARITY:
            return "not spelled plausibly like the term"
        term_tokens = {lexical(t) for t in entry.term.split()}
        for i in span:
            kind = self.protected(i)
            token = lexical(self.words[i].text)
            if kind not in (None, "name") and token not in term_tokens:
                return f"would replace a {kind} word ({self.words[i].text!r})"
        if self.state.text(span.start, span.stop - 1) == entry.term:
            return "no change"
        return None

    # --- paragraphs -----------------------------------------------------------------

    def paragraphs(self, breaks: list[int]) -> None:
        allowed = self.chunk.size // 15 + 1
        for word_id in sorted(set(breaks)):
            if not self.chunk.owns(word_id):
                self.rejections.append(
                    Rejection(
                        chunk=self.chunk.index,
                        role="edit",
                        kind="paragraph_break",
                        proposal={"word": word_id},
                        reason="outside this chunk's editable words",
                    )
                )
            elif allowed > 0:
                self.state.paragraphs.add(word_id)
                allowed -= 1


def _repeated_unit(tokens: list[str]) -> list[str]:
    """The shortest run ``tokens`` repeats whole ("I I" -> ["I"]): deleting
    several stuttered copies needs one copy of the unit to stay, not of all."""
    for size in range(1, len(tokens) + 1):
        if len(tokens) % size == 0 and tokens == tokens[:size] * (len(tokens) // size):
            return tokens[:size]
    return tokens


def _cited(
    proposed: DeletionDecision | GlossaryDecision,
    resolved: DeletionDecision | GlossaryDecision,
) -> tuple[int, int] | None:
    cited = (proposed.first_word, proposed.last_word)
    return None if cited == (resolved.first_word, resolved.last_word) else cited


def apply_edit_decisions(
    state: EditState,
    chunk: Chunk,
    decisions: EditDecisions,
    glossary: tuple[GlossaryEntry, ...],
) -> list[Rejection]:
    """Validate ``decisions`` for ``chunk`` and apply what passes, in source
    order: deletions, then formats, then glossary candidates."""
    editor = _ChunkEditor(
        state=state,
        chunk=chunk,
        glossary={entry.term.casefold(): entry for entry in glossary},
    )
    for deletion in sorted(
        decisions.deletions, key=lambda d: (d.first_word, d.last_word)
    ):
        editor.delete(deletion)
    for fmt in decisions.formats:
        editor.format(fmt)
    for candidate in decisions.glossary:
        editor.candidate(candidate)
    editor.paragraphs(decisions.paragraph_breaks)
    return editor.rejections


def chunk_model_edits(state: EditState, chunk: Chunk) -> list[Edit]:
    return [
        edit
        for edit in state.edits.values()
        if edit.chunk == chunk.index and edit.status == "applied"
    ]


def apply_review(
    state: EditState, chunk: Chunk, findings: ReviewFindings
) -> tuple[list[ReviewItem], list[Rejection]]:
    """Keep the issues that point at this chunk's words (merged where they
    overlap) and restore the source for every edit a concern names and every
    model edit an issue's span touches."""
    rejections: list[Rejection] = []

    def reject(kind: str, proposal: BaseModel, reason: str) -> None:
        rejections.append(
            Rejection(
                chunk=chunk.index,
                role="review",
                kind=kind,
                proposal=proposal.model_dump(mode="json"),
                reason=reason,
            )
        )

    items: list[ReviewItem] = []
    located: list[tuple[int, int, ReviewIssue]] = []
    for issue in findings.issues:
        span = locate_span(
            state.words, chunk, issue.first_word, issue.last_word, issue.text
        )
        if isinstance(span, str):
            reject("issue", issue, span)
        elif span[1] - span[0] + 1 > MAX_ISSUE_WORDS:
            reject("issue", issue, f"spans more than {MAX_ISSUE_WORDS} words")
        else:
            located.append((*span, issue))
    for first, last, issue in sorted(located, key=lambda x: (x[0], x[1])):
        if items and first <= items[-1].last_word + 1:
            merged = items[-1]
            merged.last_word = max(merged.last_word, last)
            if issue.category not in merged.categories:
                merged.categories.append(issue.category)
            merged.note = f"{merged.note} {issue.note}".strip()
        else:
            items.append(
                ReviewItem(
                    chunk=chunk.index,
                    first_word=first,
                    last_word=last,
                    categories=[issue.category],
                    note=issue.note.strip(),
                )
            )
    restorable = {edit.id: edit for edit in chunk_model_edits(state, chunk)}
    for concern in findings.edit_concerns:
        edit = restorable.pop(concern.edit_id, None)
        if edit is None:
            reject("edit_concern", concern, "not an accepted edit of this chunk")
        else:
            state.restore(edit, concern.note.strip())
    # A passage the review doubts keeps its source words: undo every model
    # edit that touches it, whether or not the review named the edit. Code's
    # own filler removal is not a model edit and stays.
    for item in items:
        for edit in list(restorable.values()):
            if edit.first_word <= item.last_word and item.first_word <= edit.last_word:
                del restorable[edit.id]
                state.restore(edit, f"inside a flagged passage: {item.note}")
    return items, rejections
