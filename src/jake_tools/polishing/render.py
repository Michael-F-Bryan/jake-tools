"""Deterministic output: the reader's Markdown and the word ledger.

Rendering owns everything a model never sees as an option: speaker turn
blocks (one per source speaker run, labels exactly as the source gave them,
``null`` shown as "Unassigned speaker"), timestamps, sentence-start
capitalisation, carrying a removed word's sentence-ending punctuation to the
word before it, review markers and notices for chunks that were not polished.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from .edits import EditState, ReviewItem
from .lexicon import ends_sentence, split_token
from .source import SourceWord, turns

PARAGRAPH_SECONDS = 45.0

WordStatus = Literal["retained", "removed", "in_candidate"]


class LedgerWord(BaseModel):
    """One source word and what became of it."""

    id: int
    text: str
    start_seconds: float
    end_seconds: float
    speaker: str | None
    overlap: bool
    confidence: float
    segment: int
    status: WordStatus
    display: str | None = None
    """What the reader sees for this word (``None`` unless retained, or the
    first word of a glossary candidate, which shows the candidate)."""
    edit: str | None = None
    """The edit that removed, reformatted or replaced it, if any."""


def timestamp(seconds: float) -> str:
    whole = int(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02}:{secs:02}" if hours else f"{minutes:02}:{secs:02}"


def speaker_label(speaker: str | None) -> str:
    return speaker if speaker is not None else "Unassigned speaker"


def _capitalise(display: str) -> str:
    lead, core, trail = split_token(display)
    # Only all-lower-case words: "iPhone" or "eBay" keep their shape.
    if core and core[0].islower() and core.islower():
        core = core[0].upper() + core[1:]
    return lead + core + trail


def _with_terminal(display: str, terminal: str) -> str:
    lead, core, trail = split_token(display)
    if ends_sentence(display) or not core:
        return display
    return lead + core + terminal


def sentence_ends(state: EditState, word_id: int, display: str) -> bool:
    """Whether a shown word ends a sentence. A glossary candidate's own
    ``?`` marks doubt, not a question: its last source word decides."""
    edit = state.candidates.get(word_id)
    if edit is not None:
        return ends_sentence(state.words[state.edits[edit].last_word].text)
    return ends_sentence(display)


def ledger_words(state: EditState) -> list[LedgerWord]:
    """Every window word with its final status and reader display."""
    words = state.words
    ids = list(state.window.ids())
    result: list[LedgerWord] = []
    sentence_start = True
    previous_speaker: object = object()
    for position, word_id in enumerate(ids):
        word = words[word_id]
        status: WordStatus
        display: str | None = None
        edit: str | None = None
        if word.speaker != previous_speaker:
            sentence_start = True
            previous_speaker = word.speaker
        if word_id in state.removed:
            status, edit = "removed", state.removed[word_id]
        elif word_id in state.candidates:
            edit = state.candidates[word_id]
            candidate = state.edits[edit]
            if candidate.first_word == word_id:
                status, display = "retained", candidate.after
            else:
                status = "in_candidate"
        else:
            status = "retained"
            if word_id in state.formats:
                edit, display = state.formats[word_id]
            else:
                display = word.text
            display = _carry_terminal(state, ids, position, display)
        if display is not None:
            if sentence_start:
                display = _capitalise(display)
            sentence_start = sentence_ends(state, word_id, display)
        result.append(
            LedgerWord(
                id=word.id,
                text=word.text,
                start_seconds=word.start_seconds,
                end_seconds=word.end_seconds,
                speaker=word.speaker,
                overlap=word.overlap,
                confidence=word.confidence,
                segment=word.segment,
                status=status,
                display=display,
                edit=edit,
            )
        )
    return result


def _carry_terminal(
    state: EditState, ids: list[int], position: int, display: str
) -> str:
    """A removed word that ended a sentence ("home um.") hands its full stop,
    question or exclamation mark to the retained word before it."""
    speaker = state.words[ids[position]].speaker
    terminal = ""
    for word_id in ids[position + 1 :]:
        word = state.words[word_id]
        if word_id not in state.removed or word.speaker != speaker:
            break
        trail = split_token(word.text)[2]
        marks = "".join(c for c in trail if c in ".?!")
        terminal = marks[:1] if marks else terminal
    return _with_terminal(display, terminal) if terminal else display


@dataclass(frozen=True)
class ChunkNotice:
    """A chunk whose edit or review pass did not complete, shown in place."""

    chunk: int
    first_word: int
    last_word: int
    message: str


@dataclass(frozen=True)
class RenderInput:
    state: EditState
    ledger: list[LedgerWord]
    review: list[tuple[str, ReviewItem]]
    notices: list[ChunkNotice]
    source_name: str
    source_sha256: str
    status: Literal["complete", "partial"]


def render_markdown(data: RenderInput) -> str:
    state, words = data.state, data.state.words
    by_id = {w.id: w for w in data.ledger}
    markers = _markers(data)
    notices = {n.first_word: n for n in data.notices}
    first, last = state.window.first_word, state.window.last_word
    out: list[str] = [
        "# Polished transcript",
        "",
        f"Clean-verbatim draft of `{data.source_name}` (source SHA-256 "
        f"`{data.source_sha256[:16]}…`), words {first}–{last}, "
        f"{timestamp(words[first].start_seconds)}–{timestamp(words[last].end_seconds)}.",
        "",
        "Speaker labels come from automatic diarisation and are not verified "
        "identities. Fillers, stutters and abandoned restarts were removed and "
        "punctuation adjusted; `ledger.json` records every change with its "
        "source words. `[Term?]` is a tentative glossary candidate, not a "
        "verified correction. `[R#]` points to a review item below. Unmarked "
        "text is not verified: the review had no audio and is not exhaustive.",
        "",
    ]
    if data.status == "partial":
        out += [
            "> **Partial result.** Some passages were not polished or not "
            "reviewed; each is marked where it occurs and listed under Notes.",
            "",
        ]
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            out.extend([" ".join(paragraph), ""])
            paragraph.clear()

    shown_speaker: object = object()
    heading = ""
    paragraph_start = 0.0
    sentence_start = True
    for run in turns(words, state.window):
        emitted = [i for i in run if by_id[i].display is not None]
        pending_notices = [notices[i] for i in run if i in notices]
        if not emitted and not pending_notices:
            # Only removed fillers: no block of its own (the ledger keeps them).
            continue
        flush()
        if words[run.start].speaker != shown_speaker or pending_notices:
            # The same speaker either side of an emptied turn stays one block.
            heading = _heading(words, run)
            out.extend([heading, ""])
            shown_speaker = words[run.start].speaker
            sentence_start = True
        paragraph_start = words[run.start].start_seconds
        for word_id in run:
            notice = notices.get(word_id)
            if notice is not None:
                flush()
                out.extend([f"> **Notice:** {notice.message}", ""])
                if word_id != run.start:
                    out.extend([f"{heading} _(continued)_", ""])
                paragraph_start = words[word_id].start_seconds
            entry = by_id[word_id]
            if entry.display is None:
                continue
            start = words[word_id].start_seconds
            if paragraph and (
                word_id in state.paragraphs
                or (sentence_start and start - paragraph_start > PARAGRAPH_SECONDS)
            ):
                flush()
                paragraph_start = start
            paragraph.append(
                entry.display + "".join(f" [{m}]" for m in markers.get(word_id, []))
            )
            sentence_start = sentence_ends(state, word_id, entry.display)
        flush()
    out += _review_section(data)
    out += _glossary_section(data)
    out += _notes_section(data)
    return "\n".join(out).rstrip() + "\n"


def _heading(words: tuple[SourceWord, ...], run: range) -> str:
    first = words[run.start]
    suffix = " _(overlapping speech)_" if any(words[i].overlap for i in run) else ""
    return (
        f"**[{timestamp(first.start_seconds)}] {speaker_label(first.speaker)}**{suffix}"
    )


def _markers(data: RenderInput) -> dict[int, list[str]]:
    """Each review item's marker goes after the first shown word of its span
    (or the next shown word, if all of the span was removed)."""
    shown = [w.id for w in data.ledger if w.display is not None]
    markers: dict[int, list[str]] = {}
    for item_id, item in data.review:
        anchor = next((i for i in shown if i >= item.first_word), None)
        if anchor is None and shown:
            anchor = shown[-1]
        if anchor is not None:
            markers.setdefault(anchor, []).append(item_id)
    return markers


def _review_section(data: RenderInput) -> list[str]:
    words = data.state.words
    out = ["## Review items", ""]
    if not data.review:
        out += ["None reported. That is not evidence the transcript is accurate.", ""]
    for item_id, item in data.review:
        first, last = words[item.first_word], words[item.last_word]
        categories = ", ".join(c.replace("_", " ") for c in item.categories)
        quote = data.state.text(item.first_word, item.last_word)
        out.append(
            f"- **{item_id}** [{timestamp(first.start_seconds)}–"
            f"{timestamp(last.end_seconds)}] {speaker_label(first.speaker)}, "
            f"{categories}: “{quote}” — {item.note}"
        )
    out.append("")
    return out


def _glossary_section(data: RenderInput) -> list[str]:
    words = data.state.words
    candidates = [
        e
        for e in data.state.edits.values()
        if e.kind == "glossary_candidate" and e.status == "applied"
    ]
    if not candidates:
        return []
    out = ["## Glossary candidates (tentative)", ""]
    for edit in sorted(candidates, key=lambda e: e.first_word):
        out.append(
            f"- [{timestamp(words[edit.first_word].start_seconds)}] heard "
            f"“{edit.before}”, shown as {edit.after.strip()}: {edit.reason}"
        )
    out.append("")
    return out


def _notes_section(data: RenderInput) -> list[str]:
    restored = [
        e for e in data.state.edits.values() if e.status == "restored_after_review"
    ]
    out = ["## Notes", ""]
    for notice in data.notices:
        out.append(f"- {notice.message}")
    if restored:
        out.append(
            f"- {len(restored)} edit(s) were undone after the review questioned "
            "them; the source words are shown instead (see `ledger.json`)."
        )
    out.append(
        "- Recognition can drop or mishear speech, and speaker turns can be "
        "split or merged; this draft cannot recover what the recogniser missed."
    )
    return out + [""]
