"""M22: working out who was in a meeting when the note does not say.

Resolution order is **frontmatter, then operator, then inference**. A note
carrying an `Attendees` list is never inferred over — an explicit list is
a declaration and beats a guess. `--participant` is the operator saying so
directly. Only when neither exists does this module run.

Why inference is admissible here at all, when M19 forbids it for speaker
attribution: **inferring who was present is not inferring who said what.**
A participant record is the menu the reviewer chooses from; M8's ladder
still needs a reviewed decision before a single turn carries a name. An
inferred attendee cannot put a name against a spoken word, so the failure
mode M19 guards against stays impossible. What it *can* do is stop a note
being refused at ingest for a missing frontmatter key, which is the
concrete problem it exists to solve.

Two guards make the output honest rather than merely plausible:

- every returned name must be supported by text that actually appears in
  the note or its filename -- a name from nowhere is rejected, not stored;
- the resulting records carry ``note-inferred`` provenance, a distinct
  value from ``operator``. Recording a model's guess as an operator
  assertion would be a lie about exactly the field M19 exists to protect,
  and the review pack shows the source so a reviewer can see which names
  were guessed.

The stage reads the note's filename and body only, never a transcript:
attendees are resolved at ingest, before any transcript exists, and
reading one would reintroduce the "this voice sounds like Michael"
reasoning the ladder keeps out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from ...prompting import StructuredPrompt
from ..stages import StructuredAgent, run_structured_with_retries
from .components import ParticipantDeclarationSource

#: How much of a long note the stage sees. Attendees are named early --
#: frontmatter, an opening line, a quoted email header -- and a 40-page
#: ops log would otherwise cost more than the answer is worth.
_MAX_NOTE_CHARS = 12_000

#: A name must be at least this many characters to be considered supported
#: by the note text. Without it, a one- or two-letter "name" would match
#: almost any document by accident.
_MIN_NAME_CHARS = 3


@dataclass(frozen=True)
class ExternalParticipant:
    """A participant declared from outside the note's own frontmatter.

    Carries its provenance with it so the adapter never has to guess
    which route a name arrived by -- an operator assertion and a model's
    inference are different claims and stay different records (M19/M22).
    """

    display_name: str
    declaration_source: ParticipantDeclarationSource
    declaration_evidence: str


class InferredAttendee(BaseModel):
    display_name: str
    evidence: str = ""


class AttendeeInferencePayload(BaseModel):
    attendees: list[InferredAttendee] = Field(default_factory=list)


class AttendeeInferencePrompt(StructuredPrompt[AttendeeInferencePayload]):
    response_model: ClassVar[type[BaseModel]] = AttendeeInferencePayload
    template: ClassVar[str] = """
Work out which *people* took part in the meeting this note records.

You are reading a personal meeting note. It names people in prose and in
wiki-links, and it also links plenty of things that are not people --
documents, procedures, courses, roles, organisations, dates. Only return
people who were actually part of the conversation.

Include someone if the note shows them taking part: named as who the
conversation was with, quoted, recorded as saying or agreeing something, or
listed as present. The note's author counts if the note shows them
participating.

Exclude:
- documents, policies, courses, systems, places, dates, organisations, and
  role titles, even when they are wiki-linked exactly like a person;
- people mentioned only as a topic ("we should ask Rob about X") who were
  not in the conversation;
- anyone you are guessing at. Returning fewer, certain names is better than
  returning more.

For each person, `evidence` quotes the short phrase from the note that shows
they took part. A name you cannot quote evidence for must not be returned.

Return JSON only. `display_name` must be spelled exactly as the note spells
it, without wiki-link brackets.

Note filename:
{{ filename }}

Note content:
{{ note_text }}
{% if correction %}

Your previous response was invalid: {{ correction }}
Return corrected JSON that fixes this issue.
{% endif %}
"""

    filename: str
    note_text: str
    correction: str = ""


def _normalise(text: str) -> str:
    """Casefolded, punctuation-free text for support checking.

    A note writes ``[[Joshua Jenkins|Josh Jenkins]]`` and prose writes
    ``Josh``; comparing raw strings would reject names the note plainly
    contains. Stripping to words and casefolding keeps the check strict
    about *content* while tolerant of markup.
    """
    return " ".join(re.findall(r"[\w']+", text.casefold()))


def name_is_supported(name: str, *, haystack: str) -> bool:
    """Does the note actually contain this name?

    The guard against a fabricated attendee. Requires every word of the
    name to appear in the note (not merely the surname), so ``Steven
    Crawford`` is supported by a note naming him and rejected by one that
    only mentions a different Crawford.
    """
    if len(name.strip()) < _MIN_NAME_CHARS:
        return False
    words = _normalise(name).split()
    if not words:
        return False
    return all(word in haystack.split() for word in words)


async def infer_attendees(
    agent: StructuredAgent,
    *,
    note_path: Path,
    note_text: str,
    max_attempts: int = 2,
) -> tuple[ExternalParticipant, ...]:
    """M22: infer the meeting's participants from the note and filename.

    Returns empty rather than raising when nothing can be supported: a
    note with no discernible people is a real situation, and the caller's
    own refusal ("no participants -- pass --participant") is a better
    message than one from in here.
    """
    payload, _reply = await run_structured_with_retries(
        agent,
        AttendeeInferencePrompt(
            filename=note_path.name, note_text=note_text[:_MAX_NOTE_CHARS]
        ),
        max_attempts=max_attempts,
    )
    haystack = _normalise(f"{note_path.name} {note_text}")
    resolved: list[ExternalParticipant] = []
    seen: set[str] = set()
    for attendee in payload.attendees:
        name = attendee.display_name.strip().strip("[]")
        if not name or name in seen:
            continue
        if not name_is_supported(name, haystack=haystack):
            continue
        seen.add(name)
        quoted = attendee.evidence.strip()
        resolved.append(
            ExternalParticipant(
                display_name=name,
                declaration_source=ParticipantDeclarationSource.NOTE_INFERRED,
                declaration_evidence=(
                    f"inferred from {note_path.name}"
                    + (f": {quoted!r}" if quoted else "")
                ),
            )
        )
    return tuple(resolved)


def operator_participants(names: tuple[str, ...]) -> tuple[ExternalParticipant, ...]:
    """Names the operator declared on the command line (M19/M3)."""
    return tuple(
        ExternalParticipant(
            display_name=name,
            declaration_source=ParticipantDeclarationSource.OPERATOR,
            declaration_evidence="declared on the command line",
        )
        for name in dict.fromkeys(names)
        if name.strip()
    )
