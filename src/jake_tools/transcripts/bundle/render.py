"""M17: rendering a document revision into note markdown.

A render is **pure**. It reads the bound document revision and nothing
else -- no live reads, no head lookup, no filesystem browsing, no clock.
That is what makes rendering the same revision twice byte-identical, which
the corpus verifies by double render and hash compare (§8), and it is why
this module takes a :class:`~.document.TranscriptDocumentV1` rather than a
store.

What it produces is the *owned region body* (M13), not a whole note: the
three generated sections and nothing above or around them. Frontmatter,
the note's title, authored prep notes, and recording embeds belong to the
author and are never re-emitted here -- ``apply.py`` splices this body
into the note between markers and leaves everything else byte-identical.

The M5 speaker gate lives here, at the point of publication. Every
canonical turn's effective assignment (the single M8 ladder function) must
come from a trusted provider class, a reviewed decision, or an explicit
``unclear-speaker`` -- rendered as "Unclear speaker" with a visible count.
A bare machine hypothesis fails the gate and :func:`render_meeting_note`
refuses. That refusal is the point of the whole review loop: the note
either says who spoke because someone decided, or says it does not know.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..errors import TranscriptError
from ..merge import format_timestamp
from .assignment import (
    UNCLEAR_SPEAKER_LABEL,
    AssignmentCoverage,
    SpeakerContext,
    assignment_coverage,
    canonical_turns,
    speaker_display_name,
)
from .components import (
    ChapterRecord,
    ChapterSetComponent,
    ClaimStatus,
    DestinationComponent,
    FindingKind,
    MinutesComponent,
    MinutesFinding,
    TimedTurn,
)
from .document import TranscriptDocumentV1
from .ids import TurnId
from .registry import CapabilityKey, CapabilityStatus

#: Bumped whenever this module's own output could differ for an unchanged
#: revision. Recorded on every render (M17) so a byte difference is always
#: attributable to a named change rather than to drift.
RENDERER_VERSION = "v2"

#: The meeting-note profile's own version, separate from the renderer's:
#: a layout change to this profile does not invalidate a source-note
#: render, and vice versa.
MEETING_NOTE_PROFILE_VERSION = "v2"

MEETING_NOTE_PROFILE = "meeting-note"
DUMC_VARIANT = "dumc"

#: The structural contract of the DUM-C layout: section order and heading
#: text. Hashed into every render record as M17's ``template_sha256``.
#: There is no external template file to hash -- the layout *is* this
#: string plus the code below, so changing the layout without changing
#: this string would be a silent template change, which is exactly what
#: M17's field exists to prevent.
_DUMC_LAYOUT = (
    "summary-callout|## Discussion Notes"
    "|nested-decisions|nested-next-steps|nested-risks|nested-questions"
    "|attribution-note"
    "|## Chapters|chapter-list"
    "|## Transcript|chaptered-turns"
)

#: The legacy heading set M13's migration path recognises. Imported from
#: ``merge.py`` at use sites rather than re-declared, so the two can never
#: disagree about what "a generated section" is.


class RenderError(TranscriptError):
    """Base class for every error this module raises."""


class SpeakerGateFailedError(RenderError):
    """M5: one or more turns would publish a bare machine hypothesis.

    The meeting-note profile refuses rather than shipping an attribution
    nobody confirmed. Resolve it by reviewing (``review export`` ->
    ``review apply``); an explicit ``unclear-speaker`` is a perfectly good
    resolution and satisfies the gate.
    """


class MissingProductError(RenderError):
    """M14: the profile declares a component required and it is absent."""


@dataclass(frozen=True)
class RenderResult:
    """The rendered body plus the facts M17 wants on the render record."""

    body: str
    input_capability_keys: tuple[str, ...]
    coverage: AssignmentCoverage
    unresolved_turn_ids: tuple[TurnId, ...]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.body.encode("utf-8")).hexdigest()


def template_sha256() -> str:
    """M17's template hash for the meeting-note profile."""
    return hashlib.sha256(_DUMC_LAYOUT.encode("utf-8")).hexdigest()


def destination_of(document: TranscriptDocumentV1) -> DestinationComponent | None:
    """The apply target this document declares (M13), or ``None``.

    More than one destination is treated as none: two candidate targets is
    an ambiguity, and picking one would mean writing to a note the
    operator did not choose.
    """
    destinations = document.components_of(DestinationComponent)
    return destinations[0] if len(destinations) == 1 else None


def _required_minutes(document: TranscriptDocumentV1) -> MinutesComponent:
    minutes = document.components_of(MinutesComponent)
    if len(minutes) != 1:
        raise MissingProductError(
            f"the meeting-note profile requires exactly one minutes component "
            f"(M14); this document has {len(minutes)}. Run `transform minutes`."
        )
    if document.capability_status(CapabilityKey.MINUTES) != (
        CapabilityStatus.PRESENT_VALIDATED
    ):
        raise MissingProductError(
            "the minutes component is present but did not validate (M10: every "
            "finding needs a resolvable evidence ref); refusing to render "
            "unsourced findings."
        )
    return minutes[0]


def _required_chapters(document: TranscriptDocumentV1) -> ChapterSetComponent:
    chapters = document.components_of(ChapterSetComponent)
    if len(chapters) != 1:
        raise MissingProductError(
            f"the meeting-note profile requires exactly one chapter set (M14); "
            f"this document has {len(chapters)}. Run `transform chapter`."
        )
    if document.capability_status(CapabilityKey.CHAPTERS) != (
        CapabilityStatus.PRESENT_VALIDATED
    ):
        raise MissingProductError(
            "the chapter set is present but did not validate against the canonical "
            "turn sequence (M7 exact coverage); refusing to render a table of "
            "contents that does not match the transcript."
        )
    return chapters[0]


def _finding_lines(
    findings: tuple[MinutesFinding, ...], kind: FindingKind, context: SpeakerContext
) -> list[str]:
    """One nested bullet per finding of ``kind``, notes-derived ones marked.

    D2/F22 require notes-derived claims to be visibly distinct from
    transcript-derived ones in the rendered output -- a reader must be able
    to tell "the meeting decided this" from "the provider's notes say the
    meeting decided this" without opening the bundle.
    """
    lines: list[str] = []
    for finding in findings:
        if finding.kind != kind:
            continue
        parts = [f"\t- {finding.text}"]
        if finding.owner_participant_id is not None:
            participant = context.participants.get(finding.owner_participant_id)
            if participant is not None:
                parts.append(f" — **{participant.display_names[0]}**")
        if finding.due:
            parts.append(f" (due {finding.due})")
        if finding.claim_status == ClaimStatus.NOTES_DERIVED:
            parts.append(" _(from notes, not the transcript)_")
        elif finding.claim_status == ClaimStatus.MIXED:
            parts.append(" _(from notes and transcript)_")
        lines.append("".join(parts))
    return lines


_FINDING_HEADINGS: tuple[tuple[FindingKind, str], ...] = (
    (FindingKind.DECISION, "Decisions"),
    (FindingKind.ACTION, "Next steps"),
    (FindingKind.RISK, "Risks"),
    (FindingKind.QUESTION, "Open questions"),
)


def _chapter_turns(
    chapter: ChapterRecord, turns_by_id: dict[str, TimedTurn]
) -> list[TimedTurn]:
    return [
        turns_by_id[turn_id] for turn_id in chapter.turn_ids if turn_id in turns_by_id
    ]


def render_meeting_note(
    document: TranscriptDocumentV1,
    *,
    include_transcript: bool = True,
) -> RenderResult:
    """Render the DUM-C meeting-note owned region for ``document``.

    Refuses -- never degrades -- when the M5 speaker gate fails or a
    profile-required product is missing. The result is a body string plus
    the facts a render record needs; writing it anywhere is a separate,
    separately verified act (M13).
    """
    turns = canonical_turns(document.components)
    if not turns:
        raise MissingProductError(
            "the meeting-note profile requires canonical timed turns (M14); this "
            "document has none. Run `transform normalise`."
        )
    context = SpeakerContext.from_components(document.components)
    assignments = {turn.turn_id: context.assignment_for(turn) for turn in turns}
    failures = tuple(
        turn_id
        for turn_id, assignment in assignments.items()
        if not assignment.satisfies_meeting_note_gate
    )
    if failures:
        raise SpeakerGateFailedError(
            f"{len(failures)} of {len(turns)} turns would publish an unreviewed "
            "speaker attribution (M5: a bare machine hypothesis never renders into "
            "a meeting note). Export a review pack, decide them -- 'unclear "
            f"speaker' is a valid decision -- and apply it. First: {failures[0]}"
        )
    coverage = assignment_coverage(assignments.values())
    minutes = _required_minutes(document)
    chapters = _required_chapters(document)

    lines: list[str] = ["> [!summary]"]
    lines.extend(
        f"> {line}" for line in minutes.summary.text.strip().splitlines() or [""]
    )
    lines.extend(["", "## Discussion Notes"])
    for kind, heading in _FINDING_HEADINGS:
        bullets = _finding_lines(minutes.findings, kind, context)
        if not bullets:
            continue
        lines.extend(["", f"- {heading}"])
        lines.extend(bullets)

    if coverage.unresolved:
        lines.extend(
            [
                "",
                "> [!warning] Speaker attribution",
                f"> {coverage.unresolved} of {coverage.total_turns} turns could not "
                f"be attributed to a participant and are shown as "
                f'"{UNCLEAR_SPEAKER_LABEL}".',
            ]
        )

    lines.extend(["", "## Chapters", ""])
    lines.extend(
        f"- {format_timestamp(chapter.start_ms / 1000)} — {chapter.title}"
        + (f": {chapter.summary}" if chapter.summary else "")
        for chapter in chapters.chapters
    )

    if include_transcript:
        turns_by_id = {turn.turn_id: turn for turn in turns}
        lines.extend(["", "## Transcript"])
        for chapter in chapters.chapters:
            lines.extend(
                [
                    "",
                    f"### {format_timestamp(chapter.start_ms / 1000)} — "
                    f"{chapter.title}",
                    "",
                ]
            )
            for turn in _chapter_turns(chapter, turns_by_id):
                speaker = speaker_display_name(assignments[turn.turn_id], context)
                lines.append(f"**{speaker}** {turn.text}")
                lines.append("")
            if lines and lines[-1] == "":
                lines.pop()

    body = "\n".join(lines).rstrip() + "\n"
    return RenderResult(
        body=body,
        input_capability_keys=tuple(
            sorted(
                key.value
                for key, record in document.capabilities.items()
                if record.status == CapabilityStatus.PRESENT_VALIDATED
            )
        ),
        coverage=coverage,
        unresolved_turn_ids=tuple(
            turn_id
            for turn_id, assignment in assignments.items()
            if not assignment.resolved
        ),
    )
