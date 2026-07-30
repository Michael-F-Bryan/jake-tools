"""Inference-free source adapters (Phase 2 slice).

Each adapter turns one already-ingested artefact's raw content into typed
bundle-store components -- no ASR, no diarisation, no LLM calls
(CONTRACTS.md M6/M7/M19/M20, D2, D6). An adapter's job stops at "produce
components, added to the store"; bringing them into the document is the
separate M18 ``assemble`` operation (``assemble.py``) -- adapting a
source alone never moves the head (M12).

Two of the three adapters (:func:`adapt_untimed_transcript`,
:func:`adapt_gemini_notes`) are pure over their text input: no filesystem
IO happens inside them, so they are testable with plain strings. The
third (:func:`adapt_teams_vtt`) is the deliberate exception -- it reuses
``parse.py``'s existing ``parse_teams_vtt`` primitive rather than
reimplementing VTT cue parsing (per the Phase 2 plan), and that function
itself requires a real file path (``SourceArtifact.raw_text_path``), so
this adapter does too.

Every adapter mints its own per-item IDs (turn segment IDs, participant
IDs) via :func:`.ids.mint_id` directly, before constructing a component
body -- the same precedent every ``ParticipantRecord`` in this codebase
already follows (see ``components.py``'s module docstring). Bringing
these components into existence in the store (``store.add_component``)
is still the only place their *identity* (``component_id``,
``content_hash``) is minted.

Crash-window discipline: every adapter here is a short sequence of
independent ``store.add_component`` calls (e.g. ``adapt_teams_vtt`` adds
a turn set, then a label set, then a participant set). Each call is
individually crash-safe (``BundleStore._write_json_exclusive``, M16); a
crash between two of them simply leaves the earlier component(s) written
and unreferenced by any revision -- harmless, since nothing downstream
can observe a component until an ``assemble()`` call (which has its own,
separately documented crash-window analysis) folds it into a revision.
Unlike ``add_component``'s own dedup (which compares *content*), a
retried adapter call is not itself deduplicated end to end: each retry
mints fresh per-item IDs (``source_segment_id``, ``participant_id``)
*before* calling ``add_component``, so those IDs become part of the
body's hashed content and a retry's component bodies differ from the
crashed attempt's -- the orphaned component(s) from the earlier attempt
stay behind as harmless, unreferenced garbage (the same "retry may
produce extra unreferenced records, never corruption" shape
``assemble.py`` documents for revisions).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..errors import TranscriptError
from ..models import SourceArtifact
from ..parse import parse_teams_vtt
from .components import (
    NotesComponent,
    NotesComponentBody,
    NotesKind,
    NotesSectionBody,
    ParticipantDeclarationSource,
    ParticipantRecord,
    ParticipantSetComponent,
    ParticipantSetComponentBody,
    ParticipantStatus,
    ProviderLabelSetComponent,
    ProviderLabelSetComponentBody,
    ProviderLabelSpan,
    TimedTurn,
    TimedTurnSetComponent,
    TimedTurnSetComponentBody,
    TranscriptAbsenceDeclaration,
    TranscriptAbsenceDeclarationBody,
    TrustClass,
    UntimedTurn,
    UntimedTurnSetComponent,
    UntimedTurnSetComponentBody,
)
from .ids import ArtefactId, mint_id
from .store import BundleStore


class AdapterError(TranscriptError):
    """Base class for every error a source adapter raises."""


class NoTurnsFoundError(AdapterError):
    """A markdown/VTT input produced zero usable turns."""


# -- (a) existing-untimed markdown -> untimed turn set + participants -------

_SPEAKER_TURN_RE = re.compile(
    r"^\*\*(?P<label>[^*:]+):\*\*\s*(?P<text>.+)\Z", re.DOTALL
)


def parse_untimed_markdown_turns(markdown_text: str) -> tuple[tuple[str, str], ...]:
    """Split a ``**Speaker:** text`` markdown transcript into ``(label,
    text)`` pairs, import order preserved (D6/M6). Paragraphs that do not
    match the speaker-turn shape (e.g. a leading ``# Title`` heading) are
    skipped, never guessed at.
    """
    turns: list[tuple[str, str]] = []
    for paragraph in re.split(r"\n\s*\n", markdown_text.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        match = _SPEAKER_TURN_RE.match(paragraph)
        if match is None:
            continue
        label = match.group("label").strip()
        text = " ".join(match.group("text").split())
        turns.append((label, text))
    return tuple(turns)


@dataclass(frozen=True)
class UntimedTranscriptAdaptation:
    turn_set: UntimedTurnSetComponent
    participants: ParticipantSetComponent


def adapt_untimed_transcript(
    store: BundleStore,
    *,
    source_artefact_id: ArtefactId,
    markdown_text: str,
) -> UntimedTranscriptAdaptation:
    """M14/D6: an already-timed-free speaker-labelled transcript, imported
    verbatim -- turns preserve exactly the supplied order and wording, no
    timestamps invented (D6). Distinct speaker labels become participant
    records: the operator naming this file as a transcript import *is*
    the declaration (M3's operator-assertion pattern, echoed here for
    M19), and every named speaker is ``speaking-evidenced`` -- they are
    literally shown speaking in the imported text, not merely declared
    present.
    """
    parsed = parse_untimed_markdown_turns(markdown_text)
    if not parsed:
        raise NoTurnsFoundError(
            "no '**Speaker:** text' turns found in the supplied markdown."
        )

    turn_set_body = UntimedTurnSetComponentBody(
        source_artefact_id=source_artefact_id,
        turns=tuple(
            UntimedTurn(
                source_segment_id=mint_id("seg"), speaker_label=label, text=text
            )
            for label, text in parsed
        ),
    )
    turn_set = store.add_component(turn_set_body)
    assert isinstance(turn_set, UntimedTurnSetComponent)

    seen_labels: dict[str, None] = {}
    for label, _text in parsed:
        seen_labels.setdefault(label, None)

    participant_body = ParticipantSetComponentBody(
        participants=tuple(
            ParticipantRecord(
                participant_id=mint_id("participant"),
                declaration_source=ParticipantDeclarationSource.OPERATOR,
                declaration_evidence=(
                    f"speaker label {label!r} in imported untimed transcript "
                    f"(artefact {source_artefact_id})"
                ),
                display_names=(label,),
                status=ParticipantStatus.SPEAKING_EVIDENCED,
            )
            for label in seen_labels
        )
    )
    participants = store.add_component(participant_body)
    assert isinstance(participants, ParticipantSetComponent)

    return UntimedTranscriptAdaptation(turn_set=turn_set, participants=participants)


# -- (b) Teams VTT -> timed turn set + provider label set + participants ----


@dataclass(frozen=True)
class TeamsSpeakerConfirmation:
    """M19: an explicit, operator-confirmed edge from one raw VTT ``<v>``
    label to a declared attendee's display name -- never derived by
    string-equality/casefold matching inside the adapter (F14). The
    fixture's ``Michael BRYAN`` cue label is exactly the trap this guards
    against: it differs in case from the declared attendee ``Michael
    Bryan``, so an adapter that "helpfully" matched case-insensitively
    would be doing the *inferring* M19 forbids.
    """

    raw_label: str
    participant_display_name: str


@dataclass(frozen=True)
class TeamsVttAdaptation:
    turn_set: TimedTurnSetComponent
    label_set: ProviderLabelSetComponent
    participants: ParticipantSetComponent
    warnings: tuple[str, ...] = ()


def _hash_proxy_config(room_proxy_display_names: frozenset[str]) -> str:
    canonical = json.dumps(sorted(room_proxy_display_names), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def adapt_teams_vtt(
    store: BundleStore,
    *,
    source_artefact_id: ArtefactId,
    vtt_path: Path,
    declared_attendees: tuple[str, ...],
    speaker_confirmations: tuple[TeamsSpeakerConfirmation, ...] = (),
    room_proxy_display_names: frozenset[str] = frozenset(),
) -> TeamsVttAdaptation:
    """M14: a genuine provider-attributed transcript -- parsed via the
    existing ``parse.py`` VTT primitive (reused, not reimplemented, per
    the Phase 2 plan), never through ASR/diarisation.

    ``speaker_confirmations`` is the explicit, operator-supplied raw-
    label -> declared-attendee edge (M19: "never bare string matching on
    the label text"). Only confirmed attendees become
    ``speaking-evidenced``; every other declared attendee (Des Everingham
    in the fixture) stays ``declared`` -- M19: "a declared attendee who
    never speaks stays declared."

    ``room_proxy_display_names`` is the explicit, hashed room-proxy
    config (D1/M5) -- a raw cue label present in this set gets trust
    class ``room-proxy``; every other cue defaults to
    ``per-participant-stream``, matching D1's default for a genuine
    online-meeting per-attendee VTT stream. Detection is config-driven,
    never inferred.

    Cues are re-sorted into M6 canonical order before any segment ID is
    minted (Teams VTT cues are not guaranteed to appear in chronological
    file order -- the fixture's own overlapping/interleaved cues are
    exactly this quirk); zero-length cues (legal raw evidence, M6) are
    dropped from the canonical timed turn set with a warning, since a
    canonical timed turn requires ``end_ms > start_ms`` strictly.
    """
    source = SourceArtifact(kind="msgraph-teams", raw_text_path=vtt_path)
    parsed = parse_teams_vtt(source)

    cues = sorted(
        parsed.turns,
        key=lambda turn: (round(turn.start * 1000), round(turn.end * 1000)),
    )

    # M19/F14: a confirmation is honoured only when its raw_label is
    # something the VTT actually said -- a typo'd or fabricated raw_label
    # must never silently promote a declared attendee to speaking-
    # evidenced (adversarial review MAJOR: a confirmation naming a raw
    # label absent from every cue previously still marked its attendee
    # speaking-evidenced). Checked up front, before any component is
    # built, so a bad confirmation fails the whole call rather than
    # partially applying.
    observed_labels = {cue.speaker for cue in cues}
    unknown_raw_labels = sorted(
        {
            confirmation.raw_label
            for confirmation in speaker_confirmations
            if confirmation.raw_label not in observed_labels
        }
    )
    if unknown_raw_labels:
        raise AdapterError(
            "speaker confirmation(s) name raw_label(s) that do not appear in "
            f"this VTT's cues: {unknown_raw_labels} -- available raw labels: "
            f"{sorted(observed_labels)}."
        )

    proxy_config_hash = _hash_proxy_config(room_proxy_display_names)
    confirmed_display_names = {
        confirmation.participant_display_name for confirmation in speaker_confirmations
    }
    confirmed_raw_labels = {
        confirmation.raw_label for confirmation in speaker_confirmations
    }
    declared_attendee_set = set(declared_attendees)

    turns: list[TimedTurn] = []
    spans: list[ProviderLabelSpan] = []
    warnings: list[str] = []
    unmapped_labels: set[str] = set()
    for cue in cues:
        start_ms = round(cue.start * 1000)
        end_ms = round(cue.end * 1000)
        if end_ms <= start_ms:
            warnings.append(
                f"dropped zero-length Teams VTT cue for {cue.speaker!r} at "
                f"{start_ms}ms (M6: zero-length cues are raw evidence only, "
                "never a canonical timed turn)."
            )
            continue
        segment_id = mint_id("seg")
        turns.append(
            TimedTurn(
                source_segment_id=segment_id,
                speaker_label=cue.speaker,
                text=cue.text,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
        trust_class = (
            TrustClass.ROOM_PROXY
            if cue.speaker in room_proxy_display_names
            else TrustClass.PER_PARTICIPANT_STREAM
        )
        spans.append(
            ProviderLabelSpan(
                source_segment_id=segment_id,
                raw_label=cue.speaker,
                text=cue.text,
                trust_class=trust_class,
            )
        )
        if (
            cue.speaker not in confirmed_raw_labels
            and cue.speaker not in declared_attendee_set
        ):
            unmapped_labels.add(cue.speaker)

    if not turns:
        raise NoTurnsFoundError(f"no usable (non-zero-length) cues in {vtt_path}")

    # MINOR (adversarial review): a real speaker with no roster entry and
    # no confirmation is correctly preserved as unattributed evidence
    # (no fabricated participant) -- but that should not be silent.
    for label in sorted(unmapped_labels):
        warnings.append(
            f"raw VTT speaker label {label!r} matches no declared attendee and "
            "has no speaker confirmation -- preserved as unattributed provider-"
            "label evidence only; no participant record represents them."
        )

    turn_set_body = TimedTurnSetComponentBody(
        source_artefact_id=source_artefact_id, turns=tuple(turns)
    )
    turn_set = store.add_component(turn_set_body)
    assert isinstance(turn_set, TimedTurnSetComponent)

    label_set_body = ProviderLabelSetComponentBody(
        source_artefact_id=source_artefact_id,
        proxy_config_hash=proxy_config_hash,
        spans=tuple(spans),
    )
    label_set = store.add_component(label_set_body)
    assert isinstance(label_set, ProviderLabelSetComponent)

    participant_body = ParticipantSetComponentBody(
        participants=tuple(
            ParticipantRecord(
                participant_id=mint_id("participant"),
                declaration_source=ParticipantDeclarationSource.TEAMS_ROSTER,
                declaration_evidence=(
                    f"Teams meeting metadata attendee list (artefact "
                    f"{source_artefact_id})"
                ),
                display_names=(attendee,),
                status=(
                    ParticipantStatus.SPEAKING_EVIDENCED
                    if attendee in confirmed_display_names
                    else ParticipantStatus.DECLARED
                ),
                room_proxy=attendee in room_proxy_display_names,
            )
            for attendee in declared_attendees
        )
    )
    participants = store.add_component(participant_body)
    assert isinstance(participants, ParticipantSetComponent)

    return TeamsVttAdaptation(
        turn_set=turn_set,
        label_set=label_set,
        participants=participants,
        warnings=tuple(warnings),
    )


# -- (c) Gemini notes-only markdown -> M20 notes + absence declaration ------

_HEADING_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$", re.MULTILINE)
_BULLET_RE = re.compile(r"^-\s+(?P<item>.+?)(?=\n-\s+|\Z)", re.MULTILINE | re.DOTALL)
_BOLD_RE = re.compile(r"\*\*")
_LEADING_PAREN_RE = re.compile(r"^\([^)]*\)\s*")
_NO_TRANSCRIPT_RE = re.compile(r"no\b.*\btranscript\b.*\bavailable\b", re.IGNORECASE)

_SECTION_KIND: dict[str, NotesKind] = {
    "Gemini summary": NotesKind.PROVIDER_SUMMARY,
    "Decisions": NotesKind.PROVIDER_DECISIONS,
    "Action items": NotesKind.PROVIDER_ACTIONS,
    "Details": NotesKind.PROVIDER_DETAILS,
}


def _extract_absence_statement(section: str) -> tuple[str, str]:
    """Locate the sentence stating no transcript was available inside a
    ``## Transcript`` section, returning ``(that sentence, the section
    with it removed)``.

    Only the matched sentence is ever stored as
    :class:`~.components.TranscriptAbsenceDeclaration`'s ``statement``
    (adversarial-review MINOR: storing the whole section previously
    glued any turn-shaped fake transcript content sharing the section
    into that evidence field). The remainder preserves the section's
    original layout (blank lines and all) so paragraph-shaped content in
    it -- e.g. a fabricated ``**Speaker:** text`` turn -- is still
    detectable by :func:`parse_untimed_markdown_turns` in the caller.
    Returns ``("", section)`` unchanged when the phrase is not found.
    """
    match = _NO_TRANSCRIPT_RE.search(section)
    if match is None:
        return "", section

    start = match.start()
    while start > 0 and section[start - 1] not in ".!?\n":
        start -= 1
    end = match.end()
    while end < len(section) and section[end] not in ".!?":
        end += 1
    if end < len(section):
        end += 1  # include the terminating punctuation

    statement = section[start:end].strip().strip("_").strip()
    remainder = (section[:start] + section[end:]).strip()
    return statement, remainder


def _split_markdown_sections(body: str) -> dict[str, str]:
    """``## Heading`` -> raw section text (heading excluded), in file
    order. Gemini's export shape (fixture-observed): ``## Gemini
    summary``, ``## Decisions``, ``## Action items``, ``## Details``,
    ``## Transcript``.
    """
    matches = list(_HEADING_RE.finditer(body))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        title = match.group("title").strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections[title] = body[start:end].strip()
    return sections


def _split_bullets(text: str) -> tuple[str, ...]:
    """One item per ``- `` bullet; a section with no bullets (the
    fixture's prose-shaped Gemini summary) falls back to one item
    covering the whole text -- both are real shapes this fixture uses.
    """
    items = tuple(match.group("item").strip() for match in _BULLET_RE.finditer(text))
    if items:
        return items
    return (text,) if text.strip() else ()


def _bullet_title(item: str, *, fallback_index: int) -> str:
    """A short, addressable title for one notes section (M20).

    Strips markdown bold markers and any leading ``(assignee)`` marker
    (the fixture's action items use this for an unresolved provider
    label like ``(Speaker)``/``(The group)`` -- kept verbatim in the
    section's own *text*, just not used as the title), then takes the
    text up to the first colon (the fixture's "Task Name: detail" shape)
    -- or a truncated prefix when there is no colon.
    """
    plain = _BOLD_RE.sub("", item).strip()
    plain = _LEADING_PAREN_RE.sub("", plain).strip()
    head = plain.split(":", 1)[0].strip()
    return head[:80] if head else f"Item {fallback_index}"


def _parse_frontmatter_attendees(markdown_text: str) -> tuple[str, ...]:
    """A deliberately narrow YAML-frontmatter attendee reader.

    Reads only the ``Attendees:`` block's ``- "[[Name]]"`` wikilink
    entries, matching the Gemini export shape this fixture uses -- no
    PyYAML dependency (not a declared project dependency; see
    ``pyproject.toml``/AGENTS.md's "no new dependencies").
    """
    if not markdown_text.startswith("---\n"):
        return ()
    _, _, rest = markdown_text.partition("---\n")
    frontmatter, separator, _ = rest.partition("\n---\n")
    if not separator:
        return ()
    attendees: list[str] = []
    capturing = False
    for line in frontmatter.splitlines():
        if not capturing:
            if line.startswith("Attendees:"):
                capturing = True
            continue
        if not line.startswith("  - "):
            break
        raw = line.removeprefix("  - ").strip().strip('"').strip("'")
        if raw.startswith("[[") and raw.endswith("]]"):
            raw = raw[2:-2]
        if raw:
            attendees.append(raw)
    return tuple(attendees)


def _notes_sections(items: Iterable[str]) -> tuple[NotesSectionBody, ...]:
    return tuple(
        NotesSectionBody(title=_bullet_title(item, fallback_index=index + 1), text=item)
        for index, item in enumerate(items)
    )


@dataclass(frozen=True)
class GeminiNotesAdaptation:
    notes: tuple[NotesComponent, ...]
    absence_declaration: TranscriptAbsenceDeclaration
    participants: ParticipantSetComponent


def adapt_gemini_notes(
    store: BundleStore,
    *,
    source_artefact_id: ArtefactId,
    markdown_text: str,
) -> GeminiNotesAdaptation:
    """D2/M20: a Gemini/Google notes-only export -- provider summary,
    decisions, actions, and details become typed M20 notes components;
    the note's own explicit "no transcript was available" statement
    becomes a :class:`~.components.TranscriptAbsenceDeclaration`, which
    is what lets ``transcript.timed``/``transcript.untimed`` report
    ``not-available-from-source`` rather than a bare ``absent`` (D2/M5:
    "the absence statement is itself the evidence the product must
    preserve"). Deliberately builds **no** transcript/turn-set component
    at all -- there is nothing to build one from.

    ``(Speaker)``/``(The group)`` provider-assignee markers are preserved
    verbatim inside each section's own text (never resolved into a
    participant -- they are not among the declared attendees, and this
    adapter never guesses); every declared attendee stays ``declared``,
    never ``speaking-evidenced`` -- there is no transcript evidence here
    that could evidence speech.
    """
    sections = _split_markdown_sections(markdown_text)

    notes: list[NotesComponent] = []
    for title, kind in _SECTION_KIND.items():
        text = sections.get(title, "").strip()
        if not text:
            continue
        notes_body = NotesComponentBody(
            notes_kind=kind,
            source_artefact_id=source_artefact_id,
            authored=False,
            sections=_notes_sections(_split_bullets(text)),
        )
        component = store.add_component(notes_body)
        assert isinstance(component, NotesComponent)
        notes.append(component)

    if not notes:
        raise AdapterError(
            "no provider-summary/decisions/actions/details sections found; this "
            "does not look like a Gemini notes export."
        )

    transcript_section = sections.get("Transcript", "").strip()
    statement, remainder = (
        _extract_absence_statement(transcript_section)
        if transcript_section
        else ("", "")
    )
    if not statement:
        raise AdapterError(
            "the '## Transcript' section did not state that no transcript was "
            "available -- adapt_gemini_notes only handles the notes-only case "
            "(D2); a populated transcript section needs a different adapter."
        )
    if remainder and parse_untimed_markdown_turns(remainder):
        raise AdapterError(
            "the '## Transcript' section states no transcript was available but "
            "also contains turn-shaped ('**Speaker:** text') content alongside "
            "it -- ambiguous evidence; refusing rather than silently discarding "
            "or embedding it into the absence declaration."
        )
    absence_body = TranscriptAbsenceDeclarationBody(
        source_artefact_id=source_artefact_id, statement=statement
    )
    absence_declaration = store.add_component(absence_body)
    assert isinstance(absence_declaration, TranscriptAbsenceDeclaration)

    attendees = _parse_frontmatter_attendees(markdown_text)
    if not attendees:
        raise AdapterError("no frontmatter Attendees found in the supplied note.")
    participant_body = ParticipantSetComponentBody(
        participants=tuple(
            ParticipantRecord(
                participant_id=mint_id("participant"),
                declaration_source=ParticipantDeclarationSource.NOTE_FRONTMATTER,
                declaration_evidence=(
                    f"note frontmatter Attendees (artefact {source_artefact_id})"
                ),
                display_names=(attendee,),
                status=ParticipantStatus.DECLARED,
            )
            for attendee in attendees
        )
    )
    participants = store.add_component(participant_body)
    assert isinstance(participants, ParticipantSetComponent)

    return GeminiNotesAdaptation(
        notes=tuple(notes),
        absence_declaration=absence_declaration,
        participants=participants,
    )
