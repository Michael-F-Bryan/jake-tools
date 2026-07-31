from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from fixtures_bundle import registered_source_and_artefact

from jake_tools.transcripts.bundle.adapters import (
    AdapterError,
    FfprobeFailedError,
    FfprobeNotFoundError,
    NoTurnsFoundError,
    TeamsSpeakerConfirmation,
    adapt_gemini_notes,
    adapt_local_media,
    adapt_obsidian_note,
    adapt_teams_vtt,
    adapt_untimed_transcript,
    parse_untimed_markdown_turns,
    probe_audio_metadata,
)
from jake_tools.transcripts.bundle.components import (
    DestinationComponent,
    MediaRecordingComponent,
    OwnedRegionState,
    ParticipantDeclarationSource,
    ParticipantStatus,
    RecordingReferenceSetComponent,
    TrustClass,
)
from jake_tools.transcripts.bundle.store import BundleStore
from jake_tools.transcripts.models import SourceArtifact
from jake_tools.transcripts.parse import parse_teams_vtt

# Real, immutable evaluation-corpus fixtures (CONTRACTS.md Phase 2 brief):
# read-only, absolute path -- this worktree has no _working/ of its own,
# but the corpus lives at a fixed location on this machine regardless of
# which git worktree is running the tests.
_CORPUS_FIXTURES = Path(
    "/Users/work/Documents/jake-tools/_working/"
    "transcription-overhaul-context-2026-07-29/evaluation-corpus/fixtures"
)
_HAS_CORPUS = _CORPUS_FIXTURES.is_dir()
_requires_corpus = pytest.mark.skipif(
    not _HAS_CORPUS, reason=f"evaluation corpus not present at {_CORPUS_FIXTURES}"
)


def _store(tmp_path: Path) -> BundleStore:
    store = BundleStore(tmp_path / "bundle")
    store.create_bundle()
    return store


# -- parse_untimed_markdown_turns: synthetic edge cases -----------------------


def test_parse_untimed_markdown_turns_skips_a_non_matching_paragraph() -> None:
    text = "# Title\n\n**Alice:** hello\n\n**Bob:** hi there\n"

    turns = parse_untimed_markdown_turns(text)

    assert turns == (("Alice", "hello"), ("Bob", "hi there"))


def test_parse_untimed_markdown_turns_returns_empty_for_no_matches() -> None:
    assert parse_untimed_markdown_turns("just some prose, no speaker turns") == ()


def test_parse_untimed_markdown_turns_collapses_internal_whitespace() -> None:
    text = "**Alice:** line one\nstill line one   with  extra space\n"

    turns = parse_untimed_markdown_turns(text)

    assert turns == (("Alice", "line one still line one with extra space"),)


# -- adapt_untimed_transcript --------------------------------------------------


def test_adapt_untimed_transcript_raises_when_no_turns_found(tmp_path: Path) -> None:
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=b"# just a title")

    with pytest.raises(NoTurnsFoundError):
        adapt_untimed_transcript(
            store,
            source_artefact_id=artefact.artefact_id,
            markdown_text="# just a title",
        )


def test_adapt_untimed_transcript_dedupes_participants_by_label(tmp_path: Path) -> None:
    store = _store(tmp_path)
    text = "**Alice:** hi\n\n**Bob:** hey\n\n**Alice:** again\n"
    artefact = registered_source_and_artefact(store, content=text.encode())

    result = adapt_untimed_transcript(
        store, source_artefact_id=artefact.artefact_id, markdown_text=text
    )

    assert len(result.turn_set.turns) == 3
    assert [turn.speaker_label for turn in result.turn_set.turns] == [
        "Alice",
        "Bob",
        "Alice",
    ]
    assert len(result.participants.participants) == 2
    assert all(
        participant.status == ParticipantStatus.SPEAKING_EVIDENCED
        for participant in result.participants.participants
    )


@_requires_corpus
def test_adapt_untimed_transcript_preserves_the_real_fixture_import_order(
    tmp_path: Path,
) -> None:
    text = (_CORPUS_FIXTURES / "existing-untimed-transcript" / "input.md").read_text(
        encoding="utf-8"
    )
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=text.encode("utf-8"))

    result = adapt_untimed_transcript(
        store, source_artefact_id=artefact.artefact_id, markdown_text=text
    )

    assert [turn.speaker_label for turn in result.turn_set.turns] == [
        "Michael Bryan",
        "Michael Bryan",
        "Morgan McDermont",
        "Michael Bryan",
        "David Htet",
        "Michael Bryan",
        "Michael Bryan",
    ]
    assert not hasattr(result.turn_set.turns[0], "start_ms")
    display_names = {
        participant.display_names[0] for participant in result.participants.participants
    }
    assert display_names == {"Michael Bryan", "Morgan McDermont", "David Htet"}
    assert all(
        participant.status == ParticipantStatus.SPEAKING_EVIDENCED
        for participant in result.participants.participants
    )


# -- adapt_teams_vtt: synthetic controlled cases -------------------------------


def _write_vtt(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "input.vtt"
    path.write_text("WEBVTT\n\n" + body, encoding="utf-8")
    return path


def test_adapt_teams_vtt_drops_zero_length_cues_with_a_warning(tmp_path: Path) -> None:
    vtt_path = _write_vtt(
        tmp_path,
        "00:00:00.000 --> 00:00:00.000\n<v Alice>silent</v>\n\n"
        "00:00:01.000 --> 00:00:02.000\n<v Alice>hello</v>\n",
    )
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=vtt_path.read_bytes())

    result = adapt_teams_vtt(
        store,
        source_artefact_id=artefact.artefact_id,
        vtt_path=vtt_path,
        declared_attendees=("Alice",),
    )

    assert len(result.turn_set.turns) == 1
    assert result.turn_set.turns[0].text == "hello"
    assert len(result.warnings) == 1
    assert "zero-length" in result.warnings[0]


def test_adapt_teams_vtt_raises_when_every_cue_is_zero_length(tmp_path: Path) -> None:
    vtt_path = _write_vtt(
        tmp_path, "00:00:00.000 --> 00:00:00.000\n<v Alice>silent</v>\n"
    )
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=vtt_path.read_bytes())

    with pytest.raises(NoTurnsFoundError):
        adapt_teams_vtt(
            store,
            source_artefact_id=artefact.artefact_id,
            vtt_path=vtt_path,
            declared_attendees=("Alice",),
        )


def test_adapt_teams_vtt_room_proxy_config_sets_trust_class_and_participant_flag(
    tmp_path: Path,
) -> None:
    vtt_path = _write_vtt(
        tmp_path,
        "00:00:00.000 --> 00:00:01.000\n<v Meetings Ahoy>hello everyone</v>\n\n"
        "00:00:01.000 --> 00:00:02.000\n<v Alice>hi</v>\n",
    )
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=vtt_path.read_bytes())

    result = adapt_teams_vtt(
        store,
        source_artefact_id=artefact.artefact_id,
        vtt_path=vtt_path,
        declared_attendees=("Meetings Ahoy", "Alice"),
        room_proxy_display_names=frozenset({"Meetings Ahoy"}),
    )

    trust_by_label = {
        span.raw_label: span.trust_class for span in result.label_set.spans
    }
    assert trust_by_label["Meetings Ahoy"] == TrustClass.ROOM_PROXY
    assert trust_by_label["Alice"] == TrustClass.PER_PARTICIPANT_STREAM
    proxy_participant = next(
        p
        for p in result.participants.participants
        if p.display_names[0] == "Meetings Ahoy"
    )
    assert proxy_participant.room_proxy is True


def test_adapt_teams_vtt_raises_when_a_confirmation_names_a_raw_label_absent_from_the_vtt(
    tmp_path: Path,
) -> None:
    """MAJOR (adversarial review): a TeamsSpeakerConfirmation.raw_label
    was never cross-checked against the VTT's actual cues -- a
    fabricated raw_label for a never-speaking attendee silently promoted
    them to speaking-evidenced, fabricating speech evidence (M19: "a
    declared attendee who never speaks stays declared"; D1's speech
    gate). A confirmation naming a raw_label the VTT never actually said
    must fail closed, not silently pass through or silently downgrade."""
    vtt_path = _write_vtt(
        tmp_path,
        "00:00:00.000 --> 00:00:01.000\n<v Alice>hi</v>\n\n"
        "00:00:01.000 --> 00:00:02.000\n<v Bob>hey</v>\n",
    )
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=vtt_path.read_bytes())

    with pytest.raises(AdapterError, match="do not appear in this VTT's cues"):
        adapt_teams_vtt(
            store,
            source_artefact_id=artefact.artefact_id,
            vtt_path=vtt_path,
            declared_attendees=("Alice", "Bob", "NonSpeaker"),
            speaker_confirmations=(
                TeamsSpeakerConfirmation(
                    raw_label="THIS LABEL DOES NOT EXIST IN THE VTT AT ALL",
                    participant_display_name="NonSpeaker",
                ),
            ),
        )


def test_adapt_teams_vtt_confirmed_attendee_becomes_speaking_evidenced(
    tmp_path: Path,
) -> None:
    vtt_path = _write_vtt(
        tmp_path, "00:00:00.000 --> 00:00:01.000\n<v Alice X>hi</v>\n"
    )
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=vtt_path.read_bytes())

    result = adapt_teams_vtt(
        store,
        source_artefact_id=artefact.artefact_id,
        vtt_path=vtt_path,
        declared_attendees=("Alice X", "Bob Never Speaks"),
        speaker_confirmations=(
            TeamsSpeakerConfirmation(
                raw_label="Alice X", participant_display_name="Alice X"
            ),
        ),
    )

    statuses = {p.display_names[0]: p.status for p in result.participants.participants}
    assert statuses["Alice X"] == ParticipantStatus.SPEAKING_EVIDENCED
    assert statuses["Bob Never Speaks"] == ParticipantStatus.DECLARED


# -- adapt_teams_vtt: real fixture quirk tests --------------------------------


@_requires_corpus
def test_adapt_teams_vtt_preserves_michael_bryan_casing_quirk_verbatim(
    tmp_path: Path,
) -> None:
    """The raw VTT cue label is `Michael BRYAN` (provider casing), which
    differs from the declared attendee `Michael Bryan`. The adapter must
    never silently case-fold this into a match (M19/F14) -- it is
    preserved verbatim as evidence, and only becomes an edge to the
    participant via an explicit TeamsSpeakerConfirmation."""
    vtt_path = _CORPUS_FIXTURES / "teams-attributed-vtt" / "input.vtt"
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=vtt_path.read_bytes())

    result = adapt_teams_vtt(
        store,
        source_artefact_id=artefact.artefact_id,
        vtt_path=vtt_path,
        declared_attendees=(
            "Michael Bryan",
            "Joanne Olsen",
            "Sam Lintern",
            "Des Everingham",
        ),
    )

    raw_labels = {span.raw_label for span in result.label_set.spans}
    assert "Michael BRYAN" in raw_labels
    assert "Michael Bryan" not in raw_labels
    # Without an explicit confirmation, Michael Bryan is NOT auto-matched
    # despite the labels differing only in case.
    statuses = {p.display_names[0]: p.status for p in result.participants.participants}
    assert statuses["Michael Bryan"] == ParticipantStatus.DECLARED


@_requires_corpus
def test_adapt_teams_vtt_refuses_a_fabricated_confirmation_for_a_non_speaking_attendee(
    tmp_path: Path,
) -> None:
    """The verifier's exact repro against the real fixture: Des
    Everingham is declared but never speaks in this window. A
    confirmation naming a raw_label that appears nowhere in the VTT must
    not be able to promote him to speaking-evidenced."""
    vtt_path = _CORPUS_FIXTURES / "teams-attributed-vtt" / "input.vtt"
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=vtt_path.read_bytes())

    with pytest.raises(AdapterError, match="do not appear in this VTT's cues"):
        adapt_teams_vtt(
            store,
            source_artefact_id=artefact.artefact_id,
            vtt_path=vtt_path,
            declared_attendees=(
                "Michael Bryan",
                "Joanne Olsen",
                "Sam Lintern",
                "Des Everingham",
            ),
            speaker_confirmations=(
                TeamsSpeakerConfirmation(
                    raw_label="THIS LABEL DOES NOT EXIST IN THE VTT AT ALL",
                    participant_display_name="Des Everingham",
                ),
            ),
        )


@_requires_corpus
def test_adapt_teams_vtt_flags_an_unmapped_raw_label_in_warnings(
    tmp_path: Path,
) -> None:
    """MINOR 2 (adversarial review): a raw label matching no declared
    attendee and no confirmation is correctly preserved as unattributed
    evidence (no fabricated participant) -- but must be visible to an
    operator via warnings, the same channel zero-length-cue drops use."""
    vtt_path = _CORPUS_FIXTURES / "teams-attributed-vtt" / "input.vtt"
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=vtt_path.read_bytes())

    result = adapt_teams_vtt(
        store,
        source_artefact_id=artefact.artefact_id,
        vtt_path=vtt_path,
        declared_attendees=("Michael Bryan", "Joanne Olsen", "Sam Lintern"),
    )

    assert any("Michael BRYAN" in warning for warning in result.warnings)
    assert any(
        "no participant record represents them" in warning
        for warning in result.warnings
    )


@_requires_corpus
def test_adapt_teams_vtt_explicit_confirmation_resolves_the_casing_quirk(
    tmp_path: Path,
) -> None:
    vtt_path = _CORPUS_FIXTURES / "teams-attributed-vtt" / "input.vtt"
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=vtt_path.read_bytes())

    result = adapt_teams_vtt(
        store,
        source_artefact_id=artefact.artefact_id,
        vtt_path=vtt_path,
        declared_attendees=(
            "Michael Bryan",
            "Joanne Olsen",
            "Sam Lintern",
            "Des Everingham",
        ),
        speaker_confirmations=(
            TeamsSpeakerConfirmation(
                raw_label="Michael BRYAN", participant_display_name="Michael Bryan"
            ),
            TeamsSpeakerConfirmation(
                raw_label="Joanne Olsen", participant_display_name="Joanne Olsen"
            ),
            TeamsSpeakerConfirmation(
                raw_label="Sam Lintern", participant_display_name="Sam Lintern"
            ),
        ),
    )

    statuses = {p.display_names[0]: p.status for p in result.participants.participants}
    assert statuses["Michael Bryan"] == ParticipantStatus.SPEAKING_EVIDENCED
    assert statuses["Joanne Olsen"] == ParticipantStatus.SPEAKING_EVIDENCED
    assert statuses["Sam Lintern"] == ParticipantStatus.SPEAKING_EVIDENCED
    # Des Everingham is declared but never speaks in this window (D1/M19
    # -- fixture's expected-behaviour.md).
    assert statuses["Des Everingham"] == ParticipantStatus.DECLARED


@_requires_corpus
def test_adapt_teams_vtt_preserves_every_overlapping_cue_without_loss(
    tmp_path: Path,
) -> None:
    """The fixture's raw cues are not in chronological file order and
    include genuine cross-speaker overlap (e.g. Michael's short "Mm."
    interjection during Sam's longer turn). Normalisation must not drop
    or duplicate any of them (eval corpus §"How to evaluate" 5)."""
    vtt_path = _CORPUS_FIXTURES / "teams-attributed-vtt" / "input.vtt"
    raw_cue_count = len(
        parse_teams_vtt(
            SourceArtifact(kind="msgraph-teams", raw_text_path=vtt_path)
        ).turns
    )
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=vtt_path.read_bytes())

    result = adapt_teams_vtt(
        store,
        source_artefact_id=artefact.artefact_id,
        vtt_path=vtt_path,
        declared_attendees=("Michael Bryan", "Joanne Olsen", "Sam Lintern"),
        speaker_confirmations=(
            TeamsSpeakerConfirmation(
                raw_label="Michael BRYAN", participant_display_name="Michael Bryan"
            ),
        ),
    )

    assert len(result.turn_set.turns) == raw_cue_count
    assert len(result.label_set.spans) == raw_cue_count
    # Every raw label is either an exact declared-attendee match (Joanne
    # Olsen, Sam Lintern) or explicitly confirmed (Michael BRYAN) -- no
    # unmapped-label warning, and no cue was dropped.
    assert not result.warnings


@_requires_corpus
def test_adapt_teams_vtt_orders_turns_canonically_despite_file_order_quirk(
    tmp_path: Path,
) -> None:
    vtt_path = _CORPUS_FIXTURES / "teams-attributed-vtt" / "input.vtt"
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=vtt_path.read_bytes())

    result = adapt_teams_vtt(
        store,
        source_artefact_id=artefact.artefact_id,
        vtt_path=vtt_path,
        declared_attendees=("Michael Bryan", "Joanne Olsen", "Sam Lintern"),
    )

    keys = [(turn.start_ms, turn.end_ms) for turn in result.turn_set.turns]
    assert keys == sorted(keys)


@_requires_corpus
def test_adapt_teams_vtt_label_spans_resolve_to_the_turn_sets_own_cues(
    tmp_path: Path,
) -> None:
    vtt_path = _CORPUS_FIXTURES / "teams-attributed-vtt" / "input.vtt"
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=vtt_path.read_bytes())

    result = adapt_teams_vtt(
        store,
        source_artefact_id=artefact.artefact_id,
        vtt_path=vtt_path,
        declared_attendees=("Michael Bryan", "Joanne Olsen", "Sam Lintern"),
    )

    turn_segment_ids = {turn.source_segment_id for turn in result.turn_set.turns}
    assert all(
        span.source_segment_id in turn_segment_ids for span in result.label_set.spans
    )


# -- adapt_gemini_notes: synthetic edge cases ---------------------------------


def test_adapt_gemini_notes_raises_without_a_no_transcript_statement(
    tmp_path: Path,
) -> None:
    text = (
        '---\nAttendees:\n  - "[[Alice]]"\n---\n\n'
        "## Gemini summary\n\nsome summary text\n\n"
        "## Transcript\n\nActual transcript content here.\n"
    )
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=text.encode())

    with pytest.raises(AdapterError, match="did not state"):
        adapt_gemini_notes(
            store, source_artefact_id=artefact.artefact_id, markdown_text=text
        )


def test_adapt_gemini_notes_raises_without_frontmatter_attendees(
    tmp_path: Path,
) -> None:
    text = (
        "## Gemini summary\n\nsome summary\n\n"
        "## Transcript\n\nNo transcript was available for this meeting.\n"
    )
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=text.encode())

    with pytest.raises(AdapterError, match="Attendees"):
        adapt_gemini_notes(
            store, source_artefact_id=artefact.artefact_id, markdown_text=text
        )


def test_adapt_gemini_notes_raises_without_any_notes_sections(tmp_path: Path) -> None:
    text = (
        '---\nAttendees:\n  - "[[Alice]]"\n---\n\n'
        "## Transcript\n\nNo transcript was available for this meeting.\n"
    )
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=text.encode())

    with pytest.raises(AdapterError, match="notes export"):
        adapt_gemini_notes(
            store, source_artefact_id=artefact.artefact_id, markdown_text=text
        )


# -- adapt_gemini_notes: real fixture quirk tests -----------------------------


@_requires_corpus
def test_adapt_gemini_notes_preserves_unresolved_speaker_labels_verbatim(
    tmp_path: Path,
) -> None:
    text = (_CORPUS_FIXTURES / "gemini-notes-only" / "input.md").read_text(
        encoding="utf-8"
    )
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=text.encode("utf-8"))

    result = adapt_gemini_notes(
        store, source_artefact_id=artefact.artefact_id, markdown_text=text
    )

    all_text = " ".join(
        section.text for component in result.notes for section in component.sections
    )
    assert "(Speaker)" in all_text
    assert "(The group)" in all_text
    # Neither unresolved marker becomes a participant identity.
    display_names = {
        participant.display_names[0] for participant in result.participants.participants
    }
    assert "(Speaker)" not in display_names
    assert "(The group)" not in display_names


@_requires_corpus
def test_adapt_gemini_notes_builds_exactly_the_four_notes_kinds(tmp_path: Path) -> None:
    text = (_CORPUS_FIXTURES / "gemini-notes-only" / "input.md").read_text(
        encoding="utf-8"
    )
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=text.encode("utf-8"))

    result = adapt_gemini_notes(
        store, source_artefact_id=artefact.artefact_id, markdown_text=text
    )

    kinds = {component.notes_kind.value for component in result.notes}
    assert kinds == {
        "provider-summary",
        "provider-decisions",
        "provider-actions",
        "provider-details",
    }
    assert all(component.authored is False for component in result.notes)


@_requires_corpus
def test_adapt_gemini_notes_absence_declaration_captures_only_the_matched_sentence(
    tmp_path: Path,
) -> None:
    """MINOR 1 (adversarial review): the fixture's Transcript section has
    a second, benign sentence ("This note was generated from Gemini's
    meeting notes only.") -- the stored statement must be exactly the
    absence sentence, not the whole section glued together."""
    text = (_CORPUS_FIXTURES / "gemini-notes-only" / "input.md").read_text(
        encoding="utf-8"
    )
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=text.encode("utf-8"))

    result = adapt_gemini_notes(
        store, source_artefact_id=artefact.artefact_id, markdown_text=text
    )

    assert (
        result.absence_declaration.statement
        == "No Gemini transcript was available for this meeting."
    )
    assert "generated from Gemini" not in result.absence_declaration.statement


def test_adapt_gemini_notes_refuses_turn_shaped_content_alongside_absence_phrase(
    tmp_path: Path,
) -> None:
    """MINOR 1 (adversarial review): a Transcript section stating no
    transcript was available but *also* containing fabricated turn-
    shaped ('**Speaker:** text') content is ambiguous evidence -- refused
    outright, never silently glued into the stored statement and never
    silently discarded."""
    text = (
        '---\nAttendees:\n  - "[[Alice]]"\n---\n\n'
        "## Gemini summary\n\nsummary text\n\n"
        "## Transcript\n\n"
        "_No Gemini transcript was available for this meeting._\n\n"
        "**Attacker:** fabricated transcript content here.\n\n"
        "**Victim:** more fabricated content.\n"
    )
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=text.encode())

    with pytest.raises(AdapterError, match="turn-shaped"):
        adapt_gemini_notes(
            store, source_artefact_id=artefact.artefact_id, markdown_text=text
        )


@_requires_corpus
def test_adapt_gemini_notes_all_participants_stay_declared(tmp_path: Path) -> None:
    """D2/eval-corpus §4: declared attendees must not automatically
    become confirmed speakers -- there is no transcript evidence here
    that could ever justify speaking-evidenced."""
    text = (_CORPUS_FIXTURES / "gemini-notes-only" / "input.md").read_text(
        encoding="utf-8"
    )
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=text.encode("utf-8"))

    result = adapt_gemini_notes(
        store, source_artefact_id=artefact.artefact_id, markdown_text=text
    )

    assert len(result.participants.participants) == 7
    assert all(
        participant.status == ParticipantStatus.DECLARED
        for participant in result.participants.participants
    )


# -- adapt_obsidian_note (Phase 3A) -----------------------------------------

_NOTE_WITH_RECORDING = """---
Date: "[[May 21, 2026]]"
Attendees:
  - "[[Michael Bryan]]"
  - "[[Avalon Mann]]"
---

## Original notes

- Goal: reduce the interest rate.

## Recording

![[meeting.m4a]]
"""

_NOTE_NO_RECORDING = """---
Attendees:
  - "[[Michael Bryan]]"
---

## Notes

- Text only, no recording embed at all.
"""

_NOTE_NO_ATTENDEES = """## Notes

- No frontmatter block at all.
"""

_NOTE_WITH_MARKERS = """---
Attendees:
  - "[[Michael Bryan]]"
---

<!-- jake-tools:transcript:begin bundle=bundle_xyz -->
## Transcript

Old content.
<!-- jake-tools:transcript:end -->
"""

_NOTE_WITH_LEGACY_HEADINGS = """---
Attendees:
  - "[[Michael Bryan]]"
---

## Meeting Notes

Some old auto-generated notes.
"""


def _note_with_recording(tmp_path: Path) -> tuple[Path, Path]:
    note_path = tmp_path / "note.md"
    note_path.write_text(_NOTE_WITH_RECORDING, encoding="utf-8")
    recording_path = tmp_path / "meeting.m4a"
    recording_path.write_bytes(b"not real audio, just needs to exist")
    return note_path, recording_path


def test_adapt_obsidian_note_extracts_destination_participants_and_references(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    note_path, recording_path = _note_with_recording(tmp_path)
    artefact = registered_source_and_artefact(
        store, content=note_path.read_bytes(), kind="obsidian-note"
    )

    result = adapt_obsidian_note(
        store, note_artefact_id=artefact.artefact_id, note_path=note_path
    )

    assert isinstance(result.destination, DestinationComponent)
    assert result.destination.vault_relative_path == str(note_path)
    assert result.destination.owned_region_state == OwnedRegionState.NONE
    assert [p.display_names[0] for p in result.participants.participants] == [
        "Michael Bryan",
        "Avalon Mann",
    ]
    assert result.reference_set is not None
    assert isinstance(result.reference_set, RecordingReferenceSetComponent)
    assert len(result.reference_set.references) == 1
    assert result.reference_set.references[0].resolved_path == str(
        recording_path.resolve()
    )


def test_adapt_obsidian_note_without_recordings_has_no_reference_set(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    note_path = tmp_path / "note.md"
    note_path.write_text(_NOTE_NO_RECORDING, encoding="utf-8")
    artefact = registered_source_and_artefact(
        store, content=note_path.read_bytes(), kind="obsidian-note"
    )

    result = adapt_obsidian_note(
        store, note_artefact_id=artefact.artefact_id, note_path=note_path
    )

    assert result.reference_set is None
    assert len(result.participants.participants) == 1


def test_adapt_obsidian_note_raises_without_frontmatter_attendees(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    note_path = tmp_path / "note.md"
    note_path.write_text(_NOTE_NO_ATTENDEES, encoding="utf-8")
    artefact = registered_source_and_artefact(
        store, content=note_path.read_bytes(), kind="obsidian-note"
    )

    with pytest.raises(AdapterError, match="Attendees"):
        adapt_obsidian_note(
            store, note_artefact_id=artefact.artefact_id, note_path=note_path
        )


def test_adapt_obsidian_note_raises_when_the_recording_link_does_not_resolve(
    tmp_path: Path,
) -> None:
    from jake_tools.transcripts.obsidian import RecordingResolutionError

    store = _store(tmp_path)
    note_path = tmp_path / "note.md"
    note_path.write_text(_NOTE_WITH_RECORDING, encoding="utf-8")
    # meeting.m4a deliberately not created next to the note.
    artefact = registered_source_and_artefact(
        store, content=note_path.read_bytes(), kind="obsidian-note"
    )

    with pytest.raises(RecordingResolutionError):
        adapt_obsidian_note(
            store, note_artefact_id=artefact.artefact_id, note_path=note_path
        )


def test_adapt_obsidian_note_is_idempotent_on_retry(tmp_path: Path) -> None:
    """Crash-window discipline (adapters.py's own module docstring): a
    retried adapter call over the *same* note content leaves at most
    harmless, unreferenced extra records behind, never corruption --
    the destination and reference-set components carry no per-item
    minted IDs, so those two dedup byte-for-byte across retries; the
    participant set does NOT (by the same documented design every
    existing adapter already follows: a fresh `participant_id` is minted
    per retry, so its body -- and therefore its component -- differs each
    time, which is harmless orphaned-record churn, never a correctness
    bug)."""
    store = _store(tmp_path)
    note_path, _recording_path = _note_with_recording(tmp_path)
    artefact = registered_source_and_artefact(
        store, content=note_path.read_bytes(), kind="obsidian-note"
    )

    first = adapt_obsidian_note(
        store, note_artefact_id=artefact.artefact_id, note_path=note_path
    )
    second = adapt_obsidian_note(
        store, note_artefact_id=artefact.artefact_id, note_path=note_path
    )

    assert first.destination.component_id == second.destination.component_id
    assert first.reference_set is not None and second.reference_set is not None
    assert first.reference_set.component_id == second.reference_set.component_id
    assert first.participants.component_id != second.participants.component_id
    assert [p.display_names for p in first.participants.participants] == [
        p.display_names for p in second.participants.participants
    ]


@pytest.mark.parametrize(
    ("note_text", "expected_state"),
    [
        (_NOTE_WITH_MARKERS, OwnedRegionState.MARKERS_PRESENT),
        (_NOTE_WITH_LEGACY_HEADINGS, OwnedRegionState.LEGACY_HEADINGS),
        (_NOTE_NO_RECORDING, OwnedRegionState.NONE),
    ],
)
def test_adapt_obsidian_note_detects_owned_region_state(
    tmp_path: Path, note_text: str, expected_state: OwnedRegionState
) -> None:
    """M13: detection only -- markers win outright; legacy generated
    headings (merge.py:GENERATED_HEADINGS) are the migration signal;
    neither means a first-ever write. Never writes anything itself."""
    store = _store(tmp_path)
    note_path = tmp_path / "note.md"
    note_path.write_text(note_text, encoding="utf-8")
    artefact = registered_source_and_artefact(
        store, content=note_path.read_bytes(), kind="obsidian-note"
    )
    before = note_path.read_text(encoding="utf-8")

    result = adapt_obsidian_note(
        store, note_artefact_id=artefact.artefact_id, note_path=note_path
    )

    assert result.destination.owned_region_state == expected_state
    assert (
        note_path.read_text(encoding="utf-8") == before
    )  # detection only, never writes


@_requires_corpus
def test_adapt_obsidian_note_over_the_real_single_recording_fixture(
    tmp_path: Path,
) -> None:
    """The local-single-speaker-correction fixture's own shape: an embed
    that must resolve, two declared attendees. Copies into a tmp working
    dir first -- the vault (and this fixture) are read-only; the audio
    is placed adjacent to the copied note so obsidian.py's own resolution
    (same-directory candidate) finds it, exactly as a real vault's
    Attachments-folder convention would."""
    fixture = _CORPUS_FIXTURES / "local-single-speaker-correction"
    note_path = tmp_path / "input-note.md"
    note_path.write_bytes((fixture / "input-note.md").read_bytes())
    recording_path = tmp_path / "meeting.m4a"
    recording_path.write_bytes((fixture / "audio" / "meeting.m4a").read_bytes())

    store = _store(tmp_path)
    artefact = registered_source_and_artefact(
        store, content=note_path.read_bytes(), kind="obsidian-note"
    )

    result = adapt_obsidian_note(
        store, note_artefact_id=artefact.artefact_id, note_path=note_path
    )

    assert [p.display_names[0] for p in result.participants.participants] == [
        "Michael Bryan",
        "Avalon Mann",
    ]
    assert result.reference_set is not None
    assert result.reference_set.references[0].resolved_path == str(
        recording_path.resolve()
    )


# -- probe_audio_metadata / adapt_local_media (Phase 3A) --------------------


def _fake_ffprobe_json(
    command: list[str], *, payload: str, returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        command, returncode=returncode, stdout=payload, stderr=""
    )


def test_probe_audio_metadata_parses_a_fake_ffprobe_stream(tmp_path: Path) -> None:
    payload = (
        '{"streams": [{"codec_type": "audio", "codec_name": "opus", '
        '"sample_rate": "48000", "channels": 1, "duration": "12.5"}], '
        '"format": {"duration": "12.5"}}'
    )
    audio_path = tmp_path / "clip.m4a"
    audio_path.write_bytes(b"fake bytes")

    result = probe_audio_metadata(
        audio_path, run_command=lambda cmd: _fake_ffprobe_json(cmd, payload=payload)
    )

    assert result.duration_ms == 12_500
    assert result.codec == "opus"
    assert result.sample_rate_hz == 48000
    assert result.channels == 1


def test_probe_audio_metadata_raises_when_ffprobe_is_missing(tmp_path: Path) -> None:
    def _missing(command: list[str]) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("no ffprobe")

    with pytest.raises(FfprobeNotFoundError):
        probe_audio_metadata(tmp_path / "clip.m4a", run_command=_missing)


def test_probe_audio_metadata_raises_when_ffprobe_exits_non_zero(
    tmp_path: Path,
) -> None:
    with pytest.raises(FfprobeFailedError):
        probe_audio_metadata(
            tmp_path / "clip.m4a",
            run_command=lambda cmd: _fake_ffprobe_json(cmd, payload="{}", returncode=1),
        )


def test_probe_audio_metadata_raises_with_no_audio_stream(tmp_path: Path) -> None:
    payload = '{"streams": [{"codec_type": "video"}], "format": {"duration": "1.0"}}'
    with pytest.raises(FfprobeFailedError, match="no audio stream"):
        probe_audio_metadata(
            tmp_path / "clip.m4a",
            run_command=lambda cmd: _fake_ffprobe_json(cmd, payload=payload),
        )


def test_probe_audio_metadata_raises_on_non_positive_duration(tmp_path: Path) -> None:
    payload = (
        '{"streams": [{"codec_type": "audio", "codec_name": "opus", '
        '"sample_rate": "48000", "channels": 1}], "format": {"duration": "0"}}'
    )
    with pytest.raises(FfprobeFailedError, match="non-positive"):
        probe_audio_metadata(
            tmp_path / "clip.m4a",
            run_command=lambda cmd: _fake_ffprobe_json(cmd, payload=payload),
        )


_requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed (required by AGENTS.md's external tools)",
)


@_requires_ffmpeg
def test_probe_audio_metadata_over_a_real_synthesized_wav(tmp_path: Path) -> None:
    """A real subprocess, real ffprobe, real (synthesized, silent)
    1-second audio file -- no fixture needed, no fake runner."""
    audio_path = tmp_path / "silence.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=mono",
            "-t",
            "1",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
    )

    result = probe_audio_metadata(audio_path)

    assert 900 <= result.duration_ms <= 1100
    assert result.sample_rate_hz == 48000
    assert result.channels == 1


def test_adapt_local_media_canonicalises_the_media_path(tmp_path: Path) -> None:
    payload = (
        '{"streams": [{"codec_type": "audio", "codec_name": "opus", '
        '"sample_rate": "48000", "channels": 1, "duration": "50.04"}], '
        '"format": {"duration": "50.04"}}'
    )
    audio_path = tmp_path / "meeting.m4a"
    audio_path.write_bytes(b"fake bytes")
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(
        store, content=audio_path.read_bytes(), kind="audio"
    )

    result = adapt_local_media(
        store,
        source_artefact_id=artefact.artefact_id,
        media_path=audio_path,
        run_command=lambda cmd: _fake_ffprobe_json(cmd, payload=payload),
    )

    assert isinstance(result, MediaRecordingComponent)
    assert result.media_path == str(audio_path.resolve())
    assert result.duration_ms == 50_040
    assert result.codec == "opus"


def test_adapt_local_media_is_idempotent_on_retry(tmp_path: Path) -> None:
    payload = (
        '{"streams": [{"codec_type": "audio", "codec_name": "opus", '
        '"sample_rate": "48000", "channels": 1, "duration": "10.0"}], '
        '"format": {"duration": "10.0"}}'
    )
    audio_path = tmp_path / "meeting.m4a"
    audio_path.write_bytes(b"fake bytes")
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(
        store, content=audio_path.read_bytes(), kind="audio"
    )

    first = adapt_local_media(
        store,
        source_artefact_id=artefact.artefact_id,
        media_path=audio_path,
        run_command=lambda cmd: _fake_ffprobe_json(cmd, payload=payload),
    )
    second = adapt_local_media(
        store,
        source_artefact_id=artefact.artefact_id,
        media_path=audio_path,
        run_command=lambda cmd: _fake_ffprobe_json(cmd, payload=payload),
    )

    assert first.component_id == second.component_id


def test_operator_declared_participants_stand_in_for_missing_frontmatter(
    tmp_path: Path,
) -> None:
    """M19's ``operator`` declaration source. Plenty of real notes name
    their people in the body as wikilinks, and picking those out is exactly
    the inference M19 forbids -- ``[[On-road Driving]]`` and
    ``[[Michael Bryan]]`` are indistinguishable to a parser. Naming them on
    the command line is an assertion, and is recorded as one."""
    note = tmp_path / "ops-log.md"
    note.write_text(
        "---\ntags:\n  - ops-log\n---\n\n- Chatting with [[Steven Crawford]]\n",
        encoding="utf-8",
    )
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=note.read_bytes())

    adaptation = adapt_obsidian_note(
        store,
        note_artefact_id=artefact.artefact_id,
        note_path=note,
        operator_participants=("Steven Crawford", "Matt Lavender"),
    )

    participants = adaptation.participants.participants
    assert {p.display_names[0] for p in participants} == {
        "Steven Crawford",
        "Matt Lavender",
    }
    assert all(
        p.declaration_source == ParticipantDeclarationSource.OPERATOR
        for p in participants
    )


def test_a_note_with_neither_attendees_nor_declared_participants_is_refused(
    tmp_path: Path,
) -> None:
    note = tmp_path / "ops-log.md"
    note.write_text("---\ntags:\n  - ops-log\n---\n\n- some notes\n", encoding="utf-8")
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=note.read_bytes())

    with pytest.raises(AdapterError, match="--participant"):
        adapt_obsidian_note(
            store, note_artefact_id=artefact.artefact_id, note_path=note
        )


def test_frontmatter_and_operator_participants_merge_keeping_provenance(
    tmp_path: Path,
) -> None:
    """Both routes are legal at once, deduplicated by display name -- a
    frontmatter attendee stays declared by frontmatter even when the
    operator repeats them."""
    note = tmp_path / "meeting.md"
    note.write_text(
        '---\nAttendees:\n  - "[[Michael Bryan]]"\n---\n\n- notes\n',
        encoding="utf-8",
    )
    store = _store(tmp_path)
    artefact = registered_source_and_artefact(store, content=note.read_bytes())

    adaptation = adapt_obsidian_note(
        store,
        note_artefact_id=artefact.artefact_id,
        note_path=note,
        operator_participants=("Michael Bryan", "Steven Crawford"),
    )

    by_name = {
        p.display_names[0]: p.declaration_source
        for p in adaptation.participants.participants
    }
    assert by_name == {
        "Michael Bryan": ParticipantDeclarationSource.NOTE_FRONTMATTER,
        "Steven Crawford": ParticipantDeclarationSource.OPERATOR,
    }
