from __future__ import annotations

from pathlib import Path

from jake_tools.transcription.note import parse_note, render_body

FIXTURES = Path(__file__).parent / "fixtures"
MEETING_NOTE = FIXTURES / "meeting_note.md"
OPS_LOG = FIXTURES / "ops_log.md"


def test_meeting_fixture_detects_meeting_context_from_note_meeting_tag() -> None:
    note = parse_note(MEETING_NOTE)

    assert note.context == "meeting"


def test_ops_log_fixture_detects_ops_log_context_from_ops_log_tag() -> None:
    note = parse_note(OPS_LOG)

    assert note.context == "ops-log"


def test_ops_log_tag_wins_even_when_a_meeting_tag_is_also_present(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ambiguous.md"
    path.write_text("---\ntags:\n  - note/meeting\n  - ops-log\n---\n\nBody text.\n")

    note = parse_note(path)

    assert note.context == "ops-log"


def test_tagless_note_under_dum_c_path_falls_back_to_ops_log(tmp_path: Path) -> None:
    dum_c = tmp_path / "DUM-C"
    dum_c.mkdir()
    path = dum_c / "untagged.md"
    path.write_text("No frontmatter here, just body text.\n")

    note = parse_note(path)

    assert note.context == "ops-log"
    assert note.frontmatter == {}


def test_tagless_note_outside_dum_c_path_falls_back_to_meeting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "untagged.md"
    path.write_text("No frontmatter here, just body text.\n")

    note = parse_note(path)

    assert note.context == "meeting"


def test_attendees_parse_with_wikilinks_stripped() -> None:
    note = parse_note(MEETING_NOTE)

    assert note.attendees == ["Ada Lovelace", "Grace Hopper"]


def test_missing_attendees_key_yields_empty_list() -> None:
    note = parse_note(OPS_LOG)

    assert note.attendees == []


def test_diarisation_hints_are_extracted_from_the_meeting_fixture() -> None:
    note = parse_note(MEETING_NOTE)

    assert note.diarisation_hints == [
        "Ada mostly asked questions",
        "Grace did most of the talking",
    ]


def test_diarisation_hints_are_empty_when_absent() -> None:
    note = parse_note(OPS_LOG)

    assert note.diarisation_hints == []


def test_embeds_are_extracted_in_document_order() -> None:
    note = parse_note(MEETING_NOTE)

    assert note.embeds == [
        "Recording 20260803090000.m4a",
        "Recording 20260803094500.m4a",
    ]


def test_single_embed_is_extracted_from_ops_log_preamble() -> None:
    note = parse_note(OPS_LOG)

    assert note.embeds == ["Recording 20260805183000.m4a"]


def test_sections_reconstruct_the_meeting_body_byte_for_byte() -> None:
    text = MEETING_NOTE.read_text()
    _frontmatter_end = text.index("---\n", 4) + len("---\n")
    original_body = text[_frontmatter_end:]

    note = parse_note(MEETING_NOTE)

    assert render_body(note.sections) == original_body


def test_sections_reconstruct_the_ops_log_body_byte_for_byte() -> None:
    text = OPS_LOG.read_text()
    _frontmatter_end = text.index("---\n", 4) + len("---\n")
    original_body = text[_frontmatter_end:]

    note = parse_note(OPS_LOG)

    assert render_body(note.sections) == original_body


def test_sections_capture_heading_and_level() -> None:
    note = parse_note(MEETING_NOTE)

    headings = [(section.heading, section.level) for section in note.sections]

    assert (None, 0) in headings
    assert ("Meeting Prep", 2) in headings
    assert ("Discussion Notes", 2) in headings
    assert ("Chapters", 2) in headings
    assert ("00:00 — Opening", 3) in headings
    assert ("05:00 — Wrap-up", 3) in headings


def test_note_without_frontmatter_parses_with_empty_frontmatter(
    tmp_path: Path,
) -> None:
    path = tmp_path / "no_frontmatter.md"
    path.write_text("## Just a heading\n\nSome body text.\n")

    note = parse_note(path)

    assert note.frontmatter == {}
    assert render_body(note.sections) == path.read_text()
