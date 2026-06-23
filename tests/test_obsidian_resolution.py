from datetime import UTC, datetime, timedelta

from jake_tools.transcripts.obsidian import (
    RecordingResolutionError,
    extract_recording_links,
    load_source_note,
    resolve_recording_path,
)


def test_extract_recording_links_filters_audio_only() -> None:
    body = "\n".join(
        [
            "![[meeting-a.m4a]]",
            "![[image.png]]",
            "![alt](meeting-b.mp3)",
        ]
    )

    assert extract_recording_links(body) == ["meeting-a.m4a", "meeting-b.mp3"]


def test_resolve_recording_path_prefers_note_relative_file(tmp_path) -> None:
    note = tmp_path / "Meeting.md"
    recording = tmp_path / "meeting.m4a"
    note.write_text("![[meeting.m4a]]", encoding="utf-8")
    recording.write_text("audio", encoding="utf-8")

    assert resolve_recording_path(note, "meeting.m4a") == recording.resolve()


def test_load_source_note_resolves_attachments_and_sorts_by_creation_time(
    tmp_path,
) -> None:
    vault = tmp_path / "Vault"
    attachments = vault / "Attachments"
    notes = vault / "Notes"
    attachments.mkdir(parents=True)
    notes.mkdir(parents=True)

    first = attachments / "a.m4a"
    second = attachments / "b.m4a"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")

    older = datetime.now(UTC) - timedelta(minutes=10)
    newer = datetime.now(UTC)
    first_ts = older.timestamp()
    second_ts = newer.timestamp()
    import os

    os.utime(first, (first_ts, first_ts))
    os.utime(second, (second_ts, second_ts))

    note = notes / "Meeting.md"
    note.write_text("![[b.m4a]]\n![[a.m4a]]", encoding="utf-8")

    source = load_source_note(note)

    assert source.title == "Meeting"
    assert source.attendees == []
    assert [recording.resolved_path.name for recording in source.recordings] == [
        "a.m4a",
        "b.m4a",
    ]


def test_load_source_note_parses_attendees_from_frontmatter(tmp_path) -> None:
    vault = tmp_path / "Vault"
    attachments = vault / "Attachments"
    notes = vault / "Notes"
    attachments.mkdir(parents=True)
    notes.mkdir(parents=True)

    recording = attachments / "meeting.m4a"
    recording.write_text("audio", encoding="utf-8")

    note = notes / "Call.md"
    note.write_text(
        "\n".join(
            [
                "---",
                'Date: "[[June 17, 2026]]"',
                "Attendees:",
                '  - "[[Michael Bryan]]"',
                '  - "[[Gabbey Parker|Gabbey]]"',
                "---",
                "",
                "![[meeting.m4a]]",
            ]
        ),
        encoding="utf-8",
    )

    source = load_source_note(note)

    assert source.title == "Call"
    assert source.attendees == ["Michael Bryan", "Gabbey"]


def test_load_source_note_raises_clear_error_on_missing_recording(tmp_path) -> None:
    note = tmp_path / "Meeting.md"
    note.write_text("![[missing.m4a]]", encoding="utf-8")

    try:
        load_source_note(note)
    except RecordingResolutionError as exc:
        assert "missing.m4a" in str(exc)
        assert str(note) in str(exc)
    else:
        raise AssertionError("expected RecordingResolutionError")
