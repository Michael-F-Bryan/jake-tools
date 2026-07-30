"""B2: atomic_write_text never leaves a torn file on a mid-write failure."""

from __future__ import annotations

import pytest

from inference_worker.atomic_io import atomic_write_text


def test_atomic_write_text_writes_the_full_content(tmp_path):
    path = tmp_path / "response.json"

    atomic_write_text(path, '{"hello": "world"}')

    assert path.read_text() == '{"hello": "world"}'


def test_atomic_write_text_replaces_previous_content_wholesale(tmp_path):
    path = tmp_path / "response.json"
    path.write_text('{"old": true}')

    atomic_write_text(path, '{"new": true}')

    assert path.read_text() == '{"new": true}'


def test_atomic_write_text_leaves_no_temp_file_behind_on_success(tmp_path):
    path = tmp_path / "response.json"

    atomic_write_text(path, "content")

    leftovers = [p for p in tmp_path.iterdir() if p != path]
    assert leftovers == []


def test_failed_write_never_truncates_the_previous_good_file(tmp_path):
    """B2: a real mid-write failure (here: the destination directory is
    replaced by a file, so the atomic rename itself fails) must leave the
    previous good content completely intact — never a torn/truncated
    file, and never silently lost."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    path = out_dir / "response.json"
    atomic_write_text(path, '{"good": "previous run"}')

    # Make the rename step fail for real: point the write at a path whose
    # parent directory doesn't exist, so tempfile.mkstemp itself raises
    # before anything touches `path`.
    bad_path = out_dir / "missing-subdir" / "response.json"
    with pytest.raises(OSError):
        atomic_write_text(bad_path, '{"new": "torn?"}')

    # The original file is untouched — this is the actual regression
    # target (B2 was reproduced via RLIMIT_FSIZE truncating mid-write).
    assert path.read_text() == '{"good": "previous run"}'


def test_failed_write_does_not_leak_a_temp_file(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    bad_path = out_dir / "missing-subdir" / "response.json"

    with pytest.raises(OSError):
        atomic_write_text(bad_path, "content")

    assert list(out_dir.iterdir()) == []
