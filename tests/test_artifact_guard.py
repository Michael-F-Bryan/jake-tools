"""The pure authority policy of the artifact sessions (``artifacts.guard``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jake_tools.artifacts.guard import ArtifactGuard

FILE = Path("/work/report.html")
URL = "https://claude.ai/artifact/V4iHumJLsDrW9EBEtZfqZj"


def publish_guard() -> ArtifactGuard:
    return ArtifactGuard.for_publish(FILE)


def test_publish_allows_reading_the_file_and_one_new_publish() -> None:
    guard = publish_guard()
    assert guard("Read", {"file_path": str(FILE)}) is None
    assert guard("Artifact", {"action": "publish", "file_path": str(FILE)}) is None
    # ...and an omitted action means publish, with its optional text fields.
    assert (
        publish_guard()("Artifact", {"file_path": str(FILE), "description": "d"})
        is None
    )
    # ...but a second publish never.
    assert "only one publish" in (
        guard("Artifact", {"action": "publish", "file_path": str(FILE)}) or ""
    )


@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        ("Artifact", {"action": "list"}),
        ("Artifact", {"action": "share", "url": URL}),
        ("Artifact", {"action": "delete", "url": URL}),
        ("Artifact", {"action": "read", "url": URL}),
        ("Artifact", {"action": "publish", "file_path": str(FILE), "url": URL}),
        ("Artifact", {"action": "publish", "file_path": str(FILE), "edits": []}),
        ("Artifact", {"action": "publish", "file_path": "/work/other.html"}),
        ("Artifact", {"action": "publish", "file_path": "report.html"}),
        ("Artifact", {"action": "publish", "file_path": "/work/../etc/x.html"}),
        ("Artifact", {"action": "publish"}),
        ("Artifact", {"action": 1}),
        ("Read", {"file_path": "/etc/passwd"}),
        ("Read", {"file_path": str(FILE), "offset": 1}),
        ("Read", {}),
        ("Bash", {"command": "ls"}),
        ("Write", {"file_path": str(FILE), "content": "x"}),
        ("mcp__claude_ai_Gmail__send_message", {}),
        ("WebFetch", {"url": URL}),
    ],
)
def test_publish_denies_everything_else(
    tool: str, tool_input: dict[str, object]
) -> None:
    guard = publish_guard()
    assert guard(tool, tool_input) is not None
    assert guard.denied and guard.publishes_allowed == 0


def test_a_normalised_path_is_the_same_file_but_a_sibling_is_not() -> None:
    guard = publish_guard()
    assert guard("Read", {"file_path": "/work/./report.html"}) is None
    assert guard("Read", {"file_path": "/work/report.html.bak"}) is not None


def test_the_file_may_be_read_only_a_few_times() -> None:
    guard = publish_guard()
    allowed = [guard("Read", {"file_path": str(FILE)}) for _ in range(5)]
    assert allowed.count(None) == 3


def test_read_allows_exactly_one_read_of_the_one_artifact() -> None:
    guard = ArtifactGuard.for_read(URL)
    assert guard("Artifact", {"action": "read", "url": URL}) is None
    assert guard("Artifact", {"action": "read", "url": URL}) is not None  # twice: no


def test_read_accepts_the_same_artifact_under_the_code_path() -> None:
    guard = ArtifactGuard.for_read(URL)
    code = "https://claude.ai/code/artifact/V4iHumJLsDrW9EBEtZfqZj"
    assert guard("Artifact", {"action": "read", "url": code}) is None


@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        ("Artifact", {"action": "list"}),
        ("Artifact", {"action": "read", "url": "https://claude.ai/artifact/Other"}),
        ("Artifact", {"action": "read", "url": URL, "path": "/tmp/x"}),
        ("Artifact", {"action": "read"}),
        ("Artifact", {"action": "publish", "file_path": "/work/report.html"}),
        ("Artifact", {"action": "publish", "file_path": "/work/x.html", "url": URL}),
        ("Artifact", {"action": "share", "url": URL}),
        ("Artifact", {"action": "delete", "url": URL}),
        ("Artifact", {"action": "edit_version", "url": URL}),
        ("Read", {"file_path": str(FILE)}),
        ("Bash", {"command": "echo"}),
    ],
)
def test_read_never_gets_write_authority(
    tool: str, tool_input: dict[str, object]
) -> None:
    guard = ArtifactGuard.for_read(URL)
    assert guard(tool, tool_input) is not None
    assert guard.reads_allowed == 0 and guard.publishes_allowed == 0


def test_a_guard_with_no_grant_denies_everything() -> None:
    assert ArtifactGuard()("Read", {"file_path": "/x"}) is not None


def test_every_call_is_recorded_even_when_denied() -> None:
    guard = ArtifactGuard.for_read(URL)
    guard("Artifact", {"action": "list"})
    guard("Artifact", {"action": "read", "url": URL})
    assert [c[0] for c in guard.consulted] == ["Artifact", "Artifact"]
    assert len(guard.denied) == 1
