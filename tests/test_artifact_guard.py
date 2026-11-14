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


def test_only_the_exact_canonical_string_names_the_file() -> None:
    guard = publish_guard()
    assert guard("Read", {"file_path": "/work/report.html"}) is None
    # Lexically equal, but the tool resolves symlinks first, so these may name
    # another file: no normalisation.
    for lookalike in (
        "/work/./report.html",
        "/work//report.html",
        "/work/link/../report.html",
        "/work/report.html/",
        "/work/report.html.bak",
    ):
        assert guard("Read", {"file_path": lookalike}) is not None, lookalike
        assert (
            ArtifactGuard.for_publish(FILE)(
                "Artifact", {"action": "publish", "file_path": lookalike}
            )
            is not None
        )


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


def test_two_layers_asking_about_one_call_get_one_counted_answer() -> None:
    guard = publish_guard()
    call = {"action": "publish", "file_path": str(FILE)}
    assert guard("Artifact", call, "tu-1") is None  # the PreToolUse hook
    assert guard("Artifact", call, "tu-1") is None  # can_use_tool: same call
    assert guard.publishes_allowed == 1 and len(guard.consulted) == 1
    # A different call is a different decision, and the second publish is refused.
    assert guard("Artifact", call, "tu-2") is not None
    # A denial is also stable for the call.
    assert guard("Artifact", call, "tu-2") is not None
    assert len(guard.denied) == 1


def test_an_approval_is_never_reused_for_a_changed_request_with_the_same_id() -> None:
    """Another hook may rewrite a call's input between the two layers. The
    approval of the safe form must not carry over to the changed form."""
    safe = {"action": "publish", "file_path": str(FILE)}
    guard = publish_guard()
    assert guard("Artifact", safe, "tu-1") is None  # PreToolUse: the safe request
    # can_use_tool then sees the SAME id with a rewritten input:
    for changed in (
        {"action": "share", "url": URL},
        {"action": "publish", "file_path": str(FILE), "url": URL},  # an update
        {"action": "publish", "file_path": "/etc/other.html"},
    ):
        reason = guard("Artifact", changed, "tu-1")
        assert reason is not None and "changed between enforcement layers" in reason
    # ...and the id stays denied, even for the original request afterwards.
    assert guard("Artifact", safe, "tu-1") is not None
    assert guard("Artifact", {"action": "share", "url": URL}, "tu-1") is not None
    # A different tool under the same id is a change too.
    assert guard("Read", {"file_path": str(FILE)}, "tu-1") is not None


def test_key_order_does_not_make_an_identical_request_look_changed() -> None:
    guard = publish_guard()
    assert guard("Artifact", {"action": "publish", "file_path": str(FILE)}, "a") is None
    assert guard("Artifact", {"file_path": str(FILE), "action": "publish"}, "a") is None
    assert guard.publishes_allowed == 1
