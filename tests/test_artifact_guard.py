"""The pure authority policy of the artifact sessions (``artifacts.guard``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jake_tools.artifacts.guard import ArtifactGuard

FILE = Path("/work/report.html")
URL = "https://claude.ai/artifact/V4iHumJLsDrW9EBEtZfqZj"
EXPECTED = "1794664145-2330"
GOOD_READ = (
    "[Artifact e3448f84-1ca6-4757-9dd4-3bb5616d043a (version 1794664145-2330) \u2014 "
    "owned by you, private (x); raw HTML follows]\n..."
)


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


@pytest.mark.parametrize(
    "extra",
    [
        {"__artifactReadTargetPin": "x"},
        {"__consent": True},
        {"displayTarget": "x"},
        {"updatedInput": {}},
    ],
)
def test_unexpected_keys_are_denied_including_private_looking_ones(
    extra: dict[str, object],
) -> None:
    """The guard judges the raw model-supplied input; the runtime may enrich
    input later, but a model that supplies ``__*`` metadata is denied."""
    publish = publish_guard()
    assert publish("Artifact", {"action": "publish", "file_path": str(FILE), **extra})
    read = ArtifactGuard.for_read(URL)
    assert read("Artifact", {"action": "read", "url": URL, **extra})
    assert publish.publishes_allowed == 0 and read.reads_allowed == 0


def update_guard() -> ArtifactGuard:
    return ArtifactGuard.for_update(FILE, URL, EXPECTED)


def test_update_allows_one_read_of_the_target_the_file_and_one_publish_onto_it() -> (
    None
):
    guard = update_guard()
    assert guard("Artifact", {"action": "read", "url": URL}) is None
    guard.observe_read({"action": "read", "url": URL}, GOOD_READ, False)
    assert guard("Read", {"file_path": str(FILE)}) is None
    call = {"action": "publish", "file_path": str(FILE), "url": URL}
    assert guard("Artifact", call) is None
    assert guard("Artifact", call) is not None  # never twice
    assert guard.publishes_allowed == 1


@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        ("Artifact", {"action": "publish", "file_path": str(FILE)}),  # a NEW one
        (
            "Artifact",
            {
                "action": "publish",
                "file_path": str(FILE),
                "url": "https://claude.ai/artifact/Qm3Kz7Tn5Rv9Wx2YbDfGhj",
            },
        ),
        (
            "Artifact",
            {"action": "publish", "file_path": "/work/other.html", "url": URL},
        ),
        (
            "Artifact",
            {"action": "publish", "file_path": str(FILE), "url": URL, "edits": []},
        ),
        (
            "Artifact",
            {"action": "publish", "file_path": str(FILE), "url": URL, "__x": 1},
        ),
        (
            "Artifact",
            {
                "action": "read",
                "url": "https://claude.ai/artifact/Qm3Kz7Tn5Rv9Wx2YbDfGhj",
            },
        ),
        ("Artifact", {"action": "read", "url": URL, "extra": 1}),
        ("Artifact", {"action": "share", "url": URL}),
        ("Artifact", {"action": "delete", "url": URL}),
        ("Artifact", {"action": "list"}),
        ("Read", {"file_path": "/etc/passwd"}),
        ("Bash", {"command": "ls"}),
    ],
)
def test_update_denies_everything_else(
    tool: str, tool_input: dict[str, object]
) -> None:
    guard = update_guard()
    assert guard(tool, tool_input) is not None
    assert guard.publishes_allowed == 0 and guard.reads_allowed == 0


def test_a_new_publish_and_a_read_session_still_cannot_update_or_read_via_publish() -> (
    None
):
    # The no-url rule of a NEW publish is unchanged, and a publish session has
    # no `read` action at all.
    publish = publish_guard()
    assert publish(
        "Artifact", {"action": "publish", "file_path": str(FILE), "url": URL}
    )
    assert publish("Artifact", {"action": "read", "url": URL})
    assert ArtifactGuard.for_read(URL)(
        "Artifact", {"action": "publish", "file_path": str(FILE), "url": URL}
    )


@pytest.mark.parametrize(
    ("result", "is_error", "fragment"),
    [
        (None, False, "was observed"),  # the read result never arrived
        ("not a header", False, "no recognisable header"),
        (GOOD_READ, True, "no recognisable header"),  # an error result
        (
            GOOD_READ.replace("1794664145-2330", "1794999999-live"),
            False,
            "not the expected",
        ),
        (
            GOOD_READ.replace("owned by you, private", "owned by you, shared"),
            False,
            "not a private",
        ),
        (
            GOOD_READ.replace("owned by you, private", "owned by someone else"),
            False,
            "not a private",
        ),
    ],
)
def test_an_update_publish_is_denied_until_the_target_read_is_observed_as_expected(
    result: str | None, is_error: bool, fragment: str
) -> None:
    guard = update_guard()
    guard("Artifact", {"action": "read", "url": URL})
    if result is not None:
        guard.observe_read({"action": "read", "url": URL}, result, is_error)
    reason = guard(
        "Artifact", {"action": "publish", "file_path": str(FILE), "url": URL}
    )
    assert reason is not None and fragment in reason
    assert guard.publishes_allowed == 0


def test_a_read_of_another_artifact_never_satisfies_the_precondition() -> None:
    guard = update_guard()
    guard.observe_read(
        {"action": "read", "url": "https://claude.ai/artifact/Qm3Kz7Tn5Rv9Wx2YbDfGhj"},
        GOOD_READ,
        False,
    )
    assert guard.observed_ok is False


def test_the_precondition_is_not_needed_for_new_publishes() -> None:
    assert (
        publish_guard()("Artifact", {"action": "publish", "file_path": str(FILE)})
        is None
    )
