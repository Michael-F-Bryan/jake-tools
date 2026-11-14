"""Actionable classification of failed provider calls (jake-tools#14).

The caller-facing tests run the real stdio server, worker and Claude Agent SDK
against ``tests/fake_claude.py``, which replays the error envelopes reported in
the issue byte-for-byte in shape (``subtype`` ``success`` with ``is_error``
true). They prove the mechanics and what a caller can read back; the fake is
not a provider, so nothing here is evidence about what the real provider sends
beyond those two reported envelopes. The classifier tests below them pin the
reset-time rules.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import (
    ClaudeSDKError,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
    ResultError,
)
from mcp import ClientSession
from test_claude_tasks import (
    ServerFactory,
    call,
    detail,
    poll_until,
    runs_dir,
    start,
    terminal,
)

from jake_tools.claude import DEFAULT_MAX_BUFFER_SIZE
from jake_tools.claude_runs.failures import (
    classify_exception,
    classify_result,
    classify_stream_end,
    parse_reset,
)
from jake_tools.claude_runs.models import RunResult, RunState


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


async def fail_with(
    session: ClientSession, workdir: Path, brief: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one task to its end: the final poll and the detail view."""
    started = await start(session, workdir, brief)
    final, _, _ = await poll_until(session, started["task_id"], terminal)
    return final, await detail(session, started["task_id"])


# --- the reported envelopes, through the worker --------------------------------


async def test_a_reported_usage_limit_keeps_its_words_and_claims_no_date(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        final, info = await fail_with(session, workdir, "[[scenario:reported_failure]]")
        listed = await call(session, "claude_list")

    # The headline is unchanged: this is still the worker's sdk_error.
    assert final["status"] == "failed"
    assert final["termination_reason"] == "sdk_error"
    words = "You've hit your 5-hour limit · resets 5:30am (UTC)"
    for result in (final["result"], info["result"]):
        failure = result["failure"]
        assert failure["kind"] == "usage_limit"
        # A bare time of day does not say which day: words only, no instant.
        assert failure["reset_text"] == "5:30am (UTC)"
        assert "resets_at" not in failure
        assert failure["advice"]["next_step"] == "wait_for_reset"
        assert failure["advice"]["basis"] == "inferred"
        assert "does not name one date" in failure["advice"]["text"]
        # The original evidence is preserved, not replaced by the verdict.
        evidence = failure["evidence"]
        assert evidence["source"] == "result"
        assert evidence["subtype"] == "success"
        assert evidence["is_error"] is True
        assert evidence["stop_reason"] == "stop_sequence"
        assert evidence["text"] == words
        assert evidence["matched"] == "You've hit your 5-hour limit"
        assert result["final_text"] == words
        assert "subtype='success'" in result["error"]
        assert "errors='success'" in result["error"]
    # What a supervisor sees without opening the detail view.
    assert "5-hour limit" in info["status_message"]
    assert "errors='success'" not in info["status_message"]
    summary = next(t for t in listed["tasks"] if t["task_id"] == final["task_id"])
    assert summary["failure_kind"] == "usage_limit"
    assert summary["status_message"] == info["status_message"]


async def test_a_limit_with_a_full_date_and_zone_carries_the_reset_instant(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        before = datetime.now(UTC)
        _, info = await fail_with(
            session, workdir, "[[scenario:reported_failure:limit_full_date]]"
        )

    failure = info["result"]["failure"]
    resets_at = datetime.fromisoformat(failure["resets_at"])
    assert (resets_at.hour, resets_at.minute) == (5, 30)
    assert before + timedelta(days=1) < resets_at < before + timedelta(days=3)
    assert failure["reset_text"].endswith("5:30am (UTC)")


async def test_a_reported_authentication_failure_does_not_claim_a_cause(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        final, info = await fail_with(
            session, workdir, "[[scenario:reported_failure:auth]]"
        )

    assert final["termination_reason"] == "sdk_error"
    failure = info["result"]["failure"]
    assert failure["kind"] == "authentication_or_access"
    assert failure["http_status"] == 403
    assert "resets_at" not in failure and "reset_text" not in failure
    assert failure["advice"]["next_step"] == "repair_authentication"
    assert "does not say whether credentials" in failure["advice"]["text"]
    assert "Cloudflare" in failure["evidence"]["text"]
    assert info["result"]["final_text"].endswith("| Cloudflare")
    assert "Authentication or access failure reported" in info["status_message"]


@pytest.mark.parametrize(
    ("scenario", "reset_text"),
    [
        ("limit_pst", "5:30am (PST)"),  # an abbreviation names no one zone
        ("limit_dated", "Nov 17, 5am (UTC)"),  # no year: which year?
        ("limit_no_reset", None),
    ],
)
async def test_an_uninterpretable_reset_is_kept_as_words_not_a_time(
    mcp_server: ServerFactory,
    fake_env: dict[str, str],
    workdir: Path,
    scenario: str,
    reset_text: str | None,
) -> None:
    async with mcp_server(fake_env) as session:
        _, info = await fail_with(
            session, workdir, f"[[scenario:reported_failure:{scenario}]]"
        )

    failure = info["result"]["failure"]
    assert failure["kind"] == "usage_limit"  # the limit itself was reported
    assert "resets_at" not in failure
    assert failure.get("reset_text") == reset_text
    assert failure["advice"]["next_step"] == "wait_for_reset"
    assert "does not name one date" in failure["advice"]["text"] or (
        reset_text is None and "No reset time was reported" in failure["advice"]["text"]
    )


@pytest.mark.parametrize("scenario", ["unknown", "quoted", "conflict"])
async def test_unrecognised_or_conflicting_text_stays_unknown_with_its_evidence(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path, scenario: str
) -> None:
    async with mcp_server(fake_env) as session:
        final, info = await fail_with(
            session, workdir, f"[[scenario:reported_failure:{scenario}]]"
        )

    assert final["termination_reason"] == "sdk_error"
    result = info["result"]
    failure = result["failure"]
    assert failure["kind"] == "unknown"
    assert "resets_at" not in failure and "http_status" not in failure
    assert failure["advice"]["next_step"] == "investigate"
    assert failure["evidence"]["text"] == result["final_text"]
    assert "matched" not in failure["evidence"]
    assert "agent call failed" in result["error"]
    assert "Failed (sdk_error): agent call failed" in info["status_message"]


async def test_an_oversized_sdk_message_is_a_transport_failure_with_the_original_error(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    over = DEFAULT_MAX_BUFFER_SIZE + 1
    async with mcp_server(fake_env) as session:
        final, info = await fail_with(
            session, workdir, f"[[scenario:image_result:{over}]] look"
        )

    assert final["termination_reason"] == "sdk_error"
    result = info["result"]
    failure = result["failure"]
    assert failure["kind"] == "sdk_transport"
    assert failure["evidence"]["source"] == "exception"
    assert failure["evidence"]["exception_type"] == "CLIJSONDecodeError"
    assert failure["advice"]["next_step"] == "inspect_then_resume"
    # #7's own evidence and advice are untouched.
    assert "maximum buffer size" in result["error"]
    assert "Inspect them before deciding" in result["error"]
    assert "Inspect them before deciding" in info["status_message"] or (
        "maximum buffer size" in info["status_message"]
    )


async def test_other_failures_carry_no_classification(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        _, limited = await fail_with(session, workdir, "[[scenario:max_turns]] go")
        _, done = await fail_with(session, workdir, "[[scenario:echo]] go")

    assert limited["termination_reason"] == "max_turns"
    assert "failure" not in limited["result"]
    assert done["status"] == "completed" and "failure" not in done["result"]


async def test_a_run_recorded_before_classification_still_reads_back(
    mcp_server: ServerFactory,
    fake_env: dict[str, str],
    mcp_home: Path,
    workdir: Path,
) -> None:
    async with mcp_server(fake_env) as session:
        final, _ = await fail_with(session, workdir, "[[scenario:reported_failure]]")
        task_id = final["task_id"]
        run_dir = runs_dir(mcp_home) / task_id
        for name in ("status.json", "result.json"):
            record = json.loads((run_dir / name).read_text())
            assert record.pop("failure")["kind"] == "usage_limit"
            (run_dir / name).write_text(json.dumps(record))
        old = await detail(session, task_id)
        listed = await call(session, "claude_list")

    assert old["status"] == "failed" and old["termination_reason"] == "sdk_error"
    assert "failure" not in old["result"]
    assert old["result"]["final_text"].startswith("You've hit your")
    assert "failure_kind" not in next(
        t for t in listed["tasks"] if t["task_id"] == task_id
    )
    state = RunState.model_validate_json((run_dir / "status.json").read_text())
    result = RunResult.model_validate_json((run_dir / "result.json").read_text())
    assert state.failure is None and result.failure is None


# --- the classifier: caller interpretation and reset rules ---------------------

OBSERVED = datetime(2026, 11, 10, 2, 30, tzinfo=UTC)


def classify_text(text: str | None, **extra: Any) -> Any:
    return classify_result(
        subtype=extra.pop("subtype", "success"),
        is_error=True,
        text=text,
        errors=extra.pop("errors", []),
        stop_reason="stop_sequence",
        observed_at=OBSERVED,
    )


def test_the_subtype_does_not_decide_the_kind() -> None:
    words = "You've hit your 5-hour limit · resets 5:30am (UTC)"
    assert classify_text(words, subtype="success").kind == "usage_limit"
    assert classify_text(words, subtype="error_during_execution").kind == "usage_limit"
    assert classify_text(words, errors=["x"]).kind == "usage_limit"
    assert classify_text(None).kind == "unknown"
    assert classify_text("").kind == "unknown"


@pytest.mark.parametrize(
    "text",
    [
        "You've hit your limit · Failed to authenticate",
        "You've hit your 5-hour limit. API Error: 403 Attention Required!",
        "Failed to authenticate. API Error: 403. You've hit your limit",
        "Failed to authenticate (usage limit reached)",
    ],
)
def test_one_report_stating_two_kinds_is_unknown(text: str) -> None:
    failure = classify_text(text)
    assert failure.kind == "unknown"
    assert failure.evidence.text == text and failure.evidence.matched is None


def test_two_reports_of_different_kinds_in_one_envelope_are_unknown() -> None:
    failure = classify_text("You've hit your limit", errors=["API Error: 401"])
    assert failure.kind == "unknown"


def test_a_long_text_is_not_read_as_a_report() -> None:
    long = "You've hit your limit. " + "x" * 3000
    assert classify_text(long).kind == "unknown"
    assert classify_text(long).evidence.text_truncated is True
    assert classify_text(long).evidence.text_chars == len(long)


@pytest.mark.parametrize(
    "text",
    [
        "Failed to authenticate. API Error: 403 Attention Required! | Cloudflare",
        "API Error: 401 Invalid authentication credentials",
        "Invalid API key · Please run /login",
    ],
)
def test_authentication_and_access_reports_are_recognised_without_a_cause(
    text: str,
) -> None:
    failure = classify_text(text)
    assert failure.kind == "authentication_or_access"
    assert failure.advice.next_step == "repair_authentication"
    assert failure.advice.basis == "inferred"
    assert failure.resets_at is None


def test_only_the_sdks_transport_errors_are_transport_failures() -> None:
    now = OBSERVED
    for exc in (
        CLIJSONDecodeError("x", ValueError("y")),
        CLIConnectionError("not connected"),
        ProcessError("Command failed", exit_code=1),
    ):
        failure = classify_exception(exc, observed_at=now)
        assert failure.kind == "sdk_transport"
        assert failure.evidence.exception_type == type(exc).__name__
        assert "does not show whether the task's work" in failure.advice.text
    assert classify_stream_end(observed_at=now).kind == "sdk_transport"


def test_an_sdk_exception_that_is_not_a_transport_report_is_unknown() -> None:
    now = OBSERVED
    missing = classify_exception(
        CLINotFoundError(cli_path="/x/claude"), observed_at=now
    )
    assert missing.kind == "unknown"
    assert missing.evidence.exception_type == "CLINotFoundError"
    assert missing.advice.next_step == "investigate"
    assert classify_exception(ValueError("boom"), observed_at=now).kind == "unknown"
    assert classify_exception(ClaudeSDKError("x"), observed_at=now).kind == "unknown"


def test_an_sdk_result_error_is_classified_by_the_result_it_carries() -> None:
    now = OBSERVED
    limit = ResultError(
        "Command failed",
        data={
            "subtype": "success",
            "result": "You've hit your 5-hour limit · resets 5:30am (UTC)",
            "errors": [],
        },
        exit_code=1,
    )
    failure = classify_exception(limit, observed_at=now)
    assert failure.kind == "usage_limit" and failure.reset_text == "5:30am (UTC)"
    other = ResultError("Command failed", data={"subtype": "error_x"}, exit_code=1)
    assert classify_exception(other, observed_at=now).kind == "unknown"
    # Its words win over its type when an ordinary process error says them.
    said = ProcessError("You've hit your limit · resets soon")
    assert classify_exception(said, observed_at=now).kind == "usage_limit"


@pytest.mark.parametrize(
    ("wording", "expected"),
    [
        ("Nov 17, 2026, 5:30am (UTC)", "2026-11-17T05:30:00Z"),
        ("November 17 2026 at 5pm (GMT)", "2026-11-17T17:00:00Z"),
        ("2026-11-17 05:30 (UTC)", "2026-11-17T05:30:00Z"),
        ("2026-11-17T05:30:00Z", "2026-11-17T05:30:00Z"),
        ("2026-11-17T15:30:00+10:00", "2026-11-17T05:30:00Z"),
        # An IANA zone is exact: Sydney is UTC+11 in November.
        ("Nov 17, 2026, 5:30am (Australia/Sydney)", "2026-11-16T18:30:00Z"),
        ("Nov 17, 2026, 12am (UTC)", "2026-11-17T00:00:00Z"),
    ],
)
def test_a_reset_with_a_full_date_and_explicit_zone_becomes_one_instant(
    wording: str, expected: str
) -> None:
    text, resets_at = parse_reset(f"You've hit your limit · resets {wording}", OBSERVED)
    assert text == wording
    assert resets_at == datetime.fromisoformat(expected)
    assert resets_at is not None and resets_at.utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    "wording",
    [
        "5:30am (UTC)",  # the reported text: a time, no date
        "5:30am (America/New_York)",
        "Nov 17, 5am (UTC)",  # no year
        "tomorrow at 5am (UTC)",
        "Nov 17, 2026, 5:30am (PST)",  # abbreviations name no one zone
        "Nov 17, 2026, 5:30am",  # no zone
        "Nov 17, 2026, 5:30 (UTC)",  # no am/pm and no 24-hour form
        "Nov 17, 2026, 13pm (UTC)",
        "Nov 17, 2026, 0am (UTC)",
        "Nov 17, 2026, 5:75am (UTC)",
        "Nov 31, 2026, 5am (UTC)",  # no such day
        "Foo 17, 2026, 5am (UTC)",
        "Nov 17, 2026, 5:30am (Not/AZone)",
        "2026-11-17T05:30:00",  # no offset
        "soon",
        # Not in the future, or too far to be a reset.
        "Nov 9, 2026, 5:30am (UTC)",
        "Nov 10, 2026, 2:30am (UTC)",  # exactly when it was seen
        "Dec 31, 2026, 5:30am (UTC)",
    ],
)
def test_a_reset_without_one_reliable_date_and_zone_stays_words_only(
    wording: str,
) -> None:
    text, resets_at = parse_reset(f"You've hit your limit · resets {wording}", OBSERVED)
    assert text == wording
    assert resets_at is None


@pytest.mark.parametrize(
    ("observed", "wording"),
    [
        # 2:30am does not exist in New York on 2027-03-14 (clocks go 2am→3am).
        ("2027-03-13T04:00:00+00:00", "resets Mar 14, 2027, 2:30am (America/New_York)"),
        # 1:30am happens twice on 2026-11-01 (clocks go 2am→1am).
        ("2026-10-31T03:00:00+00:00", "resets Nov 1, 2026, 1:30am (America/New_York)"),
    ],
)
def test_a_wall_clock_time_the_zone_skips_or_repeats_is_not_one_instant(
    observed: str, wording: str
) -> None:
    text, resets_at = parse_reset(wording, datetime.fromisoformat(observed))
    assert text is not None and resets_at is None
    # Either side of the change the same zone is exact (EST, UTC-5).
    ordinary = parse_reset("resets Nov 11, 2026, 1:30am (America/New_York)", OBSERVED)[
        1
    ]
    assert ordinary == datetime(2026, 11, 11, 6, 30, tzinfo=UTC)


def test_a_naive_observation_time_gives_no_instant() -> None:
    naive = datetime(2026, 11, 10, 2, 30)
    assert parse_reset("resets 5:30am (UTC)", naive) == ("5:30am (UTC)", None)


def test_text_without_a_reset_phrase_has_neither_words_nor_time() -> None:
    assert parse_reset("You've hit your limit", OBSERVED) == (None, None)


@pytest.mark.parametrize(
    "text",
    [
        "403 items were processed",
        "401k contributions: 403 rows",
        "Error 403 while parsing the report",
        "Something failed with 401",
    ],
)
def test_a_bare_status_number_is_not_an_authentication_report(text: str) -> None:
    assert classify_text(text).kind == "unknown"


def test_an_explicit_api_error_status_is_an_access_report() -> None:
    failure = classify_text("API Error: 403 Forbidden")
    assert failure.kind == "authentication_or_access" and failure.http_status == 403


def test_a_missing_tz_database_does_not_hide_the_reported_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from zoneinfo import ZoneInfoNotFoundError

    from jake_tools.claude_runs import failures

    def no_tzdata(key: str) -> Any:
        raise ZoneInfoNotFoundError(key)

    monkeypatch.setattr(failures, "ZoneInfo", no_tzdata)
    # UTC/GMT need no database: the dated reset is still read.
    utc = classify_text(
        "You've hit your weekly limit · resets Nov 17, 2026, 5:30am (UTC)"
    )
    assert utc.kind == "usage_limit"
    assert utc.resets_at == datetime(2026, 11, 17, 5, 30, tzinfo=UTC)
    # An IANA name cannot be resolved here: the limit is kept, the time is not.
    iana = classify_text(
        "You've hit your weekly limit · resets Nov 17, 2026, 5:30am (Europe/London)"
    )
    assert iana.kind == "usage_limit"
    assert iana.resets_at is None
    assert iana.reset_text == "Nov 17, 2026, 5:30am (Europe/London)"
    assert iana.evidence.matched == "You've hit your weekly limit"
