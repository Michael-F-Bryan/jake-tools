"""Advisory usage-limit cooldowns shared by runs of one declared account context.

The store tests use real files and real processes for concurrency. The
caller-facing tests run the real stdio server, worker and Claude Agent SDK
against ``tests/fake_claude.py``, which replays the reported error envelope
(``subtype`` ``success`` with ``is_error``); the label is set the way an
operator would, in the server's environment. Nothing here inspects a
credential or an account, and the fake proves mechanics only.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession
from test_claude_failures import fail_with
from test_claude_tasks import (
    ServerFactory,
    call,
    detail,
    poll_until,
    runs_dir,
    terminal,
)
from test_claude_tasks import start as start_task

from jake_tools.claude_runs import cooldowns
from jake_tools.claude_runs.models import RunSpec

NOW = datetime(2026, 11, 10, 2, 30, tzinfo=UTC)
LATER = NOW + timedelta(days=2)


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


# --- the store ------------------------------------------------------------------


def test_a_recorded_cooldown_is_visible_until_it_expires(tmp_path: Path) -> None:
    record = cooldowns.record(
        tmp_path, "acct-a", resets_at=LATER, observed_at=NOW, task_id="t1", now=NOW
    )
    assert record is not None and record.advisory is True
    assert cooldowns.active(tmp_path, "acct-a", now=NOW + timedelta(days=1)) == record
    assert cooldowns.active(tmp_path, "acct-a", now=LATER) is None  # at the reset
    assert cooldowns.active(tmp_path, "acct-a", now=LATER + timedelta(hours=1)) is None


def test_labels_are_isolated_and_no_label_means_no_sharing(tmp_path: Path) -> None:
    cooldowns.record(
        tmp_path, "acct-a", resets_at=LATER, observed_at=NOW, task_id="t1", now=NOW
    )
    assert cooldowns.active(tmp_path, "acct-b", now=NOW) is None
    assert cooldowns.active(tmp_path, None, now=NOW) is None
    assert (
        cooldowns.record(
            tmp_path, None, resets_at=LATER, observed_at=NOW, task_id="t", now=NOW
        )
        is None
    )
    assert len(list((tmp_path / ".cooldowns").glob("*.json"))) == 1


def test_the_filename_holds_a_digest_not_the_label(tmp_path: Path) -> None:
    cooldowns.record(
        tmp_path, "acct-a", resets_at=LATER, observed_at=NOW, task_id="t1", now=NOW
    )
    names = [p.name for p in (tmp_path / ".cooldowns").iterdir()]
    assert not any("acct" in name for name in names)
    mode = next((tmp_path / ".cooldowns").glob("*.json")).stat().st_mode & 0o777
    assert mode == 0o600


def test_a_reset_already_past_is_not_recorded(tmp_path: Path) -> None:
    assert (
        cooldowns.record(
            tmp_path, "acct-a", resets_at=NOW, observed_at=NOW, task_id="t", now=NOW
        )
        is None
    )
    assert cooldowns.active(tmp_path, "acct-a", now=NOW - timedelta(days=1)) is None


def test_the_newer_observation_wins_whatever_order_they_arrive_in(
    tmp_path: Path,
) -> None:
    newer = cooldowns.record(
        tmp_path,
        "acct-a",
        resets_at=LATER,
        observed_at=NOW + timedelta(hours=1),
        task_id="newer",
        now=NOW,
    )
    older = cooldowns.record(
        tmp_path,
        "acct-a",
        resets_at=LATER + timedelta(days=3),
        observed_at=NOW,
        task_id="older",
        now=NOW,
    )
    assert older == newer
    assert cooldowns.active(tmp_path, "acct-a", now=NOW).task_id == "newer"  # type: ignore[union-attr]


def test_a_success_clears_only_an_older_observation(tmp_path: Path) -> None:
    cooldowns.record(
        tmp_path, "acct-a", resets_at=LATER, observed_at=NOW, task_id="t1", now=NOW
    )
    cooldowns.record(
        tmp_path, "acct-b", resets_at=LATER, observed_at=NOW, task_id="t2", now=NOW
    )
    # A prompt sent before the observation says nothing about it.
    assert not cooldowns.clear(
        tmp_path, "acct-a", interaction_at=NOW - timedelta(hours=1), now=NOW
    )
    assert not cooldowns.clear(tmp_path, "acct-a", interaction_at=NOW, now=NOW)
    assert cooldowns.active(tmp_path, "acct-a", now=NOW) is not None
    # One sent after it, and answered, does; and only for its own context.
    assert cooldowns.clear(
        tmp_path, "acct-a", interaction_at=NOW + timedelta(hours=1), now=NOW
    )
    assert cooldowns.active(tmp_path, "acct-a", now=NOW) is None
    assert not cooldowns.clear(tmp_path, None, interaction_at=LATER, now=NOW)
    assert cooldowns.active(tmp_path, "acct-b", now=NOW) is not None


def test_a_delayed_older_failure_cannot_resurrect_a_cleared_cooldown(
    tmp_path: Path,
) -> None:
    t1, t2 = NOW, NOW + timedelta(hours=1)
    # A prompt sent at t2 is answered while no cooldown is on record yet ...
    assert not cooldowns.clear(tmp_path, "acct-a", interaction_at=t2, now=t2)
    # ... then a slow worker writes the failure it observed earlier, at t1.
    assert (
        cooldowns.record(
            tmp_path, "acct-a", resets_at=LATER, observed_at=t1, task_id="late", now=t2
        )
        is None
    )
    assert cooldowns.active(tmp_path, "acct-a", now=t2) is None
    # A failure observed after the answered prompt is a newer fact and counts.
    fresh = cooldowns.record(
        tmp_path,
        "acct-a",
        resets_at=LATER,
        observed_at=t2 + timedelta(minutes=5),
        task_id="fresh",
        now=t2,
    )
    assert fresh is not None
    assert cooldowns.active(tmp_path, "acct-a", now=t2) == fresh
    # Another context never had the watermark.
    assert (
        cooldowns.record(
            tmp_path, "acct-b", resets_at=LATER, observed_at=t1, task_id="b", now=t2
        )
        is not None
    )


def test_clearing_keeps_a_newer_failure_and_the_watermark_only_moves_forward(
    tmp_path: Path,
) -> None:
    t1, t2, t3 = NOW, NOW + timedelta(hours=1), NOW + timedelta(hours=2)
    cooldowns.record(
        tmp_path, "acct-a", resets_at=LATER, observed_at=t2, task_id="f", now=t1
    )
    # An older answered prompt (t1) does not clear a failure observed at t2 ...
    assert not cooldowns.clear(tmp_path, "acct-a", interaction_at=t1, now=t1)
    assert cooldowns.active(tmp_path, "acct-a", now=t1) is not None
    # ... a later one (t3) does, and an earlier answer never lowers the mark.
    assert cooldowns.clear(tmp_path, "acct-a", interaction_at=t3, now=t3)
    assert not cooldowns.clear(tmp_path, "acct-a", interaction_at=t1, now=t3)
    assert (
        cooldowns.record(
            tmp_path, "acct-a", resets_at=LATER, observed_at=t2, task_id="g", now=t3
        )
        is None
    )


def test_purging_expired_files_keeps_a_recent_watermark_and_drops_old_ones(
    tmp_path: Path,
) -> None:
    t = NOW
    cooldowns.clear(tmp_path, "acct-a", interaction_at=t, now=t)
    # Another write purges: acct-a has no live cooldown but a recent mark.
    cooldowns.record(
        tmp_path, "acct-b", resets_at=LATER, observed_at=t, task_id="b", now=t
    )
    assert len(list((tmp_path / ".cooldowns").glob("*.json"))) == 2
    assert (
        cooldowns.record(
            tmp_path,
            "acct-a",
            resets_at=LATER,
            observed_at=t - timedelta(seconds=1),
            task_id="a",
            now=t,
        )
        is None
    )
    # Past the retention, the mark and the files with nothing to say are gone.
    later = t + cooldowns.WATERMARK_RETENTION + timedelta(days=1)
    cooldowns.record(
        tmp_path,
        "acct-c",
        resets_at=later + timedelta(days=1),
        observed_at=later,
        task_id="c",
        now=later,
    )
    assert len(list((tmp_path / ".cooldowns").glob("*.json"))) == 1


def test_a_corrupt_or_foreign_record_reads_as_no_cooldown(tmp_path: Path) -> None:
    cooldowns.record(
        tmp_path, "acct-a", resets_at=LATER, observed_at=NOW, task_id="t1", now=NOW
    )
    path = next((tmp_path / ".cooldowns").glob("*.json"))
    path.write_text("{not json")
    assert cooldowns.active(tmp_path, "acct-a", now=NOW) is None
    # A record naming another label in this label's file is not trusted.
    path.write_text(
        json.dumps(
            {
                "account_context": "acct-b",
                "resets_at": LATER.isoformat(),
                "observed_at": NOW.isoformat(),
                "task_id": "x",
            }
        )
    )
    assert cooldowns.active(tmp_path, "acct-a", now=NOW) is None


def _write_observation(args: tuple[str, int]) -> None:
    runs, index = args
    cooldowns.record(
        Path(runs),
        "acct-a",
        resets_at=datetime.now(UTC) + timedelta(days=1),
        observed_at=NOW + timedelta(seconds=index),
        task_id=f"w{index}",
    )


def _read_many(runs: str) -> int:
    seen = 0
    for _ in range(300):
        record = cooldowns.active(Path(runs), "acct-a")
        if record is not None:
            assert record.account_context == "acct-a"  # never a partial write
            seen += 1
    return seen


def test_concurrent_writers_and_readers_leave_one_whole_newest_record(
    tmp_path: Path,
) -> None:
    (tmp_path / ".cooldowns").mkdir()
    with ProcessPoolExecutor(max_workers=6) as pool:
        readers = [pool.submit(_read_many, str(tmp_path)) for _ in range(2)]
        writers = list(
            pool.map(_write_observation, [(str(tmp_path), i) for i in range(24)])
        )
        assert len(writers) == 24
        assert all(future.result() >= 0 for future in readers)
    final = cooldowns.active(tmp_path, "acct-a")
    assert final is not None and final.task_id == "w23"
    assert len(list((tmp_path / ".cooldowns").glob("*.json"))) == 1
    assert not list((tmp_path / ".cooldowns").glob(".*.tmp"))


# --- through the real server, worker and SDK ---------------------------------------

LABEL = "JAKE_TOOLS_CLAUDE_ACCOUNT_CONTEXT"


def with_label(env: dict[str, str], label: str | None) -> dict[str, str]:
    return env if label is None else {**env, LABEL: label}


async def limited(
    session: ClientSession, workdir: Path, scenario: str
) -> dict[str, Any]:
    _, info = await fail_with(
        session, workdir, f"[[scenario:reported_failure:{scenario}]]"
    )
    return info


async def test_a_dated_limit_is_shared_with_its_context_and_not_beyond(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(with_label(fake_env, "acct-a")) as session:
        info = await limited(session, workdir, "limit_full_date")
        failed_id = info["task_id"]
        # The failed task keeps its identity, raw evidence and failure record.
        assert info["status"] == "failed" and info["termination_reason"] == "sdk_error"
        failure = info["result"]["failure"]
        assert failure["kind"] == "usage_limit"
        assert failure["evidence"]["subtype"] == "success"
        assert info["result"]["final_text"].startswith("You've hit your weekly limit")
        cooldown = info["result"]["cooldown"]
        assert cooldown["account_context"] == "acct-a"
        assert cooldown["resets_at"] == failure["resets_at"]
        assert cooldown["task_id"] == failed_id and cooldown["advisory"] is True

        # The next start in the same context sees it, and is still started.
        started = await start_task(session, workdir, "[[scenario:hang]] wait")
        assert started["cooldown"] == cooldown
        assert started["config"]["account_context"] == "acct-a"
        assert started["status"] == "working"
        await call(session, "claude_cancel", task_id=started["task_id"])

    # Another declared context, and no declared context, see nothing.
    for other in ("acct-b", None):
        async with mcp_server(with_label(fake_env, other)) as session:
            started = await start_task(session, workdir, "[[scenario:echo]] hi")
            assert "cooldown" not in started
            final, _, _ = await poll_until(session, started["task_id"], terminal)
            assert final["status"] == "completed"

    # The record's own run kept the label it launched with.
    spec = RunSpec.model_validate_json(
        (runs_dir(Path(fake_env["HOME"])) / failed_id / "spec.json").read_text()
    )
    assert spec.account_context == "acct-a"


async def test_an_undated_or_ambiguous_limit_shares_nothing(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(with_label(fake_env, "acct-a")) as session:
        for scenario in ("limit", "limit_pst", "limit_dated", "limit_no_reset"):
            info = await limited(session, workdir, scenario)
            assert info["result"]["failure"]["kind"] == "usage_limit"
            assert "cooldown" not in info["result"]
        started = await start_task(session, workdir, "[[scenario:echo]] hi")
        assert "cooldown" not in started
        await poll_until(session, started["task_id"], terminal)
    # A later answered prompt leaves a watermark file, never a cooldown.
    files = (runs_dir(Path(fake_env["HOME"])) / ".cooldowns").glob("*.json")
    assert all(json.loads(f.read_text())["cooldown"] is None for f in files)


async def test_a_limit_without_a_declared_context_is_never_shared(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        info = await limited(session, workdir, "limit_full_date")
        assert info["result"]["failure"]["resets_at"]  # the failure is classified
        assert "cooldown" not in info["result"]
        started = await start_task(session, workdir, "[[scenario:echo]] hi")
        assert "cooldown" not in started
        await poll_until(session, started["task_id"], terminal)
    assert not (runs_dir(Path(fake_env["HOME"])) / ".cooldowns").exists()


async def test_other_failures_do_not_publish_a_cooldown(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(with_label(fake_env, "acct-a")) as session:
        for scenario in ("auth", "unknown", "quoted", "conflict"):
            info = await limited(session, workdir, scenario)
            assert "cooldown" not in info["result"]
        started = await start_task(session, workdir, "[[scenario:echo]] hi")
        assert "cooldown" not in started
        await poll_until(session, started["task_id"], terminal)


async def test_a_later_answered_prompt_clears_only_its_own_context(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    home = Path(fake_env["HOME"])
    async with mcp_server(with_label(fake_env, "acct-a")) as session:
        await limited(session, workdir, "limit_full_date")
    async with mcp_server(with_label(fake_env, "acct-b")) as session:
        await limited(session, workdir, "limit_full_date")
    assert len(list((runs_dir(home) / ".cooldowns").glob("*.json"))) == 2

    async with mcp_server(with_label(fake_env, "acct-a")) as session:
        started = await start_task(session, workdir, "[[scenario:echo]] hi")
        assert started["cooldown"]["account_context"] == "acct-a"  # advisory only
        final, _, _ = await poll_until(session, started["task_id"], terminal)
        assert final["status"] == "completed"  # it ran, and was answered
        again = await start_task(session, workdir, "[[scenario:echo]] again")
        assert "cooldown" not in again
        await poll_until(session, again["task_id"], terminal)
    async with mcp_server(with_label(fake_env, "acct-b")) as session:
        started = await start_task(session, workdir, "[[scenario:hang]] wait")
        assert started["cooldown"]["account_context"] == "acct-b"
        await call(session, "claude_cancel", task_id=started["task_id"])


async def test_an_expired_record_is_ignored_by_a_start(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    runs = runs_dir(Path(fake_env["HOME"]))
    runs.mkdir(parents=True, exist_ok=True)
    past = datetime.now(UTC) - timedelta(days=3)
    cooldowns.record(
        runs,
        "acct-a",
        resets_at=past + timedelta(hours=1),
        observed_at=past,
        task_id="old",
        now=past,
    )
    async with mcp_server(with_label(fake_env, "acct-a")) as session:
        started = await start_task(session, workdir, "[[scenario:echo]] hi")
        assert "cooldown" not in started
        await poll_until(session, started["task_id"], terminal)


async def test_runs_written_before_the_label_still_read_back(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    home = Path(fake_env["HOME"])
    async with mcp_server(with_label(fake_env, "acct-a")) as session:
        info = await limited(session, workdir, "limit_full_date")
        run_dir = runs_dir(home) / info["task_id"]
        record = json.loads((run_dir / "spec.json").read_text())
        record.pop("account_context")
        (run_dir / "spec.json").write_text(json.dumps(record))
        old = await detail(session, info["task_id"])
        listed = await call(session, "claude_list")

    assert old["status"] == "failed"
    assert "cooldown" not in old["result"]  # no label on record: no context
    assert old["result"]["failure"]["kind"] == "usage_limit"
    assert any(t["task_id"] == info["task_id"] for t in listed["tasks"])


def test_an_exact_tie_keeps_the_failure_whichever_order_it_arrives_in(
    tmp_path: Path,
) -> None:
    tie = NOW + timedelta(hours=1)
    # Failure first, then the answered prompt sent at the same instant.
    first = tmp_path / "failure-first"
    first.mkdir()
    cooldowns.record(
        first, "acct-a", resets_at=LATER, observed_at=tie, task_id="f", now=NOW
    )
    assert not cooldowns.clear(first, "acct-a", interaction_at=tie, now=NOW)
    assert cooldowns.active(first, "acct-a", now=NOW) is not None
    # The answered prompt first, then the failure observed at the same instant.
    second = tmp_path / "success-first"
    second.mkdir()
    assert not cooldowns.clear(second, "acct-a", interaction_at=tie, now=NOW)
    kept = cooldowns.record(
        second, "acct-a", resets_at=LATER, observed_at=tie, task_id="f", now=NOW
    )
    assert kept is not None
    assert cooldowns.active(second, "acct-a", now=NOW) == kept
    # One tick older than the prompt is stale in both orders.
    assert (
        cooldowns.record(
            second,
            "acct-b",
            resets_at=LATER,
            observed_at=tie - timedelta(microseconds=1),
            task_id="g",
            now=NOW,
        )
        is not None
    )
    cooldowns.clear(second, "acct-b", interaction_at=tie, now=NOW)
    assert cooldowns.active(second, "acct-b", now=NOW) is None
