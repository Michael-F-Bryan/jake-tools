from __future__ import annotations

import importlib
from pathlib import Path

from click.testing import CliRunner

from jake_tools.cli import main

ai_watch_cli = importlib.import_module("jake_tools.cli.ai_watch")


def test_ai_watch_help_lists_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["ai-watch", "--help"])
    assert result.exit_code == 0
    for name in (
        "collect",
        "fetch",
        "scout",
        "curate",
        "obsidian-sync",
        "digest",
        "deliver",
        "run",
        "audit",
    ):
        assert name in result.output


def test_ai_watch_run_resolves_discord_target_from_env(
    monkeypatch, tmp_path: Path
) -> None:
    captured: list[str] = []

    def fake_run_ai_watch_command(*, options):
        captured.append(options.discord_target)
        from jake_tools.ai_watch.models import AiWatchCommandResult, RunStatus

        return AiWatchCommandResult(
            status=RunStatus.OK,
            run_id="2026-07-02",
            root=tmp_path,
            digest_path=tmp_path / "digest.md",
            summary_path=tmp_path / "summary.json",
            surfaced_count=0,
            speculative_count=0,
            failed_stages=[],
        )

    monkeypatch.setenv("AI_WATCH_DISCORD_TARGET", "discord:env-user")
    monkeypatch.setattr(ai_watch_cli, "run_ai_watch_command", fake_run_ai_watch_command)

    result = CliRunner().invoke(
        main,
        ["ai-watch", "run", "--date", "2026-07-02", "--base-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert captured == ["discord:env-user"]


def test_ai_watch_deliver_resolves_discord_target_from_env(
    monkeypatch, tmp_path: Path
) -> None:
    from jake_tools.ai_watch.audit_models import DeliveryStatus
    from jake_tools.ai_watch.delivery import DeliveryResult

    captured: list[str] = []

    def fake_run_delivery(*, options, paths):
        del paths
        captured.append(options.discord_target)
        return DeliveryResult(
            status=DeliveryStatus.SKIPPED,
            surfaced_count=0,
            speculative_count=0,
        )

    monkeypatch.setenv("AI_WATCH_DISCORD_TARGET", "discord:env-channel")
    monkeypatch.setattr(ai_watch_cli, "run_delivery", fake_run_delivery)

    result = CliRunner().invoke(
        main,
        ["ai-watch", "deliver", "--date", "2026-07-02", "--base-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert captured == ["discord:env-channel"]


def test_ai_watch_fetch_exits_nonzero_on_runtime_error(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_run_fetch(**kwargs):
        del kwargs
        raise RuntimeError("fetch validation failed")

    monkeypatch.setattr(ai_watch_cli, "run_fetch", fake_run_fetch)

    result = CliRunner().invoke(
        main,
        ["ai-watch", "fetch", "--date", "2026-07-02", "--base-dir", str(tmp_path)],
    )

    assert result.exit_code == 1


def test_ai_watch_scout_exits_nonzero_on_runtime_error(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_run_scout(**kwargs):
        del kwargs
        raise RuntimeError("scout validation failed")

    monkeypatch.setattr(ai_watch_cli, "run_scout", fake_run_scout)

    result = CliRunner().invoke(
        main,
        ["ai-watch", "scout", "--date", "2026-07-02", "--base-dir", str(tmp_path)],
    )

    assert result.exit_code == 1
