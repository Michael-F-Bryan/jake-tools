from __future__ import annotations

import importlib
from pathlib import Path

from click.testing import CliRunner

from jake_tools.ai_usage import Usage
from jake_tools.ai_watch.curate import CurateRunResult
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

    async def fake_run_ai_watch_command(*, options):
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
    assert "error: fetch validation failed" in result.output


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
    assert "error: scout validation failed" in result.output


def test_ai_watch_curate_accepts_vault_path(monkeypatch, tmp_path: Path) -> None:
    """run_curate validates the surface path against options.vault_path, so
    the CLI must let an operator set it (previously curate had no such flag
    and silently used the model default)."""
    captured: list[Path] = []

    async def fake_run_curate(*, options, paths, stages):
        del paths, stages
        captured.append(options.vault_path)
        return CurateRunResult(evaluated=0, surfaced=0, speculative=0, usage=Usage())

    monkeypatch.setattr(ai_watch_cli, "run_curate", fake_run_curate)
    custom_vault = tmp_path / "custom-vault"

    result = CliRunner().invoke(
        main,
        [
            "ai-watch",
            "curate",
            "--date",
            "2026-07-02",
            "--base-dir",
            str(tmp_path),
            "--vault-path",
            str(custom_vault),
        ],
    )

    assert result.exit_code == 0
    assert captured == [custom_vault]


def test_ai_watch_tune_help_does_not_advertise_dead_flags() -> None:
    result = CliRunner().invoke(main, ["ai-watch", "tune", "--help"])
    assert result.exit_code == 0
    assert "--max-candidates" not in result.output
    assert "--calibration-only" not in result.output


def test_ai_watch_deliver_rejects_removed_target_flag(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "ai-watch",
            "deliver",
            "--date",
            "2026-07-02",
            "--base-dir",
            str(tmp_path),
            "--target",
            "discord",
        ],
    )

    assert result.exit_code != 0
    assert "no such option" in result.output.lower()
