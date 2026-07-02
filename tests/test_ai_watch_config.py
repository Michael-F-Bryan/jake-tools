from __future__ import annotations

from jake_tools.ai_watch.config import resolve_discord_target


def test_resolve_discord_target_prefers_cli_value(monkeypatch) -> None:
    monkeypatch.setenv("AI_WATCH_DISCORD_TARGET", "discord:env")
    assert resolve_discord_target("discord:cli") == "discord:cli"


def test_resolve_discord_target_falls_back_to_env(monkeypatch) -> None:
    monkeypatch.setenv("AI_WATCH_DISCORD_TARGET", "discord:env")
    assert resolve_discord_target("") == "discord:env"


def test_resolve_discord_target_empty_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("AI_WATCH_DISCORD_TARGET", raising=False)
    assert resolve_discord_target("") == ""
