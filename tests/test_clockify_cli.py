from __future__ import annotations

import importlib

from click.testing import CliRunner

from jake_tools.cli.clockify import clockify
from jake_tools.clockify import ClockifyUser

clockify_cli = importlib.import_module("jake_tools.cli.clockify")


class FakeClockifyClient:
    def __init__(self, *, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url

    def get_user(self) -> ClockifyUser:
        return ClockifyUser(
            id="user-123",
            name="Michael Bryan",
            email="michael@example.test",
            activeWorkspace="workspace-1",
            defaultWorkspace="workspace-2",
        )


def test_whoami_prints_current_clockify_user(monkeypatch) -> None:
    monkeypatch.setattr(clockify_cli, "ClockifyClient", FakeClockifyClient)
    runner = CliRunner()

    result = runner.invoke(
        clockify,
        ["--api-key", "test-key", "whoami"],
    )

    assert result.exit_code == 0
    assert "ID: user-123" in result.output
    assert "Name: Michael Bryan" in result.output
    assert "Email: michael@example.test" in result.output
    assert "Active workspace: workspace-1" in result.output
    assert "Default workspace: workspace-2" in result.output


def test_whoami_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("CLOCKIFY_API_KEY", raising=False)
    runner = CliRunner()

    result = runner.invoke(clockify, ["whoami"])

    assert result.exit_code != 0
    assert "CLOCKIFY_API_KEY" in result.output
