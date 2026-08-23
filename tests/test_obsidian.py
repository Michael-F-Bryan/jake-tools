from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from jake_tools.transcription.obsidian import ObsidianCli, ObsidianCliError

LOADER_LINE = (
    "2026-08-23 07:39:31 Loading updated app package "
    "/Users/work/Library/Application Support/obsidian/obsidian-1.13.7.asar"
)
ADVISORY_LINE = (
    "Your Obsidian installer is out of date. Please download the latest "
    "installer which includes better CLI support: https://obsidian.md/download"
)


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout)


class FakeRunner:
    """A fake ``subprocess.run`` that dispatches on the invoked sub-command."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.responses: dict[str, subprocess.CompletedProcess[str]] = {}

    def set_response(
        self, subcommand: str, response: subprocess.CompletedProcess[str]
    ) -> None:
        self.responses[subcommand] = response

    def __call__(
        self, command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(command)
        args = [part for part in command[1:] if not part.startswith("vault=")]
        subcommand = args[0] if args else ""
        response = self.responses.get(subcommand)
        if response is None:
            raise AssertionError(f"no fake response set up for {command}")
        return response


def test_vault_root_strips_loader_and_advisory_noise_from_payload() -> None:
    runner = FakeRunner()
    runner.set_response(
        "vault",
        _completed(f"{LOADER_LINE}\n{ADVISORY_LINE}\n/Users/work/Documents/Vault\n"),
    )
    client = ObsidianCli(run=runner)

    assert client.vault_root() == Path("/Users/work/Documents/Vault")


def test_vault_root_is_only_fetched_once() -> None:
    runner = FakeRunner()
    runner.set_response("vault", _completed("/Users/work/Documents/Vault\n"))
    client = ObsidianCli(run=runner)

    client.vault_root()
    client.vault_root()

    vault_calls = [call for call in runner.calls if call[1] == "vault"]
    assert len(vault_calls) == 1


def test_invoke_passes_vault_selector_when_configured() -> None:
    runner = FakeRunner()
    runner.set_response("vault", _completed("/Users/work/Documents/Vault\n"))
    client = ObsidianCli(vault="Vault", run=runner)

    client.vault_root()

    assert runner.calls == [["obsidian", "vault=Vault", "vault", "info=path"]]


def test_resolve_embed_uses_cli_resolved_path_when_the_file_exists(
    tmp_path: Path,
) -> None:
    attachments = tmp_path / "Attachments"
    attachments.mkdir()
    target_file = attachments / "Recording X.m4a"
    target_file.write_bytes(b"fake audio")

    runner = FakeRunner()
    runner.set_response("vault", _completed(f"{str(tmp_path)}\n"))
    runner.set_response(
        "file",
        _completed(
            f"{LOADER_LINE}\n"
            "path\tAttachments/Recording X.m4a\n"
            "name\tRecording X\n"
            "extension\tm4a\n"
        ),
    )
    client = ObsidianCli(run=runner)

    resolved = client.resolve_embed("Recording X.m4a")

    assert resolved == target_file


def test_resolve_embed_falls_back_to_glob_when_cli_reports_not_found(
    tmp_path: Path,
) -> None:
    attachments = tmp_path / "Attachments"
    attachments.mkdir()
    target_file = attachments / "Recording X.m4a"
    target_file.write_bytes(b"fake audio")

    runner = FakeRunner()
    runner.set_response("vault", _completed(f"{str(tmp_path)}\n"))
    runner.set_response(
        "file",
        _completed(f'{LOADER_LINE}\nError: File "Recording X.m4a" not found.\n'),
    )
    client = ObsidianCli(run=runner)

    resolved = client.resolve_embed("Recording X.m4a")

    assert resolved == target_file


def test_resolve_embed_falls_back_to_glob_when_the_cli_binary_is_unavailable(
    tmp_path: Path,
) -> None:
    attachments = tmp_path / "Attachments"
    attachments.mkdir()
    target_file = attachments / "Recording X.m4a"
    target_file.write_bytes(b"fake audio")

    def raising_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if command[1] == "vault":
            return _completed(f"{str(tmp_path)}\n")
        raise OSError("obsidian binary not found")

    client = ObsidianCli(run=raising_run)

    resolved = client.resolve_embed("Recording X.m4a")

    assert resolved == target_file


def test_resolve_embed_glob_fallback_raises_when_no_file_matches(
    tmp_path: Path,
) -> None:
    runner = FakeRunner()
    runner.set_response("vault", _completed(f"{str(tmp_path)}\n"))
    runner.set_response("file", _completed('Error: File "missing.m4a" not found.\n'))
    client = ObsidianCli(run=runner)

    with pytest.raises(ObsidianCliError):
        client.resolve_embed("missing.m4a")


def test_resolve_embed_glob_fallback_raises_and_names_candidates_when_ambiguous(
    tmp_path: Path,
) -> None:
    first_dir = tmp_path / "Attachments"
    first_dir.mkdir()
    second_dir = tmp_path / "Archive" / "Attachments"
    second_dir.mkdir(parents=True)
    (first_dir / "Recording X.m4a").write_bytes(b"a")
    (second_dir / "Recording X.m4a").write_bytes(b"b")

    runner = FakeRunner()
    runner.set_response("vault", _completed(f"{str(tmp_path)}\n"))
    runner.set_response(
        "file", _completed('Error: File "Recording X.m4a" not found.\n')
    )
    client = ObsidianCli(run=runner)

    with pytest.raises(ObsidianCliError) as excinfo:
        client.resolve_embed("Recording X.m4a")

    message = str(excinfo.value)
    assert str(first_dir / "Recording X.m4a") in message
    assert str(second_dir / "Recording X.m4a") in message


def test_invoke_raises_when_the_cli_exits_nonzero() -> None:
    runner = FakeRunner()
    runner.set_response("vault", _completed("boom\n", returncode=1))
    client = ObsidianCli(run=runner)

    with pytest.raises(ObsidianCliError):
        client.vault_root()
