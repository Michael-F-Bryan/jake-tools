"""``jake-tools files``: the operator CLI over the same operations.

The commands build the very handlers the MCP server registers, so these tests
check the CLI's own job: configuration from the environment, argument wiring,
printing, and a nonzero exit with the full failure detail. Microsoft is the
local ``fake_drive`` on loopback (mechanics only; issue #41 is live evidence).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner, Result
from fake_drive import DRIVE_ID, FakeDrive

from jake_tools.cli import main

LINK = "https://csuses.sharepoint.com/:f:/s/team/EabcDEF?e=TOKEN"


@pytest.fixture
def fake() -> Iterator[FakeDrive]:
    service = FakeDrive().start()
    try:
        yield service
    finally:
        service.stop()


@pytest.fixture
def env(fake: FakeDrive, tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir()
    return {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_CONFIG_DIRS": str(home / "etc"),
        "XDG_STATE_HOME": str(home / "state"),
        "XDG_CACHE_HOME": str(home / "cache"),
        **fake.env(),
    }


def run(env: dict[str, str], *args: str) -> Result:
    # Hermetic: nothing inherited from the host's own jake-tools configuration.
    inherited = {
        name: None
        for name in os.environ
        if name.startswith(("JAKE_TOOLS_", "MICROSOFT_"))
    }
    return CliRunner().invoke(main, ["files", *args], env={**inherited, **env})


def test_the_group_is_registered_and_documented() -> None:
    result = CliRunner().invoke(main, ["files", "--help"])
    assert result.exit_code == 0
    for command in ("resolve", "get", "list", "download", "pdf", "upload", "replace"):
        assert command in result.output


def test_resolve_list_download_replace_and_upload_from_a_shell(
    fake: FakeDrive, env: dict[str, str], tmp_path: Path
) -> None:
    folder = fake.add_folder("Contacts")
    doc = fake.add_file("plan.docx", b"v1", parent=folder.id)
    fake.links[LINK] = folder.id
    out = tmp_path / "plan.docx"
    new = tmp_path / "new.docx"
    new.write_bytes(b"v2 bytes")

    resolved = run(env, "resolve", "--url", LINK, "--json")
    listing = run(env, "list", DRIVE_ID, folder.id, "--limit", "5")
    downloaded = run(env, "download", DRIVE_ID, doc.id, str(out), "--json")
    etag = json.loads(downloaded.output)["item"]["etag"]
    replaced = run(
        env, "replace", DRIVE_ID, doc.id, str(new), "--expected-etag", etag, "--json"
    )
    created = run(env, "upload", DRIVE_ID, folder.id, str(new), "--name", "copy.docx")

    for result in (resolved, listing, downloaded, replaced, created):
        assert result.exit_code == 0, result.output
    assert json.loads(resolved.output)["item"]["item_id"] == folder.id
    assert "plan.docx" in listing.output and "complete: True" in listing.output
    assert out.read_bytes() == b"v1"
    assert json.loads(replaced.output)["verified"] is True
    assert doc.content == b"v2 bytes"
    assert "verified: True" in created.output
    assert fake.named(folder.id, "copy.docx") is not None


def test_a_stale_replacement_exits_nonzero_and_prints_the_conflict(
    fake: FakeDrive, env: dict[str, str], tmp_path: Path
) -> None:
    doc = fake.add_file("plan.docx", b"v1")
    seen = doc.etag
    fake.edit(doc, b"their edit", "Sam Editor")
    new = tmp_path / "new.docx"
    new.write_bytes(b"mine")

    result = run(env, "replace", DRIVE_ID, doc.id, str(new), "--expected-etag", seen)

    assert result.exit_code != 0
    assert "plan_stale" in result.output and "Sam Editor" in result.output
    assert doc.content == b"their edit"


def test_an_uncertain_upload_prints_its_outcome_and_next_step(
    fake: FakeDrive, env: dict[str, str], tmp_path: Path
) -> None:
    folder = fake.add_folder("Contacts")
    source = tmp_path / "a.txt"
    source.write_bytes(b"x")
    fake.script["PUT upload"] = ["apply-then-500"]

    result = run(env, "upload", DRIVE_ID, folder.id, str(source))

    assert result.exit_code != 0
    assert "write_uncertain" in result.output and "retry_safe: False" in result.output
    assert "Do NOT repeat" in result.output


def test_missing_microsoft_configuration_names_the_variables(
    tmp_path: Path,
) -> None:
    home = tmp_path / "h"
    home.mkdir()
    env = {"HOME": str(home), "XDG_CONFIG_HOME": str(home / "c")}
    result = run(env, "get", DRIVE_ID, "x")
    assert result.exit_code != 0
    assert "missing_credentials" in result.output
    assert "MICROSOFT_REFRESH_TOKEN" in result.output
