"""``install-skills``: the real Click command, installing only into temp dirs."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from click.testing import CliRunner

from jake_tools.mcp import server
from jake_tools.mcp.install_skills import install_skill_tree, install_skills


@pytest.fixture
def skills_src(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temp skills tree, which the command reads in place of the packaged one."""
    src = tmp_path / "src-skills"
    (src / "alpha" / "refs").mkdir(parents=True)
    (src / "alpha" / "SKILL.md").write_text("alpha v1")
    (src / "alpha" / "refs" / "extra.md").write_text("extra")
    (src / "beta").mkdir()
    (src / "beta" / "SKILL.md").write_text("beta v1")
    (src / "README.md").write_text("not a skill")
    monkeypatch.setattr(server, "SKILLS_DIR", src)
    return src


def invoke(*args: str, env: dict[str, str] | None = None):
    return CliRunner().invoke(install_skills, list(args), env=env)


def test_copy_installs_whole_directories(skills_src: Path, tmp_path: Path) -> None:
    target = tmp_path / "target"

    result = invoke("--into", str(target))

    assert result.exit_code == 0, result.output
    assert "installed: alpha" in result.output
    assert "installed: beta" in result.output
    assert (target / "alpha" / "SKILL.md").read_text() == "alpha v1"
    assert (target / "alpha" / "refs" / "extra.md").read_text() == "extra"
    assert not (target / "alpha").is_symlink()
    assert not (target / "README.md").exists()


def test_link_symlinks_directories(skills_src: Path, tmp_path: Path) -> None:
    target = tmp_path / "target"

    result = invoke("--into", str(target), "--link")

    assert result.exit_code == 0, result.output
    assert "linked: alpha" in result.output
    assert (target / "alpha").is_symlink()
    assert (target / "alpha").resolve() == (skills_src / "alpha").resolve()


def test_existing_skill_is_skipped_without_force(
    skills_src: Path, tmp_path: Path
) -> None:
    target = tmp_path / "target"
    (target / "alpha").mkdir(parents=True)
    (target / "alpha" / "SKILL.md").write_text("local edit")

    result = invoke("--into", str(target))

    assert result.exit_code != 0
    assert "skipped: alpha" in result.output
    assert "installed: beta" in result.output
    assert (target / "alpha" / "SKILL.md").read_text() == "local edit"


def test_force_replaces_existing_directory(skills_src: Path, tmp_path: Path) -> None:
    target = tmp_path / "target"
    (target / "alpha").mkdir(parents=True)
    (target / "alpha" / "stale.md").write_text("stale")

    result = invoke("--into", str(target), "--force")

    assert result.exit_code == 0, result.output
    assert "replaced: alpha" in result.output
    assert (target / "alpha" / "SKILL.md").read_text() == "alpha v1"
    assert not (target / "alpha" / "stale.md").exists()


def test_force_replaces_symlink_without_touching_its_source(
    skills_src: Path, tmp_path: Path
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.txt").write_text("precious")
    (target / "alpha").symlink_to(elsewhere, target_is_directory=True)

    result = invoke("--into", str(target), "--force")

    assert result.exit_code == 0, result.output
    assert "replaced: alpha" in result.output
    assert not (target / "alpha").is_symlink()
    assert (target / "alpha" / "SKILL.md").read_text() == "alpha v1"
    assert (elsewhere / "keep.txt").read_text() == "precious"


def test_force_link_replaces_link_and_leaves_sources_intact(
    skills_src: Path, tmp_path: Path
) -> None:
    target = tmp_path / "target"
    assert invoke("--into", str(target), "--link").exit_code == 0

    result = invoke("--into", str(target), "--link", "--force")

    assert result.exit_code == 0, result.output
    assert "replaced: alpha" in result.output
    assert (skills_src / "alpha" / "SKILL.md").read_text() == "alpha v1"
    assert (target / "alpha").is_symlink()


def test_target_defaults_to_hermes_home_skills(
    skills_src: Path, tmp_path: Path, mcp_env_factory: Callable[..., dict[str, str]]
) -> None:
    hermes = tmp_path / "hermes"

    result = invoke(env=mcp_env_factory(tmp_path / "home", HERMES_HOME=str(hermes)))

    assert result.exit_code == 0, result.output
    assert (hermes / "skills" / "alpha" / "SKILL.md").is_file()


def test_missing_target_is_an_error(
    tmp_path: Path, mcp_env_factory: Callable[..., dict[str, str]]
) -> None:
    env = mcp_env_factory(tmp_path / "home")

    result = subprocess.run(
        [sys.executable, "-m", "jake_tools.mcp", "install-skills"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode != 0
    assert "--into" in result.stderr
    assert "HERMES_HOME" in result.stderr


def test_zero_packaged_skills_exits_zero(
    tmp_path: Path, mcp_env_factory: Callable[..., dict[str, str]]
) -> None:
    """Runs the real command against the real packaged tree (empty for now)."""
    if server.packaged_skills():
        pytest.skip("skills are packaged; covered by the temp-tree tests")
    target = tmp_path / "target"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jake_tools.mcp",
            "install-skills",
            "--into",
            str(target),
        ],
        env=mcp_env_factory(tmp_path / "home"),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "nothing installed" in result.stdout


def test_install_skill_tree_with_empty_source(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    assert (
        install_skill_tree(tmp_path / "src", tmp_path / "t", link=False, force=False)
        == []
    )
    assert not (tmp_path / "t").exists()
