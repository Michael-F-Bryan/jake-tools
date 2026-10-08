"""``install-skills``: filesystem semantics on ``install_skill_tree``, plus the CLI."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from click.testing import CliRunner

from jake_tools.mcp import server
from jake_tools.mcp.install_skills import (
    SkillInstallError,
    install_skill_tree,
    install_skills,
)


@pytest.fixture
def src(tmp_path: Path) -> Path:
    root = tmp_path / "src-skills"
    (root / "alpha" / "refs").mkdir(parents=True)
    (root / "alpha" / "SKILL.md").write_text("alpha v1")
    (root / "alpha" / "refs" / "extra.md").write_text("extra")
    (root / "beta").mkdir()
    (root / "beta" / "SKILL.md").write_text("beta v1")
    (root / "no-skill-md").mkdir()
    (root / "no-skill-md" / "notes.txt").write_text("ignored")
    (root / "README.md").write_text("not a skill")
    return root


def outcomes(results) -> dict[str, str]:
    return {r.name: r.outcome for r in results}


def test_copy_installs_whole_directories_with_skill_md_only(
    src: Path, tmp_path: Path
) -> None:
    target = tmp_path / "target"

    results = install_skill_tree(src, target, link=False, force=False)

    assert outcomes(results) == {"alpha": "installed", "beta": "installed"}
    assert (target / "alpha" / "refs" / "extra.md").read_text() == "extra"
    assert not (target / "alpha").is_symlink()
    assert not (target / "no-skill-md").exists()
    assert not (target / "README.md").exists()
    assert sorted(p.name for p in target.iterdir()) == ["alpha", "beta"]


def test_link_symlinks_directories(src: Path, tmp_path: Path) -> None:
    target = tmp_path / "target"

    results = install_skill_tree(src, target, link=True, force=False)

    assert outcomes(results) == {"alpha": "linked", "beta": "linked"}
    assert (target / "alpha").is_symlink()
    assert (target / "alpha").resolve() == (src / "alpha").resolve()


def test_existing_skill_is_skipped_without_force(src: Path, tmp_path: Path) -> None:
    target = tmp_path / "target"
    (target / "alpha").mkdir(parents=True)
    (target / "alpha" / "SKILL.md").write_text("local edit")

    results = install_skill_tree(src, target, link=False, force=False)

    assert outcomes(results) == {"alpha": "skipped", "beta": "installed"}
    assert (target / "alpha" / "SKILL.md").read_text() == "local edit"


def test_force_replaces_existing_directory(src: Path, tmp_path: Path) -> None:
    target = tmp_path / "target"
    (target / "alpha").mkdir(parents=True)
    (target / "alpha" / "stale.md").write_text("stale")

    results = install_skill_tree(src, target, link=False, force=True)

    assert outcomes(results)["alpha"] == "replaced"
    assert (target / "alpha" / "SKILL.md").read_text() == "alpha v1"
    assert not (target / "alpha" / "stale.md").exists()
    # No staging directories are left behind.
    assert sorted(p.name for p in target.iterdir()) == ["alpha", "beta"]


def test_force_replaces_symlink_without_touching_its_source(
    src: Path, tmp_path: Path
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.txt").write_text("precious")
    (target / "alpha").symlink_to(elsewhere, target_is_directory=True)

    results = install_skill_tree(src, target, link=False, force=True)

    assert outcomes(results)["alpha"] == "replaced"
    assert not (target / "alpha").is_symlink()
    assert (target / "alpha" / "SKILL.md").read_text() == "alpha v1"
    assert (elsewhere / "keep.txt").read_text() == "precious"


def test_force_link_replaces_link_and_leaves_sources_intact(
    src: Path, tmp_path: Path
) -> None:
    target = tmp_path / "target"
    install_skill_tree(src, target, link=True, force=False)

    results = install_skill_tree(src, target, link=True, force=True)

    assert outcomes(results)["alpha"] == "replaced"
    assert (src / "alpha" / "SKILL.md").read_text() == "alpha v1"
    assert (target / "alpha").is_symlink()


def test_failed_copy_leaves_existing_install_untouched(
    src: Path, tmp_path: Path
) -> None:
    target = tmp_path / "target"
    (target / "alpha").mkdir(parents=True)
    (target / "alpha" / "SKILL.md").write_text("old")
    # An unreadable subdirectory makes copytree fail midway.
    (src / "alpha" / "unreadable").mkdir()
    (src / "alpha" / "unreadable" / "f").write_text("x")
    (src / "alpha" / "unreadable").chmod(0)
    try:
        with pytest.raises(OSError):
            install_skill_tree(src, target, link=False, force=True)
    finally:
        (src / "alpha" / "unreadable").chmod(0o755)

    assert (target / "alpha" / "SKILL.md").read_text() == "old"
    assert sorted(p.name for p in target.iterdir()) == ["alpha"]


def test_target_inside_sources_is_refused_and_nothing_is_deleted(
    src: Path,
) -> None:
    with pytest.raises(SkillInstallError):
        install_skill_tree(src, src, link=False, force=True)
    with pytest.raises(SkillInstallError):
        install_skill_tree(src, src / "alpha", link=False, force=True)

    assert (src / "alpha" / "SKILL.md").read_text() == "alpha v1"
    assert (src / "beta" / "SKILL.md").read_text() == "beta v1"


def test_target_symlinked_to_sources_is_refused(src: Path, tmp_path: Path) -> None:
    alias = tmp_path / "hermes-skills"
    alias.symlink_to(src, target_is_directory=True)

    with pytest.raises(SkillInstallError):
        install_skill_tree(src, alias, link=False, force=True)

    assert (src / "alpha" / "SKILL.md").read_text() == "alpha v1"


def test_destination_symlinked_to_its_source_is_refused(
    src: Path, tmp_path: Path
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "beta").symlink_to(src / "beta", target_is_directory=True)

    with pytest.raises(SkillInstallError):
        install_skill_tree(src, target, link=False, force=True)

    assert (src / "beta" / "SKILL.md").read_text() == "beta v1"
    assert not (target / "alpha").exists()  # refused before anything was touched


def test_empty_or_missing_source_installs_nothing(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    for source in (tmp_path / "empty", tmp_path / "missing"):
        assert install_skill_tree(source, tmp_path / "t", link=False, force=False) == []
    assert not (tmp_path / "t").exists()


def test_cli_into_the_packaged_sources_is_refused(
    src: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "SKILLS_DIR", src)

    result = CliRunner().invoke(install_skills, ["--into", str(src), "--force"])

    assert result.exit_code != 0
    assert "resolves to or into" in result.output
    assert (src / "alpha" / "SKILL.md").read_text() == "alpha v1"


def test_cli_target_defaults_to_hermes_home_skills(
    src: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mcp_env_factory: Callable[..., dict[str, str]],
) -> None:
    monkeypatch.setattr(server, "SKILLS_DIR", src)
    hermes = tmp_path / "hermes"

    # Every variable load_config reads is overridden, so the real environment
    # (including any real HERMES_HOME) cannot leak in.
    result = CliRunner().invoke(
        install_skills,
        [],
        env=mcp_env_factory(tmp_path / "home", HERMES_HOME=str(hermes)),
    )

    assert result.exit_code == 0, result.output
    assert (hermes / "skills" / "alpha" / "SKILL.md").is_file()


def test_missing_target_is_an_error(
    tmp_path: Path, mcp_env_factory: Callable[..., dict[str, str]]
) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "jake_tools.mcp", "install-skills"],
        env=mcp_env_factory(tmp_path / "home"),
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

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jake_tools.mcp",
            "install-skills",
            "--into",
            str(tmp_path / "target"),
        ],
        env=mcp_env_factory(tmp_path / "home"),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "nothing installed" in result.stdout
