"""The Swift hook runs the Nix check on macOS and skips everywhere else."""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize(
    ("platform", "has_nix", "expected_code"),
    [("Linux", True, 0), ("Darwin", False, 0), ("Darwin", True, 42)],
)
def test_swift_hook_runs_the_nix_check_on_macos(
    tmp_path: Path, platform: str, has_nix: bool, expected_code: int
) -> None:
    config = yaml.safe_load(
        (Path(__file__).parents[1] / ".pre-commit-config.yaml").read_text()
    )
    hook = next(
        hook
        for repo in config["repos"]
        for hook in repo["hooks"]
        if hook["id"] == "swift-test"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shell = shutil.which("sh")
    assert shell is not None
    (bin_dir / "sh").symlink_to(shell)
    scripts = {"uname": f"echo {platform}"}
    if has_nix:
        scripts["nix"] = 'echo "NIX_INVOKED $*"\nexit 42'
    for name, body in scripts.items():
        executable = bin_dir / name
        executable.write_text(f"#!{shell}\n{body}\n")
        executable.chmod(0o755)
    result = subprocess.run(
        shlex.split(hook["entry"]),
        cwd=tmp_path,
        env={**os.environ, "PATH": str(bin_dir)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == expected_code, result.stderr
    invoked = (
        "NIX_INVOKED build --no-link .#checks.aarch64-darwin.transcription-service"
    )
    assert (invoked in result.stdout) == (expected_code == 42)
    if expected_code == 0:
        assert "tests skipped" in result.stderr
