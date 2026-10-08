"""``python -m jake_tools.mcp doctor``, run for real under an isolated HOME.

The only fakes are at true external boundaries: a local HTTP server standing
in for Clockify and Jira, and a stand-in ``claude`` executable on PATH (the
real CLI is exercised by the slow test at the bottom).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

type RunDoctor = Callable[..., subprocess.CompletedProcess[str]]

SENTINEL = "sk-SENTINEL-1234"

FAKE_CLAUDE = """#!/bin/sh
if [ "$1" = "--version" ]; then echo "9.9.9 (Fake)"; exit 0; fi
if [ "$1" = "auth" ]; then
  echo '{"loggedIn": true, "authMethod": "claude.ai", "email": "leak@example.com"}'
  exit 0
fi
exit 2
"""


class _Handler(BaseHTTPRequestHandler):
    routes: dict[str, tuple[int, object]] = {}

    def do_GET(self) -> None:
        status, body = self.routes.get(self.path, (404, {"message": "nope"}))
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def fake_api() -> Iterator[tuple[str, dict[str, tuple[int, object]]]]:
    routes: dict[str, tuple[int, object]] = {}
    handler = type("Handler", (_Handler,), {"routes": routes})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", routes
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "home"


@pytest.fixture
def claude_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    claude = bin_dir / "claude"
    claude.write_text(FAKE_CLAUDE)
    claude.chmod(0o755)
    return bin_dir


@pytest.fixture
def run_doctor(
    mcp_env_factory: Callable[..., dict[str, str]],
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(
        home: Path, bin_dir: Path | None, *args: str, **extra: str
    ) -> subprocess.CompletedProcess[str]:
        env = mcp_env_factory(home, **extra)
        env["PATH"] = str(bin_dir) if bin_dir is not None else str(home / "empty-path")
        return subprocess.run(
            [sys.executable, "-m", "jake_tools.mcp", "doctor", *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    return run


def check_line(output: str, name: str) -> str:
    lines = [ln for ln in output.splitlines() if f" {name}  " in ln]
    assert len(lines) == 1, output
    return lines[0].strip()


def test_defaults_pass_and_secrets_are_skipped(
    run_doctor: RunDoctor, home: Path, claude_bin: Path
) -> None:
    result = run_doctor(home, claude_bin)

    assert result.returncode == 0, result.stdout + result.stderr
    assert check_line(result.stdout, "config").startswith("OK")
    assert check_line(result.stdout, "runs_dir").startswith("OK")
    assert check_line(result.stdout, "claude_cli").startswith("OK")
    assert check_line(result.stdout, "clockify").startswith("SKIP")
    assert check_line(result.stdout, "jira").startswith("SKIP")
    assert check_line(result.stdout, "session_store").startswith("SKIP")
    # The runs dir was created for real, owner-only.
    runs = home / ".local" / "state" / "jake-tools" / "runs"
    assert runs.is_dir()
    assert runs.stat().st_mode & 0o777 == 0o700
    assert list(runs.iterdir()) == []
    assert str(runs) in result.stdout
    assert "leak@example.com" not in result.stdout


def test_settings_show_sources(
    run_doctor: RunDoctor, home: Path, claude_bin: Path
) -> None:
    config_dir = home / ".config" / "jake-tools"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.toml"
    config_file.write_text('[clockify]\nclient = "Acme"\n')

    result = run_doctor(
        home,
        claude_bin,
        JAKE_TOOLS_JIRA_PROJECT="ZZ",
        CLOCKIFY_API_KEY=SENTINEL,
    )

    assert (
        "clockify.jira_project = ZZ  [env (JAKE_TOOLS_JIRA_PROJECT)]" in result.stdout
    )
    assert f"clockify.client = Acme  [config ({config_file})]" in result.stdout
    assert "claude.max_turns = 20  [default]" in result.stdout
    assert "CLOCKIFY_API_KEY = set  [env (CLOCKIFY_API_KEY)]" in result.stdout
    assert "JIRA_API_TOKEN = unset" in result.stdout
    assert f"ok      {config_file}" in result.stdout
    assert "missing" in result.stdout


def test_secrets_never_printed(
    run_doctor: RunDoctor, home: Path, claude_bin: Path
) -> None:
    result = run_doctor(
        home,
        claude_bin,
        CLOCKIFY_API_KEY=SENTINEL,
        JIRA_API_TOKEN=SENTINEL,
        JIRA_EMAIL="me@example.com",
        JIRA_BASE_URL=f"http://127.0.0.1:1/{SENTINEL}",
        CLOCKIFY_API_BASE_URL="http://127.0.0.1:1",
    )

    assert result.returncode == 1  # both unreachable
    assert SENTINEL not in result.stdout
    assert SENTINEL not in result.stderr
    assert "FAIL  clockify" in result.stdout
    assert "FAIL  jira" in result.stdout


def test_config_parse_failure_exits_1(
    run_doctor: RunDoctor, home: Path, claude_bin: Path
) -> None:
    config_dir = home / ".config" / "jake-tools"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text("this is = = not toml")

    result = run_doctor(home, claude_bin)

    assert result.returncode == 1
    assert check_line(result.stdout, "config").startswith("FAIL")
    assert "error" in result.stdout


def test_unwritable_runs_dir_exits_1(
    run_doctor: RunDoctor, home: Path, claude_bin: Path, tmp_path: Path
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")

    result = run_doctor(home, claude_bin, "--runs-dir", str(blocker / "runs"))

    assert result.returncode == 1
    assert check_line(result.stdout, "runs_dir").startswith("FAIL")
    assert "claude.runs_dir" in result.stdout
    assert "[flag" in result.stdout


def test_runs_dir_flag_creates_directory(
    run_doctor: RunDoctor, home: Path, claude_bin: Path, tmp_path: Path
) -> None:
    target = tmp_path / "custom" / "runs"

    result = run_doctor(home, claude_bin, "--runs-dir", str(target))

    assert result.returncode == 0, result.stdout
    assert target.is_dir()
    assert target.stat().st_mode & 0o777 == 0o700


def test_claude_missing_from_path_is_required_failure(
    run_doctor: RunDoctor, home: Path
) -> None:
    result = run_doctor(home, None)

    assert result.returncode == 1
    line = check_line(result.stdout, "claude_cli")
    assert line.startswith("FAIL")
    assert "not found on PATH" in line


def test_clockify_ok_with_allowlist(
    run_doctor: RunDoctor,
    home: Path,
    claude_bin: Path,
    fake_api: tuple[str, dict[str, tuple[int, object]]],
) -> None:
    url, routes = fake_api
    routes["/user"] = (200, {"id": "u1", "name": "Me", "email": "me@example.com"})
    routes["/workspaces"] = (200, [{"id": "w1"}, {"id": "w2"}])

    result = run_doctor(
        home,
        claude_bin,
        CLOCKIFY_API_KEY=SENTINEL,
        CLOCKIFY_API_BASE_URL=url,
        JAKE_TOOLS_CLOCKIFY_WORKSPACES="w2",
    )

    assert result.returncode == 0, result.stdout
    assert check_line(result.stdout, "clockify").startswith("OK")
    assert "me@example.com" not in result.stdout


def test_clockify_empty_allowlist_warns(
    run_doctor: RunDoctor,
    home: Path,
    claude_bin: Path,
    fake_api: tuple[str, dict[str, tuple[int, object]]],
) -> None:
    url, routes = fake_api
    routes["/user"] = (200, {"id": "u1"})
    routes["/workspaces"] = (200, [{"id": "w1"}])

    result = run_doctor(
        home, claude_bin, CLOCKIFY_API_KEY=SENTINEL, CLOCKIFY_API_BASE_URL=url
    )

    assert result.returncode == 0
    line = check_line(result.stdout, "clockify")
    assert line.startswith("WARN")
    assert "allowlist empty (needed from phase 2)" in line


def test_clockify_unauthorized_fails(
    run_doctor: RunDoctor,
    home: Path,
    claude_bin: Path,
    fake_api: tuple[str, dict[str, tuple[int, object]]],
) -> None:
    url, routes = fake_api
    routes["/user"] = (401, {"message": f"bad key {SENTINEL}"})

    result = run_doctor(
        home, claude_bin, CLOCKIFY_API_KEY=SENTINEL, CLOCKIFY_API_BASE_URL=url
    )

    assert result.returncode == 1
    line = check_line(result.stdout, "clockify")
    assert line.startswith("FAIL")
    assert "CLOCKIFY_API_KEY" in line
    assert "401" in line
    assert SENTINEL not in result.stdout


def test_clockify_allowlist_mismatch_fails(
    run_doctor: RunDoctor,
    home: Path,
    claude_bin: Path,
    fake_api: tuple[str, dict[str, tuple[int, object]]],
) -> None:
    url, routes = fake_api
    routes["/user"] = (200, {"id": "u1"})
    routes["/workspaces"] = (200, [{"id": "w1"}])

    result = run_doctor(
        home,
        claude_bin,
        CLOCKIFY_API_KEY=SENTINEL,
        CLOCKIFY_API_BASE_URL=url,
        JAKE_TOOLS_CLOCKIFY_WORKSPACES="w1,w9",
    )

    assert result.returncode == 1
    line = check_line(result.stdout, "clockify")
    assert line.startswith("FAIL")
    assert "w9" in line


def test_jira_ok(
    run_doctor: RunDoctor,
    home: Path,
    claude_bin: Path,
    fake_api: tuple[str, dict[str, tuple[int, object]]],
) -> None:
    url, routes = fake_api
    routes["/rest/api/3/myself"] = (200, {"accountId": "a1"})

    result = run_doctor(
        home,
        claude_bin,
        JIRA_BASE_URL=url,
        JIRA_EMAIL="me@example.com",
        JIRA_API_TOKEN=SENTINEL,
    )

    assert result.returncode == 0, result.stdout
    assert check_line(result.stdout, "jira").startswith("OK")
    assert SENTINEL not in result.stdout
    assert url not in result.stdout


def test_jira_unauthorized_fails(
    run_doctor: RunDoctor,
    home: Path,
    claude_bin: Path,
    fake_api: tuple[str, dict[str, tuple[int, object]]],
) -> None:
    url, routes = fake_api
    routes["/rest/api/3/myself"] = (401, {"message": "no"})

    result = run_doctor(
        home,
        claude_bin,
        JIRA_BASE_URL=url,
        JIRA_EMAIL="me@example.com",
        JIRA_API_TOKEN=SENTINEL,
    )

    assert result.returncode == 1
    line = check_line(result.stdout, "jira")
    assert line.startswith("FAIL")
    assert "401" in line
    assert SENTINEL not in result.stdout


def test_jira_partially_configured_names_missing_variables(
    run_doctor: RunDoctor, home: Path, claude_bin: Path
) -> None:
    result = run_doctor(
        home, claude_bin, JIRA_BASE_URL="http://x.invalid", JIRA_EMAIL="a@b.c"
    )

    assert result.returncode == 1
    line = check_line(result.stdout, "jira")
    assert line.startswith("FAIL")
    assert "JIRA_API_TOKEN" in line
    assert "JIRA_EMAIL" not in line.split("unset:")[1]


def test_jira_unresolvable_host_leaks_neither_host_nor_path_nor_token(
    run_doctor: RunDoctor, home: Path, claude_bin: Path
) -> None:
    result = run_doctor(
        home,
        claude_bin,
        JIRA_BASE_URL="https://secret-tenant-xyz.invalid/secretpath",
        JIRA_EMAIL="me@example.com",
        JIRA_API_TOKEN="tok-DISTINCT-9876",
    )

    assert result.returncode == 1
    line = check_line(result.stdout, "jira")
    assert line.startswith("FAIL")
    assert "connection failed" in line
    for leaked in ("secret-tenant", "secretpath", "tok-DISTINCT", "xyz.invalid"):
        assert leaked not in result.stdout
        assert leaked not in result.stderr


def test_clockify_response_body_is_never_printed(
    run_doctor: RunDoctor,
    home: Path,
    claude_bin: Path,
    fake_api: tuple[str, dict[str, tuple[int, object]]],
) -> None:
    url, routes = fake_api
    routes["/user"] = (200, {"id": "u1"})
    routes["/workspaces"] = (200, {"body": "BODY-TEXT-MARKER"})

    result = run_doctor(
        home,
        claude_bin,
        CLOCKIFY_API_KEY="key-DISTINCT-5555",
        CLOCKIFY_API_BASE_URL=url,
    )

    assert result.returncode == 1
    assert "BODY-TEXT-MARKER" not in result.stdout
    assert "key-DISTINCT" not in result.stdout


def test_jira_error_body_is_never_printed(
    run_doctor: RunDoctor,
    home: Path,
    claude_bin: Path,
    fake_api: tuple[str, dict[str, tuple[int, object]]],
) -> None:
    url, routes = fake_api
    routes["/rest/api/3/myself"] = (500, {"message": "BODY-TEXT-MARKER"})

    result = run_doctor(
        home,
        claude_bin,
        JIRA_BASE_URL=url,
        JIRA_EMAIL="me@example.com",
        JIRA_API_TOKEN="tok-DISTINCT-9876",
    )

    line = check_line(result.stdout, "jira")
    assert "HTTP 500" in line
    assert "BODY-TEXT-MARKER" not in result.stdout


def test_runs_dir_with_loose_mode_warns(
    run_doctor: RunDoctor, home: Path, claude_bin: Path, tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    runs.chmod(0o755)

    result = run_doctor(home, claude_bin, "--runs-dir", str(runs))

    assert result.returncode == 0
    assert check_line(result.stdout, "runs_dir").startswith("WARN")


def test_environment_block_reports_home_hermes_and_xdg(
    run_doctor: RunDoctor, home: Path, claude_bin: Path, tmp_path: Path
) -> None:
    result = run_doctor(home, claude_bin, HERMES_HOME=str(tmp_path / "hermes"))

    assert f"HOME = {home}" in result.stdout
    assert f"HERMES_HOME = {tmp_path / 'hermes'}" in result.stdout
    assert f"XDG config home = {home / '.config'}" in result.stdout
    assert f"XDG state home = {home / '.local' / 'state'}" in result.stdout
    assert f"XDG cache home = {home / '.cache'}" in result.stdout

    unset = run_doctor(home, claude_bin)
    assert "HERMES_HOME = unset" in unset.stdout


@pytest.mark.slow
def test_live_real_claude_cli_under_isolated_home_is_not_authenticated(
    run_doctor: RunDoctor,
    home: Path,
) -> None:
    """The real installed ``claude`` has no credentials under an empty HOME."""
    claude = shutil.which("claude")
    if claude is None:
        pytest.skip("claude CLI is not installed")

    result = run_doctor(home, Path(claude).parent)

    assert result.returncode == 1, result.stdout
    line = check_line(result.stdout, "claude_cli")
    assert line.startswith("FAIL")
    assert "not authenticated" in line
