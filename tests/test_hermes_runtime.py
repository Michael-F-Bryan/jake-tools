"""Acceptance of the native plugin inside a real, installed Hermes runtime.

Opt-in (``--slow``) and blocked unless an installation is supplied:

- ``JAKE_TOOLS_HERMES``: a Hermes launcher (``.../bin/hermes``) whose runtime
  includes this jake-tools as an entry-point plugin, e.g. the nixfiles build.
- ``JAKE_TOOLS_PROJECT_ENV_SCRIPT``: the installer's project-env loader
  (direnv + nix-direnv), for the terminal backend tests.
- ``JAKE_TOOLS_TEST_NIXPKGS``: a nixpkgs flake reference the test projects'
  dev shells use (a local ``path:`` keeps them offline).

Each test runs a script on Hermes' own interpreter, with the launcher's
environment, an isolated ``HERMES_HOME``/``HOME`` and no inherited secrets,
through Hermes' real plugin discovery and tool dispatch. A reaping parent
stands in for the container's init: orphaned background shells must be
reaped for Hermes to see them exit. Only Clockify is faked, by a local HTTP
server, because it is a real external service.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

HERMES = os.environ.get("JAKE_TOOLS_HERMES")
LOADER = os.environ.get("JAKE_TOOLS_PROJECT_ENV_SCRIPT")
NIXPKGS = os.environ.get("JAKE_TOOLS_TEST_NIXPKGS")

needs_hermes = pytest.mark.skipif(
    not HERMES, reason="blocked: set JAKE_TOOLS_HERMES to an installed Hermes launcher"
)
needs_project_env = pytest.mark.skipif(
    not (HERMES and LOADER and NIXPKGS and shutil.which("nix")),
    reason="blocked: needs JAKE_TOOLS_HERMES, JAKE_TOOLS_PROJECT_ENV_SCRIPT, "
    "JAKE_TOOLS_TEST_NIXPKGS and nix",
)

REAPER = """
import ctypes, os, subprocess, sys
assert ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0
child = subprocess.Popen(sys.argv[1:])
status = 0
while True:
    try:
        pid, code = os.waitpid(-1, 0)
    except ChildProcessError:
        break
    if pid == child.pid:
        status = os.waitstatus_to_exitcode(code)
        break
sys.exit(status)
"""

PRELUDE = """
import json, os, sys, time
from hermes_cli.plugins import discover_plugins
discover_plugins()
from model_tools import handle_function_call

def call(name, args, task="acceptance"):
    return json.loads(handle_function_call(name, args, task_id=task))

def emit(label, value):
    print("RESULT " + json.dumps([label, value]), flush=True)
"""


class Hermes:
    """An isolated profile on the installed runtime."""

    def __init__(self, root: Path) -> None:
        assert HERMES is not None
        launcher = Path(HERMES).read_text().rstrip("\n").splitlines()
        assert launcher[-1].startswith("exec ") and any(
            "HERMES_PYTHON=" in line for line in launcher
        ), "JAKE_TOOLS_HERMES is not a Hermes launcher wrapper"
        self.python = root / "hermes-python"
        self.python.write_text(
            "\n".join([*launcher[:-1], 'exec "$HERMES_PYTHON" "$@"', ""])
        )
        self.python.chmod(0o755)
        self.reaper = root / "reaper.py"
        self.reaper.write_text(REAPER)
        self.home = root / "hermes-home"
        self.user_home = root / "home"
        self.user_home.mkdir(parents=True)
        self.home.mkdir()
        self.env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(self.user_home),
            "HERMES_HOME": str(self.home),
            "TMPDIR": str(root / "tmp"),
            "LANG": "C.UTF-8",
            "HERMES_DISABLE_LAZY_INSTALLS": "1",
            **{
                k: os.environ[k]
                for k in ("NIX_LD", "NIX_LD_LIBRARY_PATH", "TZDIR")
                if k in os.environ
            },
        }
        (root / "tmp").mkdir()
        if LOADER:
            self.env["JAKE_TOOLS_PROJECT_ENV_SCRIPT"] = LOADER

    def configure(self, yaml: str) -> None:
        (self.home / "config.yaml").write_text(textwrap.dedent(yaml))

    def run(self, body: str, **env: str) -> dict[str, object]:
        script = self.home.parent / "probe.py"
        script.write_text(PRELUDE + textwrap.dedent(body))
        completed = subprocess.run(
            [str(self.python), str(self.reaper), str(self.python), str(script)],
            env={**self.env, **env},
            capture_output=True,
            text=True,
            timeout=900,
        )
        results = dict(
            json.loads(line.removeprefix("RESULT "))
            for line in completed.stdout.splitlines()
            if line.startswith("RESULT ")
        )
        assert completed.returncode == 0, completed.stderr[-4000:]
        return results


@pytest.fixture
def hermes(tmp_path: Path) -> Hermes:
    return Hermes(tmp_path)


@needs_hermes
def test_the_plugin_registers_what_the_profile_grants(hermes: Hermes) -> None:
    hermes.configure(
        """
        plugins:
          enabled: [jake-tools]
          entries:
            jake-tools:
              settings:
                capabilities: [claude]
        """
    )
    out = hermes.run(
        """
        from tools.registry import registry
        from hermes_cli.plugins import get_plugin_manager
        from agent.terminal_env_registry import plugin_backend_names
        names = registry.get_all_tool_names()
        emit("tools", sorted(n for n in names if n.startswith(("claude_", "clockify_", "transcribe", "ping"))))
        emit("skills", sorted(get_plugin_manager()._plugin_skills))
        emit("backends", plugin_backend_names())
        emit("skill", call("skill_view", {"name": "jake-tools:claude-delegation"}).get("content", "")[:2000])
        """
    )
    assert out["tools"] == [
        "claude_cancel",
        "claude_list",
        "claude_poll",
        "claude_start",
        "claude_status",
        "claude_steer",
        "claude_transcript",
        "claude_update",
    ]
    assert out["skills"] == ["jake-tools:claude-delegation"]
    assert out["backends"] == ["direnv_local"]
    assert "# Delegating to Claude" in str(out["skill"])


@needs_hermes
def test_a_disabled_plugin_registers_nothing(hermes: Hermes) -> None:
    hermes.configure("plugins:\n  enabled: []\n")
    out = hermes.run(
        """
        from tools.registry import registry
        emit("tools", [n for n in registry.get_all_tool_names() if n.startswith("claude_")])
        """
    )
    assert out["tools"] == []


@needs_hermes
def test_native_tools_use_only_the_owning_profiles_credentials(
    hermes: Hermes, tmp_path: Path
) -> None:
    hermes.configure(
        """
        plugins:
          enabled: [jake-tools]
          entries:
            jake-tools:
              settings:
                capabilities: [claude, clockify]
        """
    )
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    dump = tmp_path / "worker-env.json"
    (fake_bin / "claude").write_text(
        "#!/bin/sh\nexec "
        f'{sys.executable} -c \'import json,os; json.dump(dict(os.environ), open("{dump}", "w"))\'\n'
    )
    (fake_bin / "claude").chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()
    out = hermes.run(
        f"""
        import http.server, threading
        seen = []
        class Clockify(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append(self.headers.get("X-Api-Key"))
                body = b'{{"message":"Unauthorized"}}'
                self.send_response(401)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Clockify)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        os.environ["CLOCKIFY_API_BASE_URL"] = f"http://127.0.0.1:{{server.server_port}}/api/v1"
        from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope

        def catalogue(label):
            before = len(seen)
            result = call("clockify_workspace_catalogue", {{}})
            emit(label, {{"code": result.get("code"), "sent": seen[before:], "text": json.dumps(result)}})

        set_multiplex_active(True)
        token = set_secret_scope({{"CLOCKIFY_API_KEY": "profile-a-secret"}}, profile_home="/profiles/a")
        catalogue("profile_a")
        reset_secret_scope(token)
        token = set_secret_scope({{}}, profile_home="/profiles/b")
        catalogue("profile_b")
        reset_secret_scope(token)
        catalogue("unscoped")
        set_multiplex_active(False)

        os.environ["PATH"] = "{fake_bin}:" + os.environ["PATH"]
        token = set_secret_scope({{"CLOCKIFY_API_KEY": "profile-a-secret"}}, profile_home="/profiles/a")
        started = call("claude_start", {{"brief": "probe", "working_directory": "{work}"}})
        reset_secret_scope(token)
        for _ in range(150):
            if os.path.exists("{dump}") and os.path.getsize("{dump}"):
                break
            time.sleep(0.2)
        emit("started", started)
        emit("worker_env", sorted(json.load(open("{dump}"))))
        """,
        CLOCKIFY_API_KEY="launch-profile-secret",
        OPENAI_API_KEY="hermes-provider-secret",
        JAKE_TOOLS_CLOCKIFY_WORKSPACES="0123456789abcdef01234567",
    )
    a, b, unscoped = out["profile_a"], out["profile_b"], out["unscoped"]
    assert isinstance(a, dict) and isinstance(b, dict) and isinstance(unscoped, dict)
    assert a["code"] == "upstream_error" and a["sent"] == ["profile-a-secret"]
    assert b["code"] == "missing_credentials" and b["sent"] == []
    assert unscoped["code"] == "internal_error" and unscoped["sent"] == []
    for result in (a, b, unscoped):
        for value in ("profile-a-secret", "launch-profile-secret"):
            assert value not in str(result["text"])
    worker_env = out["worker_env"]
    assert isinstance(worker_env, list)
    assert "OPENAI_API_KEY" not in worker_env and "CLOCKIFY_API_KEY" not in worker_env
    assert "PYTHONPATH" in worker_env and "HOME" in worker_env


def _project(root: Path, name: str, shell: str, envrc: bool = True) -> Path:
    project = root / name
    project.mkdir(parents=True)
    (project / "flake.nix").write_text(
        textwrap.dedent(
            f"""
            {{
              inputs.nixpkgs.url = "{NIXPKGS}";
              outputs = {{ nixpkgs, ... }}: {{
                devShells = builtins.mapAttrs (system: pkgs: {{ default = {shell}; }}) nixpkgs.legacyPackages;
              }};
            }}
            """
        )
    )
    if envrc:
        (project / ".envrc").write_text("use flake\n")
    return project


@needs_project_env
def test_terminal_commands_load_the_directorys_project_environment(
    hermes: Hermes, tmp_path: Path
) -> None:
    hermes.configure(
        f"""
        plugins:
          enabled: [jake-tools]
        terminal:
          backend: direnv_local
          cwd: {tmp_path}
        """
    )
    projects = tmp_path / "projects"
    shell = 'pkgs.mkShellNoCC { packages = [ pkgs.hello ]; PROJECT_PROBE = "loaded"; }'
    ok = _project(projects, "ok", shell)
    blocked = _project(projects, "blocked", shell)
    broken = _project(projects, "broken", 'throw "deliberately broken dev shell"')
    plain = projects / "plain"
    plain.mkdir()
    direnv_config = next(
        line.split("=", 1)[1]
        for line in Path(str(LOADER)).read_text().splitlines()
        if line.startswith("export DIRENV_CONFIG=")
    )
    for approved in (ok, broken):
        subprocess.run(
            ["direnv", "allow", str(approved)],
            env={**hermes.env, "DIRENV_CONFIG": direnv_config},
            check=True,
            capture_output=True,
        )
    steps = [
        ("enter", {"command": f"cd {ok}"}),
        (
            "in_project",
            {
                "command": "echo probe:$PROJECT_PROBE; command -v hello; "
                "hello -g 'it'\\''s \"quoted\"'"
            },
        ),
        ("export", {"command": "export KEPT=yes"}),
        ("persisted", {"command": "echo kept:${KEPT-no} pwd:$PWD"}),
        ("leave", {"command": f"cd {plain}"}),
        (
            "left",
            {
                "command": "echo probe:${PROJECT_PROBE-unset}; command -v hello || echo no-hello"
            },
        ),
        ("workdir", {"command": "echo probe:$PROJECT_PROBE", "workdir": str(ok)}),
        ("after_workdir", {"command": "echo pwd:$PWD probe:${PROJECT_PROBE-unset}"}),
        ("unapproved", {"command": "echo ran", "workdir": str(blocked)}),
        ("broken", {"command": "echo ran", "workdir": str(broken)}),
        # Hermes' foreground local runs ignore providers' strip_env_keys; the
        # backend itself keeps Jake's credentials from commands.
        ("credentials", {"command": "env | grep -c '^CLOCKIFY_API_KEY=' || true"}),
    ]
    preview_command = 'git log --format="%h %s" -n 3 && printf "%s\\n" "it\'s quoted"'
    out = hermes.run(
        f"""
        from agent.display import build_tool_preview
        for label, args in json.loads({json.dumps(json.dumps(steps))}):
            result = call("terminal", args)
            emit(label, {{"output": result.get("output", ""), "exit_code": result.get("exit_code")}})
        bg = call("terminal", {{"command": "sleep 1; echo bg:$PROJECT_PROBE", "background": True, "workdir": "{ok}"}})
        call("terminal", {{"command": "cd {blocked}"}})
        emit("background", call("process", {{"action": "wait", "session_id": bg["session_id"], "timeout": 60}}))
        from tools.process_registry import process_registry
        emit("label", getattr(process_registry.get(bg["session_id"]), "command", None))
        call("terminal", {{"command": "cd {plain}", "workdir": "{plain}"}})
        pty = call("terminal", {{"command": "tty", "background": True, "pty": True, "workdir": "{plain}"}})
        emit("pty", call("process", {{"action": "wait", "session_id": pty["session_id"], "timeout": 60}}))
        emit("preview", build_tool_preview("terminal", {{"command": {json.dumps(preview_command)}}}, max_len=400))
        """,
        CLOCKIFY_API_KEY="launch-profile-secret",
    )

    def output(label: str) -> str:
        value = out[label]
        assert isinstance(value, dict)
        return str(value["output"])

    def exit_code(label: str) -> object:
        value = out[label]
        assert isinstance(value, dict)
        return value["exit_code"]

    assert "probe:loaded" in output("in_project")
    assert "/bin/hello" in output("in_project")
    assert 'it\'s "quoted"' in output("in_project")
    assert f"kept:yes pwd:{ok}" in output("persisted")
    assert "probe:unset" in output("left") and "no-hello" in output("left")
    assert "probe:loaded" in output("workdir")
    assert f"pwd:{plain} probe:unset" in output("after_workdir")
    assert exit_code("unapproved") == 1 and "is blocked" in output("unapproved")
    assert "ran" not in output("unapproved").splitlines()
    assert exit_code("broken") == 1
    assert "deliberately broken dev shell" in output("broken")
    assert "ran" not in output("broken").splitlines()
    assert output("credentials").strip() == "0"
    background = out["background"]
    assert isinstance(background, dict)
    assert background["status"] == "exited" and background["exit_code"] == 0
    assert "bg:loaded" in str(background["output"])
    assert out["label"] == "sleep 1; echo bg:$PROJECT_PROBE"
    # Pinned Hermes runs PTY sessions for the literal `local` backend only;
    # a plugin backend's background commands get pipes.
    pty = out["pty"]
    assert isinstance(pty, dict) and "not a tty" in str(pty["output"])
    preview = str(out["preview"])
    assert preview.startswith('git log --format="%h %s" -n 3')
    assert "nix/store" not in preview and "eval" not in preview
