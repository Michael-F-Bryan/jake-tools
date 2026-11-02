"""Acceptance of the native plugin inside a real, installed Hermes runtime.

Opt-in (``--slow``) and blocked unless an installation is supplied:

- ``JAKE_TOOLS_HERMES``: a Hermes launcher (``.../bin/hermes``) whose runtime
  includes this jake-tools as an entry-point plugin, e.g. the nixfiles build.

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
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.slow

HERMES = os.environ.get("JAKE_TOOLS_HERMES")

needs_hermes = pytest.mark.skipif(
    not HERMES, reason="blocked: set JAKE_TOOLS_HERMES to an installed Hermes launcher"
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

    def configure(self, yaml: str) -> None:
        (self.home / "config.yaml").write_text(textwrap.dedent(yaml))

    def run(self, body: str, **env: str) -> dict[str, Any]:
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
    assert out["backends"] == []  # the plugin leaves the terminal alone
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


def _wait(session: str) -> str:
    return f'call("process", {{"action": "wait", "session_id": {session}["session_id"], "timeout": 60}})'


@needs_hermes
def test_under_the_local_backend_the_plugin_keeps_local_behaviour(
    hermes: Hermes, tmp_path: Path
) -> None:
    hermes.configure(
        f"""
        plugins:
          enabled: [jake-tools]
          entries:
            jake-tools:
              settings:
                capabilities: [clockify]
        terminal:
          backend: local
          cwd: {tmp_path}
        """
    )
    out = hermes.run(
        f"""
        # Each PTY command outlives its output by a second: pinned Hermes' PTY
        # reader can drop what a process writes just before it exits, with or
        # without this plugin (measured; not what this test is about).
        pty = call("terminal", {{"command": "tty; sleep 1", "background": True, "pty": True}})
        emit("pty", {_wait("pty")})
        reader = call("terminal", {{"command": "echo ready; read line; echo got:$line; sleep 1", "background": True, "pty": True}})
        # Input that arrives before the program reads is discarded, as on any terminal.
        for _ in range(300):
            if "ready" in call("process", {{"action": "poll", "session_id": reader["session_id"]}}).get("output_preview", ""):
                break
            time.sleep(0.1)
        emit("submit", call("process", {{"action": "submit", "session_id": reader["session_id"], "data": "hello"}}))
        emit("stdin", {_wait("reader")})
        emit("foreground", call("terminal", {{"command": "pwd; env | grep -c '^CLOCKIFY_API_KEY=' || true"}}))
        bg = call("terminal", {{"command": "env | grep -c '^CLOCKIFY_API_KEY=' || true", "background": True}})
        emit("background_env", {_wait("bg")})
        """,
        CLOCKIFY_API_KEY="launch-profile-secret",
    )

    def output(label: str) -> str:
        value = out[label]
        assert isinstance(value, dict), value
        return str(value.get("output", ""))

    assert "/dev/pts/" in output("pty")
    assert "got:hello" in output("stdin")
    assert output("foreground").split() == [str(tmp_path), "1"]
    # The documented baseline (README "Credentials and terminal commands"):
    # a Jake secret already in Hermes' process environment reaches terminal
    # commands, as without the plugin; pinned Hermes gives a plugin no
    # supported way to add it to the scrub (jake-tools#4).
    assert output("background_env").strip() == "1"


@needs_hermes
def test_native_compact_supervision_contract(hermes: Hermes, tmp_path: Path) -> None:
    """Real Hermes dispatch, worker and SDK; fake CLI proves mechanics only."""
    from conftest import install_fake_claude

    fake_bin, _ = install_fake_claude(tmp_path / "fake")
    work = tmp_path / "work"
    work.mkdir()
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
        f"""
        import jake_tools
        emit("source", jake_tools.__file__)
        from tools.registry import registry
        schema = registry.get_schema("claude_poll")
        emit("schema", schema)
        os.environ["PATH"] = "{fake_bin}:" + os.environ["PATH"]

        def start(brief, **extra):
            value = call("claude_start", {{"brief": brief, "working_directory": "{work}",
                "model": "fake-model", "max_turns": 5, "max_budget_usd": 1,
                "timeout_seconds": 60, **extra}})
            assert "task_id" in value, value
            return value

        def poll(started, **extra):
            return call("claude_poll", {{"task_id": started["task_id"],
                "token": started["observation_token"], "wait_seconds": 30,
                "until": "terminal", **extra}})

        emit("invalid", call("claude_poll", {{"wait_seconds": 121}}))
        empty_raw = handle_function_call("claude_list", {{}}, task_id="acceptance")
        emit("empty_raw", empty_raw)
        question = start("[[scenario:ask:Which colour?]] go", allow_questions=True)
        try:
            pending = poll(question)
            emit("pending", pending)
            qid = pending["pending_questions"][0]["request_id"]
            emit("resync", poll(question, token="invalid", wait_seconds=0))
            steer = {{"task_id": question["task_id"], "message": "Keep the answer concise",
                "message_id": "native-steer"}}
            emit("steer", call("claude_steer", steer))
            emit("steer_retry", call("claude_steer", steer))
            emit("answer", call("claude_update", {{"task_id": question["task_id"],
                "responses": {{qid: {{"answer": "green"}}}}}}))
            emit("answered", poll(question))
        finally:
            call("claude_cancel", {{"task_id": question["task_id"]}})

        large = start("[[scenario:big:30000]] go")
        try:
            final = poll(large)
            emit("large", final)
            chunks = []
            offset = 0
            while True:
                chunk = call("claude_status", {{"task_id": large["task_id"], "view": "result",
                    "offset": offset, "max_chars": 7000}})
                assert chunk["available"], chunk
                chunks.append(chunk["text"])
                if "next_offset" not in chunk:
                    break
                offset = chunk["next_offset"]
            emit("retrieved", {{"text": "".join(chunks), "chunks": len(chunks)}})
            emit("ended_cancel", call("claude_cancel", {{"task_id": large["task_id"]}}))
        finally:
            call("claude_cancel", {{"task_id": large["task_id"]}})

        running = start("[[scenario:slow_tool:30]] go")
        emit("cancel", call("claude_cancel", {{"task_id": running["task_id"]}}))
        """
    )
    schema = json.dumps(out["schema"])
    assert '"token"' in schema and '"cursor"' not in schema
    assert '"maximum": 120' in schema
    assert out["invalid"]["code"] == "invalid_argument"
    raw = out["empty_raw"]
    assert isinstance(raw, str)
    assert "next_before" not in json.loads(raw)
    assert raw == json.dumps(json.loads(raw), separators=(",", ":"), ensure_ascii=False)
    assert out["pending"]["wake_reason"] == "question"
    assert "events" not in out["pending"] and "limits" not in out["pending"]
    assert out["resync"]["pending_questions"] == out["pending"]["pending_questions"]
    assert out["resync"]["token_status"] == "resynchronised"
    assert out["steer_retry"]["replayed"] is True
    assert out["steer"]["message_id"] == out["steer_retry"]["message_id"]
    assert out["answer"]["results"][0]["outcome"] == "accepted"
    assert out["answered"]["status"] == "completed"
    assert out["large"]["result"]["final_text_truncated"] is True
    assert out["retrieved"]["chunks"] > 1
    assert out["retrieved"]["text"] == "z" * 30000
    assert len(out["retrieved"]["text"]) == out["large"]["result"]["final_text_chars"]
    assert out["ended_cancel"]["termination"] == "already_ended"
    assert out["cancel"]["termination"] == "confirmed"
    assert out["cancel"]["rollback"] == "none"
