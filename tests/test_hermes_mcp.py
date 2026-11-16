"""Acceptance of the jake-tools MCP server through a real, installed Hermes.

Opt-in (``--slow``) and blocked unless an installation is supplied:

- ``JAKE_TOOLS_HERMES``: a Hermes launcher (``.../bin/hermes``).
- ``JAKE_TOOLS_MCP_BIN`` (optional): the server executable; default the
  ``jake-tools-mcp`` script of the environment running pytest.

Each test runs a script on Hermes' own interpreter, with the launcher's
environment minus any jake-tools entry (so Jake is not importable there and
no native plugin can stand in for the server), an isolated ``HERMES_HOME``,
``HOME``, ``HERMES_MANAGED_DIR`` and XDG dirs, and no inherited secrets. The
script drives Hermes' own MCP client: ``tools.mcp_tool_discovery.discover_mcp_tools``
reads ``mcp_servers`` from ``$HERMES_HOME/config.yaml`` and registers each
server tool as ``mcp__<server>__<tool>`` (``tools/mcp_tool_schema.py``
``mcp_prefixed_tool_name``, non-``[A-Za-z0-9_]`` -> ``_``) plus the
``list_resources``/``read_resource(uri)``/``list_prompts``/
``get_prompt(name, arguments)`` utilities when the server advertises those
capabilities (``_UTILITY_TOOL_SPECS``, ``mcp_tool_registration
._select_utility_schemas``); calls go through ``model_tools.handle_function_call``.
No model is involved. Only Clockify is faked, by a local HTTP server.

Credential contract verified in ``tools/mcp_tool_config.py``
``_build_safe_env``: a stdio server receives only ``PATH``, ``HOME``,
``USER``, ``LANG``, ``LC_ALL``, ``TERM``, ``SHELL``, ``TMPDIR``, ``XDG_*``,
``HERMES_HOME`` (the active profile's), names injected by an external secret
source, and the server config's own ``env``. A secret that is only in
Hermes' process environment does not reach the server.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from clockify_fixtures import sunfish_workspace

from jake_tools.config import load_config
from jake_tools.mcp.server import SKILLS_DIR
from jake_tools.operations.catalogue import OPERATIONS

pytestmark = pytest.mark.slow

HERMES = os.environ.get("JAKE_TOOLS_HERMES")
SERVER_BIN = Path(
    os.environ.get("JAKE_TOOLS_MCP_BIN")
    or Path(sys.executable).parent / "jake-tools-mcp"
)
PREFIX = "mcp__jake_tools__"
ISSUE_URL = "https://github.com/example/repo/issues/123"

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

# Drop the launcher's jake-tools PYTHONPATH entry (a native-plugin build).
STRIP_JAKE = """
_kept=; IFS=: read -ra _parts <<< "${PYTHONPATH-}"
for _p in "${_parts[@]}"; do case $_p in *jake-tools*|*jake_tools*) ;; *) _kept=${_kept:+$_kept:}$_p;; esac; done
export PYTHONPATH=$_kept
"""

PRELUDE = """
import importlib.util, json, os, sys
jake_importable = importlib.util.find_spec("jake_tools") is not None
from hermes_cli.plugins import discover_plugins, get_plugin_manager
discover_plugins()
from model_tools import handle_function_call
from tools.mcp_tool_discovery import discover_mcp_tools
from tools.registry import registry
mcp_names = discover_mcp_tools()
P = "mcp__jake_tools__"

def call(name, args):
    raw = handle_function_call(name, args, task_id="acceptance")
    try:
        return json.loads(raw)
    except ValueError:
        return raw

def emit(label, value):
    print("RESULT " + json.dumps([label, value]), flush=True)

emit("jake_importable", jake_importable)
emit("plugins", [p["name"] for p in get_plugin_manager().list_plugins()])
emit("discovered", sorted(mcp_names))
emit("registry", sorted(registry.get_all_tool_names()))
"""


class Clockify:
    """Local Clockify peer recording the API key of every request."""

    def __init__(self, key: str) -> None:
        self.key = key
        self.workspace = sunfish_workspace(0)
        self.seen: list[str | None] = []

    @contextmanager
    def serve(self) -> Iterator[str]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                url = urlsplit(self.path)
                key = self.headers.get("X-Api-Key")
                fake.seen.append(key)
                if key != fake.key:
                    status, payload = 401, {"message": "Unauthorized"}
                else:
                    params = {k: v[0] for k, v in parse_qs(url.query).items()}
                    status, payload = fake.workspace.respond(
                        url.path.removeprefix("/api/v1"), params
                    )
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}/api/v1"
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class Hermes:
    """An isolated Hermes home on the installed runtime."""

    def __init__(self, root: Path) -> None:
        assert HERMES is not None
        assert SERVER_BIN.is_file(), f"no jake-tools-mcp at {SERVER_BIN}"
        launcher = Path(HERMES).read_text().rstrip("\n").splitlines()
        assert launcher[-1].startswith("exec ") and any(
            "HERMES_PYTHON=" in line for line in launcher
        ), "JAKE_TOOLS_HERMES is not a Hermes launcher wrapper"
        root.mkdir(parents=True)
        self.root = root
        self.python = root / "hermes-python"
        self.python.write_text(
            "\n".join([*launcher[:-1], STRIP_JAKE, 'exec "$HERMES_PYTHON" "$@"', ""])
        )
        self.python.chmod(0o755)
        (root / "reaper.py").write_text(REAPER)
        self.home = root / "hermes-home"
        dirs = {
            name: root / sub
            for name, sub in [
                ("HOME", "home"),
                ("HERMES_HOME", "hermes-home"),
                ("HERMES_MANAGED_DIR", "managed"),
                ("TMPDIR", "tmp"),
                ("XDG_CONFIG_HOME", "xdg/config"),
                ("XDG_STATE_HOME", "xdg/state"),
                ("XDG_CACHE_HOME", "xdg/cache"),
                ("XDG_DATA_HOME", "xdg/data"),
            ]
        }
        for path in dirs.values():
            path.mkdir(parents=True)
        self.env = {
            "PATH": os.environ.get("PATH", ""),
            "LANG": "C.UTF-8",
            "HERMES_DISABLE_LAZY_INSTALLS": "1",
            **{k: str(v) for k, v in dirs.items()},
            **{
                k: os.environ[k]
                for k in ("NIX_LD", "NIX_LD_LIBRARY_PATH", "TZDIR")
                if k in os.environ
            },
        }
        self.server_env: dict[str, str] = {}

    def configure(self, **server_env: str) -> dict[str, Any]:
        """Configure ``mcp_servers.jake-tools``: stdio, the executable, ``env``."""
        self.server_env = server_env
        entry = {"command": str(SERVER_BIN), "args": [], "env": server_env}
        # JSON is YAML; no allowlist or tool filter anywhere.
        (self.home / "config.yaml").write_text(
            json.dumps({"mcp_servers": {"jake-tools": entry}})
        )
        return entry

    def seen_by_server(self) -> dict[str, str]:
        """What the server's ``load_config`` sees, per Hermes' contract."""
        keep = {"PATH", "HOME", "LANG", "TMPDIR"}
        inherited = {
            k: v for k, v in self.env.items() if k in keep or k.startswith("XDG_")
        }
        return {**inherited, "HERMES_HOME": str(self.home), **self.server_env}

    def run(self, body: str, **env: str) -> tuple[dict[str, Any], str]:
        script = self.root / "probe.py"
        script.write_text(PRELUDE + textwrap.dedent(body))
        python = str(self.python)
        completed = subprocess.run(
            [python, str(self.root / "reaper.py"), python, str(script)],
            env={**self.env, **env},
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert completed.returncode == 0, completed.stderr[-4000:]
        results = dict(
            json.loads(line.removeprefix("RESULT "))
            for line in completed.stdout.splitlines()
            if line.startswith("RESULT ")
        )
        assert results["jake_importable"] is False, "Jake importable inside Hermes"
        assert not [p for p in results["plugins"] if "jake" in p.lower()]
        return results, completed.stdout + completed.stderr

    def logs(self) -> str:
        """Every file Hermes wrote under its home, excluding our config.yaml
        and Hermes' verbatim backups of it (``backups/config/``), which hold
        the configured values by design."""
        backups = self.home / "backups" / "config"
        return "".join(
            path.read_text(errors="replace")
            for path in self.home.rglob("*")
            if path.is_file()
            and path != self.home / "config.yaml"
            and not path.is_relative_to(backups)
        )


def expected_tools(hermes: Hermes) -> list[str]:
    config = load_config(hermes.seen_by_server())
    names = [op.name for op in OPERATIONS if op.available(config)] + ["ping"]
    return sorted(PREFIX + name for name in names)


def utilities() -> list[str]:
    return [PREFIX + u for u in ("get_prompt", "list_prompts", "list_resources")] + [
        PREFIX + "read_resource"
    ]


def packaged_skills_dir() -> Path:
    """The skills shipped by the installation that owns ``SERVER_BIN``."""
    if "JAKE_TOOLS_MCP_BIN" not in os.environ:
        return SKILLS_DIR
    # A Nix mkApplication output only holds bin/ symlinks into a separate
    # environment, so resolve the executable before looking beside it.
    prefix = SERVER_BIN.resolve().parent.parent
    found = sorted(prefix.glob("lib/python*/site-packages/jake_tools/skills"))
    assert found, f"cannot locate the packaged skills of {SERVER_BIN}"
    return found[0]


def assert_registered(out: dict[str, Any], expected: list[str]) -> None:
    jake = sorted(n for n in out["registry"] if n.startswith(PREFIX))
    assert jake == sorted(expected + utilities())
    assert sorted(out["discovered"]) == jake
    bare = {n.removeprefix(PREFIX) for n in expected}
    assert not bare & set(out["registry"]), "unprefixed Jake duplicates registered"


@pytest.fixture
def hermes(tmp_path: Path) -> Hermes:
    return Hermes(tmp_path / "a")


@needs_hermes
def test_discovery_guidance_and_prompts_through_hermes(hermes: Hermes) -> None:
    clockify = Clockify("discovery-clockify-key")
    with clockify.serve() as url:
        entry = hermes.configure(
            CLOCKIFY_API_KEY=clockify.key,
            CLOCKIFY_API_BASE_URL=url,
            JAKE_TOOLS_CLOCKIFY_WORKSPACES="ws-1",
        )
        out, _ = hermes.run(
            """
            emit("resources", call(P + "list_resources", {}))
            emit("guide", call(P + "read_resource", {"uri": "skill://claude-delegation"}))
            emit("reference", call(P + "read_resource",
                {"uri": "skill://claude-delegation/references/recipes.md"}))
            emit("prompts", call(P + "list_prompts", {}))
            emit("prompt", call(P + "get_prompt", {"name": "issue_discovery",
                "arguments": {"issue_url": "@ISSUE_URL@"}}))
            """.replace("@ISSUE_URL@", ISSUE_URL)
        )
    assert set(entry) == {"command", "args", "env"}
    expected = expected_tools(hermes)
    assert PREFIX + "transcribe" not in expected
    assert not [n for n in expected if n.startswith(PREFIX + "artifact_")]
    assert_registered(out, expected)

    skills = packaged_skills_dir()
    listed = json.dumps(out["resources"])
    for skill in sorted(p.name for p in skills.iterdir() if (p / "SKILL.md").is_file()):
        assert f'"skill://{skill}"' in listed, skill
    assert "skill://claude-delegation/references/recipes.md" in listed
    guide = (skills / "claude-delegation" / "SKILL.md").read_text(encoding="utf-8")
    recipes = (skills / "claude-delegation/references/recipes.md").read_text(
        encoding="utf-8"
    )
    assert resource_text(out["guide"]) == guide
    assert resource_text(out["reference"]) == recipes

    prompts = json.dumps(out["prompts"])
    for name in ("issue_discovery", "issue_planning", "issue_maintainer_review"):
        assert f'"{name}"' in prompts, name
    prompt = json.dumps(out["prompt"])
    assert ISSUE_URL in prompt
    assert "This is discovery, not implementation." in prompt
    # Retrieval performed nothing: no upstream request, no Claude run.
    assert clockify.seen == []
    runs = load_config(hermes.seen_by_server()).runs_dir.value
    assert not runs.exists() or not any(runs.iterdir())


def resource_text(result: Any) -> str:
    """``read_resource`` renders as ``{"result": "<text parts joined>"}``
    (``mcp_tool_handlers._render_read_resource``)."""
    assert isinstance(result, dict) and set(result) == {"result"}, result
    return str(result["result"])


def jake(result: Any) -> dict[str, Any]:
    """Jake's JSON payload from Hermes' rendering of a tool result
    (``mcp_tool_handlers._render_call_tool_result``): success is
    ``{"result": text[, "structuredContent": ...]}``, ``isError`` is
    ``{"error": text}``; Jake's text block is its compact JSON payload."""
    assert isinstance(result, dict), result
    text = result.get("result", result.get("error"))
    assert isinstance(text, str), result
    payload = json.loads(text)
    assert isinstance(payload, dict), result
    return payload


@needs_hermes
def test_transcribe_is_listed_only_with_an_endpoint(hermes: Hermes) -> None:
    hermes.configure(TRANSCRIPTION_API_BASE_URL="http://127.0.0.1:9/v1")
    out, _ = hermes.run("")
    expected = expected_tools(hermes)
    assert PREFIX + "transcribe" in expected
    assert_registered(out, expected)


@needs_hermes
def test_dispatch_and_credential_isolation_through_hermes(tmp_path: Path) -> None:
    """Two Hermes homes, each with its own server ``env`` key and endpoint.

    Hermes' own process environment also carries a Clockify key and Jira
    credentials that are in neither server's ``env``: per
    ``_build_safe_env`` they must not reach the server.
    """
    leaked = {
        "CLOCKIFY_API_KEY": "hermes-process-clockify-secret",
        "JIRA_BASE_URL": "https://jira.invalid",
        "JIRA_EMAIL": "hermes@example.invalid",
        "JIRA_API_TOKEN": "hermes-process-jira-secret",
    }
    homes: dict[str, tuple[Hermes, Clockify, dict[str, Any], str]] = {}
    for label in ("a", "b"):
        hermes = Hermes(tmp_path / label)
        clockify = Clockify(f"profile-{label}-clockify-secret")
        with clockify.serve() as url:
            hermes.configure(
                CLOCKIFY_API_KEY=clockify.key,
                CLOCKIFY_API_BASE_URL=url,
                JAKE_TOOLS_CLOCKIFY_WORKSPACES="ws-1",
            )
            out, output = hermes.run(
                """
                emit("ok", call(P + "clockify_workspace_catalogue", {"keys": ["SF-101"]}))
                emit("invalid", call(P + "claude_poll", {"wait_seconds": 121}))
                emit("jira", call(P + "clockify_jira_sync", {}))
                emit("ping", call(P + "ping", {}))
                """,
                **leaked,
            )
        homes[label] = (hermes, clockify, out, output)

    secrets = [*leaked.values(), *(c.key for _, c, _, _ in homes.values())]
    for label, (hermes, clockify, out, output) in homes.items():
        assert_registered(out, expected_tools(hermes))
        assert "error" not in out["ok"], out["ok"]
        ok = jake(out["ok"])
        assert ok["results"][0]["key"] == "SF-101", ok
        assert ok["results"][0]["outcome"] == "resolved", ok
        assert "error" in out["invalid"], out["invalid"]
        assert jake(out["invalid"])["code"] == "invalid_argument", out["invalid"]
        assert "error" in out["jira"], out["jira"]
        jira = jake(out["jira"])
        assert jira["code"] == "missing_credentials", jira
        assert jira["detail"]["variables"] == [
            "JIRA_BASE_URL",
            "JIRA_EMAIL",
            "JIRA_API_TOKEN",
        ]
        assert "JIRA_API_TOKEN" in jira["message"]
        assert "error" not in out["ping"], out["ping"]
        assert jake(out["ping"])["ok"] is True, out["ping"]
        # Each connection used only its own profile's key.
        assert clockify.seen and set(clockify.seen) == {clockify.key}, label
        visible = json.dumps(out) + output + hermes.logs()
        for secret in secrets:
            assert secret not in visible, (label, secret)
