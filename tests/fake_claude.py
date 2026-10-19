"""A stand-in for the ``claude`` CLI that speaks its stream-json protocol.

The deterministic delegated-task tests put this on ``PATH`` as ``claude`` so
the real server, the real worker and the real Claude Agent SDK run
unchanged; only the CLI binary, the true external boundary, is replaced.
It is not a model: every reply is mechanical, and no test treats it as
evidence of judgement (the ``slow`` tests use the real CLI for that).

Protocol (as the real CLI 2.1.x speaks it, observed with
``--replay-user-messages``): newline-delimited JSON both ways. The SDK sends
``control_request`` ``initialize`` first, then ``user`` messages; this
answers with ``control_response``, emits ``system``/``init`` for each turn,
echoes each user message (with its ``uuid``) when it enters the
conversation, and ends each turn with a ``result``. A user message that
arrives while a "tool" runs is echoed after the tool result, inside the
same turn, as the real CLI does. ``ask`` calls the worker's in-process
``ask_parent`` tool with a ``control_request`` ``mcp_message``. As the real
CLI does (observed on 2.1.293), it first performs the MCP handshake
(``initialize``, ``notifications/initialized``, ``tools/list``), gives each
request a fresh JSON-RPC id, and abandons a running call with
``notifications/cancelled`` for that id.

The scenario comes from a marker in the first user message:
``[[scenario:NAME]]`` or ``[[scenario:NAME:ARG]]``. Sessions persist as
``$FAKE_CLAUDE_STATE/<session>.json`` so ``--resume`` works across
processes and fails, like the real CLI, for an unknown session. Every
invocation's argv is saved to ``$FAKE_CLAUDE_STATE/argv-<n>.json``.
"""

from __future__ import annotations

import itertools
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

VERSION = "2.1.293 (Claude Code)"
STATE = Path(os.environ.get("FAKE_CLAUDE_STATE", Path.home() / ".fake-claude"))
MARKER = re.compile(r"\[\[scenario:([a-z_]+)(?::([^\]]*))?\]\]")

_out_lock = threading.Lock()
_inbox: queue.Queue[dict[str, Any] | None] = queue.Queue()
_responses: dict[str, queue.Queue[dict[str, Any]]] = {}
_responses_lock = threading.Lock()
_rpc_ids = itertools.count(1)
_handshake_lock = threading.Lock()
_handshake_done = False


def emit(message: dict[str, Any]) -> None:
    with _out_lock:
        sys.stdout.write(json.dumps(message) + "\n")
        sys.stdout.flush()


def reader() -> None:
    """Route stdin: control traffic is answered here, user messages queued."""
    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        kind = message.get("type")
        if kind == "control_request":
            emit(
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "success",
                        "request_id": message["request_id"],
                        "response": {"commands": []}
                        if message["request"].get("subtype") == "initialize"
                        else {},
                    },
                }
            )
        elif kind == "control_response":
            request_id = message["response"]["request_id"]
            with _responses_lock:
                waiter = _responses.get(request_id)
            if waiter is not None:
                waiter.put(message["response"])
        elif kind == "user":
            _inbox.put(message)
    _inbox.put(None)


def send_mcp(message: dict[str, Any]) -> tuple[str, queue.Queue[dict[str, Any]]]:
    """Send one JSON-RPC message to the ``jake`` SDK server; its reply queue."""
    request_id = f"cli_{uuid.uuid4().hex[:8]}"
    waiter: queue.Queue[dict[str, Any]] = queue.Queue()
    with _responses_lock:
        _responses[request_id] = waiter
    emit(
        {
            "type": "control_request",
            "request_id": request_id,
            "request": {
                "subtype": "mcp_message",
                "server_name": "jake",
                "message": message,
            },
        }
    )
    return request_id, waiter


def mcp_reply(response: dict[str, Any]) -> dict[str, Any]:
    if response.get("subtype") == "error":  # the SDK has no such server
        return {"error": {"message": str(response.get("error"))}}
    return response["response"]["mcp_response"]


def handshake() -> None:
    """``initialize``, ``notifications/initialized``, ``tools/list``; once."""
    global _handshake_done
    with _handshake_lock:
        if _handshake_done:
            return
        _, waiter = send_mcp(
            {
                "jsonrpc": "2.0",
                "id": next(_rpc_ids),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "claude-code", "version": VERSION},
                },
            }
        )
        if "error" not in mcp_reply(waiter.get()):
            _, waiter = send_mcp(
                {"jsonrpc": "2.0", "method": "notifications/initialized"}
            )
            waiter.get()
            _, waiter = send_mcp(
                {"jsonrpc": "2.0", "id": next(_rpc_ids), "method": "tools/list"}
            )
            waiter.get()
        _handshake_done = True


class Session:
    def __init__(self, args: dict[str, str | None]) -> None:
        self.args = args
        resume = args.get("resume")
        if resume:
            path = STATE / f"{resume}.json"
            if not path.exists():
                sys.stderr.write(f"No conversation found with session ID: {resume}\n")
                sys.exit(1)
            self.id = resume
            self.data = json.loads(path.read_text())
        else:
            self.id = args.get("session-id") or str(uuid.uuid4())
            self.data = {"user_messages": [], "cost": 0.0}
        self.reported_id = (
            str(uuid.uuid4())
            if os.environ.get("FAKE_CLAUDE_WRONG_SESSION")
            else self.id
        )
        self.turns = 0
        self.scenario, self.arg = "echo", None
        self.tool_ids = 0

    def save(self) -> None:
        STATE.mkdir(parents=True, exist_ok=True)
        (STATE / f"{self.id}.json").write_text(json.dumps(self.data))

    # --- outgoing messages ------------------------------------------------------

    def init(self) -> None:
        emit(
            {
                "type": "system",
                "subtype": "init",
                "session_id": self.reported_id,
                "cwd": os.getcwd(),
                "model": self.args.get("model"),
                "uuid": str(uuid.uuid4()),
            }
        )

    def echo(self, message: dict[str, Any]) -> str:
        text = message["message"]["content"]
        emit(
            {
                "type": "user",
                "message": {"role": "user", "content": text},
                "uuid": message.get("uuid") or str(uuid.uuid4()),
                "session_id": self.reported_id,
                "parent_tool_use_id": None,
            }
        )
        self.data["user_messages"].append(text)
        return text

    def say(self, *blocks: dict[str, Any]) -> None:
        emit(
            {
                "type": "assistant",
                "message": {
                    "model": "fake-model",
                    "id": f"msg_{uuid.uuid4().hex[:8]}",
                    "content": list(blocks),
                    "stop_reason": None,
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
                "session_id": self.reported_id,
                "uuid": str(uuid.uuid4()),
                "parent_tool_use_id": None,
            }
        )

    def tool(self, name: str, tool_input: dict[str, Any]) -> str:
        self.tool_ids += 1
        tool_id = f"toolu_{self.tool_ids:04d}"
        self.say({"type": "tool_use", "id": tool_id, "name": name, "input": tool_input})
        return tool_id

    def tool_result(self, tool_id: str, text: str, *, is_error: bool = False) -> None:
        emit(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": text,
                            "is_error": is_error,
                        }
                    ],
                },
                "uuid": str(uuid.uuid4()),
                "session_id": self.reported_id,
                "parent_tool_use_id": None,
            }
        )

    def result(self, text: str | None, *, subtype: str = "success") -> None:
        self.data["cost"] = round(self.data["cost"] + 0.01, 6)
        self.save()
        emit(
            {
                "type": "result",
                "subtype": subtype,
                "duration_ms": 5,
                "duration_api_ms": 4,
                "is_error": subtype != "success",
                "num_turns": self.turns,
                "session_id": self.reported_id,
                "total_cost_usd": self.data["cost"],
                # Like the real CLI: tokens per prompt, cost as a session total.
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "result": text,
                "stop_reason": "end_turn",
                "errors": [] if subtype == "success" else [f"fake {subtype}"],
                "uuid": str(uuid.uuid4()),
            }
        )

    # --- incoming messages -------------------------------------------------------

    def drain_mid_turn(self) -> list[str]:
        """Echo every user message that arrived during a tool call."""
        seen: list[str] = []
        while True:
            try:
                message = _inbox.get_nowait()
            except queue.Empty:
                return seen
            if message is None:
                _inbox.put(None)
                return seen
            seen.append(self.echo(message))

    def ask(
        self, arguments: dict[str, Any], *, abandon_after: float | None = None
    ) -> str:
        handshake()
        rpc_id = next(_rpc_ids)
        _, waiter = send_mcp(
            {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "method": "tools/call",
                "params": {"name": "ask_parent", "arguments": arguments},
            }
        )
        if abandon_after is not None:
            time.sleep(abandon_after)
            send_mcp(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/cancelled",
                    "params": {"requestId": rpc_id, "reason": "AbortError"},
                }
            )
            return "ERROR: the tool call was cancelled"
        reply = mcp_reply(waiter.get())
        if "error" in reply:  # e.g. no in-process server: the tool was not granted
            return f"ERROR: {reply['error']['message']}"
        return "\n".join(item.get("text", "") for item in reply["result"]["content"])

    # --- the conversation --------------------------------------------------------

    def run(self) -> None:
        while True:
            message = _inbox.get()
            if message is None:
                return
            self.turns += 1
            self.init()
            text = self.echo(message)
            if self.turns == 1 and not self.args.get("resume"):
                found = MARKER.search(text)
                if found:
                    self.scenario, self.arg = found.group(1), found.group(2)
            getattr(self, f"turn_{self.scenario}")(text)

    def turn_echo(self, text: str) -> None:
        self.say({"type": "text", "text": f"ECHO: {text}"})
        self.result(f"ECHO: {text}")

    def turn_thinking(self, text: str) -> None:
        self.say(
            {"type": "thinking", "thinking": "SECRET-THOUGHT", "signature": "s"},
            {"type": "text", "text": "Public narration."},
        )
        self.result("done")

    def turn_big(self, text: str) -> None:
        size = int(self.arg or 30000)
        for index in range(int(os.environ.get("FAKE_CLAUDE_BIG_EVENTS", "1"))):
            self.say({"type": "text", "text": f"{index}:" + "x" * size})
        tool_id = self.tool("Read", {"file_path": "big.txt"})
        self.tool_result(tool_id, "y" * 20000)
        self.result("z" * size)

    def turn_slow_tool(self, text: str) -> None:
        """Narrate, run a 'tool' for ARG seconds, then report what arrived."""
        seconds = float(self.arg or 3)
        if self.turns > 1:
            self.turn_echo(text)
            return
        self.say({"type": "text", "text": "Starting the slow step."})
        tool_id = self.tool("Bash", {"command": f"sleep {seconds}"})
        time.sleep(seconds)
        self.tool_result(tool_id, "slept")
        heard = self.drain_mid_turn()
        final = "Done. Heard: " + (" | ".join(heard) if heard else "nothing")
        self.say({"type": "text", "text": final})
        self.result(final)

    def turn_ask(self, text: str) -> None:
        if self.turns > 1:
            self.turn_echo(text)
            return
        question = self.arg or "Which colour?"
        self.say({"type": "text", "text": "I need to ask a question."})
        tool_id = self.tool("mcp__jake__ask_parent", {"question": question})
        answer = self.ask({"question": question, "choices": ["red", "green"]})
        self.tool_result(tool_id, answer)
        heard = self.drain_mid_turn()
        final = f"Answer was: {answer}" + (
            f" | Heard: {' | '.join(heard)}" if heard else ""
        )
        self.say({"type": "text", "text": final})
        self.result(final)

    def turn_ask_twice(self, text: str) -> None:
        answers: dict[str, str] = {}

        def one(question: str) -> None:
            answers[question] = self.ask({"question": question})

        threads = [
            threading.Thread(target=one, args=(q,)) for q in ("first?", "second?")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        final = f"first={answers['first?']} second={answers['second?']}"
        self.say({"type": "text", "text": final})
        self.result(final)

    def turn_ask_one_abandoned(self, text: str) -> None:
        """Two questions at once; the CLI abandons the first after a second."""
        answers: dict[str, str] = {}

        def one(question: str, abandon: float | None) -> None:
            answers[question] = self.ask({"question": question}, abandon_after=abandon)

        threads = [
            threading.Thread(target=one, args=("abandoned?", 1.0)),
            threading.Thread(target=one, args=("kept?", None)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        final = f"abandoned={answers['abandoned?']} kept={answers['kept?']}"
        self.say({"type": "text", "text": final})
        self.result(final)

    def turn_slow_echo(self, text: str) -> None:
        """Each prompt its own turn: a slow 'tool', then an echo. Messages that
        arrive meanwhile wait for their own turn (two model turns each)."""
        tool_id = self.tool("Bash", {"command": f"sleep {self.arg or 2}"})
        time.sleep(float(self.arg or 2))
        self.tool_result(tool_id, "slept")
        self.say({"type": "text", "text": f"ECHO: {text}"})
        self.result(f"ECHO: {text}")

    def turn_bad_ask(self, text: str) -> None:
        answer = self.ask({"question": ""})
        self.say({"type": "text", "text": f"Tool said: {answer}"})
        self.result(f"Tool said: {answer}")

    def turn_hang(self, text: str) -> None:
        """Start a 'tool shell' in its own session (as the CLI does), then hang."""
        marker = self.arg or "300"
        self.say({"type": "text", "text": "Starting a long tool."})
        self.tool("Bash", {"command": f"sleep {marker}"})
        subprocess.Popen(
            ["sleep", marker],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        while True:
            time.sleep(3600)

    def turn_max_turns(self, text: str) -> None:
        self.say({"type": "text", "text": "Partial progress before the limit."})
        self.result("partial answer", subtype="error_max_turns")

    def turn_remember(self, text: str) -> None:
        if "[[hold]]" in text:
            self.turn_hang(text)
        first = self.data["user_messages"][0]
        self.say({"type": "text", "text": f"REMEMBERED: {first}"})
        self.result(f"REMEMBERED: {first}")


def parse_args(argv: list[str]) -> dict[str, str | None]:
    args: dict[str, str | None] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token.startswith("--"):
            name, _, value = token[2:].partition("=")
            if value:
                args[name] = value
            elif index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                args[name] = argv[index + 1]
                index += 1
            else:
                args[name] = None
        index += 1
    return args


def main() -> None:
    argv = sys.argv[1:]
    if argv in (["-v"], ["--version"]):
        print(VERSION)
        return
    STATE.mkdir(parents=True, exist_ok=True)
    count = len(list(STATE.glob("argv-*.json")))
    (STATE / f"argv-{count + 1}.json").write_text(json.dumps(argv))
    args = parse_args(argv)
    session = Session(args)
    if args.get("resume") and not args.get("fork-session"):
        session.scenario = "remember"
    threading.Thread(target=reader, daemon=True).start()
    session.run()


if __name__ == "__main__":
    main()
