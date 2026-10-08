---
name: claude-delegation
description: Delegate a bounded, non-interactive task to a separate Claude Code agent with the jake-tools `claude_start`, `claude_status` and `claude_cancel` MCP tools, then check its work. Use for self-contained coding, file, or research jobs in a known directory that can run unattended for minutes, not for anything needing a conversation or mid-task questions.
---

# Delegating to Claude

Three tools from the `jake-tools` MCP server: `claude_start`, `claude_status`,
`claude_cancel`. Hermes registers them as `mcp_<server>_<tool>`; this skill
uses the bare names.

A delegated task runs once and unattended. It cannot ask you anything, you
cannot add instructions once it has started, and it can't be resumed. It runs in its own
process, so it survives the MCP server restarting, and you poll it by ID.

## 1. Write a self-contained brief

The worker sees the `brief` and the files in `working_directory`. It sees no
conversation, no Hermes memory, no skills, no MCP servers and no settings
files. Put in the brief:

- the goal, in one or two sentences;
- the context it can't discover itself (why, constraints, names, decisions
  already made);
- the exact files or commands involved, as paths relative to
  `working_directory`;
- acceptance criteria it can check itself ("`pytest -q tests/test_x.py`
  passes", "reply with only the number");
- what to put in its final message (a summary of the changes, the command
  output, or a single value). That message comes back as `final_text`.

If the job needs back-and-forth, keep it yourself instead of delegating.

## 2. Grant the smallest set of tools

`tools` is an explicit list from `Read`, `Glob`, `Grep`, `WebSearch`,
`WebFetch`, `Edit`, `Write`, `Bash`. The default is **none**: the agent can
only think and answer. Granted tools run without prompts; anything else is
denied.

| Job | Grant |
| --- | --- |
| Answer from files | `Read`, `Glob`, `Grep` |
| Research online | `WebSearch`, `WebFetch` |
| Edit files | add `Edit`, `Write` |
| Run tests or commands | add `Bash` |

`Bash` is broad. It is not a sandbox: it can run anything the server's user
can, anywhere on the filesystem, and reach the network. Grant it only when
the task needs to run commands, and only with a `working_directory` you'd be
comfortable having changed.

## 3. Start

```json
{"brief": "Read notes.txt and reply with only the number of lines it has.",
 "working_directory": "/abs/path/to/checkout",
 "tools": ["Read"]}
```

`working_directory` must be absolute and must already exist. `model`,
`effort` (`low`…`max`), `max_turns`, `timeout_seconds` and `max_budget_usd`
default to the server's configuration and are hard limits. Lower them for
small jobs. Raise `timeout_seconds` only when the job really needs it.

The result comes back immediately:

```json
{"task_id": "20261008T085540Z-e19e7f4a", "status": "working",
 "run_dir": "/…/jake-tools/runs/20261008T085540Z-e19e7f4a"}
```

At most two tasks run at once. A third start returns `capacity_exceeded`
(`detail.live`, `detail.limit`). Wait for one of the two to finish, or cancel
one. There is no queue.

## 4. Poll

Call `claude_status` with the `task_id` every 15–60 s, depending on how long
the job is. Back off rather than polling in a tight loop. It keeps working
across server restarts. While running, the result is `status: "working"`,
with `termination_reason`, `final_text` and `usage` all null.

The run ends in one of these states:

| `status` | `termination_reason` | Meaning |
| --- | --- | --- |
| `completed` | `finished` | The run ended normally. **This does not mean the work is right.** |
| `failed` | `max_turns`, `budget`, `timeout` | It hit a limit. Any partial work is still on disk. |
| `failed` | `sdk_error` | The Claude CLI or SDK failed. The reason is in `error`. |
| `failed` | `worker_died` | The worker process vanished without recording an outcome. |
| `cancelled` | `cancelled` | You cancelled it. |

Example of a finished task:

```json
{"status": "completed", "termination_reason": "finished",
 "final_text": "4",
 "usage": {"api_calls": 3, "total_tokens": 5074, "estimated_cost_usd": 0.007051},
 "error": null}
```

In that example the file had **3** lines. The agent finished normally and
was wrong. That's why step 5 exists.

On a limit, `error` names it, e.g. `"… subtype='error_max_budget_usd' …
errors='Reached maximum budget ($0.0001)'"`. Use `usage.estimated_cost_usd`
for cost. The per-model token counts in `usage.model_usage` are more complete
than the top-level token fields on failed runs.

## 5. Check the work before reporting it

`final_text` is the agent's claim, not evidence. Before telling anyone the
job is done:

- Read the files it says it changed, or run `git diff`/`git status` in
  `working_directory`.
- Re-run the acceptance check yourself (the tests, the command, the count).
- For anything surprising, read `transcript.jsonl` in `run_dir`. It holds
  one JSON object per SDK message, including every tool call and its output.
  `brief.md`, `spec.json`, `status.json` and `result.json` are there too.

Report what you verified, separately from what the agent claimed.

## 6. Cancel, and inspect before retrying

`claude_cancel` with the `task_id` stops the task: it sends SIGTERM, waits up
to 10 s, then SIGKILL. It can take up to about 15 s to return. Cancelling an
already-finished task returns its state unchanged.

Neither cancellation nor a failure undoes anything. Files edited, commands
run and network calls made before the stop all stand. Before retrying:

1. Look at the state of `working_directory` (`git status`, the files).
2. Read the end of `transcript.jsonl` to see how far it got.
3. Either write a new brief that starts from the current state, or clean up
   first. Don't blindly re-run the same brief over half-done work.

## Errors

Errors come back with `isError: true` and `{code, message, detail}`:

| `code` | Cause |
| --- | --- |
| `invalid_argument` | `detail.argument` names the field: relative or missing `working_directory`, empty `brief`, an unknown tool, or a non-positive limit. |
| `capacity_exceeded` | Two tasks are already running. |
| `unknown_task` | No such `task_id` on this server's runs directory. Check you copied it exactly. |
| `worker_failed` | The worker process could not be started. Report it; don't loop. |
