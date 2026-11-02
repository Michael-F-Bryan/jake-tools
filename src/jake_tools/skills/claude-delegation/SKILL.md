---
name: claude-delegation
description: Delegate a bounded task to a separate Claude Code agent with the jake-tools `claude_*` MCP tools, watch it, steer it, answer its questions, collect and check the result, and resume the conversation later. Use for self-contained coding, file or research jobs in a known directory.
---

# Delegating to Claude

Tools from the `jake-tools` MCP server (Hermes registers them as
`mcp_<server>_<tool>`; this skill uses the bare names):

| Tool | Use |
| --- | --- |
| `claude_start` | Start a task, or resume a finished one as a new task. |
| `claude_poll` | Wait (up to 120 s) until the task needs you; a compact snapshot. |
| `claude_update` | Answer the task's questions (only with `allow_questions`). |
| `claude_steer` | Send a running task an extra message. |
| `claude_status` | Detail now (config, usage, denials, messages), or the full result text. |
| `claude_transcript` | Page through the task's history (its normalised events). |
| `claude_cancel` | Stop the task and its whole process tree. |
| `claude_list` | Find tasks, newest first (labels, status, resumable), e.g. after a restart. |

The shortest path: `claude_start` → `claude_poll` with the
`observation_token` it returned → (`claude_update` / `claude_steer` if the
poll says so) → `claude_poll` again with the new token → read `result` →
check the work → optionally `claude_start(resume_task_id=...)`. Read
`claude_status` or `claude_transcript` only when you need the detail.

Every task runs in its own process and survives MCP server restarts; task
IDs, tokens, cursors and message IDs stay valid across them. `claude_*`
results leave out fields that are null: an absent field is null.

## Discovery and side effects

Use the host's tool search/list and schema-description facility to discover
`claude_*` tools and inspect current arguments; prefixes are host-dependent. Read
`skill://claude-delegation` for this server's packaged guidance if the installed
copy may be stale. No local checkout or agent-side credential lookup is needed;
the worker uses the server's configured Claude installation/authentication.
Report unavailable tools or authentication rather than guessing credentials.

`claude_status`, `claude_poll` and `claude_transcript` observe tasks without
instructing the worker to change its work. `claude_start` runs a real task;
`claude_steer` and `claude_update` change its instructions/input, and
`claude_cancel` stops it without rollback. There is no preview/apply parameter:
a read-only review requires a read-only brief and minimal grants. Authorisation
for delegation does not imply permission to commit, push or perform business
writes. Put the approved scope in the brief and verify resulting state yourself.

## 1. Write a self-contained brief

The worker sees the `brief` and the files in `working_directory`. It sees no
conversation, no Hermes memory, no skills, no MCP servers and no settings
files. Put in the brief: the goal; context it can't discover; exact paths
(relative to `working_directory`) and commands; acceptance criteria it can
check itself; and what its final message must contain (that comes back as
`final_text`).

## 2. Grant the smallest set of tools

Two separate lists:

- `tools` is the Claude Code built-in tools the worker can **see**, by
  name: for example `Read`, `Glob`, `Grep`, `Edit`, `Write`, `Bash`,
  `WebFetch`, `WebSearch`, `NotebookEdit`, `Monitor`. The default is
  **none**: the worker gets no tools at all. Any well-formed name is
  passed to Claude Code, which decides what exists. This server keeps no
  list of tool names. If the session doesn't provide a tool you named, the
  task ends at once as `failed` / `tools_unavailable`, and `error` names
  the missing tools. The tool may be unknown to that Claude Code version,
  or unavailable to the account or surface. For example, `Artifact` is off
  by default in Agent SDK sessions, and delegated tasks are those.
- `allowed_tools` is the permission rules that **approve** calls without a
  prompt. Each rule must be for a tool in `tools`:
  - `"Read"` approves every `Read` call.
  - `"Bash(git status)"` approves only that command.
  - `"Bash(uv run pytest *)"` approves that command with any arguments.
  - `"Edit(/src/**)"` approves edits to paths matching the pattern.
  - `"WebFetch(domain:example.com)"` approves fetches from that domain.

  If you omit `allowed_tools`, every tool in `tools` is approved outright.
  A rule may not contain a comma.

Anything not approved is **denied**. The worker never waits on a prompt.
Some calls run even with no rule, because Claude Code never asks about
them: reads inside `working_directory`, and read-only Bash commands such as
`ls`, `cat` or `git status`. So `tools: ["Bash"]` with
`allowed_tools: ["Bash(uv run pytest *)"]` lets the worker run the tests
and inspect files. It cannot run `touch`, `rm`, `git commit` or other
commands that change things.

Approval patterns that constrain arguments are fragile, because the
command text has to match the pattern as written. `Bash` is not a sandbox:
an approved command can do anything the server's user can, anywhere, with
network access. That includes `uv run pytest`, which runs project code.

`allow_questions: true` gives the worker one more tool, `ask_parent`, to ask
**you** a question. Set it only when you will watch with `claude_poll`: an
unanswered question blocks the task until its timeout. It grants no
permissions; a worker cannot use it to get more tools.

## 3. Start

```json
{"brief": "…", "working_directory": "/abs/path",
 "tools": ["Read", "Grep", "Bash"],
 "allowed_tools": ["Read", "Grep", "Bash(uv run pytest *)"],
 "allow_questions": true, "request_id": "review-2026-10-19-1"}
```

`model`, `effort` (`low`…`max`), `max_turns`, `timeout_seconds` and
`max_budget_usd` default to the server's configuration and are hard limits
for this invocation:

- `timeout_seconds` is wall-clock time for the whole task, **including time
  spent waiting for your answers**.
- `max_turns` counts model turns across the whole task, steered turns
  included.
- `max_budget_usd` is this invocation's spend.

Steering never changes any limit, grant or rule. The result returns once
the task is durably created: `task_id`, `status: "working"`, `session_id`,
`run_dir`, `replayed: false`, an `observation_token` for `claude_poll`, and
`config`, the configuration as normalised (defaults applied).
`config.allowed_tools` is your request's list (absent when you omitted it,
`[]` when you sent an empty one); `config.approval_rules` is what was passed
to the runtime. `claude_status`'s `limits.exposed_tools` is what the session
actually exposed, as the runtime reported it. Whether a rule approves a
particular call is the runtime's decision; it reports the calls it refuses.

`request_id` (your choice, 1–128 of `A-Z a-z 0-9 . _ : -`) makes the call safe
to retry after a timeout or disconnect. The same `request_id` with the same
request returns the original task (`replayed: true`, its current `status`),
even when capacity is full. The same `request_id` with a different request
is `request_conflict`.

At most two tasks run at once (`capacity_exceeded`, no queue).

## 4. Watch with `claude_poll`

Call `claude_poll {"task_id": …, "token": <the latest observation_token>}`.
It waits up to `wait_seconds` (default 30, max 120; Hermes allows a call
300 s) and returns a **compact snapshot**, never a page of history:

- `status`, `termination_reason`, `wake_reason`, `waited_seconds`;
- `progress` (one line), `last_activity_at` (real activity, not a
  heartbeat), `running_tool` (the call it is waiting on, and since when),
  `turns_used`/`max_turns`;
- `pending_questions` (every open one, until answered or expired);
- `changes` since your token: grouped `denials`, `warnings`, `steering`
  and `questions` changes, and counts of `events` and routine `activity`;
- `result`, once the task has ended (section 7);
- a new `observation_token`. Pass it to the next poll.

**When it returns.** At once for a pending question or an ended task,
whatever `until` says: a question can block the outcome you are waiting
for. With `until: "attention"` (the default) also for a new distinct
denial, a material warning (turn limit near, rate limit) or steering that
could not be delivered. Routine work (narration, tool calls, API retries,
counters) does not wake it. `until: "terminal"` waits only for questions
and the end; `until: "progress"` also wakes on routine activity, for when
you really want checkpoints. At the deadline it returns the current
snapshot (`wake_reason: "deadline"`); judge it by `last_activity_at` and
`running_tool`. The wait is not the task's timeout, and cancelling a poll
never affects the task. Don't sleep in a shell between polls: poll again.

**Tokens.** Reading consumes nothing: retrying with the same token returns
the same changes (plus newer ones), and several readers can each keep
their own. Without a token you get the current state, open questions and
result, not old warnings. Advancing a token never dismisses an unanswered
question. A token that is malformed, from another task or no longer valid
is not an error: the poll resynchronises (`token_status:
"resynchronised"`, `resync_reason`) and returns a fresh token.

**Bounds.** Lists keep at most 10 entries; `changes.omitted` counts the
rest and `changes.retrieve` is the exact `claude_transcript` call that
reads the whole range. Repeated identical denials are one group with a
`count`; `new: false` marks a repeat of one you have already seen (it does
not wake you). A denial's `source` is `runtime` (the CLI reported it, with
its `runtime_reason`, e.g. `mode`: no rule approved it under `dontAsk`) or
`inferred_from_tool_error` (only the call's error text said so, e.g. its
own guard's `Blocked: …`).

History is `claude_transcript` (its own `cursor`, not a token; `limit` up
to 200, `max_text_chars` up to 50 000). Event kinds: `assistant_text`
(public narration only, never private thinking), `tool_use`,
`tool_result`, `turn_result`, `permission_denied`, `notice`, `message_*`,
`input_*`, `cancel_requested`, `handoff_*` and `finished`. Tool results are
stored up to 8000 characters; `raw_transcript_path` is the raw SDK stream,
for diagnosis only.

### Watch several tasks (select-style)

Supply `tasks` instead of `task_id`/`token`, each with its own token (task
IDs, not Claude session IDs):

```json
{"tasks": [{"task_id": "…", "token": "…"}, {"task_id": "…"}],
 "wait_seconds": 60, "until": "attention"}
```

The result is `{tasks: [<snapshot>, …], ready: [task_id, …], wake_reason:
ready|deadline, waited_seconds}`: every watched task's snapshot in request
order, under one shared deadline, with `ready` naming the ones that woke
it. Take each task's new token. 1–16 distinct tasks; every ID is checked
before waiting. Remove ended tasks after collecting them, or they keep
returning at once. Cancelling the wait does not cancel any worker.

## 5. Steer

`claude_steer {"task_id", "message", "message_id"}` adds a user message to
the same conversation. `message_id` is yours: retrying with the same ID and
text is safe (`replayed: true`); the same ID with different text is
`request_conflict`. `state` progresses:

| `state` | Meaning |
| --- | --- |
| `queued` | Stored; not yet sent to Claude. |
| `delivered` | Written to Claude's input. |
| `in_conversation` | Claude Code added it to the conversation. Sent mid-turn, it lands after the running tool call finishes (never interrupting it); sent at a turn's end, it starts a new turn. |
| `not_delivered` | The task ended first; `reason` says why. |

None of these proves the model acted on it. Check its later events and
output. An ended task returns `task_not_running`.

## 6. Answer questions

With `allow_questions`, the task may go `input_required`. A poll returns at
once and lists `pending_questions[]`: `request_id`, `question` (cut to 2000
characters, `question_truncated`; `claude_status`'s `pending_inputs` has it
whole) and optional `choices` (suggestions only). Reply with:

```json
{"task_id": "…", "responses": {"q1-1a2b3c4d": {"answer": "green"}}}
```

or `{"action": "decline"}`. You may answer some and leave others pending.
Each key reports `accepted`, `duplicate` (already applied identically),
`conflict` (settled differently), `not_pending` (expired when the task ended
or its tool call was abandoned) or `unknown`. The task continues in the same
turn with your answer.

## 7. Collect the result, then check it

Once ended, the poll's `result` holds `outcome`, `termination_reason`,
`final_text`, `error`, `session_id`, `resumable`, `invocation_cost_usd`
and, after a `max_turns` stop, `handoff_status`/`handoff_text`:

- `final_text` is inline in full up to 4000 characters. Longer, it is the
  first 2000 with `final_text_truncated: true` and `final_text_chars`, and
  `retrieve` is the exact call for the rest:
  `claude_status {"task_id", "view": "result", "part": "final_text"}`
  returns it in chunks (`offset`, `max_chars` up to 30 000; follow
  `next_offset` until it is absent). `part: "handoff"` and `part: "error"`
  work the same way. This works from any machine; `final_text_path` (in
  `claude_status`) is only the server's local file.
- A failed or cancelled task can still have useful `final_text`, and
  `last_narration` shows how far it got.
- `diagnostics` are mechanically detected inconsistencies, not judgements:
  `completed_without_final_text`, `no_model_turn_recorded`,
  `background_tasks_running_at_end` (it claimed to finish while a
  background command was still running; that command is stopped when the
  task ends), `result_before_brief_ignored` (the runtime answered an old
  background task with an empty result before taking the brief; the
  worker waited for the brief's own turn).
- `handoff` (only after a `max_turns` stop): the same session, resumed once
  with no tools for a single bounded reply, reports completed work, paths
  and commits, checks run, blockers, uncertain state and the remaining
  delta. `written`, `failed` or `skipped` (too little time, budget, or
  unknown spend); it is `worker_reported`: a claim, like `final_text`. The
  task stays `failed`/`max_turns`. Never after a cancel, timeout or budget
  stop.

`claude_status` (detail view) adds `usage`, `cost_complete` (false when
the task ended mid-prompt, so its cost was never reported), the full
`handoff` record and `transcript_path`.

| `status` | `termination_reason` | Meaning |
| --- | --- | --- |
| `completed` | `finished` | Ended normally. **Not proof the work is right.** |
| `failed` | `max_turns`, `budget`, `timeout` | Hit a limit; partial work stays on disk. |
| `failed` | `sdk_error` | The CLI/SDK failed (auth problems show here); see `error`. |
| `failed` | `resume_failed` | Claude could not resume the session (it may have been deleted). Nothing ran. |
| `failed` | `tools_unavailable` | The session lacked a tool in `tools`; `error` names it and what was exposed. Stopped at the start. Retry without it. |
| `failed` | `worker_died` | The worker vanished without recording an outcome. |
| `cancelled` | `cancelled` | You cancelled it. |

`final_text` is the agent's claim, including any "tests pass". Before you
report the work as done, read the changed files or `git diff` and re-run the
acceptance checks yourself. Report what you verified separately from what the
agent claimed.

## 8. Resume

`claude_start {"brief": "…", "resume_task_id": "<task>", "tools": […]}` continues
that task's Claude conversation as a **new** task, with the same
`session_id` and `resumed_from` set:

- Nothing is inherited except the conversation. `tools`,
  `allowed_tools`, limits and `allow_questions` are this call's own.
- `working_directory` must be the parent's (omit it to use the parent's).
- Only the session's latest task can be resumed. An older one returns
  `resume_unavailable` with `detail.latest_task_id`.
- A still-running parent, or another live task on the session, returns
  `session_busy`.
- A missing session ends as `resume_failed`; it never silently starts a
  fresh conversation.
- `invocation_cost_usd` is this invocation's spend.
  `invocation_cost_exact: false` means the parent ended mid-turn, so the
  figure may be overstated.

## 9. Cancel, and inspect before retrying

`claude_cancel` records `cancel_requested_at`, sends SIGTERM to every process
group the task used, waits up to 10 s, then sends SIGKILL (up to about 15 s
in total). `termination` says what it established: `confirmed` (every
process seen gone; `status: cancelled`), `unconfirmed` (`status: failed`,
`error` names what survived), `already_ended`, or `ended_before_cancel`
(the task reached its own end first; that outcome stands). Queued messages
become `not_delivered`, pending questions expire, and partial output is
kept. `rollback` is always `none`: before retrying, look at
`working_directory`, the result's `last_narration` and the transcript.

## Errors

Errors come back with `isError: true` and `{code, message, detail}`:

| `code` | Cause |
| --- | --- |
| `invalid_argument` | `detail.argument` names it. Causes: bad path, empty brief, a malformed tool name, a rule in `tools`, an `allowed_tools` rule that is malformed or names a tool not in `tools`, bad limit, a value outside its schema bounds or enum, a bad transcript cursor, malformed response. (A bad poll token is not an error: it resynchronises.) |
| `capacity_exceeded` | Two tasks are already running. |
| `unknown_task` | No such task ID in this server's runs directory. |
| `request_conflict` | `request_id`/`message_id` reused with a different payload. |
| `session_busy` | The session to resume is being written by a live task. |
| `resume_unavailable` | No session on record, a failed resume, a different directory, or not the latest task (`detail.latest_task_id`). |
| `task_not_running` | The task can't take messages any more. |
| `worker_failed` | The worker process could not start. Report it; don't loop. |

## Supervision judgement

- One implementation owner per piece of work. Stabilise an interface
  before starting parallel consumers of it.
- Ask for early durable checkpoints in the brief (commits, files), so a
  limit or cancel leaves usable evidence.
- Poll with the latest token and a long `wait_seconds`; use `request_id`
  on `claude_start` and `message_id` on `claude_steer` so retries are safe.
- Repeated steering of the same point, or a denial group whose `count`
  keeps growing, means the brief or the grant is wrong: stop and
  re-evaluate rather than pushing harder. Never broaden a grant just to
  get past a denial without deciding that the wider grant is acceptable.
- To resume, give the concrete remaining delta (from `handoff`,
  `last_narration` and your own checks), not "continue".
- `completed` is not verified. A cancel does not roll anything back:
  inspect the working directory before retrying.

## Retention, trust and versions

- **Retention:** run directories (results, events, transcripts) are kept
  until someone deletes them; nothing is pruned. Claude Code keeps its
  sessions under its own retention, so a resume can fail with
  `resume_failed` even when the run directory still exists.
- **Sensitivity:** `brief.md`, `events.jsonl`, `transcript.jsonl`,
  `final.md` and `handoff.md` hold whatever the task saw or wrote,
  including tool output; treat a runs directory as sensitive.
- **Accounting:** cost comes from the CLI's running session total. A turn
  cut off mid-prompt (a cancel, a timeout, a stop the worker enforces)
  never reports its cost, so `invocation_cost_usd` can understate it;
  `invocation_cost_exact: false` marks a resume whose baseline is uncertain.
- **Notifications:** this server sends no notifications and keeps no
  subscriptions. Nothing wakes the parent except its own poll returning. Whether a host surfaces
  anything else is the host's business.
- **Trust:** task IDs carry 128 random bits, but the trust boundary is the
  local user. Anyone who can call this stdio server, or read its runs
  directory, can observe and steer every task in it.
- **Versions:** older runs stay readable (`legacy_run: true`, no events,
  no steering). Restart every server sharing a runs directory together
  when upgrading: an older server does not understand `input_required` or
  the longer task IDs.
- **Native MCP Tasks:** see the next section. Everything above is the
  ordinary-tool contract, which every client gets.

## Native MCP Tasks (2026-07-28)

Only for a client on a 2026-07-28 connection that declares
`io.modelcontextprotocol/tasks` in the request's own `_meta` capabilities.
Hermes connects with `initialize`, so it always gets the ordinary tools, and
nothing below changes them. Hosts are not verified against this server; only
the server side is tested.

- `claude_start` returns `{"resultType": "task", "taskId", "status",
  "createdAt", "lastUpdatedAt", "ttlMs": null, "pollIntervalMs"}` once the
  task is durable. `status` is normally `working` (a replayed `request_id`
  returns the original task as it is now). `taskId` is the same `task_id`
  every `claude_*` tool accepts. A start that fails (bad argument,
  capacity) is still an ordinary `isError` tool result, not a task.
- `tasks/get` returns `working`, `input_required` (with `inputRequests`),
  `completed` (with `result`, a `CallToolResult` whose `content` is the
  final text, or the error when there is none, and whose
  `structuredContent` is `claude_status`'s detail `result`, with up to
  20 000 characters of final text) or
  `cancelled`. Every worker outcome other than `finished` (limits, SDK
  errors, a dead worker) is `completed` with `result.isError: true`.
  `failed` is reserved for JSON-RPC errors and is never returned.
- `inputRequests` are keyed by the same `request_id` as `pending_inputs`.
  Each is an `elicitation/create` form asking for one string, `answer`
  (`choices` appear as `examples`). Answer with `tasks/update
  {"taskId", "inputResponses": {"<request_id>": {"action": "accept",
  "content": {"answer": "…"}}}}`, or `"action": "decline"`/`"cancel"`
  (both decline). Partial answers are fine. Keys that are not outstanding
  are ignored; a bad action or a missing `content.answer` is `-32602`.
- `tasks/cancel` acknowledges at once; poll `tasks/get` for `cancelled`.
  It is the same cancellation as `claude_cancel`.
- `pollIntervalMs` is a suggestion (5000). Run directories are never
  purged, so `ttlMs` is `null`.
- Not part of the extension, so use the ordinary tools for them:
  `claude_steer`, `claude_poll` (compact supervision), `claude_transcript`,
  and resume.
  `notifications/tasks` subscriptions are not offered.
- Errors: no capability on the request is `-32021`; an unknown task ID is
  `-32602`; an unreadable task is `-32603`.
