# Delegation recipes: launch and completion-only recovery

Copyable `claude_start` arguments, each run as written against the real
runtime (see "Evidence"). Copy one, replace `/abs/path/to/project` and the
file names, keep the shape. The caller supplies the working directory:
never guess it, and never let the brief stand in for one. Anything the task
needs that you do not know is a question for the user, not something to
discover through the worker.

Every recipe states the same six things, because a missing one is where
tasks went wrong:

| Part | Where |
| --- | --- |
| Working directory | `working_directory`, absolute; paths in the brief are relative to it. A resume reuses its parent's. |
| Exposed tools | `tools`: only what the job needs. |
| Explicit grants | `allowed_tools`: written out, never omitted (omitting approves every exposed tool). |
| Inputs and output path | named in the brief; one output location, nothing else writable. |
| Acceptance | what the final message and the files must show, checkable by the worker. |
| Stopping conditions | the limits, plus "if a call is denied, stop and report it". |

A denial is a decision, not a bug to route around. Never widen a grant, add
a tool or steer the worker to another tool to get past one: `claude_steer`
changes no grant. Decide whether the wider grant is acceptable, then start or
resume with it.

## 1. Read-only review

No write tool is exposed and none is granted, so the review cannot change
files. If you want the findings in a file, save `final_text` yourself; do not
add `Write` to a review.

<!-- recipe: read-only-review -->
```json
{
 "brief": "Review src/orders.py for correctness bugs. Read only: do not modify, create or delete anything, and do not run anything.\nFinal message: one line per finding as `file:line - defect - evidence`, most serious first; or `NO FINDINGS`. If you need a tool you were not given, say which and stop; do not look for another way.",
 "working_directory": "/abs/path/to/project",
 "tools": [
  "Read",
  "Grep",
  "Glob"
 ],
 "allowed_tools": [
  "Read",
  "Grep",
  "Glob"
 ],
 "max_turns": 10,
 "timeout_seconds": 300,
 "max_budget_usd": 0.25,
 "request_id": "review-orders-1"
}
```

## 2. Scoped output with a tiny checkpoint

Only for work that produces a file. Grant writes to one directory with
`Edit(...)` (see the rules below: it covers `Write` too), and have the worker
save a placeholder **first**. A wrong grant then costs one denied call, before
any drafting, and the worker stops with nothing to recover.

<!-- recipe: scoped-output -->
```json
{
 "brief": "Input: data/inventory.csv (columns sku,qty,price_cents). Output: reports/summary.md, the only file you may create or change.\n1. FIRST, before reading anything: Write reports/summary.md containing exactly `PENDING`. If that call is denied, stop at once: reply `SAVE DENIED: <path>` and do nothing else. Do not try another tool, path or command.\n2. Read data/inventory.csv. Count total units (sum of qty) and out-of-stock SKUs (qty 0).\n3. Read reports/summary.md, then Write it again as two lines: `total_units: <n>` and `out_of_stock: <n>`.\nAcceptance: the file has exactly those two lines. Final message: the path and the two numbers.",
 "working_directory": "/abs/path/to/project",
 "tools": [
  "Read",
  "Write",
  "Edit"
 ],
 "allowed_tools": [
  "Edit(./reports/**)"
 ],
 "max_turns": 10,
 "timeout_seconds": 300,
 "max_budget_usd": 0.25,
 "request_id": "summary-1"
}
```

If the checkpoint is denied (`SAVE DENIED`), nothing was drafted: fix the
grant and start a new task. The checkpoint is a placeholder the final write
replaces; it is not permission to write elsewhere, and a read-only job never
gets one.

## 3. Completion-only continuation

For a task that drafted its deliverable and then failed to save it (a grant
mistake, or one launched without a checkpoint): the work exists in the
conversation, so resume the same session and save it; do not rerun the job.

1. Inspect, don't assume. `claude_status` (detail) lists
   `permission_denials` (`tool`, `requested`) and the task's `limits`;
   `final_text` (or `claude_status {"view": "result", "part": "final_text"}`
   for the whole of it) holds what the worker reported. Check that the
   deliverable is really there and the target file is really missing.
2. State the remaining delta in one sentence: "save the existing memo to
   `reports/memo.md`".
3. Resume with the corrected grant and only the tools the delta needs. The
   recipe below exposes no `Read`, `Bash` or search tool, so the worker cannot
   re-derive the content from the sources, and it is told to save what it has.
4. Check the file yourself against what the first task produced.

There is no handoff to read unless the first task stopped on `max_turns`
(`result.handoff_status` and `handoff_text` are absent otherwise): use the
denials, `final_text` and `last_narration`, and do not ask for or invent a
handoff. After a `max_turns` stop, read its `handoff_text` first; the remaining
delta still goes in the brief.

<!-- recipe: completion-only-continuation -->
```json
{
 "brief": "Your earlier save of reports/memo.md was denied because the grant was wrong. It is corrected. Do exactly one thing: Write the memo you already produced, unchanged, to reports/memo.md. Do not read any file, redraft, or change a word, and do not write anywhere else. If the Write is denied, stop and reply `SAVE DENIED: <path>`. Final message: `SAVED reports/memo.md`.",
 "resume_task_id": "<task_id of the task that failed to save>",
 "tools": [
  "Write",
  "Edit"
 ],
 "allowed_tools": [
  "Edit(./reports/**)"
 ],
 "max_turns": 4,
 "timeout_seconds": 120,
 "max_budget_usd": 0.1,
 "request_id": "memo-save-1"
}
```

Fill in `resume_task_id` with the failed task's `task_id` (only the session's
latest task resumes) and omit `working_directory`. `tools`, `allowed_tools`
and the limits are this call's own.

## Path and Bash rules, as observed

CLI 2.1.318, `dontAsk` (no prompts: anything unapproved is denied), no
settings files, Haiku. The `claude_start` that ran each row and its outcome
are in the receipts (`tests/fixtures/delegation_recipes/receipts/`). The
runtime decides; these rows are what it did, and they are re-checked by
`uv run pytest --slow tests/test_delegation_recipes_live.py`.

Each task made the same calls: `Write out/rel.txt`, `Write` the absolute path
`<project>/out/abs.txt`, `Write outside/b.txt`, and `Edit out/existing.txt`
(after `Read`). `tools` was `["Read", "Write", "Edit"]`.

| Grant in `allowed_tools` | `out/` writes and edit | `outside/b.txt` |
| --- | --- | --- |
| `Edit(./out/**)` | all three ran | denied |
| `Edit(out/**)` | all three ran | denied |
| `Edit(/out/**)` | all three ran | denied |
| `Edit(//abs/path/to/project/out/**)` | all three ran | denied |
| `Edit(/abs/path/to/project/out/**)` | **all denied** | denied |
| `Write(./out/**)` | **all denied** | denied |

- Write path rules are `Edit(...)`: it covers `Write` and `Edit`. A
  `Write(...)` path rule approved nothing, as on earlier versions.
- A rule path is relative to the working directory (`./out`, `out`, and
  `/out` are the same place). Matching is on the resolved path, so a call that
  names the file absolutely is approved by a relative rule.
- A single leading slash is **not** absolute here: `/abs/path/...` is read
  from the project root, so it matched nothing. Absolute is `//abs/path/...`.
- An `Edit` of an existing file needs the file read first (`Read` is
  approved inside the working directory without a rule).

Bash, with `tools: ["Bash"]` and `allowed_tools: ["Bash(touch approved-*)"]`:

| Command | Result |
| --- | --- |
| `touch approved-1` | ran |
| `touch approved-3 && touch approved-4` | ran: both parts match |
| `touch approved-6 \| cat` | ran: `cat` is read-only |
| `ls` | ran with no rule (read-only) |
| `cd sub && touch approved-2` | **denied as a whole**: nothing ran |
| `touch approved-5; touch denied-5` | **denied as a whole**: `approved-5` was not created either |

A compound command ran only when every part matched a rule or was read-only;
one part outside the grant denied all of it. So grant simple commands, ask
for them one call at a time, and do not widen a rule to cover a compound.
`Bash` is still not a sandbox: whatever a rule approves can do anything the
server's user can.

## Evidence and limits

- Each recipe above is run verbatim (only `working_directory`, `model` and
  `effort` are filled in) by `tests/test_delegation_recipes_live.py`
  against synthetic local files in `tests/fixtures/delegation_recipes/`.
  Receipts keep each run's exact brief, grants, exposed tools, tool calls,
  denials and result files, with the temporary path replaced by
  `/abs/path/to/project`. Hermetic tests pin that every documented recipe
  matches its receipt.
- Haiku at low effort followed an explicit call list in these runs. That is
  an acceptance probe on a few small examples, not a measure of reliability
  and not a claim about any other model, version, account or task size.
- A tested rule is not a permission policy for your task. Static checks
  cannot say what the runtime will approve; when a path or command shape
  matters, run the check above or read the denials.
