---
name: clockify-jira-sync
description: Reconcile Jira work items into Clockify projects and tasks with the jake-tools `clockify_jira_sync` MCP tool. Use when asked to sync, tidy, or check Clockify against Jira, or when Clockify tasks are missing, misnamed, closed too early, or still open after the Jira issue is done.
---

# Jira → Clockify sync

One tool, `clockify_jira_sync`, from the `jake-tools` MCP server. Hermes
registers it as `mcp_<server>_clockify_jira_sync`; this skill uses the bare
name. It needs Clockify and Jira credentials in the server's environment, so
it only works where Jira is configured.

It creates projects and tasks, renames them, reactivates tasks, and marks
tasks done. It never deletes anything, and it has no other Clockify or Jira
write surface.

## The loop: preview, read, apply with the digest

1. **Preview.** Call with no arguments to cover active Jira work assigned to
   the authenticated user, or `issues: ["SF-304", ...]` for exact keys
   regardless of assignee. Preview is read-only. `apply` defaults to false.
2. **Read the report** before doing anything else (fields below). If
   `actions` is empty and `conflicts` is empty, there is nothing to do; say
   so and stop.
3. **Apply** by calling again with the *same* `issues` (or none), plus
   `apply: true` and `plan_digest` copied from the preview. Only apply when
   the person asked for changes, or has seen the planned actions and agreed.
4. **Check the apply report**: every action should have `applied: true` and
   `verified: true`, and `failure` should be `null`.

Without a `plan_digest`, apply refuses with `invalid_argument` and writes
nothing.

## Reading a report

```json
{
  "mode": "preview",
  "scope": {"kind": "assigned-active", "jira_project": "SF", "issue_keys": []},
  "inventory": {"active_issues": 1, "jira_issues": 3, "projects": 1, "tasks": 2},
  "actions": [
    {"kind": "REACTIVATE_TASK", "jira_key": "SF-427",
     "current_name": "SF-427 Evaluate PX4 external control methods",
     "desired_name": "SF-427 Evaluate PX4 external control methods",
     "project_key": "SF-131", "project_id": "p-131", "task_id": "t-427",
     "jira_status": "", "message": "", "applied": false, "verified": false},
    {"kind": "MARK_TASK_DONE", "jira_key": "SF-304", "jira_status": "Done",
     "task_id": "t-304", "applied": false, "verified": false}
  ],
  "conflicts": [],
  "plan_digest": "17842a21…",
  "applied": false,
  "failure": null
}
```

- `actions[].kind` is one of `CREATE_PROJECT`, `RENAME_PROJECT`,
  `CREATE_TASK`, `REACTIVATE_TASK`, `RENAME_TASK`, `MARK_TASK_DONE`. Renames
  show `current_name` → `desired_name`; creates have an empty
  `current_name`.
- `conflicts` lists what the plan refuses to touch, each with a `jira_key` and
  a `message` (e.g. `"multiple active Clockify tasks reference SF-304"`).
  **Any conflict blocks the whole apply.**
- `applied` at the top level means "an apply was attempted", not that
  everything was written. Trust the per-action flags and `failure`.
- An empty `jira_status` or `message` just means there's nothing extra to say.

When summarising for a person, list the actions by Jira key and kind, call
out renames with both names, and show conflicts on their own.

## Failures

Errors come back with `isError: true` and `{code, message, detail}`:

| `code` | Meaning | What to do |
| --- | --- | --- |
| `plan_stale` | Jira or Clockify changed since the preview (`detail.expected`/`actual` digests). Nothing was written. | Preview again, show the new plan, then apply with the new digest. Don't loop on apply. |
| `invalid_argument` with `detail.conflicts` | The plan has conflicts. Nothing was written. | Report the conflicting keys and the preview's conflict messages. Fixing them is a human edit in Clockify or Jira. |
| `invalid_argument` with `detail.argument` | A bad argument: a missing `plan_digest`, or a malformed key in `issues`. | Fix the call. |
| `missing_credentials` | `detail.variables` names the unset environment variables. | Tell the person which variables are missing. This instance cannot sync until they are configured. Don't retry. |
| `upstream_error` | Clockify or Jira refused or was unreachable (`detail.integration`). The message gives the operation and the HTTP status or error class. | Report it. Retry once later at most. A 401/403 is a credentials problem, not a transient one. |

**Partial failure is not an error result.** An apply that fails partway
returns a normal report with `failure` set:

```json
"failure": {"jira_key": "SF-304", "kind": "MARK_TASK_DONE",
            "message": "Clockify error: PUT /workspaces/ws-1/projects/p-131/tasks/t-304 returned HTTP 500"}
```

Actions before the failure have `applied: true, verified: true`; they were
written and read back. The action named in `failure`, and everything after
it, show `applied: false`. The failing action **may still have been partly
written** (the write landed, then the read-back failed). So after a partial
failure:

1. Tell the person what was applied and what failed.
2. Preview again. The new plan shows what is still outstanding, including
   whether the failed action actually landed.
3. Apply the new plan with its new digest only if the failure looks
   transient and the person wants it finished.

## Don't

- Don't apply without first showing or summarising the preview, unless the
  person explicitly asked for the sync to be applied.
- Don't reuse a digest across different `issues` scopes, or after a
  `plan_stale`.
- Don't try to resolve conflicts by applying a narrower `issues` list that
  skips them unless the person asks for that.
