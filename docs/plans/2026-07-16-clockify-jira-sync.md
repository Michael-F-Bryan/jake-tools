# Clockify Jira Sync Implementation Plan

> **For Hermes:** Use test-driven development and commit each coherent task before moving on.

**Goal:** Add a deterministic `jake-tools clockify jira-sync` workflow that plans and safely applies Jira-to-Clockify project, task, naming, and completion changes.

**Architecture:** Keep Click orchestration thin. Extend the existing typed Clockify HTTP boundary, add an `acli`-backed Jira boundary, and put reconciliation in pure typed functions. The default command is a read-only dry run; `--apply` executes the emitted plan in dependency order and verifies each write through the returned API representation.

**Tech stack:** Python 3.14, Click, Pydantic, Requests, Atlassian CLI (`acli`), pytest.

---

### Task 0: Preserve the Jira naming foundation

**Objective:** Commit the existing focused naming helper and tests before building the sync workflow on top of it.

**Files:**
- `src/jake_tools/clockify.py`
- `src/jake_tools/cli/clockify.py`
- `tests/test_clockify.py`
- `tests/test_clockify_cli.py`

**Verification:**

```bash
env -u PYTHONPATH uv run pytest -q tests/test_clockify.py tests/test_clockify_cli.py
```

**Commit:** `feat(clockify): add Jira-backed naming helpers`

Status: completed as commit `ae63e3a`.

### Task 1: Add typed Clockify project and task operations

**Objective:** Extend the existing HTTP client to list, create, and update the exact Clockify entities required by reconciliation.

**Files:**
- Modify: `src/jake_tools/clockify.py`
- Test: `tests/test_clockify.py`

**Steps:**

1. Write failing tests for listing active projects, listing active and done tasks, creating projects/tasks, renaming projects/tasks, and changing task status.
2. Run each focused test and confirm it fails because the operation is absent.
3. Add typed `ClockifyProject`, `ClockifyTask`, and `ClockifyClientRecord` models.
4. Generalise the HTTP boundary to support array responses and query parameters without weakening typed public methods.
5. Add only the required API methods. Preserve project metadata on updates and require task names on task updates, matching Clockify's API contract.
6. Run focused tests, then the Clockify test files.
7. Commit as `feat(clockify): add project and task operations`.

### Task 2: Add the typed Jira boundary

**Objective:** Query active assigned Jira issues and hydrate issue details through the authenticated `acli` command.

**Files:**
- Create: `src/jake_tools/clockify_jira_sync.py`
- Create: `tests/test_clockify_jira_sync.py`

**Steps:**

1. Write failing tests around a callable command-runner seam for:
   - active assigned issue search using project `SF` and statuses In Progress, Blocked, and In Review;
   - full issue hydration for issue type and parent;
   - batch status/summary lookup by Jira key;
   - non-zero exits and malformed JSON with command context preserved.
2. Run focused tests and confirm expected failures.
3. Add frozen Pydantic Jira models and an `AcliJiraClient` that invokes `acli` with argument arrays, captures stdout/stderr, and parses JSON once at the boundary.
4. Keep `acli` chatter out of structured command output.
5. Run focused tests and commit as `feat(clockify): add Jira query boundary`.

### Task 3: Build the pure reconciliation plan

**Objective:** Convert current Jira and Clockify state into a stable, reviewable list of typed actions.

**Files:**
- Modify: `src/jake_tools/clockify_jira_sync.py`
- Test: `tests/test_clockify_jira_sync.py`

**Steps:**

1. Write failing behaviour tests for:
   - Project / Phase issues mapping to summary-only Clockify projects with Jira notes;
   - ordinary issues mapping to Jira-key-prefixed tasks under their parent project;
   - missing project/task creation;
   - Jira summary renames;
   - Jira Done-category issues marking active tasks done;
   - active assigned issues reactivating done tasks;
   - archived projects being absent from the reconciliation input;
   - To Do or reassigned issues not being deactivated;
   - ambiguous active duplicates producing an explicit conflict rather than a mutation.
2. Run tests and confirm each fails for the missing behaviour.
3. Add `SyncActionKind`, `SyncAction`, and `SyncPlan` models plus a pure `plan_jira_sync(...)` function.
4. Sort actions deterministically: conflicts, projects, tasks, renames/reactivations, completions.
5. Run focused tests and commit as `feat(clockify): plan Jira reconciliation`.

### Task 4: Add plan execution and verification

**Objective:** Apply a previously constructed plan safely and make reruns idempotent.

**Files:**
- Modify: `src/jake_tools/clockify_jira_sync.py`
- Modify: `tests/test_clockify_jira_sync.py`

**Steps:**

1. Write failing tests for execution ordering, conflict refusal, project IDs returned from creation feeding dependent task creation, and exact returned records verifying each mutation.
2. Run the tests and confirm expected failures.
3. Add an executor that refuses plans containing conflicts, performs no deletes, and applies actions in dependency order.
4. Return typed applied-action results instead of mutable inspection state.
5. Run focused tests and commit as `feat(clockify): apply Jira reconciliation plans`.

### Task 5: Add the operator-facing command

**Objective:** Expose the workflow as a dry-run-first CLI suitable for humans, scripts, agents, and cron.

**Files:**
- Modify: `src/jake_tools/cli/clockify.py`
- Modify: `tests/test_clockify_cli.py`
- Modify: `README.md`

**Steps:**

1. Write failing CLI tests for human dry-run output, stable JSON output, `--apply`, zero-change runs, conflicts, and backend errors.
2. Run focused tests and confirm expected failures.
3. Add:

```bash
jake-tools clockify jira-sync --dry-run
jake-tools clockify jira-sync --dry-run --json
jake-tools clockify jira-sync --apply
```

4. Default to dry-run. Use project key `SF` and Clockify client `Sunfish Robotics` as explicit documented defaults while allowing overrides.
5. Keep stdout clean JSON under `--json`; direct human diagnostics to Click errors.
6. Document authentication and unattended-run constraints.
7. Run focused tests and commit as `feat(clockify): add Jira sync command`.

### Task 6: Validate the complete workflow

**Objective:** Prove the implementation against repository checks and current live Jira/Clockify state without mutating either service.

**Steps:**

1. Run the full repository gate:

```bash
env -u PYTHONPATH uv run pre-commit run --all-files
```

2. Retrieve the Clockify API key at runtime without persisting it.
3. Run the live dry run:

```bash
CLOCKIFY_API_KEY="$(op read 'op://Smart-Home/Clockify API Key for Cursor/API Key')" \
  env -u PYTHONPATH uv run jake-tools clockify jira-sync --dry-run --json
```

4. Inspect the actual JSON and confirm it identifies the expected four completions and two renames, with no project/task creations for SF-1 or SF-427.
5. Run the human output form and check that it is concise and matches the JSON plan.
6. Inspect `git diff`, commit contents, and branch status. Do not apply the live plan during validation.
