# Claude tools, permissions and artifact publishing: evidence

What `claude_start`'s `tools`/`allowed_tools` contract rests on, and the
evidence behind the experimental artifact tools (last section). Observed on 2026-10-21 with
`claude-agent-sdk` 0.2.168 (bundled CLI 2.1.301, which does not start on
this Nix host) driving the installed Claude Code CLI 2.1.293, signed in with
a claude.ai Max login (`claude auth status`: `authMethod: claude.ai`,
`apiProvider: firstParty`). Re-check when either version changes.

## Tool exposure: the runtime is the authority

- The SDK has no tool catalogue. `ClaudeAgentOptions.tools` is
  `list[str] | ToolsPreset | None`, sent as `--tools a,b,c`
  (`_internal/transport/subprocess_cli.py`); nothing validates the names.
- The CLI does not reject unknown names either. It drops anything it does
  not provide, silently. Requested `Read,Artifact,NoSuchToolXyz,Glob,Grep`,
  the session's `init` message listed `Glob, Grep, Read`. Requested
  `Agent,Monitor,NotebookEdit,LSP,PowerShell,TodoWrite,TaskCreate,AskUserQuestion,Artifact`,
  it listed `Task, Monitor, NotebookEdit, TaskCreate`: `Agent` is reported
  as `Task`, and the rest were absent.
- What exists varies by version, model, account, provider and surface (the
  [tools reference](https://code.claude.com/docs/en/tools-reference) marks
  many tools as conditional). Even the full `claude_code` preset in an SDK
  session lacked tools named in the docs (`Glob`, `Grep`, `Artifact`) and
  had some that are not (`DesignSync`).

So `tools` accepts any well-formed name, and the worker compares the
request with the `init` tool list (`SystemMessageInit.tools`, the only
per-session record). A missing tool ends the task `tools_unavailable`,
naming it. `Agent`/`Task` is the one alias handled, because it was
observed.

## Approval is separate from exposure

From [Configure permissions](https://code.claude.com/docs/en/agent-sdk/permissions)
and the tools reference:

- `tools` decides what the model sees. `allowed_tools` holds allow rules
  in the `ToolName(specifier)` format (`Bash(npm run *)`, `Edit(/src/**)`,
  `WebFetch(domain:example.com)`). A bare name approves every call. A scoped
  rule approves only the calls it matches.
- `dontAsk` denies any call that would prompt. Calls that need no approval
  still run with no rule: reads inside the working directory, read-only Bash
  commands, and tools such as `Agent`. So "not approved by our rules" does
  not mean "forbidden". Deny rules (`disallowed_tools`) are the SDK's way to
  forbid those calls, and this server does not expose them.
- The SDK joins `allowed_tools` with commas. A specifier containing a
  comma would reach the CLI as two rules, so such rules are refused.

`test_live_narrow_bash_rule_approves_only_matching_commands` covers this
against the real CLI. With `tools=["Bash"]` and
`allowed_tools=["Bash(touch approved-*)"]`:

- `touch approved-1` ran.
- `touch denied-1` was denied: it is in `permission_denials` and the file
  does not exist.
- `ls` ran with no rule.

### Scoped writes, compound commands and denial reports

`test_live_scoped_write_and_compound_bash_denials_are_reported` (CLI
2.1.293, haiku, `dontAsk`, no settings files), with
`tools=["Bash", "Write", "Edit"]` and
`allowed_tools=["Edit(./allowed/**)", "Bash(touch approved-*)"]`:

- A `Write` to `allowed/a.txt` ran: path rules are written as `Edit(...)`
  and cover every file-editing tool, as the permissions docs say. An
  earlier run with `Write(./allowed/**)` (and `tools=["Bash", "Write"]`)
  approved nothing: the same write was denied. This server validates rule
  shape only; it does not rewrite rules.
- A `Write` to `outside/b.txt` was denied.
- `cd sub && touch approved-2` was denied: `Bash(touch approved-*)` does
  not approve a compound command that merely contains a matching part.
- `python3 -c "open('py.txt', 'w')..."` was denied.

Each refused call is in the result message's `permission_denials`
(`tool_name`, `tool_use_id`, `tool_input`), which gives no reason. The
CLI's error result for the same call does: "Permission to use Write has
been denied because Claude Code is running in don't ask mode. ...". The
worker joins the two by `tool_use_id` into `permission_denials` on the
task view (`requested`, `runtime_message`, `source`). The request's rules
are reported as requested (`limits.allowed_tools`, absent when omitted)
and as passed (`limits.approval_rules`), with `limits.exposed_tools` from
the session's `init`; whether a rule approved a call is only known from
the runtime's own reports. On 2.1.293 a denial is known when its error
result arrives (inferred from its text) and confirmed by its turn's
result. `tests/fake_claude.py`'s `deny` scenario replays that shape for the
deterministic tests; it proves the plumbing, not the runtime's decisions.

### Re-checked on CLI 2.1.309 (2026-11-02)

Haiku, `dontAsk`, no settings files, through the real server and worker
(probe evidence kept outside the repository):

- `tools=["Write"]`, `allowed_tools=["Write(./allowed/**)"]`: a `Write`
  to `allowed/a.txt` and one to `outside.txt` were **both** denied, so a
  `Write(...)` path rule still approves nothing; write path rules as
  `Edit(...)`. The server passes the rule through as requested and reports
  it so (`config.allowed_tools`); it does not rewrite or second-guess it.
- 2.1.309 reports each refusal as it happens, as a `system` message with
  subtype `permission_denied`: `tool_name`, `tool_use_id`,
  `decision_reason_type` (`mode` here: nothing approved it under `dontAsk`)
  and the same `message` the model gets. The worker records the denial
  from it at once (`source: runtime`, `runtime_reason`) instead of waiting
  for the turn's result, and the result's own record of the same call is
  not counted again.
- With `tools=["Bash"]` and `allowed_tools` omitted (Bash approved
  outright), `touch` and `rm -rf` outside the working directory ran, and a
  `cd` outside it was undone ("Shell cwd was reset"). A `sleep 40 && echo`
  call was refused by the CLI's own guard with
  `<tool_use_error>Blocked: sleep 40 followed by: …` and **no** denial
  record or system message. The worker records such a call from those
  exact error openings, labelled `source: inferred_from_tool_error`.

### Re-checked on CLI 2.1.318 (2026-11-14): recipes and the rule matrix

Haiku, `dontAsk`, no settings files, through the real server and worker, on
synthetic local files. `tests/test_delegation_recipes_live.py` (`--slow`) runs
the copyable recipes in the packaged skill reference
`claude-delegation/references/recipes.md` verbatim and the matrix below; the
sanitised briefs, grants, tool calls, denials and files are kept in
`tests/fixtures/delegation_recipes/receipts/`.

- `Edit(./out/**)`, `Edit(out/**)`, `Edit(/out/**)` and
  `Edit(//abs/path/out/**)` approved `Write` and `Edit` under `out/`, whether
  the call named the path relative or absolute, and denied `outside/b.txt`.
  `Edit(/abs/path/out/**)` (one slash: relative to the project) and
  `Write(./out/**)` approved nothing.
- With `Bash(touch approved-*)`: `touch approved-1`,
  `touch approved-3 && touch approved-4` and `touch approved-6 | cat` ran;
  `cd sub && touch approved-2` and `touch approved-5; touch denied-5` were
  denied as a whole (nothing ran, including the matching `touch`).
- A task whose grant was `Write(./reports/**)` drafted a memo, had its save
  denied and ended `completed` with no handoff; resuming it with
  `tools: [Write, Edit]` and `Edit(./reports/**)` made one `Write` of the same
  text and nothing else (hermetically pinned against the receipts). With the
  checkpoint first, the same wrong grant stopped the worker after one denied
  call, before it read anything.

These are a few small runs of one model: they establish what the runtime did
here, not a general rule.

## Artifact publishing: off by default in SDK sessions; an experimental route

The first useful slice (issue #15) is to publish an existing self-contained
HTML file as a private claude.ai artifact, return its identity and URL, and
read back the exact hosted content; updating the same identity follows.
Programmatic public/private sharing is a separate question and not a
prerequisite: new artifacts are private, and sharing can stay manual. The
slice needs the publish and read operations to work through the normal Agent
SDK path. Under the documented options inspected, none could be exposed or
verified. These findings cover only that path, under the versions and login
above. They do not cover interactive Claude Code or the claude.ai web app.

1. **Publish/update: not exposed in the inspected SDK sessions.** The
   [artifacts page](https://code.claude.com/docs/en/artifacts#availability)
   says artifacts are "Off by default in the Agent SDK, the GitHub Action,
   and MCP-server contexts, when you run Claude Code with `-p`". We found
   no documented SDK option or setting that turns them on. `enableArtifact`
   is documented as off-only: "no file can turn it back on". The
   [env-vars page](https://code.claude.com/docs/en/env-vars) lists only
   disabling and tuning variables. Third-party reports mention an
   undocumented `CLAUDE_CODE_ARTIFACT` variable; the 2026-11-14 re-check
   below examines it.
   Probes through the SDK, signed in with a claude.ai Max login (which
   meets the documented plan and auth requirements), never exposed
   `Artifact`:
   - `tools=["Artifact", ...]`
   - the full `claude_code` preset, with `setting_sources=[]`
   - the full `claude_code` preset, with `["user"]`
2. **Visibility (separate from the first slice): no documented SDK operation found.** The docs describe
   sharing as a viewer operation: "A new artifact is visible only to you.
   To share it, open the artifact in your browser and use the **Share**
   control in the page header." We found no documented SDK option or tool
   input that reads or sets public or private.
3. **Read-back: no supported SDK path found.** Interactive Claude Code can
   read artifacts natively. In an SDK session that depends on the
   `Artifact` tool, which is not exposed there. The organisation Compliance
   API (list, get version, delete) is an admin surface, not the requested
   SDK integration. It does not show that no other read-back exists.

### Re-checked 2026-11-14: SDK 0.2.182, CLI 2.1.325 and 2.1.318

Issue #15 asks whether a newer or differently configured SDK session can
expose `Artifact`. Same claude.ai Max login (`firstParty`), parent-session
`CLAUDE*` variables unset, haiku, `dontAsk`, a fresh working directory, no
model call to `Artifact` except in the last row. Each probe read the
session's `init` `tools`. Exact scripts, options and sanitised
machine-readable output: `docs/evidence/artifact-sdk-exposure-2026-11-14/`.

| Launch | CLI | `Artifact` in `init.tools` |
| --- | --- | --- |
| SDK 0.2.182, `tools=["Read","Artifact"]` | 2.1.325 (bundled) | no |
| the same, `cli_path` = installed CLI | 2.1.318 | no |
| SDK, `claude_code` preset, `setting_sources=["user"]` | 2.1.325 | no (25 tools) |
| `claude -p --tools Read,Artifact` (not via the SDK; effective entrypoint not observed) | 2.1.325 | no |
| SDK as row 1 plus **undocumented** `env={"CLAUDE_CODE_ARTIFACT": "1"}` | 2.1.325 | **yes**, with `ArtifactComments` and `ArtifactData` |

These four documented launches did not expose the tool; they do not cover
every possible option or account state.

The reason is readable in the shipped CLI (2.1.325, source reading only).
Its availability check runs in outline in this order, and the first failing
step withholds the tool:

1. switched off: `enableArtifact: false`, `CLAUDE_CODE_DISABLE_ARTIFACT`, or
   an `Artifact` deny rule;
2. surface excluded (Cowork or local-agent hosts);
3. not first-party Anthropic API, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`,
   or `CLAUDE_CODE_ARTIFACT` explicitly false;
4. `sdk_default_off`: `CLAUDE_CODE_ENTRYPOINT` is `sdk-py`, `sdk-ts`,
   `sdk-cli`, `mcp` or `claude-code-github-action` (the "off by default" in
   the docs), unless `CLAUDE_CODE_ARTIFACT` is truthy or the Remote Control
   bridge has granted `CLAUDE_CODE_BRIDGE_CHILD_ARTIFACT`;
5. a server-side feature flag, then the organisation's artifact policy.

**Demonstrated, undocumented.** Setting `CLAUDE_CODE_ARTIFACT=1` through
`ClaudeAgentOptions.env`, for that one process only, was enough on this
account for the session to list `Artifact`, `ArtifactComments` and
`ArtifactData`, and for one `{"action": "list"}` call to return without
error. Inspected privately (not stored), that result was a real listing of
the account's artifacts, not operation help: a header with the total, 25
entries of the form `(kind) title: claude.ai/code/artifact/<id>: date`, and a
trailing note that more exist and mentioning a `limit` input. So for this
account, version and launch, step 4 was the blocker and steps 1 to 3 and 5
were not. Nothing was published, edited or shared.

That variable appears in no official page and not in the
`ClaudeAgentOptions` documentation. It is an implementation detail that can
change without notice, so whether Jake may depend on it is a product and
support decision, not something this evidence makes. It is not a supported
integration and nothing in Jake sets it. Overriding `CLAUDE_CODE_ENTRYPOINT`
would be spoofing a launch identity and was not tried.

Not established by the re-check alone: how plain `claude -p` was refused
(its effective entrypoint was not observed, so no gate beyond step 4 is
inferred) and whether an upstream-supported opt-in exists. Publish and
readback are covered below.

The Artifact tool's inputs, from the same binary, name publish operations
(`publish_version`, `edit_version`, `publish_files`, `create_from_type`)
and read operations (`get_artifact`, `read_artifact`, `list_artifacts`,
`list_files`, `read_file`, `list_types`). The probe's list call used
`{"action": "list"}`. No visibility operation appears in those lists;
whether publish inputs carry one was not read. `claude.ai Claude Docs` is
a `claudeai` connector in SDK sessions; it creates new documents rather
than publishing an existing file, and was not examined further.

### Demonstrated 2026-11-14: publish, readback and update of one disposable artifact

Authorised by Michael for one disposable synthetic private artifact (a
`~400` byte page, no real content), through the SDK with the process-local
override above, no private endpoints, no TUI, entrypoint untouched, no
sharing change. Receipts: `docs/evidence/artifact-sdk-exposure-2026-11-14/`.

- **Publish** is `Artifact` with `{action: "publish", file_path, ...}`. The
  result text is `Published <path> at <URL> (Version 1, version id ...)` and
  says the artifact is private and that sharing cannot be changed from the
  tool. `title` was ignored: the file's `<title>` names the artifact.
- **Readback** in a fresh session is `{action: "read", url}`. The result
  starts `[Artifact <uuid> (version <id>) \u2014 owned by you, private (...)...]`
  and returns the page inline between `<cowritten-artifact-html>` tags; it
  saved no local file. The hosted HTML is **not** the file: it is the file
  inside a fixed 531-byte document shell (`parse.SHELL_PREFIX`) and a
  `\n</body></html>` suffix. With exactly that shell removed, v1 (395 bytes,
  sha256 `1619f473...`) and v2 (439 bytes, `fea743c0...`) were byte-identical to
  the local files. The same shell came back from CLI 2.1.318 and the
  SDK-bundled 2.1.325. The shell is not documented or promised.
- **Identity and visibility.** The URL `.../artifact/V4iHumJLsDrW9EBEtZfqZj`
  and the artifact id `e3448f84-1ca6-4757-9dd4-3bb5616d043a` stayed the same
  from v1 to v2; each readback reported `private`. That is the state the
  service reported at read time, not an inference from the documented default.
- **Update** (not shipped) passes `url` to a publish from another session.
  The tool refuses until that session has read the live version, and refused
  an unchanged resend; the third attempt succeeded. An update needs a
  pre-read, a conflict policy and an expected-version check, so it is a later
  slice.
- **Listing.** `{"action": "list"}` returns the account's artifact list (title
  and URL per entry), not operation help.

### Pre-execution authority (why a prompt is not enough)

A model can be steered by hosted HTML or a tool result, so the shipped sessions
do not rely on the prompt or on an after-the-fact audit. A new private publish
needs no approval of its own (the artifacts page says so), so no permission rule
can gate it; only code that runs before the call can. `AgentSpec.tool_guard`
runs a caller's check in a `PreToolUse` hook before every tool call. The
sessions keep a blanket allow rule for their two tools (`allowed_tools`,
`dontAsk`); the hook is what denies. It judges the **raw model-supplied input**:
any key outside the permitted set, including a model-supplied `__*` key, is
denied, and private runtime metadata is never broadly allowed.

Evidence on the real runtime (CLI 2.1.318 with the SDK pinned in `uv.lock`;
`tests/test_artifacts_live.py` and the `*.json` receipts here), harmless calls
only (a canary `Read`; a denied read-only `Artifact` `list` and `read` of
another URL):

- the hook ran for calls that an allow rule approves, and its deny stopped them;
- `disableAllHooks` and `allowManagedHooksOnly` set through `--settings` and
  through project settings did **not** disable the SDK hook;
- a call the tool rejects as malformed (`<tool_use_error>`) fails input
  validation first, never reaches the hook and never executes, so the service
  counts such calls separately instead of treating them as unguarded.

A second layer was tried and **dropped**. `can_use_tool` behind an `ask` rule
did consult the guard in the matrix (`enforcement-layers.json`), but the native
runtime enriches that callback's input with private metadata (for example an
`__artifactReadTargetPin`, consent flags and a display target), which broke a
real read. Matching or allowing those keys would couple Jake to undocumented
internals, so there is one layer, the hook.

Boundaries, stated plainly:

- A real **managed (policy)** settings file was not tested: the CLI's
  `CLAUDE_CODE_MANAGED_SETTINGS_PATH` was not honoured (a managed
  `enableArtifact: false` did not remove the tool), and writing the system
  location would change the runtime. Whether a managed policy can disable the
  SDK hook is unproven either way.
- This holds for the tested CLI and flags. A Claude Code upgrade needs the live
  tests re-run.
- The service also compares the guard's record of what it saw with the tool
  calls in the stream. That is **detection, not enforcement**: it marks a result
  unverified if a call escaped the hook, but by then the call may have run.

`artifacts.guard` permits one `Read` of the one file (its exact canonical path,
no normalisation, so a symlink cannot redirect it) and one new `publish` of it
(no `url`, `edits` or sharing), or one `read` of the one URL, and denies
everything else. User, project and managed settings stay in force, so
`enableArtifact: false` or an `Artifact` deny rule still turns the tool off.

### The experimental tools

`artifact_publish(file_path)` and `artifact_read(url, ...)` (also
`jake-tools artifact publish|read`) use that route behind an off-by-default
`artifacts.enabled` setting and a separate `artifact` Hermes capability. They
publish only a new artifact, never update, share, list or delete, and never
retry. Read their `verified`, `private_confirmed` and `problems` fields, not a
URL alone. The mechanism is undocumented and may change without notice; whether
to depend on it is a support decision. The shipped `artifact_publish` was run
once live on a synthetic 383-byte file (head `4b41207`, CLI 2.1.318):
`https://claude.ai/artifact/DsCmww9ENTTTXhzK6t2yqZ`, artifact id
`68301be6-d480-4f3e-8417-f9d137223e58`, version `1794671223-9c52`, `verified: true`,
readback `private`, no denied calls, no problems. A separate `artifact read`
saved the hosted file (pinned shell removed) and `cmp` found it byte-identical to
the source (sha256 `e5411532...5161`). Receipts:
`docs/evidence/artifact-sdk-exposure-2026-11-14/final-acceptance/`. Updates are
not shipped.

Per the task's constraints, nothing here uses private endpoints, TUI
scraping, browser automation or spoofed interactive sessions. One disposable
artifact (above) was created and updated; no other artifact was touched, and
no sharing was changed.
`test_live_unavailable_tool_fails_the_task_by_name` still gates delegated
tasks: the override is set only for the artifact sessions, never for
`claude_start`, so those still lack `Artifact`.
