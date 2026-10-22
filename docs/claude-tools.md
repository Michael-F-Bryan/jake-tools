# Claude tools, permissions and artifact publishing: evidence

What `claude_start`'s `tools`/`allowed_tools` contract rests on, and why
there is no artifact-publishing tool. Observed on 2026-10-21 with
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
task view (`requested`, `runtime_message`), with `limits` as the
effective policy. Nothing else is inferred: a denial is reported only once
its turn's result arrives, so a turn still running may have unreported
ones. `tests/fake_claude.py`'s `deny` scenario replays that shape for the
deterministic tests; it proves the plumbing, not the runtime's decisions.

## Artifact publishing: blocked, no tool shipped

The requested tool would publish a local HTML file as a claude.ai artifact,
update it in place, and set and verify public or private visibility. It
needs every one of those operations to work through the normal Agent SDK
path. None of them could be exposed or verified through the inspected SDK
and runtime. These findings cover only that path, under the versions and
login above. They do not cover interactive Claude Code or the claude.ai
web app.

1. **Publish/update: not exposed in the inspected SDK sessions.** The
   [artifacts page](https://code.claude.com/docs/en/artifacts#availability)
   says artifacts are "Off by default in the Agent SDK, the GitHub Action,
   and MCP-server contexts, when you run Claude Code with `-p`". We found
   no documented SDK option or setting that turns them on. `enableArtifact`
   is documented as off-only: "no file can turn it back on". The
   [env-vars page](https://code.claude.com/docs/en/env-vars) lists only
   disabling and tuning variables. Third-party reports mention an
   undocumented `CLAUDE_CODE_ARTIFACT` variable, and we did not use it.
   Probes through the SDK, signed in with a claude.ai Max login (which
   meets the documented plan and auth requirements), never exposed
   `Artifact`:
   - `tools=["Artifact", ...]`
   - the full `claude_code` preset, with `setting_sources=[]`
   - the full `claude_code` preset, with `["user"]`
2. **Visibility: no documented SDK operation found.** The docs describe
   sharing as a viewer operation: "A new artifact is visible only to you.
   To share it, open the artifact in your browser and use the **Share**
   control in the page header." We found no documented SDK option or tool
   input that reads or sets public or private.
3. **Read-back: no supported SDK path found.** Interactive Claude Code can
   read artifacts natively. In an SDK session that depends on the
   `Artifact` tool, which is not exposed there. The organisation Compliance
   API (list, get version, delete) is an admin surface, not the requested
   SDK integration. It does not show that no other read-back exists.

Per the task's constraints, nothing here uses private endpoints, TUI
scraping, browser automation or spoofed interactive sessions. No artifact
was created, and no existing artifact was touched.
`test_live_unavailable_tool_fails_the_task_by_name` doubles as the gate: if
a session ever lists `Artifact`, re-run these checks. Visibility control
through the SDK would still need its own check.
