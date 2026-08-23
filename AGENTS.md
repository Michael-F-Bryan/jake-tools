# Jake's Tools

Internal Python CLI tools for Jake. Stack: Python 3.14, `uv`, Click, Pydantic,
and the Claude Agent SDK (`claude-agent-sdk`) for LLM calls. Package layout
lives under `src/jake_tools/`. Human onboarding is in [README.md](README.md).

When in doubt, run `jake-tools <command> --help` for current flags.

## Project layout

```text
src/jake_tools/
  cli/           # Click commands (keep thin)
  claude.py      # Claude Agent SDK wrapper — the only LLM seam
  prompting.py   # typed Jinja prompts bound to a response model
  newsletters.py # SharePoint Graph client
tests/           # mirrors packages above
```

## Developing

Install once:

```bash
uv sync
uv run pre-commit install
```

Before finishing work, all of these must pass:

```bash
uv run pre-commit run --all-files
```

That runs `ruff check --fix`, `ruff format`, `pyright`, `pytest -q`, and a
`uv lock` consistency check. For a quicker loop on Python changes:

```bash
uv run ruff check --fix src tests
uv run ruff format src tests
uv run pyright
uv run pytest -q
```

### Code conventions

- Keep CLI commands thin; put orchestration and domain logic in package modules.
- Match surrounding code. Prefer typed Pydantic models over ad hoc dicts.
- Ruff rules: `E`, `F`, `I`, `UP`, `B`, `SIM`, `C4` (see `pyproject.toml`).
- Pyright must pass (`pyrightconfig.json`).
- All LLM calls go through `ClaudeAgent` in `claude.py`. Nothing else imports
  `claude_agent_sdk` directly. No command currently calls the LLM; the seam
  and the conventions below stand for the next one that does.
- Commands that call the LLM use the `agent_options` decorator in
  `cli/options.py` (adds `--model`/`--effort` and injects a typed
  `AgentOptions`) and `@coro`, applied closest to the callback, which runs the
  async callback with `asyncio.run`. The handler builds the agent itself —
  `agent_options.agent()` for a `ClaudeAgent`, or `.spec()` for just the
  `AgentSpec` — there is no shared factory seam to route through, so a test
  fakes the call by monkeypatching `AgentOptions.agent` (or the dependency it
  returns), not by injecting a factory. `--effort`'s choices come from
  `typing.get_args(EffortLevel)`, imported from `claude.py` (never
  `claude_agent_sdk` directly).
- `AgentSpec.tools` defaults to an empty tuple, which is genuinely tool-less.
  Never pass `tools=None` to `ClaudeAgentOptions`: the SDK then omits `--tools`
  and the agent inherits Claude Code's full default toolset.
- Tests inject a fake at the `run_query` seam (see `tests/test_claude_agent.py`),
  so prompt rendering, schema injection, and usage accounting stay real.
- There is no shared `ctx.obj` context object carrying factories. Every CLI
  dependency is built by the command that needs it: a decorator (see
  `cli/transcript_options.py`, `cli/options.py`, `cli/clockify.py`) stacks the
  relevant `click.option`s, pops their parsed values, and injects a typed
  Pydantic options model with dependency-constructor methods — e.g.
  `ObsidianOptions.vault_client()`, `ClockifyOptions.inventory_client()`,
  `AgentOptions.agent()` — via `ctx.invoke`. When a dependency takes no CLI
  flags at all (e.g. `NewsletterClient`), the handler just constructs it
  directly at the top of the function. The one exception is `clockify`'s
  `--api-key`/`--api-base-url`, which are group-level flags shared by several
  subcommands: Click only threads group state to subcommands via `ctx.obj`,
  so the group callback builds the `ClockifyOptions` once and stores it
  there — but handlers still never touch `ctx.obj` themselves. A
  `clockify_options` decorator (`cli/clockify.py`) is the one place that
  reads it back and injects it as a typed argument, the same shape as every
  other options decorator; it's a single typed value with no factories, not
  a context-object seam.
- CLI tests stay thin: they monkeypatch the constructor method on an options
  model (e.g. `ClockifyOptions.inventory_client`) or the client/orchestration
  symbol in the CLI module, and assert flag parsing and delegation. Real
  behaviour — reconciliation logic, HTTP clients, etc. — is tested at the
  library seam with injected fakes, not through the CLI.

## External dependencies

| Tool               | Used by                                               |
| ------------------ | ----------------------------------------------------- |
| `uv`                | dependency management and script runner                |
| `claude-agent-sdk`  | LLM calls; drives the local `claude` CLI               |
| `az` (Azure CLI)    | `newsletter` commands (Microsoft Graph token)          |

If a required external tool is missing, report the blocker. Do not mock
preflight checks or skip them silently.

## Commands

### `clockify`

```bash
jake-tools clockify jira-sync --dry-run --json
jake-tools clockify jira-sync --issue SF-304 --dry-run --json
jake-tools clockify jira-sync --issue SF-304 --apply --json
```

`jira-sync` defaults to active Jira work assigned to `currentUser()`. Repeat
`--issue KEY` to reconcile exact work items regardless of assignee. Dry-run is
the default; `--apply` creates, renames, reactivates, or completes records and
then re-fetches every changed Clockify object for verification. The command
never deletes records.

### `newsletter`

```bash
jake-tools newsletter list --limit 10
jake-tools newsletter list --body --json
jake-tools newsletter add "Title" < body.txt
jake-tools newsletter edit ITEM_ID --title "New title" < body.txt
```

`list` is read-only. `add` and `edit` mutate the CSU Weekly Newsletter
SharePoint list via Microsoft Graph. Body text is read from stdin for `add`
(required) and `edit` (optional). `--attach` can be supplied multiple times.

Requires `az login` to the CSU tenant for a Graph access token.

## Boundaries

- Do not commit secrets, tokens, or credentials.
- `newsletter` writes to SharePoint by design. `--dry-run` suppresses those
  writes where the command offers it.
