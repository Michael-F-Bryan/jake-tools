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
- Commands that call the LLM use the `@agent` decorator in `cli/options.py`
  (adds `--model` and `--effort`) and `@coro`, applied closest to the callback,
  which runs the async callback with `asyncio.run`. `@agent` resolves the
  agent through the typed `AppContext` on `ctx.obj` (`cli/context.py`): it
  always builds an `AgentSpec` from `--model`/`--effort` and hands it to
  `AppContext.agent_factory`, so an injected factory still sees the flags
  instead of silently ignoring them. `--effort`'s choices come from
  `typing.get_args(EffortLevel)`, imported from `claude.py` (never
  `claude_agent_sdk` directly).
- `AgentSpec.tools` defaults to an empty tuple, which is genuinely tool-less.
  Never pass `tools=None` to `ClaudeAgentOptions`: the SDK then omits `--tools`
  and the agent inherits Claude Code's full default toolset.
- Tests inject a fake at the `run_query` seam (see `tests/test_claude_agent.py`),
  so prompt rendering, schema injection, and usage accounting stay real.
- `cli/context.py`'s `AppContext` carries factories for every client a CLI
  command builds (`agent_factory`, `clockify_client_factory`,
  `jira_client_factory`, `newsletter_client_factory`). CLI tests inject fakes
  via `CliRunner(...).invoke(cmd, args, obj=AppContext(...))`, not by
  monkeypatching the client class on the CLI module.

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
