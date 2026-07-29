# Jake's Tools

Internal Python CLI tools for Jake. Stack: Python 3.14, `uv`, Click, Pydantic,
and the Claude Agent SDK (`claude-agent-sdk`) for LLM calls. Package layout
lives under `src/jake_tools/`. Human onboarding is in [README.md](README.md).

When in doubt, run `jake-tools <command> --help` for current flags.

## Project layout

```text
src/jake_tools/
  cli/           # Click commands (keep thin)
  ai_watch/      # collect→fetch→scout→curate→obsidian→digest→deliver
  transcripts/   # Obsidian recording pipeline
  claude.py      # Claude Agent SDK wrapper — the only LLM seam
  prompting.py   # typed Jinja prompts bound to a response model
  newsletters.py # SharePoint Graph client
tests/           # mirrors packages above
.agents/skills/ai-watch/  # deep workflow guidance (not CLI code)
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
  `claude_agent_sdk` directly.
- Commands that call the LLM use the `@agent` decorator in `cli/options.py`
  (adds `--model` and `--effort`) and `@coro`, applied closest to the callback,
  which runs the async callback with `asyncio.run`.
- `AgentSpec.tools` defaults to an empty tuple, which is genuinely tool-less.
  Never pass `tools=None` to `ClaudeAgentOptions`: the SDK then omits `--tools`
  and the agent inherits Claude Code's full default toolset.
- Tests inject a fake at the `run_query` seam (`tests/agent_fakes.py`), so
  prompt rendering, schema injection, and usage accounting stay real.

## External dependencies

| Tool               | Used by                                               |
| ------------------ | ----------------------------------------------------- |
| `uv`                | dependency management and script runner                |
| `claude-agent-sdk`  | LLM calls; drives the local `claude` CLI               |
| `hermes-agent`      | `ai-watch` `web_search`/`web_extract` tool calls only  |
| `ffmpeg`, `scribe`  | `transcribe obsidian-recording` audio pipeline         |
| `yt-dlp`            | YouTube caption and metadata source adapter            |
| `az` (Azure CLI)    | `newsletter` commands (Microsoft Graph token)          |

If a required external tool is missing, report the blocker. Do not mock
preflight checks or skip them silently.

## Commands

### `ai-watch`

```bash
jake-tools ai-watch run --date YYYY-MM-DD
jake-tools ai-watch run --date today --dry-run --max-candidates 10
jake-tools ai-watch collect --date today
jake-tools ai-watch deliver --date today --target discord --dry-run
jake-tools ai-watch audit --since 7d
```

Low-noise AI developments radar. Archives checked articles, surfaces almost
nothing, syncs curated items to Obsidian, and DMs a digest on Discord when items
clear the bar.

- Pipeline stages: collect, fetch, scout, curate, obsidian-sync, digest,
  deliver. `run` executes all stages sequentially; individual subcommands call
  the same domain functions.
- Discovery and fetch use Hermes `web_search` / `web_extract` (no hand-rolled
  HTTP); these are deterministic tool calls, not LLM calls, and are the one
  place `hermes-agent` is still imported. Scout uses `--scout-model`
  (`claude-haiku-4-5`); curator uses `--curator-model` (`claude-sonnet-5`).
- `--dry-run` skips vault writes and live Discord send; still writes run
  artefacts including `delivery-payload.txt`.
- Empty main digest is **silent** (no Discord message; audit records still
  written). Speculative items go to `speculative.md` only (not Discord in v1).
- Discord target from `--discord-target` or `AI_WATCH_DISCORD_TARGET`.
- `--max-candidates` and `--cost-cap-usd` guard run cost. Exits `1` when any
  stage fails.

Key artefacts under `_working/ai-watch/YYYY-MM-DD/`: `manifest.json`,
`summary.json`, `digest.md`, `speculative.md`, stage JSONL files, `articles/`,
and `delivery-payload.txt`.

For checkpoint gates, calibration replay, artefact layout, and audit-driven
tuning, see [.agents/skills/ai-watch/SKILL.md](.agents/skills/ai-watch/SKILL.md).

### `transcribe`

```bash
jake-tools transcribe obsidian-recording NOTE.md
jake-tools transcribe obsidian-recording --dry-run --json NOTE.md
jake-tools transcribe polish TRANSCRIPT.txt
```

`obsidian-recording` rewrites the Obsidian note in place unless `--dry-run`.
Output sections: `## Meeting Notes`, `## Chapters`, `## Transcript`.
`polish` writes polished text to stdout only.

Both subcommands accept `--model` and `--effort` via the `@agent` decorator.

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
- `ai-watch`, `transcribe`, and `transcript` write to Obsidian notes, and
  `newsletter` writes to SharePoint, by design. `--dry-run` suppresses those
  writes where the command offers it.
