# Jake's Tools

Internal Python CLI tools for Jake. Stack: Python 3.14, `uv`, Click, Pydantic,
and Hermes (`hermes-agent`) for LLM calls. Package layout lives under
`src/jake_tools/`. Human onboarding is in [README.md](README.md).

When in doubt, run `jake-tools <command> --help` for current flags.

## Project layout

```text
src/jake_tools/
  cli/           # Click commands (keep thin)
  ai_watch/      # collect→fetch→scout→curate→obsidian→digest→deliver
  daily_report/  # coordinator, lanes, synthesis, Himalaya preflight
  transcripts/   # Obsidian recording pipeline
  hermes.py      # Hermes wrapper and structured prompts
  newsletters.py # SharePoint Graph client
tests/           # mirrors packages above
.agents/skills/ai-watch/      # deep workflow guidance (not CLI code)
.agents/skills/daily-report/  # deep workflow guidance (not CLI code)
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
- Hermes-injected commands use the `@hermes` decorator in `cli/options.py`.
  `daily-report` constructs `Hermes()` directly with `--judgement-model` and
  `--evidence-model`.

## External dependencies

| Tool               | Used by                                               |
| ------------------ | ----------------------------------------------------- |
| `uv`               | dependency management and script runner               |
| `hermes-agent`     | LLM calls (editable path dep in `pyproject.toml`)     |
| `himalaya`         | daily-report inbox lane preflight and envelope export |
| `ffmpeg`, `scribe` | `transcribe obsidian-recording` audio pipeline        |
| `az` (Azure CLI)   | `newsletter` commands (Microsoft Graph token)         |

If a required external tool is missing, report the blocker. Do not mock
preflight checks or skip them silently.

## Commands

### `daily-report`

```bash
jake-tools daily-report --date YYYY-MM-DD
jake-tools daily-report --date YYYY-MM-DD --json
```

Read-only outside `_working/daily-report-YYYY-MM-DD/` under the current working
directory. Does not mutate memory, skills, email, cron, Obsidian, SharePoint,
or git.

- All six lanes always run: session hindsight, memory candidates, skill review,
  failure patterns, transcripts and DUM-C, and inbox triage.
- Inbox triage is envelope-only via Himalaya; it does not read message bodies,
  draft replies, send mail, move mail, or delete mail.
- `--judgement-model` drives standard-tier lanes; `--evidence-model` drives
  cheap-tier lanes. `--provider` defaults to `openrouter`.
- `--json` emits a machine-readable summary; `summary.json` includes token and
  estimated cost totals.
- Exits `1` when any lane fails.

Key artefacts: `report.md`, `summary.json`, `manifest.json`,
`lane-events.jsonl`, plus `evidence/`, `subtasks/`, `prompts/`, and `logs/`.

For lane orchestration, validation gates, and usage forensics, see
[.agents/skills/daily-report/SKILL.md](.agents/skills/daily-report/SKILL.md).

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
  HTTP). Scout uses `--scout-model`; curator uses `--curator-model`. Provider
  defaults to `openai-codex`.
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

Both subcommands accept `--default-model` and `--provider` via the `@hermes`
decorator.

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
- `daily-report` is read-only outside its dated work directory; other commands
  may write to Obsidian notes or SharePoint by design.
- Do not assume the whole repo is read-only because daily-report is.
