# Jake's Tools

This repository contains a collection of tools that Jake uses to help him with his work.

When in doubt, check the help text.

```console
$ jake-tools --help
Usage: jake-tools [OPTIONS] COMMAND [ARGS]...

Options:
  -h, --help  Show this message and exit.

Commands:
  daily-report  Run the deterministic daily-report coordinator.
  newsletter    Read and update the CSU Weekly Newsletter list.
  transcribe    Tools for transcribing audio files.
```

*(update this help text as the CLI changes)*

## Daily report CLI

```console
$ jake-tools daily-report --help
Usage: jake-tools daily-report [OPTIONS]

  Run the deterministic daily-report coordinator.

Options:
  --date YYYY-MM-DD       Target local date for the report.  [required]
  --provider TEXT         LLM provider for lane workers.  [default:
                          openrouter]
  --judgement-model TEXT  Model for judgement-heavy lanes.  [default:
                          openrouter/auto]
  --evidence-model TEXT   Model for evidence-fed lanes.  [default:
                          openrouter/auto]
  --json                  Emit a machine-readable JSON summary only.
  -h, --help              Show this message and exit.
```

Notes:

- all six lanes always run: session hindsight, memory candidates, skill review,
  failure patterns, transcripts and DUM-C, and inbox triage
- inbox triage is envelope-only via Himalaya; it does not read message bodies or
  mutate mail
- the workflow does not write memory, skills, email, cron, Obsidian,
  SharePoint, or git
- artefacts live under `_working/daily-report-YYYY-MM-DD/`
- use `--json` for automation; `summary.json` includes token and estimated cost
  totals
