---
name: daily-report
description: Use when running, implementing, auditing, or refining Jake's date-scoped daily report workflow, including multi-lane hindsight reports, evidence validation, safe inbox triage, daily-report CLI work, or Hermes usage/cost forensics for report runs.
---

# Daily Report

## Overview

Daily reports are date-scoped hindsight runs. They review one day's work, gather evidence from multiple sources, and produce a report that stages future action without silently mutating memory, skills, email, cron, Obsidian, SharePoint, or git.

Use a thin skill / thick repo-code split:

- this skill defines triggers, safety boundaries, lane shape, output expectations, and validation gates;
- `jake-tools` code should own orchestration, prompts, artefact writing, manifests, validators, synthesis, and tests.

## When to use

Use this skill when the user asks for any of these:

- a daily report, daily review, hindsight report, or yesterday/today work review;
- implementation or refinement of `jake-tools daily-report`;
- validation or critique of a daily-report run;
- migration of daily-report prototype scripts into repo code;
- token, quota, or dollar accounting for daily-report runs;
- inbox-triage, DUM-C, transcript, Obsidian, memory, or skill-review lanes inside a daily report.

Do not use this for a simple weekly work summary unless the user wants the daily-report workflow specifically.

## Safety defaults

- Report and propose only.
- Do not auto-write memory.
- Do not auto-patch skills.
- Do not auto-send, auto-reply to, or mark email.
- Inbox triage is envelope-only unless the user gives an explicit body-reading boundary for the current session.
- Do not draft replies from envelope metadata alone.
- Treat sub-agent summaries as untrusted until their claimed artefacts have been read from disk.

## Standard lanes

A full daily-report run uses these lanes:

1. `session-hindsight` — what happened that day, grounded in session transcripts.
2. `memory-candidates` — durable memory or Obsidian candidates to propose, not apply.
3. `skill-review` — skills used, missing steps, and patch candidates.
4. `failure-patterns` — repeated friction, retries, or legibility issues.
5. `transcripts-and-dumc` — relevant transcript, DUM-C, or domain artefacts.
6. `inbox-triage` — important unreplied messages, envelope-only by default.
7. Optional `final-critic` — verifies the report and challenges weak claims after all lane artefacts exist.

Use the strongest available model for judgement-heavy lanes and final verification. Use cheaper models only for bounded evidence lanes with strict mechanical validators.

## Working directory contract

Use a predictable dated working directory:

```text
_working/daily-report-YYYY-MM-DD/
  report.md
  summary.json
  manifest.json
  notes.md
  plan.md
  lane-results.json
  prompts/
  logs/
  subtasks/
  evidence/
  drafts/
```

Write lane state incrementally, ideally as JSONL start/completion events. Do not wait until every lane finishes before recording durable state.

Each lane owns exactly one artefact path and any declared evidence path. Do not manually rerun a lane against a file while a coordinator-spawned lane may still be running.

## Validation gates

Before final critic or final synthesis:

1. Build a manifest containing expected lane artefacts, evidence files, writer model/provider, return code, byte size, timestamps, validation status, and caveats.
2. Confirm every expected artefact exists and is non-empty.
3. Confirm required headings, allowing mild normalisation such as `Rejected / weak claims` for `Rejected/weak claims`.
4. Confirm declared evidence files exist.
5. Parse machine-readable outputs such as `summary.json`.
6. Scan for banned placeholder phrasing, while allowing quoted prompt/history evidence.
7. Reject “no findings” claims unless the lane cites exact searches/checks performed.
8. Reject “tool unavailable” claims unless the lane cites a preflight check.
9. Reject envelope-only lanes that draft replies or imply resolution.
10. Read back the final report, summary, manifest, runbook, and final critic before claiming completion.

## Report synthesis rules

Clearly separate:

- verified activity from the target date;
- nearby-session or retrospective context;
- current-run prototype or workflow failures;
- blocked or partial findings;
- envelope-only triage leads.

Prefer honest gaps over confident filler. Prototype failures can be useful migration notes, but do not frame them as events from the target date.

## Repo implementation guidance

When implementing `jake-tools daily-report`, keep the Click command thin: parse options, construct concrete stages, call `run_daily_report(options, stages)`, and render the typed result.

Prefer typed objects over ad hoc dictionaries:

- `LaneSpec`: lane name, model tier, prompt template, artefact path, evidence path, required sections, timeout, and safety mode.
- `LaneEvent`: JSONL start/completion/validation records.
- `LaneResult`: return code, log path, artefact path, evidence path, byte sizes, and validation status.
- `Manifest`: lane results, required-file checks, banned-placeholder scan results, and machine-readable caveats.

See `references/cli-implementation-plan.md` for the current implementation plan.

## Usage and cost forensics

Daily-report runs may spawn worker agents. A single Hermes session row is not enough for token/cost accounting.

For run-family reconstruction:

1. Query the exact session row.
2. Check children via `parent_session_id`.
3. If the session is a child, inspect sibling sessions under the same parent.
4. For one-shot CLI workers without parent links, search message content for working directory, lane name, or report slug.
5. Aggregate per run and provider class.
6. Keep subscription-included quota separate from direct dollar spend.

See `references/usage-forensics.md` for SQL patterns and phrasing.

## Reference router

- `references/prototype.md` — original daily-report prototype shape, lanes, safe defaults, and validation gates.
- `references/lane-orchestration.md` — coordinator contract, hard rejection rules, inbox preflight, and synthesis separation.
- `references/workflow-findings.md` — durable lessons from the 2026-06-20 multi-lane prototype run.
- `references/cli-implementation-plan.md` — concrete `jake-tools daily-report` implementation plan.
- `references/usage-forensics.md` — reconstructing Hermes token/quota/cost across parent, child, and one-shot worker sessions.

## Common mistakes

- Treating `session_search` as a date-range database. Use a deterministic session manifest where possible; use `session_search` for cited transcript evidence.
- Letting agents write artefacts directly when repo code should own writes from structured outputs.
- Accepting a self-report that a file was written without reading the file.
- Producing “no findings” without listing the searches/checks performed.
- Reading email bodies or drafting replies without an explicit current-session boundary.
- Mixing target-date facts with retrospective workflow feedback.
- Reporting OpenRouter dollar spend from unreliable negative or unknown ledger estimates.
