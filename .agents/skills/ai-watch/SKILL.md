---
name: ai-watch
description: Use when running, implementing, auditing, or tuning Jake's AI Watch daily radar pipeline — collect, fetch, scout, curate, Obsidian sync, digest, and Discord delivery — including checkpoint gates, calibration replay, and audit-driven prompt tuning.
---

# AI Watch

## Overview

AI Watch is a low-noise, high-recall daily radar for AI developments. It archives everything checked, surfaces almost nothing, copies curated items into Obsidian, and DMs a terse digest to Michael on Discord when items clear the bar.

Use a thin skill / thick repo-code split:

- this skill defines triggers, safety boundaries, pipeline shape, artefact layout, checkpoint gates, and tuning loops;
- `jake-tools` code owns orchestration, Hermes web tools, structured prompts, validators, manifests, and tests.

## When to use

Use this skill when the user asks for any of these:

- running or reviewing the AI Watch daily pipeline;
- implementation or refinement of `jake-tools ai-watch`;
- tuning scout/curator prompts or source queries from audit evidence;
- checkpoint validation (CP1–CP4) before progressing implementation;
- calibration replay against golden articles;
- Discord delivery behaviour or silent empty digest policy.

Do not use this for general news aggregation, RSS parsers, or unrelated daily-report workflows.

## Safety defaults

- Archive and propose only unless explicitly running live delivery or Obsidian sync without `--dry-run`.
- Empty main digest is **silent**: no Discord message; audit artefacts still written.
- Speculative lane (`speculative.md`) is file-only in v1 — not included in Discord notifications.
- `needs_human_review` decisions stay in audit output only.
- Do not auto-send Discord messages without a configured target (`--discord-target` or `AI_WATCH_DISCORD_TARGET`).
- Treat sub-agent summaries as untrusted until artefacts have been read from disk.
- No secrets in repo; Discord target from env/CLI only.

## Pipeline stages

A full run executes these stages in order:

1. **collect** — discover candidates via Hermes `web_search` (`site:` queries per source).
2. **fetch** — archive new candidates via Hermes `web_extract` (markdown + metadata).
3. **scout** — cheap model scores fit/novelty/noise; promotes to curator on recall.
4. **curate** — smart model decides `surface`, `speculative_watch`, `reject`, etc.
5. **obsidian-sync** — create vault notes for `surface` decisions only.
6. **digest** — render `digest.md`, `speculative.md`, `rejected.md`.
7. **deliver** — Discord DM when surfaced items exist; silent when empty.

Subcommands and `run` call the same domain functions. Prefer `run` for production; use individual stages for debugging or checkpoint gates.

## Working directory contract

Per-run artefacts live under a dated directory. Persistent deduplication state is cross-run.

```text
_working/ai-watch/
  state/                          # persistent seen/fetch/surface indexes
  YYYY-MM-DD/
    manifest.json
    summary.json
    digest.md
    speculative.md
    rejected.md
    delivery-payload.txt          # written on deliver (dry-run or live)
    candidates.jsonl
    fetch-results.jsonl
    scout-evaluations.jsonl
    curator-decisions.jsonl
    obsidian-sync.jsonl
    delivery.jsonl
    articles/
      <candidate-id>.md
      <candidate-id>.metadata.json
    raw/
    prompts/
    evidence/
```

Write stage state incrementally as JSONL (`sort_keys=True`, flushed + fsync). Per-run JSONL files truncate at stage start.

Candidate IDs: `sha256(canonical_url)`.

## Validation gates (mechanical)

`validation.py` enforces anti-hallucination and vocabulary rules after fetch, scout, and curate:

- Scout `evidence_quotes` must be substrings of archived article text.
- Curator `digest_summary` ≤ 500 chars; `reason` must not be generic ("interesting article").
- `surface` decisions require `obsidian_recommendation.should_create_note == true`.
- Reject reasons use vocabulary from audit-log-design.
- Coordinator fails the stage on non-empty error lists.

## Checkpoint gates (coordinator-run)

Implementers do not self-certify quality. The coordinator runs these gates and logs results to `_working/ai-watch-brainstorm/validation-log.md`. **STOP** on failure until fixed or Michael approves an override.

### CP1 — Live web smoke (after fetch stage lands)

```bash
jake-tools ai-watch collect --date today
jake-tools ai-watch fetch --date today
```

**Pass:** ≥3 candidates from ≥2 sources; ≥1 archived article with body >500 chars; fetch failures recorded with reasons; raw responses saved to `fixtures/live-smoke/`.

### CP2 — Calibration replay (after curate stage lands)

```bash
jake-tools ai-watch scout --date today --calibration-only
jake-tools ai-watch curate --date today --calibration-only
```

Golden cases in `tests/fixtures/ai_watch/calibration-cases.json` (Anthropic harness, Claude Code HTML, generative UI).

**Pass:** all 3 articles scout `fit_score ≥ 4` and `promote_to_curator`; harness surfaces to main digest; at least one UI/HTML article surfaces or goes speculative; no `reject` with `generic_ai_news` or `vendor_fluff`; every curator reason cites a concrete transferable pattern. Save outputs to `fixtures/calibration-runs/YYYY-MM-DD/`.

### CP3 — Live dry-run pipeline (after `run` coordinator lands)

```bash
jake-tools ai-watch run --date today --dry-run --max-candidates 10
```

**Pass:** stage JSONL present; `manifest.json` status not `fail`; digest items have title, summary, why-it-matters, URL; `summary.json` has token/cost totals; mechanical validation passes. Coordinator writes a 3–5 sentence quality assessment to `validation-log.md`.

### CP4 — Delivery dry-run (after docs/integration tests land)

```bash
jake-tools ai-watch deliver --date today --discord-target discord --dry-run
```

**Pass:** `delivery-payload.txt` readable and under Discord limits; empty digest → `status: skipped` in `delivery.jsonl`, no send; surfaced items include Obsidian paths in payload.

**Optional live send:** only if Michael explicitly approves after reviewing CP3/CP4 artefacts.

## Tuning loop from audit log

After each checkpoint, append notes to `_working/ai-watch-brainstorm/feedback-log.md`:

```markdown
## YYYY-MM-DD HH:MM AWST — CP2 calibration replay
- Pass/fail: ...
- False positives: ...
- False negatives: ...
- Prompt/threshold changes needed: ...
```

Inspect recent runs:

```bash
jake-tools ai-watch audit --since 7d
```

Use audit JSONL to tune:

- **Source queries** (`sources.py`) — missed domains, stale `site:` results, per-source failure rates in collect/fetch records.
- **Scout prompt** — false negatives (good articles not promoted); evidence quote grounding failures in `scout-evaluations.jsonl`.
- **Curator prompt** — false positives (generic AI news surfacing); weak `reason` fields; placement paths outside vault.
- **Thresholds** — `promote_to_curator_score`, `surface_score` in config; `--max-candidates` and `--cost-cap-usd` on `run`.

Compare calibration drift using saved runs in `fixtures/calibration-runs/` to bisect scout vs curator regressions.

## Discord delivery

- Target: `--discord-target` or `AI_WATCH_DISCORD_TARGET` env var.
- Sender: Hermes CLI `hermes send --to <target> --file delivery-payload.txt` unless `--dry-run`.
- Payload: terse digest from `digest.md` (title, summary, why-it-matters, URL, Obsidian path per item).
- Empty digest: `delivery.jsonl` records `status: skipped`, `reason: empty_digest`; no message sent.

If Hermes send is unavailable, `--dry-run` still writes `delivery-payload.txt`; live send raises a clear error.

## Cost guardrails

- Default `--max-candidates`: 80 (production), 20 for live smokes.
- `--cost-cap-usd` on `run` aborts when estimated cost exceeds cap (recorded in manifest failed stages).
- Token/cost totals in `summary.json` via `ai_usage.py`.

## Repo implementation guidance

Keep the Click command thin: parse options, construct `AiWatchCommandOptions`, call domain runners.

Prefer typed Pydantic models over ad hoc dicts. Inject external dependencies for tests:

- `WebTools` / `FakeWebTools` for collect + fetch
- `AiWatchStages` / fake Hermes stages for scout + curate
- `MessageSender` / `FakeSender` for delivery

Exit `1` when `run` status is `fail`.

## Common mistakes

- Hitting live web in unit tests — use `FakeWebTools` and fixtures; live smoke is `@pytest.mark.live` only.
- Writing Obsidian notes in tests without `tmp_path` vault — never touch the real vault in pytest.
- Sending Discord on empty digest — must stay silent.
- Accepting scout evidence quotes that are paraphrases — they fail mechanical validation.
- Surfacing before CP2 passes — do not wire real vault writes until calibration replay succeeds.
- Trusting exit code alone at checkpoints — coordinator must read artefacts on disk.
