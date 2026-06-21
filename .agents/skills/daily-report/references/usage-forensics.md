# Hermes daily-report usage forensics example

This reference captures the reusable pattern from reconstructing two Hermes daily-report runs from `~/.hermes/state.db`.

## Situation

The user first asked about one session ID, then clarified that the real question was the total token/quota and dollar cost for two daily-report runs from the previous day. Both runs spawned worker agents.

The initial single-session answer was incomplete because a Hermes session row does not automatically include sub-agent sessions.

## Reconstruction pattern

1. Query the exact session row first to understand model, provider, billing mode, and token buckets.
2. Check whether the session has children:

   ```sql
   SELECT * FROM sessions WHERE parent_session_id = :session_id;
   ```

3. Check whether the session is itself a child:

   ```sql
   SELECT parent_session_id FROM sessions WHERE id = :session_id;
   ```

4. If it is a child, inspect sibling sessions under the same parent. This often reveals the full run family.
5. For one-shot CLI workers that were not linked by `parent_session_id`, search message content for task-specific markers such as working directory, lane name, or report slug:

   ```sql
   SELECT DISTINCT s.id, s.parent_session_id, s.title, s.source, s.model,
          s.billing_provider, s.billing_mode,
          datetime(s.started_at,'unixepoch','localtime') AS started
   FROM sessions s
   JOIN messages m ON m.session_id = s.id
   WHERE date(s.started_at,'unixepoch','localtime') = :local_date
     AND lower(m.content) LIKE :marker
   ORDER BY s.started_at;
   ```

6. Aggregate per run and per provider class. Keep Codex/subscription quota separate from direct dollar spend.

## Lessons from the daily-report run

- A compressed continuation can appear as a child row with its own session ID and title, while still being part of the same logical run.
- Some worker sessions launched through `delegate_task` are linked by `parent_session_id`.
- Other one-shot CLI workers may have no parent link and must be found by prompt text or working directory.
- `openai-codex` / `subscription_included` should be reported as zero marginal API spend but non-zero quota usage.
- `openrouter/auto` can have exact local token usage but unreliable local cost if Hermes records `cost_status='unknown'` or negative estimates. Do not use the negative values.
- Exact `openrouter/auto` dollars require OpenRouter’s analytics API or generation-level accounting. The analytics endpoint requires a management key; inference keys return 403.

## Example phrasing

> Tokens are known locally. Dollar spend is exact only for rows with `actual_cost_usd` or credible `estimated_cost_usd`. The `openrouter/auto` workers have known token usage but unknown dollars in the Hermes ledger; OpenRouter analytics needs a management key to recover exact spend.
