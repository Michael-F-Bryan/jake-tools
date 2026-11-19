## Review: SYN-705 (subtle variant)

**Judgement: Implementable with specific corrections.**

The signal fix (Part 1) is the right, smallest correction to the actual defect and is backed by data (7/9 recent alerts were non-consecutive failures that the new rule would not have fired on, both real outages still would). I'd ship that as proposed.

The responder change (Part 2) is where the plan has outgrown the problem it's solving. The underlying defect — `run_once` can't record "I checked and nothing changed" without also posting — is real and worth fixing, but the fix needed is a one-line decoupling inside the existing function, not a versioned-state migration plus a new CI job. That's a material jump in what the maintainer has to own, and it should be narrowed before implementation, or explicitly approved as a deliberate larger investment.

### 1. The board-silence fix doesn't need a new schema, a migration, or a second on-disk file

Look at why the board goes stale:

```python
def run_once(now: datetime) -> Outcome:
    report = render(signal.repeated_failures(load_rows(), now))
    digest = sha256(report.encode()).hexdigest()
    state = load_state()
    if digest == state.get("last_digest"):
        return Outcome.UNCHANGED                 # nothing written
    post_to_channel("#ops-alerts", report)
    save_state({"last_digest": digest, "last_seen": now.isoformat(), "acknowledged": True})
    return Outcome.DELIVERED
```

`last_seen` is only ever written inside the "posted a new report" branch. The bug is that two unrelated facts — *"I ran and checked"* and *"I posted a new report"* — are written by the same `save_state` call. Once the signal fix lands, long healthy stretches will produce an unchanging "All clear" report for well over 30 minutes at a stretch (previously masked because the flaky signal kept forcing new posts), so the board will show "responder silent" essentially all the time the system is quiet and fine — worse than today, not better.

The fix for that is: write `last_seen`/`last_checked` on *every* call, and only gate the post + `last_digest` update on the change check. That requires no new file, no `schema` field, no v1→v2 migration, and no second JSON file left on disk forever:

```python
def run_once(now: datetime) -> Outcome:
    report = render(signal.repeated_failures(load_rows(), now))
    digest = sha256(report.encode()).hexdigest()
    state = load_state()
    changed = digest != state.get("last_digest")
    save_state({**state, "last_seen": now.isoformat()})   # always, regardless of changed
    if not changed:
        return Outcome.UNCHANGED
    post_to_channel("#ops-alerts", report)
    save_state({"last_digest": digest, "last_seen": now.isoformat(), "acknowledged": True})
    return Outcome.DELIVERED
```

`board.py` keeps reading `last_seen` from the same file, unchanged. This is a same-day implementation detail, not a design decision — I'd just make this correction during implementation rather than adopting the `schema`/`responder_state_v2.json`/in-memory-upgrade machinery the plan proposes for it.

### 2. The checked-in JSON-schema + CI job is new standing infrastructure the ticket doesn't call for

The plan adds "a CI step validates `responder_state_v2.json` against a checked-in JSON schema… on every PR touching `state/`." That's a new artifact (`state/schema.json`) and a new CI job that someone has to keep in sync with the state shape forever, for a 58-line module with exactly one writer and one reader (`board.py` reading `last_seen`). The unit tests already planned (state upgrade, assess/publish, recovery) cover the actual risk of a malformed state transition; the schema-CI step adds a second, parallel verification path for the same risk without covering anything those tests don't.

This is the kind of operated CI job the review brief calls out as a material expansion of ownership, not a routine implementation choice — it should be a decision for Michael, not something folded in silently. Given the minimal fix in §1 removes the need for a versioned schema at all, my recommendation is to drop this unless Michael specifically wants standing schema governance for this file independent of SYN-705.

### 3. `open_incidents` has no identified consumer and isn't required by the stated behaviour

The plan adds a persistent `open_incidents` list "for recovery wording," but recovery is already satisfied today: when all alerts clear, `render()` returns "All clear…", the digest changes, and a post fires. Nothing in the issue's must-keep list ("recovery from a reported outage is still announced") asks for per-target resolved messages, and no consumer of `open_incidents` is named in the plan (board.py only reads `last_seen`/`last_assessed`). This looks like scope added in service of the larger state-v2 design rather than something the ticket requires. If Michael wants named-target recovery wording, that's a reasonable separate ask, but it shouldn't ride in as unstated persistent state on this ticket. Recommend dropping it unless he confirms the requirement.

### 4. Leaving the v1 state file in place indefinitely is a dangling artifact

Contingent on the above: if any versioned-state approach is kept, "the first write... creates v2 and leaves the v1 file in place" with no stated cleanup leaves two files on disk indefinitely, one dead. A future maintainer has to know which one is live. This only matters if §1's minimal fix is rejected in favor of the v2 design — flagging it so it isn't silently inherited if that design is chosen.

### Summary

- Slice 1 (signal fix): ready as proposed.
- Slice 2 (responder): the diagnosed defect is real, but the fix is a same-function reordering of an existing write, not new schema/migration/state fields. Recommend narrowing to that before implementation.
- Slice 3 (schema CI job): drop, or bring back explicitly as a separate decision — it's new operated infrastructure disproportionate to one small state file, and becomes unnecessary once Slice 2 is narrowed.

None of this blocks starting on the signal fix now; the responder scope should be settled (narrowed per §1, with §2/§3 as explicit opt-ins only if Michael wants them) before that part is implemented.
