**Plan for SYN-705** (builds on the discovery comment)

## Recommendation

Two changes: correct the signal, and separate assessment from publication in the
responder so an unchanged recurrence can complete without another post. Together they
remove the misleading warnings and the "responder silent" board state.

## 1. Signal

`repeated_failures` alerts only when a target's most recent 3 rows are all failures
(consecutive), replacing the windowed count. Both genuine outages in the last 30 days
alert within 3 minutes; the 7 isolated cases do not alert.

## 2. Responder: assess, then publish

```python
def run_once(now):
    assessment = assess(now)              # always runs; always records last_assessed
    if assessment.needs_publish:          # report changed, or recovery
        publish(assessment.report)        # records last_published, digest
```

State moves to `state/responder_state_v2.json`:

| field | meaning |
|---|---|
| `schema` | `2` |
| `last_assessed` | updated every run (board silence check reads this) |
| `last_published`, `digest` | updated only when a report is posted |
| `open_incidents` | list of targets currently reported, for recovery wording |

`load_state()` accepts the v1 file and upgrades it in memory; the first write after that
creates v2 and leaves the v1 file in place. `board.py` reads `last_assessed`.

## 3. Verification

Unit tests for the new signal rule (consecutive, interleaved, boundary), for
assess/publish with an unchanged report (no post, `last_assessed` advances), recovery
announcement, and v1-state upgrade. A CI step validates `responder_state_v2.json` against
a checked-in JSON schema (`state/schema.json`) on every PR touching `state/`, so future
state changes are caught.

## Delivery

Slice 1: signal rule. Slice 2: assess/publish split, state v2 and the board change.
Slice 3: schema CI step. The ticket allows responder behaviour changes "where necessary";
the board bug and the limitation above make this one necessary.
