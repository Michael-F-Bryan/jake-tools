**Discovery for SYN-705** (inspected revision `a1b2c3d`, synthetic)

## Conclusion

`signal.repeated_failures` counts failed rows per target inside a 15-minute window and
alerts at 3, whether or not they were consecutive. Three unrelated one-off failures in 15
minutes therefore alert.

## Evidence

Last 30 days of `#ops-alerts`: 9 `repeated_failure` alerts. In 7 the target had successes
between its three failures and was healthy; in 2 (`billing-api`, down 41 and 95 minutes)
the failures were consecutive.

## Existing contracts

`probe.py` appends one row per target per minute. `signal.py` reduces rows to `Alert`s.
`responder.py` renders them as one report and posts to `#ops-alerts` when the report
differs from the last one posted, then writes `state/responder_state.json`, which only
`board.py` reads (for `last_seen`).

## A responder limitation found while tracing the 95-minute outage

`run_once` can only record that it looked (`last_seen`) by posting. When the report is
unchanged it returns `UNCHANGED` before writing anything, so after 30 minutes of an
unchanged outage the board shows "responder silent" although the responder is healthy.
In other words the responder cannot complete an unchanged recurrence without delivering
another report.

## Recommendation carried forward

The responder needs to separate *assessing* from *publishing*, and its state should be
versioned so that can be done safely.
