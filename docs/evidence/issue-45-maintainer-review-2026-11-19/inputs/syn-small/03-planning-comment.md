**Plan for SYN-small**

## Recommendation

Extend `_delay` so a non-integer value is parsed as an HTTP date and converted to
seconds from now. Nothing else changes: no new module, dependency or setting.

## Behaviour

- Integer value: unchanged.
- HTTP date in the future: seconds until then, rounded up, capped at `MAX_DELAY`.
  A date with no zone is read as UTC (naive results from `parsedate_to_datetime` are
  given UTC before subtracting).
- HTTP date in the past: `0`, as for `Retry-After: 0`.
- Anything else (`ValueError` from the parser): the same `1` second that `get` already
  uses when the header is missing, so a garbled header is not treated more aggressively
  than an absent one.
- `_delay(value, now=None)`: when `now` is `None` it is resolved inside the call
  (`datetime.now(timezone.utc)`), never in the signature, so a long-lived process does
  not freeze it at import time. Tests pass `now` explicitly.

## Verification

Table-driven test for `_delay`: integer, future date, date exactly at the cap, past
date, zone-less date, and a garbage string. No change to the retry loop; the existing
retry test still covers it.
