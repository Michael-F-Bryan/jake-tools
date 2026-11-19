## Review: synthetic-case:syn-small

**Ready to implement.**

The underlying need is clear and well-separated from the proposed mechanism: the partner API legitimately sends `Retry-After` as an HTTP-date per RFC 9110, and the client must honor that value (not just avoid crashing) so 429/503 backoff still works. A cheaper fix — e.g. catching `ValueError` in `get` and silently falling back to a fixed delay — would stop the crash but would defeat the actual purpose of `Retry-After` (respecting the server's requested wait), so it's not a credible smaller alternative here. Extending the single existing parsing point, `_delay`, with stdlib date parsing is the right-sized intervention: no new module, no dependency, no new persistent state, and the only caller (`get`) is unaffected by the added optional `now` parameter.

Two points worth a quick look during implementation, neither blocking:

**1. Garbage-string fallback extends behavior slightly beyond the reported bug, but is the right side effect of the required change, not scope creep.**
Today, any non-integer `Retry-After` (not just an HTTP-date) crashes the same way — `int("garbage")` raises `ValueError` too. Once `_delay` has to distinguish "not an int" from "is an HTTP-date" via a `try/except`, it necessarily needs *some* behavior for strings that are neither. The plan's choice — fall back to the same `1` second already used for a missing header — is the natural, minimal default and keeps the function total instead of leaving a second crash path for a different malformed input. This can be done during implementation as proposed; no decision needed.

**2. Pre-existing negative-delay crash is correctly out of scope, but worth a one-line follow-up note, not a blocker here.**
`min(int(value), MAX_DELAY)` already accepts something like `Retry-After: -5` and would later call `time.sleep(-5)`, which raises. That bug predates this change and the discovery comment correctly scopes `MAX_DELAY`/retry-count behavior as unrelated. The new date-parsing path already clamps past dates to `0` (matching `Retry-After: 0`), so it doesn't introduce this problem — it just doesn't fix the orthogonal pre-existing one. No action needed for this issue; flagging only so it isn't later attributed to this change if it surfaces.

Everything else in the plan — reusing `email.utils.parsedate_to_datetime`, treating a zone-less date as UTC, capping at `MAX_DELAY`, resolving `now` inside the call rather than at a default-argument/import boundary, and the table-driven test set (integer, future date, at-cap date, past date, zone-less date, garbage) — matches the single call site and the stated Python ≥3.14 parsing semantics from discovery, and needs no redesign.
