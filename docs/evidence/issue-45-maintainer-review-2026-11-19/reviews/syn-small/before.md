**Ready to implement.**

The framing is solid and the plan matches it: the issue's own text ("RFC 9110 allows either delay-seconds or an HTTP-date... Handle both forms") makes date parsing a requirement, not a suggested mechanism inherited uncritically — so extending `_delay` rather than, say, just catching-and-ignoring the exception is the right scope. The plan correctly leaves retry count and `MAX_DELAY` alone (confirmed out of scope in discovery and untouched by the proposed change at `http_client.py:41`), introduces no new module/dependency/setting, and keeps `_delay` private with its only caller (`get`) unaffected by the new optional `now` parameter. The `now=None`-resolved-inside-the-call design is the right call for a long-lived process and is explicitly justified rather than asserted. I don't find a defect that should block implementation.

Two non-blocking observations for the implementer, not corrections to the plan:

**1. Silent fallback for garbled headers loses the signal that caused this bug.**

The plan's `1`-second fallback for anything that's neither a valid int nor a valid HTTP-date is reasonable and consistent with the existing missing-header default — I'm not asking to change that behavior. But today, a malformed `Retry-After` crashes loudly (that's literally how this bug was found). After the fix, a *different* malformed value (e.g. a fourth upstream sending something neither RFC form permits) will be swallowed into a quiet 1-second retry with no trace. Given this issue exists because one upstream deviated from the expected format, it's plausible another will too, and the next person debugging "why do we retry so fast against this new partner" will have nothing to go on.

Suggest a debug/warning-level log at the fallback branch noting the raw header value. This is an improvement, not a requirement — fine to decide during implementation rather than up front.

**2. Minor structural nit: keep date/int parsing and cap-clamping visually separate.**

`_delay` currently does one thing (`min(int(value), MAX_DELAY)`); it's about to do "parse as int, or as date-converted-to-seconds, or fall back to 1" and *then* clamp. Reusing the existing `min(..., MAX_DELAY)` for all three paths is right (one cap mechanism, not three) — but if the int/date/fallback branching gets written inline ahead of that `min()`, the function stops being skimmable in one glance. Not blocking — just a preference the implementer can apply or skip; no need to split into a separate module or add a new public helper for it.

Test plan is adequate for the surface described (int, future date, boundary-at-cap, past date, zone-less date, garbage string) and correctly treats the retry loop as already covered. No reason to add more stages or revisit the existing `test_delay_seconds` / `test_delay_capped` coverage.
