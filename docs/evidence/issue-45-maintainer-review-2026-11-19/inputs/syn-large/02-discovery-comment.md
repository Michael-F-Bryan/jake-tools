**Discovery for SYN-large** (inspected revision `7f8a9b0`, synthetic)

All three incidents are in the hand-written pagination loops, each different (see the
excerpts). The API's documented contract: `has_more` is authoritative; `next_cursor` can be
non-null on the last page; a cursor repeats the last item of the previous page (boundary
overlap) because items are ordered by `(updated_at, id)`; 429 carries `Retry-After`.

Callers: exactly these three jobs. No other code pages this API. A fourth (`reconcile_orders`)
is on the roadmap for next quarter but has no ticket.

Existing tests: each job has tests with one-page fake responses; none has a multi-page
case, an exact-multiple case, a boundary-overlap case or a 429 case. Fixtures for the API
exist in `tests/fixtures/api/` (recorded from staging, 3 pages with overlap).
