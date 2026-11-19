**Ready to implement, with one specific correction before/while building `pager.py`.**

The shared-module approach is the right size here, not an over-build. Three real callers each produced a *different* bug class from hand-rolled loops, and the issue explicitly asks to stop fixing them job by job (a fourth caller is already on the roadmap). A per-job patch of the three loops would fix the three incidents but not the stated requirement — the next job would re-derive the same termination/resume/retry logic from scratch. Moving loop ownership (not just a validation helper) into `pager.py` is what actually satisfies "same bug classes should not need fixing job by job." The explicit exclusions (no generic retry lib, no checkpoint store, no `reconcile_orders` work) correctly resist scope creep, and keeping cursor persistence (`save(page.cursor)`) in the caller avoids pager taking on new persistent state. Size (~90 lines, no new dependency, no new CI job) is proportionate.

**1. The resume/dedup contract is ambiguous in exactly the way that would reproduce INC-2.**

The plan says: *"Items already seen at the previous page boundary (same id) are dropped."* That's described in terms of "the previous page," which is in-process state. But INC-2 happened on a **cold resume**: `sync_customers` crashed, restarted, and called into the API with a saved cursor — there was no "previous page" in memory at all, only a cursor from a prior process.

```python
cursor = state.load_cursor()          # survives a crash
page = client.get(..., cursor=cursor) # first item == last item already upserted
```

If `pager.pages(fetch, resume=saved_cursor)` implements dedup by literally comparing the current page's first item to "the last page's last item seen in this run," then on resume there is no prior page to compare against, the overlap item is **not** dropped, and `upsert_customer` creates a duplicate again — the exact incident this project exists to prevent, now happening inside the code that was supposed to centralize the fix.

The fix should be stated as a contract, not left to however the loop happens to be coded: *any* non-null cursor passed to `fetch`, whether supplied via `resume=` on the first call or produced internally between pages, identifies an item that has already been delivered to a caller (this run or a prior one) and must be dropped from the page it returns. That makes dedup a property of the cursor, not of in-memory loop state, so cold resume and warm continuation behave identically.

Concretely, Slice 1's test list (staging fixture, exact-multiple, overlap, 429) should add one more case before this is called done: call `pager.pages(fetch, resume=X)` as the *first* call in a fresh process (no prior page in memory) and assert the item matching `X` is dropped. The currently-listed "overlap" case alone doesn't prove this, since it's plausible to pass that case via in-memory adjacent-page comparison while still failing on cold resume.

This is a bounded correction to state and verify during Slice 1 — it doesn't need Michael's decision, just needs to not be left implicit, since it's the one piece of the plan where a reasonable-sounding reading reintroduces the motivating bug.

**2. Minor: each job's migration needs to actively remove its old "succeed anyway" path, not just swap the loop.**

For `audit_users`, the bug wasn't just "loop logic" — it's that `report()` runs unconditionally after the loop regardless of how it exited. Swapping the `while` loop for `async for page in pager.pages(...)` without also conditioning `report()` (and the exit code) on `pager.coverage.complete` would leave the old "report partial results as complete" behavior intact even with correct pagination underneath. The plan's described behavior ("jobs fail their run when `complete` is false") implies this, and Slice 2's "each adding the regression test for its incident" should catch it if the `audit_users` regression test specifically asserts *no report call / non-zero exit* on a 429-exhausted run rather than just checking page counts. Worth calling out explicitly in that PR's description so it isn't just "loop replaced, tests still green" — this is an implementation reminder, not a design gap, and resolves naturally during that migration.

No other boundary concerns: `fetch`/`process` stay owned by each job (HTTP client, auth, business logic untouched by pager), `has_more`-only termination directly matches the documented contract and closes INC-1, and the 429/503 retry-then-stop-loudly behavior closes INC-3 without pager acquiring any responsibility (auth, client config, scheduling) beyond "how to page this one API."
