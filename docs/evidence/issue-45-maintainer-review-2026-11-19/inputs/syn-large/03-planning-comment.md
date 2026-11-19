**Plan for SYN-large**

## Recommendation

Add one small module, `pager.py`, that owns how this API is paged, and move the three
jobs onto it. Each job then supplies "fetch one page" and "process one item" and no longer
decides when paging is finished, how to resume, or what to do on a 429.

## Behaviour

```python
async for page in pager.pages(fetch, resume=saved_cursor):   # fetch(cursor) -> RawPage
    for item in page.items:      # boundary-overlap items already removed
        process(item)
    save(page.cursor)
result = pager.coverage       # Coverage(pages, items, complete: bool, stopped_because)
```

- Termination is by `has_more` only (never page length, never `next_cursor` truthiness).
- Items already seen at the previous page boundary (same id) are dropped.
- 429/503: wait `Retry-After` (cap 60 s), retry the same cursor up to 5 times; then stop
  with `complete=False, stopped_because="rate_limited"`, never silently.
- `Coverage.complete` is true only if the last page had `has_more=False`. The jobs fail
  their run (non-zero exit) when it is false.

Size: roughly 90 lines plus tests. Each job's loop (18, 24 and 31 lines today) shrinks to
a 4-6 line loop; `audit_users`' bespoke 429 handler is deleted.

## Delivery

Slice 1: `pager.py` with tests from the recorded staging fixture plus synthetic exact
multiple, overlap and 429 cases. Slice 2: migrate the three jobs, one PR each, each
adding the regression test for its incident. No new dependency, no new CI job, no
configuration.

## Not proposed

No generic retry library, no checkpoint store, no support for other APIs, and no
`reconcile_orders` integration until it has a ticket.
