# Repository excerpts (synthetic)

## jobs/export_invoices.py (loop)
```python
cursor = None
while True:
    page = client.get("/v2/invoices", cursor=cursor)
    for item in page["items"]:
        writer.write(item)
    if len(page["items"]) < PAGE_SIZE:          # stops on a short page
        break
    cursor = page["next_cursor"]
```

## jobs/sync_customers.py (loop)
```python
cursor = state.load_cursor()                     # saved after every page
while True:
    page = client.get("/v2/customers", cursor=cursor)
    for item in page["items"]:
        upsert_customer(item)                    # not idempotent on first sight: creates
    cursor = page["next_cursor"]
    state.save_cursor(cursor)
    if not cursor:
        break
```

## jobs/audit_users.py (loop)
```python
cursor = None
results = []
while True:
    try:
        page = client.get("/v2/users", cursor=cursor)
    except HTTPError as exc:
        log.warning("page failed: %s", exc)      # 429 lands here
        break                                    # then report() runs and the job exits 0
    results.extend(page["items"])
    if not page["has_more"]:
        break
    cursor = page["next_cursor"]
report(results)
```

## Current tests
`tests/test_export_invoices.py` (4 tests, one-page fakes), `tests/test_sync_customers.py`
(5, one-page), `tests/test_audit_users.py` (3, one-page). Recorded staging fixtures:
`tests/fixtures/api/users_page{1,2,3}.json` (page 1 ends with the id page 2 starts with).
