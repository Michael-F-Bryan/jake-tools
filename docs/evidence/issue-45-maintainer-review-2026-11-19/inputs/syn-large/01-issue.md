# SYN-large: Exports silently miss records; three jobs paginate differently

(Synthetic case invented for this evaluation.)

Three scheduled jobs page through the same internal REST API (`/v2/<resource>?cursor=`
responses carry `items`, `next_cursor`, `has_more`) and each has produced an incident in
the last quarter in which a "successful" run was missing records:

- INC-1: `export_invoices` dropped the final page when the total was an exact multiple of
  the page size.
- INC-2: `sync_customers` resumed after a crash from a saved cursor and re-processed one
  boundary record twice, creating a duplicate customer.
- INC-3: `audit_users` hit HTTP 429 on page 7 of 12, caught the error, logged a warning
  and exited 0 with 6 pages of results, reported to the auditors as complete.

Stop the recurrence. The same bug classes should not need fixing job by job.
