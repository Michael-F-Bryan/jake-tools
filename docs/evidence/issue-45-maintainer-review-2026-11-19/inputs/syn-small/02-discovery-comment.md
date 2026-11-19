**Discovery for SYN-small** (inspected revision `d4e5f6a`, synthetic)

`_delay` in `http_client.py` is the only place `Retry-After` is read. It is called from
`get` for status 429 and 503, with the header value and nothing else. Nothing else in the
package parses the header. Tests for it: `test_delay_seconds`, `test_delay_capped`.
Retry count and `MAX_DELAY = 60` are existing behaviour and out of scope. The project
requires Python >= 3.14. There, `email.utils.parsedate_to_datetime` raises `ValueError`
for an unparseable string, and returns a *naive* datetime when the string has no zone
(so it cannot be subtracted from an aware `now` without first being given a zone). No
dependency is needed. A missing header already defaults to `"1"` in `get`.
