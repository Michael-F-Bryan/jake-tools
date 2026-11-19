# Repository excerpts (synthetic)

## http_client.py (62 lines total)
```python
MAX_DELAY = 60
RETRY_STATUSES = {429, 503}

def _delay(value: str) -> int:
    return min(int(value), MAX_DELAY)          # line 41

class HttpClient:
    def get(self, url: str) -> Response:
        for attempt in range(self.retries + 1):
            resp = self._session.get(url)
            if resp.status_code not in RETRY_STATUSES or attempt == self.retries:
                return resp
            time.sleep(_delay(resp.headers.get("Retry-After", "1")))
        raise AssertionError("unreachable")
```
`_delay` has no other callers (`grep -rn _delay` finds only `get` and the two tests).
