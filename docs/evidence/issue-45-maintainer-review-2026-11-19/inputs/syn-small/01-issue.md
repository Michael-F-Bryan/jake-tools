# SYN-small: `Retry-After` given as an HTTP date crashes the client

(Synthetic case invented for this evaluation.)

`HttpClient.get` raises `ValueError: invalid literal for int()` when a server answers
429/503 with `Retry-After: Wed, 21 Oct 2026 07:28:00 GMT`. RFC 9110 allows either
delay-seconds or an HTTP-date. One upstream (the partner status API) now sends the date
form; the stack trace is below. Requests that should have backed off and retried fail
outright instead.

```
  File "http_client.py", line 41, in _delay
    return min(int(value), MAX_DELAY)
ValueError: invalid literal for int() with base 10: 'Wed, 21 Oct 2026 07:28:00 GMT'
```

Handle both forms.
