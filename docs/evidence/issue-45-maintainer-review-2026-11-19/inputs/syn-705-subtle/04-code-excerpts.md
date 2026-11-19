# Repository excerpts at the inspected revision (synthetic)

Everything below is the relevant code; the rest of the repository is unrelated to this
ticket. Line counts: `signal.py` 31, `responder.py` 58, `board.py` 24, `probe.py` 44;
`tests/` has 14 tests for these modules; CI has three jobs (`lint`, `unit`, `docs`).

## signal.py
```python
WINDOW = timedelta(minutes=15)
THRESHOLD = 3

@dataclass(frozen=True)
class Alert:
    target: str
    kind: str          # only "repeated_failure" exists today
    count: int

def repeated_failures(rows: list[ProbeRow], now: datetime) -> list[Alert]:
    recent = [r for r in rows if now - r.ts <= WINDOW]
    failed = Counter(r.target for r in recent if not r.ok)
    return [Alert(t, "repeated_failure", n) for t, n in sorted(failed.items())
            if n >= THRESHOLD]
```

## responder.py
```python
STATE = Path("state/responder_state.json")   # {"last_digest": str, "last_seen": iso, "acknowledged": bool}

class Outcome(Enum):
    DELIVERED = "delivered"
    UNCHANGED = "unchanged"

def render(alerts: list[Alert]) -> str:
    if not alerts:
        return "All clear: no repeated failures."
    return "\n".join(f"REPEATED FAILURE {a.target} x{a.count}" for a in alerts)

def run_once(now: datetime) -> Outcome:
    report = render(signal.repeated_failures(load_rows(), now))
    digest = sha256(report.encode()).hexdigest()
    state = load_state()
    if digest == state.get("last_digest"):
        return Outcome.UNCHANGED                 # nothing written
    post_to_channel("#ops-alerts", report)       # also what announces recovery ("All clear")
    save_state({"last_digest": digest, "last_seen": now.isoformat(), "acknowledged": True})
    return Outcome.DELIVERED
```

## board.py
```python
SILENT_AFTER = timedelta(minutes=30)

def responder_status(now: datetime) -> str:
    state = load_state()
    seen = datetime.fromisoformat(state["last_seen"])
    return "responder silent" if now - seen > SILENT_AFTER else "ok"
```
Only `responder.py` writes `state/responder_state.json`; only `board.py` reads
`last_seen` from it. No other consumer exists.

## Existing tests (names)
`test_signal_alerts_at_threshold`, `test_signal_ignores_old_rows`,
`test_responder_posts_changed_report`, `test_responder_skips_unchanged_report`,
`test_responder_announces_recovery`, `test_board_reports_silence`, and 8 more
for `probe.py` and rendering.
