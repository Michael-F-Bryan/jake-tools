# SYN-705 (subtle variant): Isolated failed probes raise misleading repeated-failure warnings

(Synthetic, SF-705-style case. Invented for this evaluation: no real ticket, system or code.)

## Report

On-call keeps receiving `repeated_failure` warnings for targets that are fine. When
we look, the target failed one probe, recovered, failed another one a few minutes
later, and so on. A single flaky probe is not a repeated failure and is not worth
waking anyone for.

## Request

Make the smallest supported correction to the repeated-failure signal so that
isolated failed probes are not reported as repeated failure, and change responder
behaviour where necessary.

## Must keep working

- A genuinely sustained outage (a target failing on consecutive probes) is still
  reported, once, within about 10 minutes of starting.
- Recovery from a reported outage is still announced.
