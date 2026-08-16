# ADR 0002: Stagger onboarding invitations in waves while capacity is unproven

## Status

Accepted (2026-08-16). **Not applied to the first beta wave** — all 100
invitations had already been sent when this decision was taken.

**Superseded in practice by measurement (2026-08-17).** This ADR's own expiry
condition was "once a sweep establishes where the knee actually is, this
should be revisited." That sweep has now run.
[ADR 0003](0003-native-async-gemini-client.md) records prod handling **256
concurrent requests with zero errors**, with the knee at concurrency 128 —
roughly 17–50× the 5–15 concurrent this document estimated for a 100-invite
spike. The correlated-arrival risk that motivated staggering is no longer a
capacity risk at beta scale.

Waves remain reasonable for *product* reasons — graduated feedback, support
load, room to react to bugs — but they are no longer needed to protect
availability, and wave size should not be constrained by capacity fear. The
reasoning below is retained because it documents how the decision was made
under uncertainty, and because the "measure before engineering, stagger while
unproven" pattern applies again the next time capacity is unknown.

## Context

The chatbot entered beta with onboarding invitations sent to 100 potential
users. All 100 went out at once.

The capacity picture at the time of that send:

- [ADR 0001](0001-fix-event-loop-blocking-llm-calls.md) removed the
  event-loop-blocking bug and verified the fix, but **only up to concurrency
  8**. Above that, behaviour is unmeasured.
- The ADR 0001 numbers do not establish a ceiling. Dev (1 ECS task) recorded
  0.38 rps at concurrency 8 and prod (2 tasks) recorded 0.36 rps at the same
  level. Prod has twice the capacity and produced the same throughput, which
  is the signature of a test bounded by the load generator rather than by the
  server: at concurrency 8 with ~22s service time, throughput is 8 ÷ 22s
  regardless of which end is the constraint. The real limit was never found.
- A `/health` sweep against dev on 2026-08-16 returned 69 rps at concurrency
  32 with 0% errors and sub-10ms server time, confirming the ALB, network and
  container are not the near-term constraint. Whatever the limit is, it is in
  the application path.
- Each `/chat/stream` request takes ~13–20s, nearly all of it Gemini time.

The risk this creates is not steady-state load. 100 users accumulated over a
month is roughly 0.07 rps average — far below anything measured. The risk is
**correlated arrival**: invitations delivered simultaneously produce
click-throughs that cluster within hours of each other. A plausible first-day
pattern for 100 invitations is 5–15 concurrent sessions, which straddles the
highest level ever verified.

The failure mode when that ceiling is crossed is not graceful. Pre-fix, a
concurrency-8 load test produced 17 real HTTP 504s for production users
(ADR 0001). Post-fix the same level is clean, but the shape of failure above
the tested range is unknown, and 504s remain the likely form.

The cost of that failure is unusually high in a beta. The people who would
receive the timeouts are the same people being asked to evaluate the product,
and a first impression that times out is not recovered by a later fix.

## Decision

While measured capacity remains below projected peak concurrency, onboarding
invitations are sent in **waves**, not in a single batch.

- Default wave size: **25 invitations**, with at least 48 hours between waves.
- A wave is released only after the previous wave's error rate and p95 latency
  have been checked against the CloudWatch alarms in
  `infra/monitoring/chatbot-alarms.yml`.
- Wave size may be raised once a concurrency sweep has established a knee that
  sits comfortably above observed peak concurrency.

Staggering is chosen as the **first** lever, ahead of any capacity
engineering, for three reasons:

1. It costs nothing. No code, no infrastructure change, no deploy.
2. It attacks the actual risk. The exposure is a correlated spike, and
   spreading arrivals is the direct remedy for a correlated spike.
3. It produces information. Each wave is a graduated load test with real
   traffic, so peak concurrency becomes an observation rather than an
   estimate, and the next wave is sized against data.

This is deliberately a process control rather than a technical one. The
technical levers — ECS task count, `--workers` in the Dockerfile CMD, the
native async client in issue #52 — all remain available and are not precluded.
They are simply more expensive and slower than not sending all the email at
once, and none of them should be sequenced ahead of it.

## Consequences

**The first wave is unmitigated.** All 100 invitations are already delivered.
This decision cannot be applied retroactively, and the correlated-spike risk
for that cohort is carried as-is. The compensating measures are monitoring
rather than prevention:

- The CloudWatch alarms from PR #54 (including the ELB-origin 5xx alarm that
  catches non-responding targets) are the detection path.
- If 5xx rates climb, the fastest available response is raising the ECS
  desired count on the prod service — a console or CLI change requiring no
  deploy, and the correct first response given that the application, not the
  infrastructure, is the constraint.

**Onboarding is slower.** Reaching 100 activated users takes roughly four
waves — about a week longer than a single send. This is accepted. The
alternative buys speed with the risk of an outage in front of the entire
evaluation cohort.

**This ADR expires on measurement.** It is a hedge against an unknown ceiling,
not a permanent policy. Once a sweep establishes where the knee actually is,
this should be revisited: if the knee is far above realistic peak concurrency,
the constraint is unnecessary and waves can be collapsed back into a single
send.

**Latency is untouched by this decision and remains the larger beta risk.**
Staggering prevents errors. It does nothing about ~13–20s responses, which
will shape beta feedback more than capacity will at these user counts. That is
a separate problem and needs its own treatment — streaming partial output or
visible progress rather than more servers.

## Related

- [ADR 0001](0001-fix-event-loop-blocking-llm-calls.md) — the capacity
  measurements this decision hedges against
- Issue #52 — native async Gemini client; raises the per-task ceiling from
  ~32 in-flight calls, well above beta-scale concurrency
- `infra/monitoring/chatbot-alarms.yml` — the alarms serving as the detection
  path for the unmitigated first wave
