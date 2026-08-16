# ADR 0003: Issue Gemini calls through native async clients instead of asyncio.to_thread

## Status

Accepted, verified on dev, and deployed and re-verified on prod (2026-08-17).
Implements issue #52. Supersedes the mechanism — though not the reasoning — of
[ADR 0001](0001-fix-event-loop-blocking-llm-calls.md) for Gemini calls.
BigQuery calls keep ADR 0001's treatment unchanged.

## Context

ADR 0001 removed the event-loop-blocking bug by wrapping every synchronous
`generate_content()` call in `asyncio.to_thread`. That was the right immediate
fix and it worked, but it was explicitly a workaround: `asyncio.to_thread`
hands work to Python's default `ThreadPoolExecutor`, which is capped at
`min(32, cpu_count + 4)` workers.

ADR 0001 recorded that cap as "not a limit at the concurrency levels tested
(1–8)". A concurrency sweep against dev on 2026-08-16, pushing further than
ADR 0001 did, found that it becomes the limit well before we expected:

| conc | rps | p50 | p95 | errors |
|------|------|-------|-------|--------------------|
| 1 | 0.08 | 12.1s | 15.1s | 0% |
| 4 | 0.14 | 11.6s | 45.4s | 0% |
| 16 | **0.44** | 29.6s | 39.4s | 0% |
| 32 | 0.24 | 42.7s | 59.5s | **54.7%** (35× 504) |

Throughput peaked at concurrency 16 and went *backwards* at 32, which is
saturation, not headroom. Two details identify the thread pool as the
constraint:

1. **The arithmetic matches a pool of ~5.** Peak throughput of 0.44 rps
   against ~12s per call implies roughly 5.3 requests genuinely in flight
   (0.44 × 12). The dev ECS task is allocated 512 CPU units, so
   `min(32, cpu_count + 4)` resolves to about 5–6 workers, not 32. The `32`
   in that formula is a ceiling, not the value — a detail easy to misread,
   and the reason ADR 0001 under-weighted this risk.
2. **The failures are queueing, not crashes.** The 35 errors were HTTP 504
   clustered at a hard 60s boundary (p99 60.2s, max 60.4s) — the ALB idle
   timeout cutting off requests that were still waiting for a thread.

So the cap ADR 0001 dismissed as distant is roughly six times closer than its
`32` reading implied, and it is the active ceiling today.

## Decision

Replace `asyncio.to_thread`-wrapped Gemini calls with the SDKs' native async
entry points, so a call in flight yields the event loop instead of holding a
thread-pool worker. Two SDKs are in play and get different treatment.

**`google.genai` (v2.6.0) — use `client.aio.models.generate_content`:**

- `agent_v2.py` — the router, refusal, and main generation calls
- `document_selector.py` — company extraction
- `financial_utils.py` — the cached-content branch of `parse_user_query`

**`google.generativeai` (v0.8.6, deprecated) — use `generate_content_async`:**

- `financial_utils.py` — the legacy fallback branch of `parse_user_query`,
  and `format_response`

**Not changed — `asyncio.to_thread` remains correct for BigQuery:**
`google-cloud-bigquery` has no first-party async client, so `query_data`
(`main.py`) and the interactions-log insert (`interaction_log.py`) keep
ADR 0001's treatment. A thread is the right tool when the library has no
async path; it was only the wrong tool when a native one existed.

### Structural consequences

`extract_companies_from_query` is called from both a synchronous subtree
(`semantic_document_selection` → `auto_load_relevant_documents`) and the async
request path. Rather than convert the whole subtree, it is split into a sync
function and an `extract_companies_from_query_async` variant sharing
`_build_extraction_request` / `_parse_extraction_response` helpers, so the
prompt exists in exactly one place.

`FinancialDataManager.parse_user_query` and `.format_response` become genuinely
`async def`. ADR 0001 deliberately avoided this, on the grounds that they were
called synchronously elsewhere. That reasoning no longer holds: the only live
synchronous callers were tests. `app/_archive/streaming_financial_chat.py`
still calls them synchronously but is not imported by the running application
(see `_archive/README.md`), and its own tests already import a module path that
no longer exists.

## Consequences

**Verified on dev (same 1-task environment, same levels, 2026-08-17).** The
stated hypothesis was that the plateau at concurrency 16 should rise and the
504s at 32 should disappear if the thread pool was the binding constraint. It
did, and they did:

| conc | rps before | rps after | p50 before | p50 after | errors before | errors after |
|------|-----------|-----------|-----------|-----------|---------------|--------------|
| 1 | 0.08 | 0.04 † | 12.1s | 27.7s † | 0% | 0% |
| 4 | 0.14 | 0.31 | 11.6s | 11.9s | 0% | 0% |
| 16 | 0.44 | 0.92 | 29.6s | 12.2s | 0% | 0% |
| 32 | 0.24 | **1.83** | 42.7s | **13.2s** | **54.7%** | **0%** |

† The concurrency-1 row is a cold-start artefact, not a regression. It is two
requests against a container that had just been replaced, and its `queue*`
attribution was 18.6s against 0.4s in the before run — the cost of warming,
not of serving. Every subsequent level settles at ~12–13s p50.

The headline results at concurrency 32: throughput **7.6× higher** (0.24 →
1.83 rps), p50 **3.2× faster** (42.7s → 13.2s), and **zero errors where 55%
of requests previously died** as HTTP 504s. Peak measured throughput rose
4.2× (0.44 → 1.83 rps), and **no saturation knee exists in the tested range**
any more — throughput was still climbing at the highest level.

The mechanism check confirms the diagnosis rather than merely the outcome.
Effective in-flight requests, computed as throughput × service time, went from
0.44 × 12s ≈ **5.3** to 1.83 × 13.2s ≈ **24**. A jump from ~5 to ~24 is
precisely what removing a ~5-worker thread-pool cap predicts. The constraint
was the pool, as argued, and not something correlated with it.

Notably, p50 stayed flat (~12–13s) from concurrency 4 to 32 while throughput
rose 6×. That is the signature of genuine parallelism: added load is absorbed
rather than queued.

**Cost of measurement is negligible.** The before sweep cost $0.24 and the
after sweep $0.37 (higher only because zero requests failed, so more real
answers were generated). Re-running this is not a spend decision.

**The new ceiling is soft, and sits near concurrency 128 on a single task.**
A follow-up sweep pushed further (2026-08-17, same 1-task dev environment):

| conc | rps | p50 | p95 | errors | scaling efficiency |
|------|------|-------|-------|--------|--------------------|
| 32 | 1.69 | 13.6s | 23.4s | 0% | — |
| 64 | 2.94 | 14.2s | 24.1s | 0% | 87% |
| 128 | 3.69 | 17.6s | 30.2s | 0.4% (1×504) | **63%** |

There is no collapse anywhere in this range — a meaningful contrast with the
pre-change build, which lost 55% of requests at concurrency 32. But the
ceiling is visible as deceleration rather than a cliff. Doubling concurrency
from 32 to 64 bought 1.74× throughput (87% efficient); doubling again to 128
bought only 1.26× (63% efficient). Three independent signals agree that
saturation is beginning around 128:

1. **Throughput is flattening** toward roughly 4 rps per task.
2. **Queueing has appeared.** The `queue*` attribution held flat at ~890ms
   through concurrency 64, then rose to 2172ms at 128 — the first evidence of
   requests waiting to be picked up rather than being served immediately.
3. **The first timeout returned.** One HTTP 504, with a max latency of 54.1s,
   approaching the 60s ALB idle timeout that converted saturation into mass
   failure in the pre-change build.

Effective in-flight requests at concurrency 128 is 3.69 × 17.6s ≈ **65**, not
128 — so roughly half the offered load is queueing. That is the mechanism
behind the deceleration.

The concurrency-32 row also reproduces the previous sweep (1.69 vs 1.83 rps,
about 8% run-to-run variance), which is worth knowing before reading small
differences in any future comparison as signal.

**Prod (2 tasks × 1024 CPU) re-verified after deploy, 2026-08-17.** 1002 paid
requests across six concurrency levels:

| conc | rps | p50 | p95 | errors |
|------|------|-------|-------|--------|
| 1 | 0.09 | 11.5s | 12.1s | 0% |
| 4 | 0.27 | 13.0s | 18.8s | 0% |
| 16 | 0.96 | 12.2s | 18.3s | 0% |
| 32 | 1.75 | 13.5s | 20.8s | 0% |
| 64 | 2.43 | 15.4s | 24.9s | 0% |
| 128 | **4.89** | 16.5s | 28.0s | 0% |
| 256 | 5.48 | 25.2s | 40.7s | 0% |

**Zero errors at every level, including 256 concurrent.** For scale, ADR 0001
recorded prod losing 17 requests to HTTP 504 at concurrency *8* before that
fix; concurrency 256 now completes cleanly. No CloudWatch alarm changed state
during the sweep — including `prod-gemini-quota-exhausted` and the
ELB-origin 5xx alarm added in PR #54 — and both ECS tasks stayed up with no
restarts.

Prod's knee is at concurrency **128**, peaking near **5.5 rps**: 64→128 scaled
at essentially 100% efficiency (2.43 → 4.89 rps), while 128→256 returned only
12% more throughput for double the load, with `queue*` rising 1740ms → 4563ms.

### The scaling result is the actionable finding

Prod has **4× dev's total CPU** (2 × 1024 units vs 1 × 512) but delivers only
**1.5× the throughput** (5.48 vs 3.69 rps). Throughput tracks *task count*
(2× tasks → 1.5× throughput) far better than it tracks CPU, and the extra 512
CPU units per task bought close to nothing.

That is the signature of an **I/O-bound workload**, which is what this is: the
request is dominated by waiting on a network call to Gemini, not by local
computation. The consequence for capacity planning is concrete and slightly
counterintuitive:

> **Scale out, not up.** Raising `cpu`/`memory` on the task is close to wasted
> spend. Adding ECS tasks — or uvicorn `--workers`, which adds event loops
> within a task — is the lever that moves throughput.

This also revises the guess in the dev section above. CPU is ruled out. The
remaining candidates are the per-process event loop and upstream Gemini
quota, and the sublinearity (2× tasks buying 1.5×, not 2×) hints at a shared
constraint starting to bind. The `gemini-quota-exhausted` alarm staying OK is
weak evidence against quota, but that alarm's threshold has not been checked
against these rates, so it is not conclusive. The broken
`ai_request_duration_seconds` metric remains what prevents settling this.

**Test doubles had to change shape.** Mocks now have to be awaitable —
`AsyncMock` on `client.aio.models.generate_content` and on
`model.generate_content_async`. A regression back to blocking calls would
leave those mocks un-awaited and fail loudly, which is the intended tripwire.

**The deprecated SDK is still on the critical path.** `financial_utils.py`
continues to import `google.generativeai`, which has ended all support and
emits a deprecation warning on import. `generate_content_async` narrows the
damage but does not remove the dependency. Migrating that module onto
`google.genai` is deliberately out of scope here and warrants its own issue.

**Known gaps carried forward from ADR 0001, still unaddressed:**

- The `ai_request_duration_seconds` histogram still reads 0ms, so the load
  test's attribution table cannot split Gemini time from app time. All 45s of
  server time at concurrency 32 landed in "other". The diagnosis above rests
  on wall-clock, throughput and a code read — not on that metric.
- Neither uvicorn `--workers` nor ECS task count is changed here. If the
  re-run shows the ceiling moved but is still too low, those are the next
  levers, in that order.
- The 60s ALB idle timeout is what converts saturation into 504s. Raising it
  would turn fast failures into slow ones, which is not obviously better, but
  it should be a deliberate choice rather than an inherited default.

## How to reproduce this measurement

```bash
python loadtest/loadtest.py --base-url <dev-url> --endpoint health --levels 1,4,16,32
python loadtest/loadtest.py --base-url <dev-url> --endpoint chat_stream --levels 1,4,16,32 --max-cost-usd 10.00
```

## Related

- [ADR 0001](0001-fix-event-loop-blocking-llm-calls.md) — the `asyncio.to_thread`
  fix this supersedes for Gemini calls
- [ADR 0002](0002-stagger-onboarding-waves-during-testing.md) — the onboarding
  hedge taken while this ceiling was unknown
- Issue #52 — the change implemented here
- Issue #51 — `/fast_chat_v2` cost/cache reporting in `loadtest.py` is broken,
  so that endpoint's cost numbers are untrustworthy (latency/throughput/errors
  are fine)
