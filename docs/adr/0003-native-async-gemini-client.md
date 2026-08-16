# ADR 0003: Issue Gemini calls through native async clients instead of asyncio.to_thread

## Status

Proposed (2026-08-17). Implements issue #52. Supersedes the mechanism —
though not the reasoning — of
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

**Verification is pending.** The dev sweep above is the *before* measurement.
The same sweep must be re-run against dev after deploy, at the same levels
(1,4,16,32), and the results recorded here. The hypothesis this ADR rests on
is falsifiable and should be treated as unproven until then:

> If the thread pool was the binding constraint, the plateau at concurrency 16
> should rise and the 504s at 32 should reduce or disappear. If throughput
> still peaks near 0.44 rps, the constraint is elsewhere — most likely
> upstream Gemini rate limits or the single uvicorn worker — and this change
> is correctness hygiene only.

**Cost of measurement is negligible.** The full 106-request sweep cost $0.24,
so re-running it is not a spend decision.

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
