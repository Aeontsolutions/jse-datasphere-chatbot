# ADR 0001: Run blocking Gemini/BigQuery calls in a thread instead of on the event loop

## Status

Accepted. Fixed in `agent_v2.py` (`/chat/stream`) and `main.py` (`/fast_chat_v2`).

## Context

A concurrency-sweep load test (`loadtest/loadtest.py`) was run against prod
on 2026-08-15 to measure real capacity ahead of increased traffic. The
`/health` baseline was clean — 71 rps at concurrency 32, sub-10ms server
time, no errors — ruling out the ALB, network, and general infra as
bottlenecks.

The `/chat/stream` sweep told a different story:

| conc | p50    | p95    | rps  | errors            |
|------|--------|--------|------|-------------------|
| 1    | 13.0s  | 14.6s  | 0.08 | 0%                |
| 2    | 24.5s  | 27.4s  | 0.08 | 0%                |
| 4    | 33.3s  | 57.8s  | 0.08 | 12.5%             |
| 8    | —      | —      | 0.00 | **100%** (17×504) |

Throughput never moved off ~0.08 rps regardless of concurrency, and p50
scaled almost linearly with concurrency — the signature of full
serialization, not parallel handling. At concurrency 8 the service round-
tripped into complete failure: 17 real HTTP 504s, i.e. real prod users
timing out, caused by this test.

Root cause, confirmed by reading the code the sweep pointed at:

- `fastapi_app/Dockerfile` runs `uvicorn app.main:app` with no `--workers`
  — one process, one event loop per ECS task. Prod runs 2 tasks; dev runs 1.
- `AgentV2._fast_path()` and `AgentV2.run()` (`app/agent_v2.py`) called
  `self.client.models.generate_content(...)` — a synchronous, blocking SDK
  call — directly inside `async def` methods. A blocking call on the event
  loop stops that loop from doing anything else, so with one worker each
  task served **one LLM request at a time**, regardless of how many clients
  arrived. `AgentV2._extract_grounded_symbols()` already used
  `asyncio.to_thread` for exactly this reason — the fix pattern already
  existed in the file, just not applied everywhere.
- The same pattern existed in `/fast_chat_v2` (`app/main.py`), worse: three
  *sequential* blocking calls per request — `FinancialDataManager.
  parse_user_query()` (Gemini), `.query_data()` (BigQuery), and
  `.format_response()` (Gemini) — all synchronous `def` methods called
  directly from the async endpoint handler, none overlapped.

## Decision

Wrap each blocking call at its call site in `asyncio.to_thread`, rather than
converting the underlying methods to `async def`:

- `agent_v2.py`: the router call, the refusal call, and the main generation
  call in `AgentV2._fast_path()`/`run()`.
- `main.py`: `parse_user_query`, `query_data`, and `format_response` in the
  `/fast_chat_v2` handler.

`FinancialDataManager`'s methods were left synchronous rather than made
`async def` because they're called synchronously elsewhere (unit tests,
archived orchestrator code) and converting them would ripple out into
call sites unrelated to this fix. Wrapping at the call site is the smaller,
lower-risk change and matches the pattern already established by
`_extract_grounded_symbols`.

Not changed: the single-uvicorn-worker-per-task deployment shape. Threading
the blocking calls off the loop raises the ceiling from "1 request served at
a time" to "N requests interleaved on N threads, still 1 process," which the
verification sweep confirmed is enough to remove the serialization signature
at the concurrency levels tested (1–8). If traffic grows past that, the next
lever is `--workers` in the Dockerfile CMD or more ECS tasks — deliberately
out of scope here since it wasn't the demonstrated bottleneck.

## Consequences

**Verified on dev** (1 task, i.e. *half* prod's capacity) after the fix:

| conc | p50   | p95   | rps  | errors |
|------|-------|-------|------|--------|
| 1    | 13.5s | 15.3s | 0.07 | 0%     |
| 2    | 14.6s | 18.3s | 0.13 | 0%     |
| 4    | 12.8s | 15.7s | 0.27 | 0%     |
| 8    | 17.8s | 24.1s | 0.38 | 0%     |

Throughput now climbs with concurrency (0.07 → 0.38 rps, 5.4x) instead of
staying flat, p50 grew only 1.3x from concurrency 1→8 instead of scaling
linearly, and there were zero errors at every level including the one that
previously produced a total outage on prod. No saturation knee was found in
the tested range.

**Known gaps, not addressed by this change:**

- The Prometheus `ai_request_duration_seconds` histogram read `0ms` in both
  sweeps despite 6–15s of measured server time per request — the metric
  isn't being recorded on this path (or is misnamed), so the load-test
  attribution table couldn't actually split "gemini time" from "other."
  Worth fixing separately; the fix in this ADR was diagnosed from the
  wall-clock/throughput numbers and a direct code read, not from that
  attribution.
- `financial_manager.parse_user_query()`/`format_response()` still fall
  back to legacy non-cached `self.model.generate_content()` calls in some
  branches (see `financial_utils.py`); those are threaded now like
  everything else in this path, but weren't otherwise audited.
- This ADR does not raise task count or add uvicorn workers. If load grows
  beyond what interleaving on threads can absorb, that's the next lever,
  and should be measured with the same `loadtest/loadtest.py` script rather
  than assumed.

## How to reproduce this measurement

```bash
pip install httpx
python loadtest/loadtest.py --base-url <target> --endpoint health --levels 1,4,16,32
python loadtest/loadtest.py --base-url <target> --endpoint chat_stream --levels 1,2,4,8 --max-cost-usd 2.00
```

See `loadtest/README.md` for the full explanation of the attribution table
and cost-safety behavior.
