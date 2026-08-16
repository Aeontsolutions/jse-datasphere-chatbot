# Load testing

One script, `loadtest.py`, answers both questions at once: **what can the API
handle**, and **where does the latency actually go**.

It drives the API at increasing concurrency levels and, at each level, diffs
the Prometheus counters already exposed at `/metrics`. So instead of a single
"p95 was 4 seconds" number, you get 4 seconds split into Gemini time,
BigQuery time, app time, and queueing time — which is what tells you whether
to optimise the model call or the server.

## Run it

```bash
pip install httpx

# Cheapest useful run: free endpoint, no LLM cost, baselines the infra
python loadtest/loadtest.py --base-url https://your-api --endpoint health --levels 1,4,16,32

# The real thing — 30 paid requests, ~2 minutes
python loadtest/loadtest.py --base-url https://your-api --levels 1,2,4,8

# How fast is a cache hit, and how many can we serve?
python loadtest/loadtest.py --base-url https://your-api --cache-mode hot --levels 1,4,16
```

Defaults: `--endpoint chat_stream`, `--levels 1,2,4,8`, and
`--requests-per-level` set to twice the concurrency (the cheapest run that
still measures queueing). Add `--json-out results.json` to save a
machine-readable record you can diff after a change.

## Cost safety

`/chat/stream` and `/fast_chat_v2` make real, paid Gemini calls, so the script
is built to bound spend rather than duration:

- Every sweep is a fixed **request count**, never a duration — you always know
  the bill before you start.
- Actual USD is read from each response's `cost_summary` and totalled live.
- The sweep aborts as soon as measured spend passes `--max-cost-usd`
  (default `$2.00`).
- Paid endpoints prompt for confirmation unless you pass `--yes`.

Point it at a **dev/staging** deployment, not prod. It writes a row to the
BigQuery interactions table for every request, and it will show up in Langfuse.

## Reading the output

```
 conc   reqs    ok    err%      rps       p50       p90       p95       p99
    1      8     8    0.0%     2.83      353m      354m      354m      354m
    2      8     8    0.0%     2.85      701m      702m      703m      704m
    4      8     8    0.0%     2.85     1402m     1403m     1405m     1407m
```

Throughput flat while p50 doubles at every step is the classic serialisation
signature: the server has one effective lane, so users queue. If throughput
climbs with concurrency and latency stays flat, you have headroom.

The attribution table splits per-request time:

| column | means |
| --- | --- |
| `server` | total time inside the app, from the app's own middleware |
| `gemini` | time in Gemini calls, spread across requests in that level |
| `bigquery` | time in BigQuery queries |
| `other` | app code, serialisation, everything not the two above |
| `queue*` | client latency minus server latency: network, load balancer, and time waiting to be picked up |

`gemini` staying flat while `other` or `queue*` grows with concurrency means
the bottleneck is your server, not the model. `gemini` dominating and staying
flat means the model call is the floor and you need caching or a smaller model
to go faster.

## Two things worth knowing before you read results

Both are visible in the code, and both shape what the numbers will look like:

**The container runs a single uvicorn worker.** `fastapi_app/Dockerfile` ends
with `uvicorn app.main:app --host 0.0.0.0 --port 8000` — no `--workers`. One
process, one event loop.

**The Gemini calls are async; the BigQuery calls run on threads.** Gemini calls
go through the SDKs' native async clients
([ADR 0003](../docs/adr/0003-native-async-gemini-client.md)), so they yield the
event loop. BigQuery has no first-party async client, so `query_data` and the
interactions-log insert stay wrapped in `asyncio.to_thread`
([ADR 0001](../docs/adr/0001-fix-event-loop-blocking-llm-calls.md)) and still
consume a worker from a pool capped at `min(32, cpu_count + 4)` — which on a
512-CPU-unit ECS task is about 5–6 workers, not 32.

Two failure signatures to recognise in the output:

- **Throughput flat while p50 scales with concurrency** — serialisation, the
  original ADR 0001 bug. Fixed, but worth spotting if it returns.
- **Throughput peaking and then falling while p95 climbs to ~60s** —
  saturation, with the 60s cluster being the ALB idle timeout converting
  queued requests into HTTP 504s. This is what a dev sweep found at
  concurrency 32 on 2026-08-16.

Run the sweep and let it confirm or refute your theory on the real deployment
before changing anything — the point of measuring is to not guess.
