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

**The Gemini calls are synchronous inside `async` endpoints.**
`app/agent_v2.py` calls `self.client.models.generate_content(...)` directly in
`_fast_path()` and `run()` — no `await`, no `asyncio.to_thread`. A blocking
call on the event loop stops that loop from doing anything else, so with one
worker the API serves roughly **one LLM request at a time** regardless of how
many clients arrive. `_extract_grounded_symbols()` already uses
`asyncio.to_thread`, so the pattern for fixing it exists in the file.

Run the sweep first and let it confirm or refute this on the real deployment
before changing anything — the point of measuring is to not guess.
