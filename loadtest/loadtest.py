#!/usr/bin/env python3
"""Concurrency-sweep load test for the JSE Datasphere chatbot API.

Answers two questions in one run:

1. **What can it handle?** — drives the API at increasing concurrency levels
   (1, 2, 4, 8, ...) and reports throughput, error rate and latency
   percentiles at each level. The level where p95 climbs but throughput
   stops climbing is the saturation knee.

2. **Where is the latency?** — scrapes ``/metrics`` before and after each
   level and diffs the Prometheus histograms, so wall-clock latency is
   split into time spent in Gemini calls, BigQuery queries, and everything
   else (queueing + app code).

Cost safety: every request to ``/chat/stream`` and ``/fast_chat_v2`` is a
real, paid Gemini call. The sweep is bounded by an explicit request count
(never a duration), actual USD is summed from each response's
``cost_summary``, and anything above ``--max-cost-usd`` aborts the run.

Usage
-----
    # Local, against a dev server
    python loadtest/loadtest.py --base-url http://localhost:8000

    # Deployed, cold path (distinct queries defeat the response cache)
    python loadtest/loadtest.py \
        --base-url https://api.example.com \
        --endpoint chat_stream \
        --levels 1,2,4,8 \
        --requests-per-level 8

    # Cached/hot path — how fast is a cache hit, and how many can we serve?
    python loadtest/loadtest.py --base-url ... --cache-mode hot

    # Baseline the infrastructure with no LLM cost at all
    python loadtest/loadtest.py --base-url ... --endpoint health
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import httpx
except ImportError:  # pragma: no cover - dependency guard
    sys.exit("httpx is required: pip install httpx")


# ---------------------------------------------------------------------------
# Query pool
# ---------------------------------------------------------------------------

# Distinct queries so a sweep exercises the cold path. Each one is a real
# question the API is meant to answer, not a synthetic string, so the work
# per request is representative.
COLD_QUERIES: tuple[str, ...] = (
    "What was NCBFG revenue for 2024?",
    "Show me GraceKennedy profit for the last three years",
    "How did JMMB perform in 2023?",
    "Compare Carreras and Jamaica Broilers net profit",
    "What is Sagicor Group's total assets?",
    "Summarise Wisynco's latest financial results",
    "What were Supreme Ventures earnings in 2024?",
    "How has Massy Holdings revenue trended recently?",
    "What is Seprod's profit margin?",
    "Give me an overview of Barita Investments performance",
    "What were Caribbean Cement's 2024 results?",
    "How did Fontana perform last financial year?",
)

# One fixed query for the hot path: after the first request populates the
# response cache, every subsequent one should be a cache hit.
HOT_QUERY = "What was NCBFG revenue for 2024?"


# ---------------------------------------------------------------------------
# Endpoint definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Endpoint:
    """How to call one API endpoint under load."""

    name: str
    method: str
    path: str
    paid: bool

    def build_payload(self, query: str) -> dict[str, Any] | None:
        """Build the request body for this endpoint, or None for GETs."""
        if self.method == "GET":
            return None
        if self.name == "chat_stream":
            return {
                "query": query,
                "conversation_history": [],
                "memory_enabled": False,
                "enable_web_search": True,
                "enable_financial_data": True,
            }
        if self.name == "fast_chat_v2":
            return {
                "query": query,
                "conversation_history": [],
                "memory_enabled": False,
            }
        raise ValueError(f"No payload builder for endpoint {self.name!r}")


ENDPOINTS: dict[str, Endpoint] = {
    "chat_stream": Endpoint("chat_stream", "POST", "/chat/stream", paid=True),
    "fast_chat_v2": Endpoint("fast_chat_v2", "POST", "/fast_chat_v2", paid=True),
    "health": Endpoint("health", "GET", "/health", paid=False),
    "root": Endpoint("root", "GET", "/", paid=False),
}


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    """Outcome of a single request."""

    latency_ms: float
    status: int | None
    ok: bool
    error: str | None = None
    cost_usd: float = 0.0
    cache_hit: bool = False
    response_chars: int = 0


@dataclass
class ServerDelta:
    """Server-side timing, diffed from Prometheus histograms across a level."""

    http_count: float = 0.0
    http_seconds: float = 0.0
    ai_count: float = 0.0
    ai_seconds: float = 0.0
    bigquery_count: float = 0.0
    bigquery_seconds: float = 0.0

    @property
    def available(self) -> bool:
        return self.http_count > 0

    @property
    def mean_http_ms(self) -> float:
        return (self.http_seconds / self.http_count) * 1000 if self.http_count else 0.0

    @property
    def mean_ai_ms_per_request(self) -> float:
        """Gemini seconds spread over the requests in this level."""
        return (self.ai_seconds / self.http_count) * 1000 if self.http_count else 0.0

    @property
    def mean_bigquery_ms_per_request(self) -> float:
        return (
            (self.bigquery_seconds / self.http_count) * 1000 if self.http_count else 0.0
        )

    @property
    def ai_calls_per_request(self) -> float:
        return self.ai_count / self.http_count if self.http_count else 0.0


@dataclass
class LevelResult:
    """Aggregated outcome of one concurrency level."""

    concurrency: int
    results: list[RequestResult]
    wall_seconds: float
    server: ServerDelta = field(default_factory=ServerDelta)

    @property
    def ok_results(self) -> list[RequestResult]:
        return [r for r in self.results if r.ok]

    @property
    def latencies(self) -> list[float]:
        return sorted(r.latency_ms for r in self.ok_results)

    @property
    def error_rate(self) -> float:
        if not self.results:
            return 0.0
        return 1.0 - (len(self.ok_results) / len(self.results))

    @property
    def throughput_rps(self) -> float:
        if self.wall_seconds <= 0:
            return 0.0
        return len(self.ok_results) / self.wall_seconds

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    @property
    def cache_hits(self) -> int:
        return sum(1 for r in self.ok_results if r.cache_hit)

    def pct(self, p: float) -> float:
        """Percentile of successful-request latency, in ms."""
        return percentile(self.latencies, p)


def percentile(sorted_values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile over an already-sorted sequence."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (p / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


# ---------------------------------------------------------------------------
# Prometheus scraping
# ---------------------------------------------------------------------------

_METRIC_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>\S+)$"
)


def parse_metrics(text: str) -> dict[str, float]:
    """Sum Prometheus histogram ``_sum``/``_count`` series across all labels.

    Label dimensions are collapsed on purpose: we want total time spent in
    Gemini across every model, not a per-model breakdown.
    """
    totals: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE.match(line)
        if not match:
            continue
        name = match.group("name")
        if not (name.endswith("_sum") or name.endswith("_count")):
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        totals[name] = totals.get(name, 0.0) + value
    return totals


async def scrape_metrics(
    client: httpx.AsyncClient, base_url: str
) -> dict[str, float] | None:
    """Fetch and parse ``/metrics``; None if the endpoint is unreachable."""
    try:
        response = await client.get(f"{base_url}/metrics", timeout=10.0)
        response.raise_for_status()
        return parse_metrics(response.text)
    except Exception:  # noqa: BLE001 - metrics are best-effort
        return None


def diff_metrics(
    before: dict[str, float] | None, after: dict[str, float] | None
) -> ServerDelta:
    """Subtract two metric snapshots into a per-level server-side delta."""
    if before is None or after is None:
        return ServerDelta()

    def delta(key: str) -> float:
        return max(0.0, after.get(key, 0.0) - before.get(key, 0.0))

    return ServerDelta(
        http_count=delta("http_request_duration_seconds_count"),
        http_seconds=delta("http_request_duration_seconds_sum"),
        ai_count=delta("ai_request_duration_seconds_count"),
        ai_seconds=delta("ai_request_duration_seconds_sum"),
        bigquery_count=delta("bigquery_query_duration_seconds_count"),
        bigquery_seconds=delta("bigquery_query_duration_seconds_sum"),
    )


# ---------------------------------------------------------------------------
# Request driving
# ---------------------------------------------------------------------------


def extract_cost(body: Any) -> float:
    """Pull actual USD spent out of a response's cost_summary, if present."""
    if not isinstance(body, dict):
        return 0.0
    summary = body.get("cost_summary")
    if isinstance(summary, dict):
        try:
            return float(summary.get("total_cost_usd") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def looks_cached(body: Any) -> bool:
    """A cache hit returns no cost_summary (the response never hit Gemini)."""
    return isinstance(body, dict) and body.get("cost_summary") is None


async def send_one(
    client: httpx.AsyncClient,
    base_url: str,
    endpoint: Endpoint,
    query: str,
    timeout_s: float,
) -> RequestResult:
    """Issue one request and time it end to end."""
    url = f"{base_url}{endpoint.path}"
    payload = endpoint.build_payload(query)
    start = time.perf_counter()
    try:
        if endpoint.method == "GET":
            response = await client.get(url, timeout=timeout_s)
        else:
            response = await client.post(url, json=payload, timeout=timeout_s)
        elapsed_ms = (time.perf_counter() - start) * 1000
        ok = 200 <= response.status_code < 300
        body: Any = None
        if ok:
            try:
                body = response.json()
            except (json.JSONDecodeError, ValueError):
                body = None
        text = body.get("response", "") if isinstance(body, dict) else ""
        return RequestResult(
            latency_ms=elapsed_ms,
            status=response.status_code,
            ok=ok,
            error=None if ok else f"HTTP {response.status_code}",
            cost_usd=extract_cost(body),
            cache_hit=endpoint.paid and ok and looks_cached(body),
            response_chars=len(text) if isinstance(text, str) else 0,
        )
    except Exception as exc:  # noqa: BLE001 - any failure is a failed request
        elapsed_ms = (time.perf_counter() - start) * 1000
        return RequestResult(
            latency_ms=elapsed_ms,
            status=None,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_level(
    client: httpx.AsyncClient,
    base_url: str,
    endpoint: Endpoint,
    concurrency: int,
    total_requests: int,
    queries: Iterable[str],
    timeout_s: float,
) -> LevelResult:
    """Run ``total_requests`` through ``concurrency`` workers, closed-loop.

    Closed-loop (each worker sends the next request only when the previous
    one returns) matches how real clients behave and never piles unbounded
    work onto a saturated server.
    """
    query_list = list(queries)
    queue: asyncio.Queue[int] = asyncio.Queue()
    for index in range(total_requests):
        queue.put_nowait(index)

    results: list[RequestResult] = []

    async def worker() -> None:
        while True:
            try:
                index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            query = query_list[index % len(query_list)]
            result = await send_one(client, base_url, endpoint, query, timeout_s)
            results.append(result)
            marker = "." if result.ok else "x"
            print(marker, end="", flush=True)

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall_seconds = time.perf_counter() - start
    print()
    return LevelResult(
        concurrency=concurrency, results=results, wall_seconds=wall_seconds
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(
    levels: list[LevelResult], endpoint: Endpoint, cache_mode: str
) -> str:
    """Render the sweep as a human-readable report."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 96)
    lines.append(
        f"LOAD TEST RESULTS — {endpoint.method} {endpoint.path}  (cache mode: {cache_mode})"
    )
    lines.append("=" * 96)
    lines.append("")

    header = (
        f"{'conc':>5}  {'reqs':>5}  {'ok':>4}  {'err%':>6}  {'rps':>7}  "
        f"{'p50':>8}  {'p90':>8}  {'p95':>8}  {'p99':>8}  {'max':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for level in levels:
        lines.append(
            f"{level.concurrency:>5}  {len(level.results):>5}  {len(level.ok_results):>4}  "
            f"{level.error_rate * 100:>5.1f}%  {level.throughput_rps:>7.2f}  "
            f"{level.pct(50):>7.0f}m  {level.pct(90):>7.0f}m  {level.pct(95):>7.0f}m  "
            f"{level.pct(99):>7.0f}m  {max(level.latencies, default=0):>7.0f}m"
        )
    lines.append("")
    lines.append("  (latencies in ms, measured client-side end to end)")
    lines.append("")

    # --- Server-side attribution -------------------------------------------
    if any(level.server.available for level in levels):
        lines.append("LATENCY ATTRIBUTION (from /metrics, per request averages)")
        attribution_header = (
            f"{'conc':>5}  {'server':>9}  {'gemini':>9}  {'bigquery':>9}  "
            f"{'other':>9}  {'queue*':>9}  {'ai calls':>9}"
        )
        lines.append(attribution_header)
        lines.append("-" * len(attribution_header))
        for level in levels:
            server = level.server
            if not server.available:
                continue
            other = max(
                0.0,
                server.mean_http_ms
                - server.mean_ai_ms_per_request
                - server.mean_bigquery_ms_per_request,
            )
            client_mean = statistics.fmean(level.latencies) if level.latencies else 0.0
            queue_ms = max(0.0, client_mean - server.mean_http_ms)
            lines.append(
                f"{level.concurrency:>5}  {server.mean_http_ms:>8.0f}m  "
                f"{server.mean_ai_ms_per_request:>8.0f}m  "
                f"{server.mean_bigquery_ms_per_request:>8.0f}m  {other:>8.0f}m  "
                f"{queue_ms:>8.0f}m  {server.ai_calls_per_request:>9.1f}"
            )
        lines.append("")
        lines.append(
            "  server = time inside the app; gemini/bigquery = time in those calls;\n"
            "  other  = app code + serialisation;\n"
            "  queue* = client latency minus server latency — network plus any time\n"
            "           the request spent waiting to be picked up (load balancer,\n"
            "           event loop, worker queue). A queue* that grows with\n"
            "           concurrency is the clearest sign of a concurrency ceiling."
        )
        lines.append("")

    # --- Saturation knee ---------------------------------------------------
    lines.append("READING")
    if not any(level.ok_results for level in levels):
        lines.append(
            "  Every request failed, so there is nothing to say about capacity.\n"
            "  Check the errors below — wrong --base-url, the service being down, or\n"
            "  a timeout shorter than the endpoint's normal response time are the\n"
            "  usual causes."
        )
        lines.extend(_error_lines(levels))
        lines.append("")
        return "\n".join(lines)

    knee = find_knee(levels)
    if knee is not None:
        after_knee = levels[levels.index(knee) + 1]
        lines.append(
            f"  Saturation knee at concurrency {knee.concurrency}. Going to "
            f"{after_knee.concurrency}, throughput\n"
            f"  moved {knee.throughput_rps:.2f} -> {after_knee.throughput_rps:.2f} rps "
            f"while p95 moved {knee.pct(95):.0f} -> {after_knee.pct(95):.0f}ms.\n"
            f"  Past this point added load buys latency, not capacity."
        )
    else:
        lines.append(
            "  No saturation knee inside the tested range — throughput was still\n"
            "  climbing at the highest concurrency level. Re-run with higher\n"
            "  --levels to find the ceiling."
        )

    if len(levels) >= 2:
        first, last = levels[0], levels[-1]
        concurrency_ratio = last.concurrency / first.concurrency
        lines.append("")
        if first.pct(50) > 0:
            lines.append(
                f"  p50 grew {last.pct(50) / first.pct(50):.1f}x going from concurrency "
                f"{first.concurrency} to {last.concurrency}."
            )
        # Throughput that stays flat while concurrency multiplies means the
        # requests are being served one at a time, whatever the client does.
        if first.throughput_rps > 0 and concurrency_ratio >= 4:
            throughput_ratio = last.throughput_rps / first.throughput_rps
            if throughput_ratio < 1.3:
                lines.append(
                    f"  Throughput barely moved ({first.throughput_rps:.2f} -> "
                    f"{last.throughput_rps:.2f} rps) while concurrency went up "
                    f"{concurrency_ratio:.0f}x.\n"
                    "  That is the signature of requests being serialised rather than\n"
                    "  served in parallel: every extra user waits behind the ones ahead\n"
                    "  of them, and the box has one effective lane no matter how many\n"
                    "  clients arrive."
                )

    total_cost = sum(level.total_cost_usd for level in levels)
    total_hits = sum(level.cache_hits for level in levels)
    total_ok = sum(len(level.ok_results) for level in levels)
    lines.append("")
    lines.append(
        f"  Requests: {total_ok} ok / {sum(len(lvl.results) for lvl in levels)} sent"
        f" | cache hits: {total_hits} | measured spend: ${total_cost:.4f}"
    )
    lines.extend(_error_lines(levels))
    lines.append("")
    return "\n".join(lines)


def _error_lines(levels: list[LevelResult]) -> list[str]:
    """Render a deduplicated error tally, or nothing if the run was clean."""
    errors = [
        r.error for level in levels for r in level.results if not r.ok and r.error
    ]
    if not errors:
        return []
    counts: dict[str, int] = {}
    for error in errors:
        counts[error] = counts.get(error, 0) + 1
    lines = ["", "ERRORS"]
    for error, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {count:>4}x  {error}")
    return lines


def find_knee(levels: list[LevelResult]) -> LevelResult | None:
    """First level whose throughput gain is under 20% while p95 rose over 20%."""
    for previous, current in zip(levels, levels[1:]):
        if previous.throughput_rps <= 0 or previous.pct(95) <= 0:
            continue
        throughput_gain = (
            current.throughput_rps - previous.throughput_rps
        ) / previous.throughput_rps
        latency_gain = (current.pct(95) - previous.pct(95)) / previous.pct(95)
        if throughput_gain < 0.20 and latency_gain > 0.20:
            return previous
    return None


def results_to_json(
    levels: list[LevelResult], endpoint: Endpoint, cache_mode: str
) -> dict:
    """Machine-readable form of the sweep, for diffing runs over time."""
    return {
        "endpoint": endpoint.name,
        "path": endpoint.path,
        "cache_mode": cache_mode,
        "levels": [
            {
                "concurrency": level.concurrency,
                "requests": len(level.results),
                "ok": len(level.ok_results),
                "error_rate": round(level.error_rate, 4),
                "throughput_rps": round(level.throughput_rps, 3),
                "wall_seconds": round(level.wall_seconds, 2),
                "latency_ms": {
                    "p50": round(level.pct(50), 1),
                    "p90": round(level.pct(90), 1),
                    "p95": round(level.pct(95), 1),
                    "p99": round(level.pct(99), 1),
                    "max": round(max(level.latencies, default=0), 1),
                },
                "cache_hits": level.cache_hits,
                "cost_usd": round(level.total_cost_usd, 6),
                "server_ms_per_request": {
                    "total": round(level.server.mean_http_ms, 1),
                    "gemini": round(level.server.mean_ai_ms_per_request, 1),
                    "bigquery": round(level.server.mean_bigquery_ms_per_request, 1),
                },
                "errors": [r.error for r in level.results if not r.ok],
            }
            for level in levels
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_levels(raw: str) -> list[int]:
    """Parse ``1,2,4,8`` into a sorted list of positive concurrency levels."""
    levels = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise argparse.ArgumentTypeError("concurrency levels must be >= 1")
        levels.append(value)
    if not levels:
        raise argparse.ArgumentTypeError("at least one concurrency level is required")
    return sorted(set(levels))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Concurrency-sweep load test for the JSE Datasphere chatbot API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--endpoint",
        choices=sorted(ENDPOINTS),
        default="chat_stream",
        help="Which endpoint to load (default: chat_stream)",
    )
    parser.add_argument(
        "--levels",
        type=parse_levels,
        default=[1, 2, 4, 8],
        help="Comma-separated concurrency levels (default: 1,2,4,8)",
    )
    parser.add_argument(
        "--requests-per-level",
        type=int,
        default=0,
        help=(
            "Requests to send at each level. Default 0 means 'twice the "
            "concurrency', which is the cheapest run that still measures queueing."
        ),
    )
    parser.add_argument(
        "--cache-mode",
        choices=("cold", "hot"),
        default="cold",
        help=(
            "cold: distinct queries, measures real work (default). "
            "hot: one repeated query, measures the response-cache path."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Per-request timeout in seconds (default: 180)",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=2.00,
        help="Abort the sweep if measured spend exceeds this (default: 2.00)",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=2.0,
        help="Pause between levels so the server drains (default: 2)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Also write machine-readable results to this path",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the cost confirmation prompt (for CI)",
    )
    return parser


async def main_async(args: argparse.Namespace) -> int:
    base_url = args.base_url.rstrip("/")
    endpoint = ENDPOINTS[args.endpoint]
    queries = [HOT_QUERY] if args.cache_mode == "hot" else list(COLD_QUERIES)

    plan = [
        (level, args.requests_per_level if args.requests_per_level > 0 else level * 2)
        for level in args.levels
    ]
    total_requests = sum(count for _, count in plan)

    print(f"Target      : {base_url}")
    print(f"Endpoint    : {endpoint.method} {endpoint.path}")
    print(f"Cache mode  : {args.cache_mode}")
    print(f"Levels      : {', '.join(f'{c}x{n}' for c, n in plan)}")
    print(f"Total reqs  : {total_requests}")

    if endpoint.paid:
        print(
            f"\n!! {endpoint.path} makes paid Gemini calls. This run will send "
            f"{total_requests} of them.\n"
            f"   The sweep aborts if measured spend passes ${args.max_cost_usd:.2f}."
        )
        if not args.yes and sys.stdin.isatty():
            answer = input("   Continue? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("Aborted.")
                return 1
    print()

    levels: list[LevelResult] = []
    spent = 0.0

    limits = httpx.Limits(
        max_connections=max(args.levels) * 2, max_keepalive_connections=0
    )
    async with httpx.AsyncClient(limits=limits, follow_redirects=True) as client:
        for concurrency, request_count in plan:
            print(
                f"[concurrency {concurrency}] sending {request_count} requests ",
                end="",
                flush=True,
            )
            before = await scrape_metrics(client, base_url)
            level = await run_level(
                client,
                base_url,
                endpoint,
                concurrency,
                request_count,
                queries,
                args.timeout,
            )
            after = await scrape_metrics(client, base_url)
            level.server = diff_metrics(before, after)
            levels.append(level)

            spent += level.total_cost_usd
            print(
                f"  -> p50 {level.pct(50):.0f}ms  p95 {level.pct(95):.0f}ms  "
                f"{level.throughput_rps:.2f} rps  "
                f"{level.error_rate * 100:.0f}% errors  ${spent:.4f} spent"
            )

            if spent > args.max_cost_usd:
                print(
                    f"\n!! Stopping: measured spend ${spent:.4f} exceeded "
                    f"--max-cost-usd ${args.max_cost_usd:.2f}."
                )
                break

            if args.settle_seconds > 0 and (concurrency, request_count) != plan[-1]:
                await asyncio.sleep(args.settle_seconds)

    report = format_report(levels, endpoint, args.cache_mode)
    print(report)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(results_to_json(levels, endpoint, args.cache_mode), indent=2)
            + "\n"
        )
        print(f"Wrote {args.json_out}")

    # Non-zero exit if any level failed outright, so CI can gate on it.
    return 1 if any(level.error_rate >= 0.5 for level in levels) else 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
