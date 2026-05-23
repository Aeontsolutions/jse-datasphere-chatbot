"""Latency and cost utility functions, shared across the eval suite."""

from __future__ import annotations

import statistics
from typing import Any

from pydantic import BaseModel


class LatencyStats(BaseModel):
    """Latency statistics over a sample of durations."""

    min_ms: float
    max_ms: float
    avg_ms: float
    p95_ms: float
    count: int


class CostInfo(BaseModel):
    """Cost and token counts extracted from a chatbot response."""

    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


def latency_stats(durations_ms: list[float]) -> LatencyStats:
    """Compute min/max/avg/p95 latency statistics."""
    if not durations_ms:
        return LatencyStats(min_ms=0, max_ms=0, avg_ms=0, p95_ms=0, count=0)

    sorted_d = sorted(durations_ms)
    p95_index = int(len(sorted_d) * 0.95)
    p95 = sorted_d[p95_index] if p95_index < len(sorted_d) else sorted_d[-1]

    return LatencyStats(
        min_ms=min(durations_ms),
        max_ms=max(durations_ms),
        avg_ms=statistics.mean(durations_ms),
        p95_ms=p95,
        count=len(durations_ms),
    )


def extract_cost_from_response(response: dict[str, Any]) -> CostInfo:
    """Pull cost + token counts from a chatbot response's cost_summary block."""
    cost_summary = response.get("cost_summary")
    if not isinstance(cost_summary, dict):
        return CostInfo()
    return CostInfo(
        cost_usd=cost_summary.get("total_cost_usd"),
        input_tokens=cost_summary.get("total_input_tokens"),
        output_tokens=cost_summary.get("total_output_tokens"),
    )
