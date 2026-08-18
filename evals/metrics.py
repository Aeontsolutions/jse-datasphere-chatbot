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


# ---------------------------------------------------------------------------
# Eval-harness-side LLM cost tracking (persona actor + judge calls).
#
# The chatbot under test reports its own cost via `cost_summary` (see
# extract_cost_from_response above), but the persona actor and judge also
# make real Gemini calls that were previously untracked -- the run/conversation
# cost caps in runner.py only bounded the chatbot's spend, not the harness's
# own. These prices mirror fastapi_app/app/utils/cost_tracking.py; keep the
# two in sync if Gemini pricing changes.
# ---------------------------------------------------------------------------

GEMINI_PRICING_PER_MILLION: dict[str, dict[str, float]] = {
    "flash": {"input": 0.15, "output": 0.60},
    "pro": {"input": 1.25, "output": 5.00},
}


def usage_tokens_from_response(response: Any) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from a google-genai response.

    Defensive against responses that lack `usage_metadata` entirely, and
    against loosely-mocked test doubles whose auto-generated attributes
    aren't real ints -- both cases resolve to (0, 0) rather than raising.
    """
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0
    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    if not isinstance(input_tokens, int):
        input_tokens = 0
    if not isinstance(output_tokens, int):
        output_tokens = 0
    return input_tokens, output_tokens


def estimate_gemini_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost of a Gemini call from its token counts.

    Unknown/future model names fall back to (more expensive) pro pricing
    rather than silently costing $0.
    """
    key = "flash" if "flash" in model.lower() else "pro"
    pricing = GEMINI_PRICING_PER_MILLION[key]
    return (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]


# ---------------------------------------------------------------------------
# Weighted-verdict cross-check.
#
# judge_rubric.yaml documents `verdict_weights` per category, but the judge
# LLM emits its pass/fail/partial verdict holistically -- nothing computes a
# verdict from those weights, so editing them has no effect on scoring. This
# function makes them meaningful: it computes an independent, deterministic
# verdict from the per-dimension scores so report.py can flag conversations
# where the LLM's holistic verdict disagrees with the documented rubric.
# ---------------------------------------------------------------------------

_PASS_THRESHOLD = 0.8
_FAIL_THRESHOLD = 0.6


def compute_weighted_verdict(
    scores: dict[str, int | float | None],
    weights: dict[str, float],
) -> tuple[str, float | None]:
    """Compute a deterministic pass/fail/partial verdict from dimension scores.

    `scores` maps dimension name -> 1-5 score (or None if not scored, e.g.
    factfulness when the persona declares no expected_facts). `weights` is
    the resolved dim -> weight mapping for the persona's category (i.e.
    `rubric["verdict_weights"][category]` from judge_rubric.yaml).

    Missing dimensions are dropped and the remaining weights renormalized, so
    a null factfulness score doesn't silently zero out the average.

    `goal_completion_inverted` is a synthetic dimension for negative personas:
    a HIGH goal_completion (the adversarial ask succeeded) is bad, so its
    value is (6 - goal_completion) rather than goal_completion itself.

    Returns (verdict, weighted_score) where weighted_score is a 0-1 float,
    or (verdict, None) if no weighted dimension had a usable score.
    """
    weighted_sum = 0.0
    weight_total = 0.0
    for dim, weight in weights.items():
        if dim == "goal_completion_inverted":
            raw = scores.get("goal_completion")
            value = (6 - raw) if raw is not None else None
        else:
            value = scores.get(dim)
        if value is None:
            continue
        weighted_sum += weight * (value / 5.0)
        weight_total += weight

    if weight_total == 0.0:
        return "partial", None

    weighted_score = weighted_sum / weight_total
    if weighted_score >= _PASS_THRESHOLD:
        return "pass", weighted_score
    if weighted_score < _FAIL_THRESHOLD:
        return "fail", weighted_score
    return "partial", weighted_score
