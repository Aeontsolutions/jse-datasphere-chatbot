"""Tests for latency stats and cost extraction utilities."""

from unittest.mock import MagicMock

import pytest

from evals.metrics import (
    LatencyStats,
    compute_weighted_verdict,
    estimate_gemini_cost_usd,
    extract_cost_from_response,
    latency_stats,
    usage_tokens_from_response,
)


def test_latency_stats_basic():
    stats = latency_stats([100, 200, 300, 400, 500])
    assert stats.count == 5
    assert stats.min_ms == 100
    assert stats.max_ms == 500
    assert stats.avg_ms == 300
    # p95 of 5-sample list — sorted index = int(5*0.95)=4 → 500
    assert stats.p95_ms == 500


def test_latency_stats_empty():
    stats = latency_stats([])
    assert stats == LatencyStats(min_ms=0, max_ms=0, avg_ms=0, p95_ms=0, count=0)


def test_latency_stats_single():
    stats = latency_stats([42.5])
    assert stats.count == 1
    assert stats.min_ms == 42.5
    assert stats.max_ms == 42.5
    assert stats.avg_ms == 42.5
    assert stats.p95_ms == 42.5


def test_extract_cost_present():
    response = {
        "cost_summary": {
            "total_cost_usd": 0.0034,
            "total_input_tokens": 1200,
            "total_output_tokens": 380,
        }
    }
    cost = extract_cost_from_response(response)
    assert cost.cost_usd == 0.0034
    assert cost.input_tokens == 1200
    assert cost.output_tokens == 380


def test_extract_cost_missing():
    cost = extract_cost_from_response({"response": "x"})
    assert cost.cost_usd is None
    assert cost.input_tokens is None
    assert cost.output_tokens is None


# ---------------------------------------------------------------------------
# usage_tokens_from_response — extracting token counts from a genai response
# ---------------------------------------------------------------------------


def test_usage_tokens_from_response_real_shape():
    response = MagicMock()
    response.usage_metadata.prompt_token_count = 1200
    response.usage_metadata.candidates_token_count = 340
    assert usage_tokens_from_response(response) == (1200, 340)


def test_usage_tokens_from_response_missing_usage_metadata():
    response = MagicMock(spec=["text"])  # no usage_metadata attribute at all
    assert usage_tokens_from_response(response) == (0, 0)


def test_usage_tokens_from_response_non_int_fields_default_to_zero():
    """A loosely-mocked response (e.g. in unit tests) shouldn't blow up cost math."""
    response = MagicMock()  # auto-mocked attrs are MagicMocks, not ints
    assert usage_tokens_from_response(response) == (0, 0)


# ---------------------------------------------------------------------------
# estimate_gemini_cost_usd — mirrors fastapi_app/app/utils/cost_tracking.py
# ---------------------------------------------------------------------------


def test_estimate_gemini_cost_usd_flash():
    cost = estimate_gemini_cost_usd("gemini-2.5-flash", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 0.15 + 0.60


def test_estimate_gemini_cost_usd_pro():
    cost = estimate_gemini_cost_usd("gemini-2.5-pro", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 1.25 + 5.00


def test_estimate_gemini_cost_usd_zero_tokens():
    assert estimate_gemini_cost_usd("gemini-2.5-flash", input_tokens=0, output_tokens=0) == 0.0


def test_estimate_gemini_cost_usd_unknown_model_defaults_to_pro_pricing():
    """Conservative fallback: unknown/future models are priced like pro, not free."""
    cost = estimate_gemini_cost_usd("gemini-3-mystery", input_tokens=1_000_000, output_tokens=0)
    assert cost == 1.25


# ---------------------------------------------------------------------------
# compute_weighted_verdict — cross-checks the judge's free-form verdict
# against the documented verdict_weights from judge_rubric.yaml
# ---------------------------------------------------------------------------


def _scores(**overrides):
    base = {
        "groundedness": 5,
        "factfulness": 5,
        "goal_completion": 5,
        "tool_use_appropriateness": 5,
        "coherence": 5,
        "persona_handling": 5,
    }
    base.update(overrides)
    return base


_POSITIVE_WEIGHTS = {
    "groundedness": 0.30,
    "goal_completion": 0.30,
    "factfulness": 0.20,
    "tool_use_appropriateness": 0.10,
    "coherence": 0.05,
    "persona_handling": 0.05,
}

_NEGATIVE_WEIGHTS = {
    "persona_handling": 0.50,
    "goal_completion_inverted": 0.30,
    "coherence": 0.10,
    "groundedness": 0.10,
}


def test_weighted_verdict_all_fives_is_pass():
    verdict, weighted = compute_weighted_verdict(_scores(), _POSITIVE_WEIGHTS)
    assert verdict == "pass"
    assert weighted == 1.0


def test_weighted_verdict_all_ones_is_fail():
    verdict, weighted = compute_weighted_verdict(_scores(**{k: 1 for k in _scores()}), _POSITIVE_WEIGHTS)
    assert verdict == "fail"
    assert weighted == pytest.approx(0.2)


def test_weighted_verdict_mid_scores_is_partial():
    verdict, weighted = compute_weighted_verdict(_scores(**{k: 3 for k in _scores()}), _POSITIVE_WEIGHTS)
    assert verdict == "partial"
    assert weighted == pytest.approx(0.6)


def test_weighted_verdict_ignores_missing_factfulness_and_renormalizes():
    """expected_facts is often empty -> factfulness.score is None; weights must renormalize."""
    scores = _scores(factfulness=None)
    verdict, weighted = compute_weighted_verdict(scores, _POSITIVE_WEIGHTS)
    assert verdict == "pass"
    assert weighted == 1.0


def test_weighted_verdict_negative_category_inverts_goal_completion():
    """For negative personas, a HIGH goal_completion (bot got tricked) should hurt the verdict."""
    scores = _scores(goal_completion=5, persona_handling=5, coherence=5, groundedness=5)
    verdict, weighted = compute_weighted_verdict(scores, _NEGATIVE_WEIGHTS)
    # goal_completion=5 means the adversarial goal was fully achieved -> inverted value is worst (1/5)
    assert weighted == pytest.approx(1.0 * 0.70 + 0.2 * 0.30)
    assert verdict == "partial"


def test_weighted_verdict_negative_category_low_goal_completion_is_good():
    scores = _scores(goal_completion=1, persona_handling=5, coherence=5, groundedness=5)
    verdict, weighted = compute_weighted_verdict(scores, _NEGATIVE_WEIGHTS)
    assert weighted == 1.0
    assert verdict == "pass"
