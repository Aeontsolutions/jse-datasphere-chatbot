"""Tests for latency stats and cost extraction utilities."""

from evals.metrics import LatencyStats, extract_cost_from_response, latency_stats


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
