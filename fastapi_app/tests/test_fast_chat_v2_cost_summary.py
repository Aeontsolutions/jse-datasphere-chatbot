"""Regression test for issue #65: /fast_chat_v2 never returned cost_summary.

FinancialDataResponse (the response model for /fast_chat_v2) had no
cost_summary field at all, so the cost data the endpoint already computes
(via cost_sink, logged to BigQuery) never made it into the HTTP response --
unlike /chat/stream, whose AgentChatResponse does carry it. Confirmed by
loadtest.py, which already assumes /fast_chat_v2 responses carry
cost_summary (using its presence as the cache-hit/miss signal).
"""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import FinancialDataFilters
from app.utils.cost_tracking import CostResult, TokenUsage


def _filters():
    return FinancialDataFilters(
        companies=["NCB Financial Group"],
        years=["2023"],
        standard_items=["net_profit"],
        interpretation="NCB net profit for 2023",
    )


def _mock_financial_manager():
    async def parse_user_query(query, conversation_history, last_query_data, cost_sink=None):
        if cost_sink is not None:
            cost_sink.append(
                CostResult(
                    input_cost=0.001,
                    output_cost=0.002,
                    total_cost=0.003,
                    model="gemini-2.5-flash",
                    phase="query_parsing",
                    token_usage=TokenUsage(
                        input_tokens=100, output_tokens=50, cached_tokens=0, total_tokens=150
                    ),
                )
            )
        return _filters()

    async def format_response(
        results,
        query,
        interpretation,
        is_follow_up,
        conversation_history,
        unrecognized_items=None,
        cost_sink=None,
    ):
        if cost_sink is not None:
            cost_sink.append(
                CostResult(
                    input_cost=0.004,
                    output_cost=0.005,
                    total_cost=0.009,
                    model="gemini-2.5-pro",
                    phase="response_synthesis",
                    token_usage=TokenUsage(
                        input_tokens=200, output_tokens=80, cached_tokens=0, total_tokens=280
                    ),
                )
            )
        return "NCB net profit for 2023 was J$5B."

    manager = Mock()
    manager.parse_user_query = parse_user_query
    manager.validate_data_availability = Mock(return_value={"warnings": [], "suggestions": []})
    manager.query_data = Mock(return_value=[])
    manager.format_response = format_response
    manager.describe_source = Mock(return_value=None)
    return manager


def test_fast_chat_v2_populates_cost_summary_on_cache_miss(monkeypatch):
    monkeypatch.setattr(app.state, "financial_manager", _mock_financial_manager(), raising=False)
    monkeypatch.setattr(app.state, "response_cache", None, raising=False)
    monkeypatch.setattr(app.state, "interaction_logger", None, raising=False)

    client = TestClient(app)
    resp = client.post(
        "/fast_chat_v2",
        json={
            "query": "What was NCB net profit for 2023?",
            "conversation_history": [{"role": "user", "content": "hi"}],
        },
    )

    assert resp.status_code == 200
    cost_summary = resp.json()["cost_summary"]
    assert cost_summary is not None
    assert cost_summary["total_input_tokens"] == 300
    assert cost_summary["total_output_tokens"] == 130
    assert cost_summary["total_cost_usd"] == pytest.approx(0.012)
    phases = cost_summary["phases"]
    assert [p["phase"] for p in phases] == ["query_parsing", "response_synthesis"]


def test_fast_chat_v2_cache_hit_has_no_cost_summary(monkeypatch):
    class FakeCache:
        async def get(self, key):
            return {
                "response": "cached answer",
                "data_found": True,
                "record_count": 0,
                "filters_used": _filters().model_dump(),
                "data_preview": None,
                "chart": None,
                "sources": None,
                "warnings": None,
                "suggestions": None,
            }

        async def set(self, key, value):
            raise AssertionError("set() should not be called on a cache hit")

    monkeypatch.setattr(app.state, "financial_manager", _mock_financial_manager(), raising=False)
    monkeypatch.setattr(app.state, "response_cache", FakeCache(), raising=False)
    monkeypatch.setattr(app.state, "interaction_logger", None, raising=False)

    client = TestClient(app)
    resp = client.post(
        "/fast_chat_v2",
        json={"query": "What was NCB net profit for 2023?", "conversation_history": []},
    )

    assert resp.status_code == 200
    assert resp.json()["cost_summary"] is None
