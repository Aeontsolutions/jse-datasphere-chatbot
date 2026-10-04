"""/fast_chat_v2 interaction-log cost fields.

The row used to carry no thinking tokens, an input + output `total_tokens`, and
`phase_costs_json` holding phase *names* only, so a per-phase cost split could
not be rebuilt from BigQuery.
"""

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from fastapi.testclient import TestClient

from app.interaction_log import InteractionLogger
from app.main import (
    app,
    get_financial_manager,
    get_interaction_logger_dep,
    get_response_cache_dep,
)
from app.models import FinancialDataFilters
from app.utils.cost_tracking import CostResult, TokenUsage


def _cost(phase, model, **usage):
    return CostResult(
        input_cost=0.001,
        output_cost=0.002,
        total_cost=0.003,
        model=model,
        phase=phase,
        token_usage=TokenUsage(**usage),
    )


def _financial_manager():
    async def parse_user_query(query, history, last_query_data, cost_sink=None):
        cost_sink.append(
            _cost(
                "parse_user_query",
                "parser-model",
                input_tokens=100,
                output_tokens=20,
                thinking_tokens=30,
                total_tokens=150,
            )
        )
        return FinancialDataFilters(
            companies=["NCB"], years=["2023"], standard_items=["net_profit"], interpretation="x"
        )

    async def format_response(results, query, interpretation, is_follow_up, history, **kwargs):
        kwargs["cost_sink"].append(
            _cost(
                "format_response",
                "formatter-model",
                input_tokens=200,
                output_tokens=40,
                thinking_tokens=10,
                total_tokens=250,
            )
        )
        return "NCB earned J$5B."

    manager = Mock()
    manager.parse_user_query = parse_user_query
    manager.format_response = format_response
    manager.validate_data_availability = Mock(return_value={"warnings": [], "suggestions": []})
    manager.query_data = Mock(return_value=[])
    manager.describe_source = Mock(return_value=None)
    return manager


def _logged_kwargs():
    fake_logger = MagicMock()
    fake_logger.log = AsyncMock()
    app.dependency_overrides[get_interaction_logger_dep] = lambda: fake_logger
    app.dependency_overrides[get_financial_manager] = _financial_manager
    app.dependency_overrides[get_response_cache_dep] = lambda: None
    try:
        with patch.object(InteractionLogger, "build_row", wraps=InteractionLogger.build_row) as spy:
            resp = TestClient(app).post("/fast_chat_v2", json={"query": "NCB profit 2023?"})
        assert resp.status_code == 200
        return spy.call_args.kwargs
    finally:
        app.dependency_overrides.clear()


def test_fast_chat_v2_logs_thinking_and_api_reported_total():
    kwargs = _logged_kwargs()

    assert kwargs["input_tokens"] == 300
    assert kwargs["output_tokens"] == 60
    assert kwargs["thinking_tokens"] == 40
    # input + output alone would say 360 and silently drop the thinking.
    assert kwargs["total_tokens"] == 400


def test_fast_chat_v2_logs_per_phase_records_that_survive_json():
    kwargs = _logged_kwargs()
    row = InteractionLogger.build_row(
        endpoint="fast_chat_v2",
        query="q",
        response="r",
        **{
            k: v
            for k, v in kwargs.items()
            if k in {"phase_costs", "input_tokens", "output_tokens", "cost_usd"}
        },
    )

    phases = json.loads(row["phase_costs_json"])
    assert [p["phase"] for p in phases] == ["parse_user_query", "format_response"]
    assert [p["model"] for p in phases] == ["parser-model", "formatter-model"]
    assert phases[0]["thinking_tokens"] == 30
    assert sum(p["total_cost_usd"] for p in phases) == kwargs["cost_usd"]
