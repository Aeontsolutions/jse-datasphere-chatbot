"""Endpoint wiring tests for /chat/stream financial-data integration (ATS-334)."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


def _result():
    return {
        "response": "NCB 2023 revenue was J$123.5M.",
        "data_found": True,
        "record_count": 1,
        "needs_clarification": False,
        "clarification_question": None,
        "tools_executed": ["query_financial_data"],
        "sources": None,
        "filters_used": None,
        "data_preview": None,
        "chart": None,
        "web_search_results": None,
        "suggestions": None,
        "conversation_history": None,
        "warnings": None,
        "cost_summary": None,
    }


def test_chat_stream_forwards_enable_financial_data():
    from app.main import app

    with patch("app.main.AgentV2") as MockAgent:
        instance = MockAgent.return_value
        instance.run = AsyncMock(return_value=_result())
        client = TestClient(app)
        resp = client.post(
            "/chat/stream",
            json={"query": "What was NCB revenue in 2023?", "enable_financial_data": True},
        )

    assert resp.status_code == 200
    assert resp.json()["tools_executed"] == ["query_financial_data"]
    # enable_financial_data forwarded to AgentV2.run
    run_kwargs = instance.run.call_args.kwargs
    assert run_kwargs["enable_financial_data"] is True
