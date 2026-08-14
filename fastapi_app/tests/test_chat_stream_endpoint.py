"""Endpoint wiring tests for /chat/stream.

/chat/stream constructs AgentV2 with no dependencies (it no longer has a
BigQuery/financial-data branch — see app/_archive/README.md, 2026-08-15 entry)
and forwards only query/conversation_history/enable_web_search to .run().
"""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


def _result():
    return {
        "response": "Here is some recent JSE news.",
        "data_found": True,
        "record_count": 0,
        "needs_clarification": False,
        "clarification_question": None,
        "tools_executed": ["google_search"],
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


def test_chat_stream_constructs_agent_with_no_args():
    from app.main import app

    with patch("app.main.AgentV2") as MockAgent:
        instance = MockAgent.return_value
        instance.run = AsyncMock(return_value=_result())
        client = TestClient(app)
        resp = client.post("/chat/stream", json={"query": "Any recent JSE news?"})

    assert resp.status_code == 200
    MockAgent.assert_called_once_with()


def test_chat_stream_forwards_enable_web_search():
    from app.main import app

    with patch("app.main.AgentV2") as MockAgent:
        instance = MockAgent.return_value
        instance.run = AsyncMock(return_value=_result())
        client = TestClient(app)
        resp = client.post(
            "/chat/stream",
            json={"query": "Any recent JSE news?", "enable_web_search": False},
        )

    assert resp.status_code == 200
    run_kwargs = instance.run.call_args.kwargs
    assert run_kwargs["enable_web_search"] is False
    assert "enable_financial_data" not in run_kwargs


def test_chat_stream_accepts_but_ignores_legacy_enable_financial_data():
    # Backward compatible with Django callers still sending this field
    # (see AgentChatRequest / CHATBOT_INTEGRATION_SPEC.md): the request is
    # accepted, but the flag is no longer forwarded to AgentV2.run.
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
    run_kwargs = instance.run.call_args.kwargs
    assert "enable_financial_data" not in run_kwargs
