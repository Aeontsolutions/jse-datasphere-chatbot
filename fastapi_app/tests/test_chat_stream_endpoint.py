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


# ---------------------------------------------------------------------------
# Generation-outcome logging (issue #72)
# ---------------------------------------------------------------------------


def _log_spy_call(agent_result):
    """Drive /chat/stream with a stub agent and return build_row's kwargs."""
    from unittest.mock import MagicMock

    from app.interaction_log import InteractionLogger
    from app.main import app, get_interaction_logger_dep

    fake_logger = MagicMock()
    fake_logger.log = AsyncMock()
    app.dependency_overrides[get_interaction_logger_dep] = lambda: fake_logger
    try:
        with (
            patch("app.main.AgentV2") as MockAgent,
            patch.object(InteractionLogger, "build_row", wraps=InteractionLogger.build_row) as spy,
        ):
            MockAgent.return_value.run = AsyncMock(return_value=agent_result)
            resp = TestClient(app).post("/chat/stream", json={"query": "top 3 stocks to buy"})
        assert resp.status_code == 200
        return spy.call_args.kwargs
    finally:
        app.dependency_overrides.clear()


def test_chat_stream_logs_finish_reason():
    result = _result()
    result["finish_reason"] = "MAX_TOKENS"
    assert _log_spy_call(result)["finish_reason"] == "MAX_TOKENS"


def test_chat_stream_logs_none_finish_reason_when_absent():
    """A cache hit or an older agent result carries no outcome — stay null, not False."""
    assert _log_spy_call(_result())["finish_reason"] is None


def test_chat_stream_logs_thinking_tokens_from_cost_summary():
    from app.models import CostSummary, PhaseCost

    result = _result()
    result["finish_reason"] = "MAX_TOKENS"
    result["cost_summary"] = CostSummary(
        total_input_tokens=392,
        total_output_tokens=9,
        total_thinking_tokens=243,
        total_cost_usd=0.0001,
        phases=[PhaseCost(phase="refusal", model="gemini-2.5-flash")],
    )
    kwargs = _log_spy_call(result)
    assert kwargs["thinking_tokens"] == 243
    assert kwargs["total_tokens"] == 392 + 9 + 243
