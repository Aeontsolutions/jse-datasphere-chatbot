"""Tests for AgentStreamClient (`/chat/stream`).

The endpoint is non-streaming despite the name; see
`evals/client/agent_stream.py` module docstring for the verification.
"""

import json

import httpx
import pytest
import respx

from evals.client.agent_stream import AgentStreamClient


@pytest.mark.asyncio
async def test_send_returns_response_text():
    body = {
        "response": "NCB grew net interest income to J$50B in FY2023.",
        "tools_executed": ["financial_data_query"],
        "sources": [{"title": "NCB FY2023 Annual Report"}],
        "needs_clarification": False,
        "data_found": True,
        "record_count": 12,
        "cost_summary": {
            "total_cost_usd": 0.012,
            "total_input_tokens": 4500,
            "total_output_tokens": 800,
        },
    }

    async with respx.mock(assert_all_called=True) as router:
        router.post("http://localhost:8000/chat/stream").respond(
            status_code=200, json=body
        )
        client = AgentStreamClient(base_url="http://localhost:8000", timeout_s=10)
        result = await client.send(
            query="How did NCB do in FY2023?",
            conversation_history=[],
            api_options={"enable_financial_data": True, "enable_web_search": False},
        )

    assert "J$50B" in result.chatbot_text
    assert result.chatbot_metadata["tools_executed"] == ["financial_data_query"]
    assert result.chatbot_metadata["data_found"] is True
    assert result.cost_usd == 0.012
    assert result.ttfb_ms is None  # endpoint is non-streaming
    assert result.latency_ms > 0


@pytest.mark.asyncio
async def test_send_raises_on_5xx():
    async with respx.mock() as router:
        router.post("http://localhost:8000/chat/stream").respond(status_code=500)
        client = AgentStreamClient(base_url="http://localhost:8000", timeout_s=10)
        with pytest.raises(httpx.HTTPStatusError):
            await client.send("q", [], {})


@pytest.mark.asyncio
async def test_send_passes_options():
    captured = {}

    async with respx.mock() as router:
        def cb(request):
            captured["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"response": "ok"})

        router.post("http://localhost:8000/chat/stream").mock(side_effect=cb)
        client = AgentStreamClient(base_url="http://localhost:8000", timeout_s=10)
        await client.send(
            "test",
            [],
            {"enable_web_search": True, "enable_financial_data": False, "memory_enabled": True},
        )

    assert captured["payload"]["enable_web_search"] is True
    assert captured["payload"]["enable_financial_data"] is False
    assert captured["payload"]["memory_enabled"] is True
