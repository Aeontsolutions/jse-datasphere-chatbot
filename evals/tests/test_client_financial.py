"""Tests for FinancialClient (`/fast_chat_v2`)."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from evals.client.financial import FinancialClient


@pytest.mark.asyncio
async def test_send_returns_response_text(fixtures_dir: Path):
    body = json.loads((fixtures_dir / "financial_response.json").read_text())

    async with respx.mock(assert_all_called=True) as router:
        router.post("http://localhost:8000/fast_chat_v2").respond(
            status_code=200, json=body
        )
        client = FinancialClient(base_url="http://localhost:8000", timeout_s=10)
        result = await client.send(
            query="NCB net interest income last 2 years",
            conversation_history=[],
            api_options={"memory_enabled": True},
        )

    assert "NCB net interest income" in result.chatbot_text
    assert result.chatbot_metadata["data_found"] is True
    assert result.cost_usd == 0.0034
    assert result.input_tokens == 1200
    assert result.output_tokens == 380
    assert result.ttfb_ms is None  # non-streaming
    assert result.latency_ms > 0


@pytest.mark.asyncio
async def test_send_raises_on_5xx():
    async with respx.mock() as router:
        router.post("http://localhost:8000/fast_chat_v2").respond(status_code=500)
        client = FinancialClient(base_url="http://localhost:8000", timeout_s=10)
        with pytest.raises(httpx.HTTPStatusError):
            await client.send("q", [], {})


@pytest.mark.asyncio
async def test_send_passes_conversation_history():
    captured = {}

    async with respx.mock() as router:
        def callback(request):
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "response": "ok",
                    "data_found": False,
                    "record_count": 0,
                    "filters_used": {
                        "companies": [], "symbols": [], "years": [],
                        "standard_items": [], "interpretation": "",
                        "data_availability_note": "", "is_follow_up": False,
                        "context_used": "",
                    },
                },
            )

        router.post("http://localhost:8000/fast_chat_v2").mock(side_effect=callback)
        client = FinancialClient(base_url="http://localhost:8000", timeout_s=10)
        await client.send(
            "follow up",
            [{"role": "user", "content": "first"}, {"role": "assistant", "content": "answer"}],
            {"memory_enabled": False},
        )

    assert captured["payload"]["query"] == "follow up"
    assert len(captured["payload"]["conversation_history"]) == 2
    assert captured["payload"]["memory_enabled"] is False
