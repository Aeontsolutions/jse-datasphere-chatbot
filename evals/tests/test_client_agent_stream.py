"""Tests for AgentStreamClient (`/chat/stream`)."""

from pathlib import Path

import httpx
import pytest
import respx

from evals.client.agent_stream import AgentStreamClient


@pytest.mark.asyncio
async def test_send_assembles_final_payload(fixtures_dir: Path):
    raw = (fixtures_dir / "stream_response.txt").read_text()

    async with respx.mock(assert_all_called=True) as router:
        router.post("http://localhost:8000/chat/stream").respond(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=raw.encode("utf-8"),
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
    assert result.ttfb_ms is not None
    assert result.ttfb_ms <= result.latency_ms


@pytest.mark.asyncio
async def test_send_raises_when_no_final_event():
    no_final = "data: {\"type\": \"progress\", \"step\": \"x\"}\n\n"

    async with respx.mock() as router:
        router.post("http://localhost:8000/chat/stream").respond(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=no_final.encode("utf-8"),
        )
        client = AgentStreamClient(base_url="http://localhost:8000", timeout_s=10)
        with pytest.raises(RuntimeError, match="no final"):
            await client.send("q", [], {})


@pytest.mark.asyncio
async def test_send_passes_options():
    captured = {}

    async with respx.mock() as router:
        def cb(request):
            import json as _json
            captured["payload"] = _json.loads(request.content)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b'data: {"type": "final", "payload": {"response": "ok"}}\n\n',
            )

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
