"""Tests for FinancialClient (`/fast_chat_v2`)."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

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


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retries_on_503_then_succeeds(monkeypatch, fixtures_dir: Path):
    """Client retries up to twice on 503 and returns the eventual success."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    good_body = json.loads((fixtures_dir / "financial_response.json").read_text())

    call_count = 0

    def respond(request):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return httpx.Response(503)
        return httpx.Response(200, json=good_body)

    async with respx.mock() as router:
        router.post("http://localhost:8000/fast_chat_v2").mock(side_effect=respond)
        client = FinancialClient(base_url="http://localhost:8000", timeout_s=10)
        result = await client.send("q", [], {})

    assert "NCB net interest income" in result.chatbot_text
    assert call_count == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_raises_after_max_retries_on_502(monkeypatch):
    """Client raises HTTPStatusError after exhausting 2 retries on persistent 502."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    call_count = 0

    def always_502(request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(502)

    async with respx.mock() as router:
        router.post("http://localhost:8000/fast_chat_v2").mock(side_effect=always_502)
        client = FinancialClient(base_url="http://localhost:8000", timeout_s=10)
        with pytest.raises(httpx.HTTPStatusError):
            await client.send("q", [], {})

    assert call_count == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_does_not_retry_on_400():
    """Client raises immediately on 4xx without retrying."""
    call_count = 0

    def respond_400(request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(400)

    async with respx.mock() as router:
        router.post("http://localhost:8000/fast_chat_v2").mock(side_effect=respond_400)
        client = FinancialClient(base_url="http://localhost:8000", timeout_s=10)
        with pytest.raises(httpx.HTTPStatusError):
            await client.send("q", [], {})

    assert call_count == 1


@pytest.mark.asyncio
async def test_does_not_retry_on_500():
    """Client raises immediately on 500 (not a gateway error)."""
    call_count = 0

    def respond_500(request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(500)

    async with respx.mock() as router:
        router.post("http://localhost:8000/fast_chat_v2").mock(side_effect=respond_500)
        client = FinancialClient(base_url="http://localhost:8000", timeout_s=10)
        with pytest.raises(httpx.HTTPStatusError):
            await client.send("q", [], {})

    assert call_count == 1


@pytest.mark.asyncio
async def test_retries_on_read_timeout_then_succeeds(monkeypatch, fixtures_dir: Path):
    """Client retries after ReadTimeout and returns result on eventual success."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    good_body = json.loads((fixtures_dir / "financial_response.json").read_text())

    call_count = 0

    def respond(request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ReadTimeout("timed out")
        return httpx.Response(200, json=good_body)

    async with respx.mock() as router:
        router.post("http://localhost:8000/fast_chat_v2").mock(side_effect=respond)
        client = FinancialClient(base_url="http://localhost:8000", timeout_s=10)
        result = await client.send("q", [], {})

    assert "NCB net interest income" in result.chatbot_text
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_logs_each_attempt(monkeypatch, capsys: pytest.CaptureFixture):
    """Each retry prints a progress line alongside per-conversation output."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    def always_503(request):
        return httpx.Response(503)

    async with respx.mock() as router:
        router.post("http://localhost:8000/fast_chat_v2").mock(side_effect=always_503)
        client = FinancialClient(base_url="http://localhost:8000", timeout_s=10)
        with pytest.raises(httpx.HTTPStatusError):
            await client.send("q", [], {})

    captured = capsys.readouterr()
    assert "retry 1/2" in captured.out
    assert "retry 2/2" in captured.out
