"""HTTP client for the chatbot's `/chat/stream` endpoint.

Despite the URL name, `/chat/stream` is NOT a streaming endpoint — it
runs the agent to completion and returns a plain JSON `AgentChatResponse`.
Verified against `fastapi_app/app/main.py` lines 987-1015. If a real
streaming variant is added in the future, fork a separate client rather
than reintroducing SSE parsing here.

The class name `AgentStreamClient` is kept for stability of imports
across the eval suite.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from evals.client.base import ChatClientResult
from evals.metrics import extract_cost_from_response


class AgentStreamClient:
    """Client targeting `POST /chat/stream`.

    The endpoint name implies SSE, but the response is a single JSON
    body (`AgentChatResponse`). This client behaves like `FinancialClient`:
    one POST in, full JSON out. `ttfb_ms` is always `None` because there
    is no first-byte signal distinct from full completion.
    """

    def __init__(self, base_url: str, timeout_s: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def send(
        self,
        query: str,
        conversation_history: list[dict[str, str]],
        api_options: dict[str, Any],
    ) -> ChatClientResult:
        payload = {
            "query": query,
            "conversation_history": conversation_history,
            "memory_enabled": api_options.get("memory_enabled", True),
            "enable_web_search": api_options.get("enable_web_search", True),
            "enable_financial_data": api_options.get("enable_financial_data", True),
        }
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(f"{self._base_url}/chat/stream", json=payload)
            response.raise_for_status()
            data = response.json()
        elapsed_ms = (time.perf_counter() - start) * 1000

        cost = extract_cost_from_response(data)
        return ChatClientResult(
            chatbot_text=data.get("response", ""),
            chatbot_metadata=data,
            latency_ms=elapsed_ms,
            ttfb_ms=None,
            cost_usd=cost.cost_usd,
            input_tokens=cost.input_tokens,
            output_tokens=cost.output_tokens,
        )
