"""HTTP client for the chatbot's `/fast_chat_v2` (non-streaming) endpoint."""

from __future__ import annotations

import time
from typing import Any

import httpx

from evals.client._retry import send_with_retry
from evals.client.base import ChatClientResult
from evals.metrics import extract_cost_from_response


class FinancialClient:
    """Non-streaming client targeting `POST /fast_chat_v2`."""

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
        }
        url = f"{self._base_url}/fast_chat_v2"

        async def _post() -> dict:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()

        start = time.perf_counter()
        data = await send_with_retry(_post, label="/fast_chat_v2")
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
