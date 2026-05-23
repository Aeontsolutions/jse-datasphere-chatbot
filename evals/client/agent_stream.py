"""HTTP client for the chatbot's `/chat/stream` (SSE) endpoint.

UNVERIFIED ENVELOPE WARNING:
This client assumes `/chat/stream` emits SSE events of the form
`data: {"type": "progress"|"final", "payload": ...}`. The real endpoint's
shape has NOT been verified against a running chatbot yet. Before the first
live run, confirm the real envelope with:

    curl -v -N -X POST http://localhost:8000/chat/stream \\
      -H "Content-Type: application/json" \\
      -d '{"query":"test","conversation_history":[]}' | head -20

If the response is plain JSON (not SSE), `/chat/stream` is effectively a
non-streaming endpoint and this client should be rewritten to call it
like `FinancialClient`. If the SSE envelope differs (e.g., uses
`event: result\\ndata: <json>` instead of the `type` field), update the
parser in `send()` to match.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from evals.client.base import ChatClientResult
from evals.metrics import extract_cost_from_response


class AgentStreamClient:
    """Streaming client targeting `POST /chat/stream` (SSE).

    See module docstring for the unverified envelope warning. Consumes
    `data: <json>` events; the event with `type == "final"` carries
    the assembled `AgentChatResponse` under `payload`. Records TTFB and
    total elapsed time separately.
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

        final_payload: dict[str, Any] | None = None
        start = time.perf_counter()
        ttfb_ms: float | None = None

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            async with client.stream(
                "POST", f"{self._base_url}/chat/stream", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if ttfb_ms is None:
                        ttfb_ms = (time.perf_counter() - start) * 1000
                    if not line.startswith("data:"):
                        continue
                    body = line[len("data:") :].strip()
                    if not body:
                        continue
                    try:
                        event = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "final":
                        final_payload = event.get("payload") or {}

        elapsed_ms = (time.perf_counter() - start) * 1000

        if final_payload is None:
            raise RuntimeError(
                "stream ended with no final event; the chatbot may have failed mid-stream"
            )

        cost = extract_cost_from_response(final_payload)
        return ChatClientResult(
            chatbot_text=final_payload.get("response", ""),
            chatbot_metadata=final_payload,
            latency_ms=elapsed_ms,
            ttfb_ms=ttfb_ms,
            cost_usd=cost.cost_usd,
            input_tokens=cost.input_tokens,
            output_tokens=cost.output_tokens,
        )
