"""Common protocol shared by the financial and stream chat clients."""

from __future__ import annotations

from typing import Any, Protocol


class ChatClientResult:
    """Outcome of a single chatbot API call."""

    def __init__(
        self,
        chatbot_text: str,
        chatbot_metadata: dict[str, Any],
        latency_ms: float,
        ttfb_ms: float | None,
        cost_usd: float | None,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> None:
        self.chatbot_text = chatbot_text
        self.chatbot_metadata = chatbot_metadata
        self.latency_ms = latency_ms
        self.ttfb_ms = ttfb_ms
        self.cost_usd = cost_usd
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class ChatClient(Protocol):
    """Abstract chatbot client. Each endpoint gets one implementation."""

    async def send(
        self,
        query: str,
        conversation_history: list[dict[str, str]],
        api_options: dict[str, Any],
    ) -> ChatClientResult:
        """Send a single user turn; return the chatbot's full response."""
        ...
