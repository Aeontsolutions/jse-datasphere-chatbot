"""Per-conversation transcript and turn data models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    """One persona ↔ chatbot exchange."""

    turn_index: int
    persona_utterance: str
    chatbot_text: str
    chatbot_metadata: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float
    ttfb_ms: float | None = None
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    persona_actor_cost_usd: float | None = None
    """Cost of the persona actor's Gemini call that produced this turn's
    utterance. Distinct from `cost_usd`, which is the chatbot-under-test's
    own reported spend."""


class TerminationReason(BaseModel):
    """Why a conversation ended."""

    reason: Literal["done", "cap", "error"]
    at_turn: int
    persona_done_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class Transcript(BaseModel):
    """Full record of a single conversation, before judging."""

    conversation_id: str
    persona_id: str
    replicate_index: int
    endpoint: Literal["fast_chat_v2", "chat_stream"]
    turns: list[ChatTurn] = Field(default_factory=list)
    termination: TerminationReason

    def totals(self) -> dict[str, float | int]:
        """Aggregate latency, cost, and turn count across turns.

        `cost_usd` is the chatbot-under-test's spend PLUS the persona actor's
        spend generating each turn's utterance -- both are real Gemini calls
        this run paid for. It does not include the judge's cost, which is
        billed once per conversation rather than per turn; see
        ConversationArtifact.judge_cost_usd for that.
        """
        return {
            "turns": len(self.turns),
            "latency_ms": sum(t.latency_ms for t in self.turns),
            "cost_usd": sum(
                (t.cost_usd or 0.0) + (t.persona_actor_cost_usd or 0.0) for t in self.turns
            ),
        }
