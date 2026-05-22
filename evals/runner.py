"""Conversation runner — orchestrates persona ↔ chatbot turns."""

from __future__ import annotations

from typing import Any

from evals.client.base import ChatClient
from evals.persona import PersonaSpec
from evals.persona_actor import PersonaActor
from evals.transcript import ChatTurn, TerminationReason, Transcript


async def run_conversation(
    persona: PersonaSpec,
    replicate_index: int,
    chat_client: ChatClient,
    persona_actor: PersonaActor,
    max_cost_usd: float,
) -> Transcript:
    """Run one persona ↔ chatbot conversation to completion."""
    conversation_id = f"{persona.id}__rep{replicate_index + 1:02d}"
    turns: list[ChatTurn] = []
    chatbot_history: list[dict[str, str]] = []
    persona_history: list[dict[str, str]] = []
    running_cost = 0.0

    termination: TerminationReason | None = None

    for turn_index in range(persona.max_turns):
        try:
            persona_turn = await persona_actor.act(
                persona=persona,
                transcript_history=persona_history,
                replicate_index=replicate_index,
            )
        except Exception as exc:
            termination = TerminationReason(
                reason="error",
                at_turn=turn_index,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            break

        try:
            result = await chat_client.send(
                query=persona_turn.utterance,
                conversation_history=chatbot_history,
                api_options=persona.api_options,
            )
        except Exception as exc:
            termination = TerminationReason(
                reason="error",
                at_turn=turn_index,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            break

        chat_turn = ChatTurn(
            turn_index=turn_index,
            persona_utterance=persona_turn.utterance,
            chatbot_text=result.chatbot_text,
            chatbot_metadata=result.chatbot_metadata,
            latency_ms=result.latency_ms,
            ttfb_ms=result.ttfb_ms,
            cost_usd=result.cost_usd,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        turns.append(chat_turn)
        chatbot_history.append({"role": "user", "content": persona_turn.utterance})
        chatbot_history.append({"role": "assistant", "content": result.chatbot_text})
        persona_history.append(
            {"persona_utterance": persona_turn.utterance, "chatbot_text": result.chatbot_text}
        )
        running_cost += result.cost_usd or 0.0

        if running_cost > max_cost_usd:
            termination = TerminationReason(
                reason="error",
                at_turn=turn_index,
                error_type="CostCapExceeded",
                error_message=f"per-conversation cost cap ${max_cost_usd:.2f} exceeded at ${running_cost:.4f}",
            )
            break

        if persona_turn.done:
            termination = TerminationReason(
                reason="done",
                at_turn=turn_index,
                persona_done_reason=persona_turn.done_reason,
            )
            break

    if termination is None:
        termination = TerminationReason(reason="cap", at_turn=persona.max_turns - 1)

    return Transcript(
        conversation_id=conversation_id,
        persona_id=persona.id,
        replicate_index=replicate_index,
        endpoint=persona.endpoint,
        turns=turns,
        termination=termination,
    )
