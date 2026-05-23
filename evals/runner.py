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


import asyncio
from typing import Awaitable, Callable

from evals.judge import Judge, JudgeOutput


class ConversationArtifact:
    """One transcript plus its judge output (if successful)."""

    def __init__(
        self,
        transcript: Transcript,
        judge_output: JudgeOutput | None,
        judge_failed: bool,
        judge_error: str | None,
    ) -> None:
        self.transcript = transcript
        self.judge_output = judge_output
        self.judge_failed = judge_failed
        self.judge_error = judge_error


class RunArtifacts:
    """All conversations from a single run, plus a cost-cap flag."""

    def __init__(
        self,
        conversations: list[ConversationArtifact],
        cost_capped: bool,
    ) -> None:
        self.conversations = conversations
        self.cost_capped = cost_capped


async def run_simulation(
    personas: list[PersonaSpec],
    replicates: int,
    concurrency: int,
    max_cost_usd_per_run: float,
    max_cost_usd_per_conversation: float,
    chat_client_factory: Callable[[str], ChatClient],
    persona_actor: PersonaActor,
    judge: Judge,
) -> RunArtifacts:
    """Run all personas × replicates concurrently with a global cost cap."""
    semaphore = asyncio.Semaphore(concurrency)
    judge_semaphore = asyncio.Semaphore(concurrency * 2)

    running_cost = 0.0
    cost_lock = asyncio.Lock()
    cost_capped = False
    cancel_event = asyncio.Event()

    async def one(persona: PersonaSpec, rep: int) -> ConversationArtifact | None:
        nonlocal running_cost, cost_capped
        if cancel_event.is_set():
            return None
        async with semaphore:
            if cancel_event.is_set():
                return None
            chat_client = chat_client_factory(persona.endpoint)
            transcript = await run_conversation(
                persona=persona,
                replicate_index=rep,
                chat_client=chat_client,
                persona_actor=persona_actor,
                max_cost_usd=max_cost_usd_per_conversation,
            )

            convo_cost = float(transcript.totals()["cost_usd"])
            async with cost_lock:
                running_cost += convo_cost
                if running_cost > max_cost_usd_per_run:
                    cost_capped = True
                    cancel_event.set()

        async with judge_semaphore:
            try:
                output = await judge.evaluate(persona=persona, transcript=transcript)
                return ConversationArtifact(transcript, output, False, None)
            except Exception as exc:
                return ConversationArtifact(transcript, None, True, f"{type(exc).__name__}: {exc}")

    tasks = [
        asyncio.create_task(one(persona, rep))
        for persona in personas
        for rep in range(replicates)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    conversations = [r for r in results if r is not None]

    return RunArtifacts(conversations=conversations, cost_capped=cost_capped)
