"""Tests for the runner — the conversation loop and orchestration."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from evals.client.base import ChatClientResult
from evals.persona import PersonaSpec
from evals.runner import run_conversation


def _persona(max_turns: int = 3) -> PersonaSpec:
    return PersonaSpec(
        id="p1",
        name="P1",
        category="positive",
        endpoint="fast_chat_v2",
        character="A skeptical analyst.",
        goal="Find NCB revenue growth.",
        max_turns=max_turns,
    )


def _client_result(text: str = "OK answer", cost: float = 0.001) -> ChatClientResult:
    return ChatClientResult(
        chatbot_text=text,
        chatbot_metadata={"data_found": True},
        latency_ms=500,
        ttfb_ms=None,
        cost_usd=cost,
        input_tokens=100,
        output_tokens=40,
    )


@pytest.mark.asyncio
async def test_run_conversation_persona_signals_done():
    """A 2-turn conversation where the persona says done after the 2nd reply."""
    persona = _persona(max_turns=5)
    client = MagicMock()
    client.send = AsyncMock(side_effect=[_client_result("a1"), _client_result("a2")])

    actor = MagicMock()
    from evals.persona_actor import PersonaTurn
    actor.act = AsyncMock(
        side_effect=[
            PersonaTurn(utterance="q1", done=False),
            PersonaTurn(utterance="q2", done=True, done_reason="satisfied"),
        ]
    )

    transcript = await run_conversation(
        persona=persona,
        replicate_index=0,
        chat_client=client,
        persona_actor=actor,
        max_cost_usd=1.0,
    )

    assert len(transcript.turns) == 2
    assert transcript.termination.reason == "done"
    assert transcript.termination.persona_done_reason == "satisfied"
    assert transcript.turns[0].persona_utterance == "q1"
    assert transcript.turns[1].chatbot_text == "a2"


@pytest.mark.asyncio
async def test_run_conversation_hits_max_turns():
    persona = _persona(max_turns=2)
    client = MagicMock()
    client.send = AsyncMock(return_value=_client_result())

    from evals.persona_actor import PersonaTurn
    actor = MagicMock()
    actor.act = AsyncMock(
        side_effect=[PersonaTurn(utterance=f"q{i}", done=False) for i in range(5)]
    )

    transcript = await run_conversation(
        persona=persona,
        replicate_index=0,
        chat_client=client,
        persona_actor=actor,
        max_cost_usd=1.0,
    )
    assert len(transcript.turns) == 2
    assert transcript.termination.reason == "cap"
    assert transcript.termination.at_turn == 1


@pytest.mark.asyncio
async def test_run_conversation_aborts_on_api_error():
    persona = _persona()
    client = MagicMock()
    client.send = AsyncMock(side_effect=RuntimeError("HTTP 500"))

    from evals.persona_actor import PersonaTurn
    actor = MagicMock()
    actor.act = AsyncMock(return_value=PersonaTurn(utterance="q", done=False))

    transcript = await run_conversation(
        persona=persona,
        replicate_index=0,
        chat_client=client,
        persona_actor=actor,
        max_cost_usd=1.0,
    )
    assert transcript.termination.reason == "error"
    assert "HTTP 500" in transcript.termination.error_message


@pytest.mark.asyncio
async def test_run_conversation_respects_per_convo_cost_cap():
    persona = _persona(max_turns=10)
    expensive = _client_result(cost=0.6)
    client = MagicMock()
    client.send = AsyncMock(return_value=expensive)

    from evals.persona_actor import PersonaTurn
    actor = MagicMock()
    actor.act = AsyncMock(
        side_effect=[PersonaTurn(utterance=f"q{i}", done=False) for i in range(10)]
    )

    transcript = await run_conversation(
        persona=persona,
        replicate_index=0,
        chat_client=client,
        persona_actor=actor,
        max_cost_usd=0.5,
    )
    # First turn already exceeds the cap; loop exits after capturing it.
    assert transcript.termination.reason == "error"
    assert "cost cap" in transcript.termination.error_message.lower()


import asyncio  # noqa: E402

from evals.runner import RunArtifacts, run_simulation  # noqa: E402


@pytest.mark.asyncio
async def test_run_simulation_produces_one_transcript_per_replicate_per_persona():
    persona_a = _persona(max_turns=1).model_copy(update={"id": "a"})
    persona_b = _persona(max_turns=1).model_copy(update={"id": "b"})

    from evals.persona_actor import PersonaTurn
    actor = MagicMock()
    actor.act = AsyncMock(return_value=PersonaTurn(utterance="q", done=True))

    client = MagicMock()
    client.send = AsyncMock(return_value=_client_result())

    fake_judge = MagicMock()
    from evals.judge import (
        DimensionScore, FactfulnessScore, JudgeOutput, JudgeScores, ToolUseScore,
    )
    fake_judge.evaluate = AsyncMock(
        return_value=JudgeOutput(
            scores=JudgeScores(
                groundedness=DimensionScore(score=4, justification="x"),
                factfulness=FactfulnessScore(score=None, facts_satisfied=[], justification="n/a"),
                goal_completion=DimensionScore(score=4, justification="x"),
                tool_use_appropriateness=ToolUseScore(score=4, justification="x"),
                coherence=DimensionScore(score=4, justification="x"),
                persona_handling=DimensionScore(score=4, justification="x"),
            ),
            verdict="pass",
            verdict_reason="ok",
        )
    )

    def client_for(endpoint: str):
        return client

    artifacts = await run_simulation(
        personas=[persona_a, persona_b],
        replicates=2,
        concurrency=2,
        max_cost_usd_per_run=10.0,
        max_cost_usd_per_conversation=1.0,
        chat_client_factory=client_for,
        persona_actor=actor,
        judge=fake_judge,
    )

    assert isinstance(artifacts, RunArtifacts)
    assert len(artifacts.conversations) == 4  # 2 personas × 2 replicates
    ids = {c.transcript.conversation_id for c in artifacts.conversations}
    assert ids == {"a__rep01", "a__rep02", "b__rep01", "b__rep02"}
    assert all(c.judge_output is not None for c in artifacts.conversations)
    assert artifacts.cost_capped is False


@pytest.mark.asyncio
async def test_run_simulation_respects_concurrency_limit():
    """No more than `concurrency` conversations should run simultaneously."""
    persona = _persona(max_turns=1)
    in_flight = 0
    max_observed = 0
    lock = asyncio.Lock()

    async def slow_send(*args, **kwargs):
        nonlocal in_flight, max_observed
        async with lock:
            in_flight += 1
            max_observed = max(max_observed, in_flight)
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        return _client_result()

    from evals.persona_actor import PersonaTurn
    actor = MagicMock()
    actor.act = AsyncMock(return_value=PersonaTurn(utterance="q", done=True))

    client = MagicMock()
    client.send = slow_send

    fake_judge = MagicMock()
    from evals.judge import JudgeOutput, JudgeScores, DimensionScore, FactfulnessScore, ToolUseScore
    fake_judge.evaluate = AsyncMock(
        return_value=JudgeOutput(
            scores=JudgeScores(
                groundedness=DimensionScore(score=4, justification="x"),
                factfulness=FactfulnessScore(score=None, facts_satisfied=[], justification="n/a"),
                goal_completion=DimensionScore(score=4, justification="x"),
                tool_use_appropriateness=ToolUseScore(score=4, justification="x"),
                coherence=DimensionScore(score=4, justification="x"),
                persona_handling=DimensionScore(score=4, justification="x"),
            ),
            verdict="pass",
            verdict_reason="ok",
        )
    )

    await run_simulation(
        personas=[persona],
        replicates=8,
        concurrency=3,
        max_cost_usd_per_run=10.0,
        max_cost_usd_per_conversation=1.0,
        chat_client_factory=lambda _: client,
        persona_actor=actor,
        judge=fake_judge,
    )

    assert max_observed <= 3


@pytest.mark.asyncio
async def test_run_simulation_cost_cap_marks_artifacts_and_skips_remaining():
    persona = _persona(max_turns=1)
    expensive = _client_result(cost=0.6)
    client = MagicMock()
    client.send = AsyncMock(return_value=expensive)

    from evals.persona_actor import PersonaTurn
    actor = MagicMock()
    actor.act = AsyncMock(return_value=PersonaTurn(utterance="q", done=True))

    from evals.judge import JudgeOutput, JudgeScores, DimensionScore, FactfulnessScore, ToolUseScore
    fake_judge = MagicMock()
    fake_judge.evaluate = AsyncMock(
        return_value=JudgeOutput(
            scores=JudgeScores(
                groundedness=DimensionScore(score=4, justification="x"),
                factfulness=FactfulnessScore(score=None, facts_satisfied=[], justification="n/a"),
                goal_completion=DimensionScore(score=4, justification="x"),
                tool_use_appropriateness=ToolUseScore(score=4, justification="x"),
                coherence=DimensionScore(score=4, justification="x"),
                persona_handling=DimensionScore(score=4, justification="x"),
            ),
            verdict="pass",
            verdict_reason="ok",
        )
    )

    artifacts = await run_simulation(
        personas=[persona],
        replicates=10,
        concurrency=1,
        max_cost_usd_per_run=1.0,                 # only ~1-2 convos fit
        max_cost_usd_per_conversation=1.0,
        chat_client_factory=lambda _: client,
        persona_actor=actor,
        judge=fake_judge,
    )

    assert artifacts.cost_capped is True
    assert len(artifacts.conversations) < 10
    assert len(artifacts.conversations) >= 1
