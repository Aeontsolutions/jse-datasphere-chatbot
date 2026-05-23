"""Tests for ChatTurn / Transcript data models + serialization."""

import pytest

from evals.transcript import ChatTurn, TerminationReason, Transcript


def test_chat_turn_roundtrip():
    turn = ChatTurn(
        turn_index=0,
        persona_utterance="Show me NCB profitability",
        chatbot_text="Net interest income was J$50B in FY2023.",
        chatbot_metadata={"data_found": True, "record_count": 12},
        latency_ms=1820,
        ttfb_ms=None,
        cost_usd=0.0034,
        input_tokens=1200,
        output_tokens=380,
    )
    d = turn.model_dump()
    back = ChatTurn(**d)
    assert back == turn


def test_transcript_totals_computed():
    turns = [
        ChatTurn(
            turn_index=i,
            persona_utterance=f"q{i}",
            chatbot_text=f"a{i}",
            chatbot_metadata={},
            latency_ms=1000.0,
            ttfb_ms=None,
            cost_usd=0.001,
            input_tokens=100,
            output_tokens=50,
        )
        for i in range(3)
    ]
    t = Transcript(
        conversation_id="test__rep01",
        persona_id="test",
        replicate_index=1,
        endpoint="fast_chat_v2",
        turns=turns,
        termination=TerminationReason(reason="done", at_turn=2),
    )
    totals = t.totals()
    assert totals["turns"] == 3
    assert totals["latency_ms"] == 3000.0
    assert totals["cost_usd"] == pytest.approx(0.003)


def test_transcript_json_roundtrip():
    t = Transcript(
        conversation_id="x__rep01",
        persona_id="x",
        replicate_index=1,
        endpoint="chat_stream",
        turns=[],
        termination=TerminationReason(reason="error", at_turn=0, error_message="boom"),
    )
    j = t.model_dump_json()
    back = Transcript.model_validate_json(j)
    assert back == t
