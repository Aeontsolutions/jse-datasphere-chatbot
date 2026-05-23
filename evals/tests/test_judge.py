"""Tests for the Judge component."""

import json
from unittest.mock import MagicMock

import pytest

from evals.judge import Judge, JudgeOutput
from evals.persona import PersonaSpec
from evals.transcript import ChatTurn, TerminationReason, Transcript


def _persona() -> PersonaSpec:
    return PersonaSpec(
        id="p1",
        name="P1",
        category="positive",
        endpoint="fast_chat_v2",
        character="Senior analyst.",
        goal="Find NCB revenue growth.",
        max_turns=4,
        expected_facts=["NCB is mentioned", "Revenue figures included"],
    )


def _transcript() -> Transcript:
    return Transcript(
        conversation_id="p1__rep01",
        persona_id="p1",
        replicate_index=0,
        endpoint="fast_chat_v2",
        turns=[
            ChatTurn(
                turn_index=0,
                persona_utterance="Show me NCB revenue last 2 years.",
                chatbot_text="NCB revenue was J$50B in FY2023, J$45B in FY2022.",
                chatbot_metadata={
                    "data_found": True,
                    "record_count": 2,
                    "sources": [{"title": "NCB FY2023 Annual Report"}],
                    "tools_executed": ["financial_data_query"],
                },
                latency_ms=1800,
                cost_usd=0.003,
            ),
        ],
        termination=TerminationReason(reason="done", at_turn=0, persona_done_reason="got the numbers"),
    )


def _judge_response_json() -> str:
    return json.dumps({
        "scores": {
            "groundedness": {"score": 5, "justification": "Both numbers cited."},
            "factfulness": {
                "score": 5,
                "facts_satisfied": [True, True],
                "justification": "Both expected facts present.",
            },
            "goal_completion": {"score": 5, "justification": "Got NCB revenue."},
            "tool_use_appropriateness": {
                "score": 5,
                "observed_tools": ["financial_data_query"],
                "justification": "Correct tool fired.",
            },
            "coherence": {"score": 5, "justification": "One turn, no contradictions."},
            "persona_handling": {"score": 4, "justification": "Crisp, matches analyst style."},
        },
        "verdict": "pass",
        "verdict_reason": "All dimensions strong.",
        "notable_moments": [
            {"turn": 0, "type": "good_citation", "note": "Cites annual report."}
        ],
    })


def _mock_genai_response(text: str) -> MagicMock:
    r = MagicMock()
    r.text = text
    return r


def _async_value(v):
    async def _c():
        return v
    return _c()


@pytest.mark.asyncio
async def test_evaluate_returns_structured_output():
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock(
        return_value=_async_value(_mock_genai_response(_judge_response_json()))
    )
    judge = Judge(
        client=fake_client,
        model="gemini-2.5-pro",
        temperature=0.2,
    )
    output = await judge.evaluate(persona=_persona(), transcript=_transcript())

    assert isinstance(output, JudgeOutput)
    assert output.verdict == "pass"
    assert output.scores.groundedness.score == 5
    assert output.scores.factfulness.score == 5
    assert output.scores.factfulness.facts_satisfied == [True, True]
    assert len(output.notable_moments) == 1


@pytest.mark.asyncio
async def test_factfulness_null_when_no_expected_facts():
    persona = _persona().model_copy(update={"expected_facts": []})
    body = json.loads(_judge_response_json())
    body["scores"]["factfulness"]["score"] = None
    body["scores"]["factfulness"]["facts_satisfied"] = []
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock(
        return_value=_async_value(_mock_genai_response(json.dumps(body)))
    )

    judge = Judge(client=fake_client, model="gemini-2.5-pro", temperature=0.2)
    output = await judge.evaluate(persona=persona, transcript=_transcript())
    assert output.scores.factfulness.score is None
    assert output.scores.factfulness.facts_satisfied == []


@pytest.mark.asyncio
async def test_evaluate_retries_on_invalid_schema():
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock(
        side_effect=[
            _async_value(_mock_genai_response("not parseable")),
            _async_value(_mock_genai_response(_judge_response_json())),
        ]
    )
    judge = Judge(client=fake_client, model="gemini-2.5-pro", temperature=0.2)
    output = await judge.evaluate(persona=_persona(), transcript=_transcript())
    assert output.verdict == "pass"


@pytest.mark.asyncio
async def test_evaluate_raises_after_second_failure():
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock(
        side_effect=[
            _async_value(_mock_genai_response("bad")),
            _async_value(_mock_genai_response("still bad")),
        ]
    )
    judge = Judge(client=fake_client, model="gemini-2.5-pro", temperature=0.2)
    with pytest.raises(RuntimeError, match="judge_failed"):
        await judge.evaluate(persona=_persona(), transcript=_transcript())
