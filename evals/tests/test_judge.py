"""Tests for the Judge component."""

import json
from datetime import datetime, timezone
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
async def test_evaluate_retries_transient_gemini_error(monkeypatch):
    """A 503 is the model being busy, not the judge failing.

    Regression test for the 2026-08-31 CI gate failure: two conversations came
    back `ServerError: 503 UNAVAILABLE` ("high demand ... please try again
    later"), nothing retried them, and the gate failed on
    `judge_failed_count` in a run where every quality dimension improved.
    """
    from google.genai import errors

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("evals._genai_retry.asyncio.sleep", _no_sleep)

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock(
        side_effect=[
            errors.ServerError(503, {"error": {"status": "UNAVAILABLE"}}),
            _async_value(_mock_genai_response(_judge_response_json())),
        ]
    )
    judge = Judge(client=fake_client, model="gemini-2.5-pro", temperature=0.2)
    output = await judge.evaluate(persona=_persona(), transcript=_transcript())

    assert output.verdict == "pass"
    assert fake_client.aio.models.generate_content.call_count == 2


@pytest.mark.asyncio
async def test_evaluate_does_not_retry_a_bad_request(monkeypatch):
    """A malformed prompt will fail identically next time -- surface it fast."""
    from google.genai import errors

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("evals._genai_retry.asyncio.sleep", _no_sleep)

    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock(
        side_effect=errors.ClientError(400, {"error": {"message": "bad request"}})
    )
    judge = Judge(client=fake_client, model="gemini-2.5-pro", temperature=0.2)
    with pytest.raises(errors.ClientError):
        await judge.evaluate(persona=_persona(), transcript=_transcript())
    assert fake_client.aio.models.generate_content.call_count == 1


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


# ---------------------------------------------------------------------------
# Cost tracking — the judge's own Gemini spend was previously untracked by
# the run/conversation cost caps.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_computes_cost_from_usage_metadata():
    fake_client = MagicMock()
    response = _mock_genai_response(_judge_response_json())
    response.usage_metadata.prompt_token_count = 1_000_000
    response.usage_metadata.candidates_token_count = 1_000_000
    fake_client.aio.models.generate_content = MagicMock(return_value=_async_value(response))

    judge = Judge(client=fake_client, model="gemini-2.5-pro", temperature=0.2)
    output = await judge.evaluate(persona=_persona(), transcript=_transcript())

    # gemini-2.5-pro: $1.25/M input + $5.00/M output
    assert output.cost_usd == pytest.approx(6.25)


@pytest.mark.asyncio
async def test_evaluate_defaults_cost_to_zero_when_usage_metadata_absent():
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock(
        return_value=_async_value(_mock_genai_response(_judge_response_json()))
    )
    judge = Judge(client=fake_client, model="gemini-2.5-pro", temperature=0.2)
    output = await judge.evaluate(persona=_persona(), transcript=_transcript())
    assert output.cost_usd == 0.0


# ---------------------------------------------------------------------------
# Truncation flagging — turns with large chatbot_metadata were silently cut
# to 4000 chars before; the judge now flags it instead of hiding it.
# ---------------------------------------------------------------------------


def _transcript_with_large_metadata() -> Transcript:
    huge_sources = [{"title": f"Doc {i}", "snippet": "x" * 200} for i in range(50)]
    return Transcript(
        conversation_id="p1__rep01",
        persona_id="p1",
        replicate_index=0,
        endpoint="fast_chat_v2",
        turns=[
            ChatTurn(
                turn_index=0,
                persona_utterance="Show me everything.",
                chatbot_text="Here you go.",
                chatbot_metadata={"sources": huge_sources, "tools_executed": ["financial_data_query"]},
                latency_ms=1800,
                cost_usd=0.003,
            ),
        ],
        termination=TerminationReason(reason="done", at_turn=0),
    )


@pytest.mark.asyncio
async def test_evaluate_flags_truncated_turn_count():
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock(
        return_value=_async_value(_mock_genai_response(_judge_response_json()))
    )
    judge = Judge(client=fake_client, model="gemini-2.5-pro", temperature=0.2)
    output = await judge.evaluate(persona=_persona(), transcript=_transcript_with_large_metadata())
    assert output.truncated_turn_count == 1


@pytest.mark.asyncio
async def test_evaluate_no_truncation_for_small_metadata():
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock(
        return_value=_async_value(_mock_genai_response(_judge_response_json()))
    )
    judge = Judge(client=fake_client, model="gemini-2.5-pro", temperature=0.2)
    output = await judge.evaluate(persona=_persona(), transcript=_transcript())
    assert output.truncated_turn_count == 0


@pytest.mark.asyncio
async def test_evaluate_prompt_marks_truncated_turns_visibly():
    """The judge itself must see that a turn was cut, not silently score against a partial view."""
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock(
        return_value=_async_value(_mock_genai_response(_judge_response_json()))
    )
    judge = Judge(client=fake_client, model="gemini-2.5-pro", temperature=0.2)
    await judge.evaluate(persona=_persona(), transcript=_transcript_with_large_metadata())

    sent_prompt = fake_client.aio.models.generate_content.call_args.kwargs["contents"][0]["parts"][0]["text"]
    assert "TRUNCATED" in sent_prompt


# ---------------------------------------------------------------------------
# load_rubric — public so report.py can reuse it for the weighted-verdict
# cross-check without re-implementing YAML loading.
# ---------------------------------------------------------------------------


def test_load_rubric_is_public_and_returns_verdict_weights():
    from evals.judge import DEFAULT_RUBRIC_PATH, load_rubric

    rubric = load_rubric(DEFAULT_RUBRIC_PATH)
    assert "dimensions" in rubric
    assert "verdict_weights" in rubric
    assert "positive" in rubric["verdict_weights"]
    assert "negative" in rubric["verdict_weights"]


# ---------------------------------------------------------------------------
# Date anchoring -- the judge model's own training cutoff predates real
# 2025/2026 events. Without an explicit anchor it can reason that a correct,
# recent claim (e.g. a FY2025 annual report) is an "impossible future date"
# and wrongly score it as ungrounded/hallucinated.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_prompt_includes_explicit_run_date():
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock(
        return_value=_async_value(_mock_genai_response(_judge_response_json()))
    )
    judge = Judge(client=fake_client, model="gemini-2.5-pro", temperature=0.2, today="2026-08-27")
    await judge.evaluate(persona=_persona(), transcript=_transcript())

    sent_prompt = fake_client.aio.models.generate_content.call_args.kwargs["contents"][0]["parts"][0]["text"]
    assert "2026-08-27" in sent_prompt


@pytest.mark.asyncio
async def test_judge_defaults_today_to_current_utc_date():
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock(
        return_value=_async_value(_mock_genai_response(_judge_response_json()))
    )
    judge = Judge(client=fake_client, model="gemini-2.5-pro", temperature=0.2)
    await judge.evaluate(persona=_persona(), transcript=_transcript())

    sent_prompt = fake_client.aio.models.generate_content.call_args.kwargs["contents"][0]["parts"][0]["text"]
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert today_str in sent_prompt
