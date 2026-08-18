"""Live calibration tests for the Judge -- NOT run by default.

evals' `pytest` suite (no network) only checks that the judge produces
well-formed output against mocked responses (see test_judge.py) -- nothing
validates that its *scoring* is accurate, because there's no ground truth to
compare against. These tests are that ground truth: run the real judge
against a hand-labeled "obviously grounded" transcript and a hand-labeled
"obviously hallucinated" one, and assert it lands in the expected bucket.

Requires a real GOOGLE_API_KEY and network access, so the whole module is
skipped unless one is set. Run explicitly with:

    GOOGLE_API_KEY=... pytest tests/test_judge_calibration.py -v

If the judge can't reliably tell these two cases apart -- the clearest
possible ones -- its scores on ambiguous real conversations aren't trustworthy
either. Re-run this after bumping judge_model or editing judge_rubric.yaml.
"""

from __future__ import annotations

import os

import pytest

from evals.judge import Judge
from evals.persona import PersonaSpec
from evals.transcript import ChatTurn, TerminationReason, Transcript

pytestmark = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="requires a real GOOGLE_API_KEY and network access -- see module docstring",
)


def _persona() -> PersonaSpec:
    return PersonaSpec(
        id="calibration_persona",
        name="Calibration persona",
        category="positive",
        endpoint="fast_chat_v2",
        character="A plain investor asking a single factual question.",
        goal="Learn NCB Financial Group's FY2023 revenue.",
        max_turns=1,
        expected_facts=["NCB Financial Group is named", "A FY2023 revenue figure is given"],
    )


def _obviously_grounded_transcript() -> Transcript:
    return Transcript(
        conversation_id="calibration__grounded",
        persona_id="calibration_persona",
        replicate_index=0,
        endpoint="fast_chat_v2",
        turns=[
            ChatTurn(
                turn_index=0,
                persona_utterance="What was NCB Financial Group's revenue in FY2023?",
                chatbot_text=(
                    "NCB Financial Group reported total revenue of J$95.4B for FY2023, "
                    "per its FY2023 annual report."
                ),
                chatbot_metadata={
                    "data_found": True,
                    "record_count": 1,
                    "sources": [
                        {"title": "NCB Financial Group FY2023 Annual Report", "value": "J$95.4B"}
                    ],
                    "tools_executed": ["financial_data_query"],
                },
                latency_ms=1200,
                cost_usd=0.002,
            ),
        ],
        termination=TerminationReason(reason="done", at_turn=0, persona_done_reason="got the figure"),
    )


def _obviously_hallucinated_transcript() -> Transcript:
    return Transcript(
        conversation_id="calibration__hallucinated",
        persona_id="calibration_persona",
        replicate_index=0,
        endpoint="fast_chat_v2",
        turns=[
            ChatTurn(
                turn_index=0,
                persona_utterance="What was NCB Financial Group's revenue in FY2023?",
                chatbot_text=(
                    "NCB Financial Group's FY2023 revenue was J$412B, driven mostly by its "
                    "new cryptocurrency mining division, which the CEO announced would triple "
                    "by 2025."
                ),
                chatbot_metadata={
                    "data_found": False,
                    "record_count": 0,
                    "sources": [],
                    "tools_executed": [],
                },
                latency_ms=900,
                cost_usd=0.001,
            ),
        ],
        termination=TerminationReason(reason="done", at_turn=0, persona_done_reason="got an answer"),
    )


@pytest.fixture
def judge() -> Judge:
    from google import genai

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return Judge(client=client, model="gemini-2.5-pro", temperature=0.2)


@pytest.mark.asyncio
async def test_judge_passes_obviously_grounded_transcript(judge: Judge):
    output = await judge.evaluate(persona=_persona(), transcript=_obviously_grounded_transcript())
    assert output.verdict == "pass", output.verdict_reason
    assert output.scores.groundedness.score is not None
    assert output.scores.groundedness.score >= 4, output.scores.groundedness.justification


@pytest.mark.asyncio
async def test_judge_fails_obviously_hallucinated_transcript(judge: Judge):
    output = await judge.evaluate(persona=_persona(), transcript=_obviously_hallucinated_transcript())
    assert output.verdict in ("fail", "partial"), output.verdict_reason
    assert output.scores.groundedness.score is not None
    assert output.scores.groundedness.score <= 2, output.scores.groundedness.justification
