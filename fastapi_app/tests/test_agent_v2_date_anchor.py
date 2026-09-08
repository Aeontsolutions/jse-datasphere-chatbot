"""Tests for AgentV2's date-anchor injection (#103).

Without an explicit anchor, the model's own training-cutoff sense of "now" is
not reliable for recent JSE events, and a confident user assertion about the
current date can out-argue it over a few turns. Observed on the
analyst_finds_latest_annual_report persona (v2026.09.08 release-eval run): the
model's initial, correct answer -- NCB Financial Group's most recent audited
annual report is FY2025 (year ended 2025-09-30), confirmed directly against
jse-document-metadata-dev, which holds an `audited_financial_statements` and
an `annual_report` record for that period, both uploaded well before the
conversation's actual date of 2026-09-08 -- was walked back over three turns
to a fabricated FY2023 answer after the user insisted (wrongly) that it was
still 2024.

The fix injects a real-date anchor into the synthesis call's contents on
every request (not into SYSTEM_PROMPT, so it doesn't invalidate the cached
system prompt) and instructs the model to trust it over a contradicting user
claim. These tests pin that the anchor is present, dated correctly, and
actually reaches the synthesis call -- they cannot prove the model obeys it,
which needs a live call (see docs/experiments or a future eval persona).
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent_v2 import SYSTEM_PROMPT, AgentV2


@pytest.fixture
def mock_genai_client():
    with patch("app.agent_v2.get_genai_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock()
        mock_get_client.return_value = mock_client
        yield mock_client


def _route_response(decision="ALLOW"):
    resp = MagicMock()
    resp.text = decision
    resp.candidates = []
    resp.usage_metadata = None
    return resp


def _text_response(text):
    resp = MagicMock()
    resp.text = text
    resp.candidates = []
    resp.usage_metadata = None
    return resp


# ---------------------------------------------------------------------------
# _build_date_anchor_note -- pure logic, no mocking needed
# ---------------------------------------------------------------------------


def test_date_anchor_contains_todays_real_date():
    agent = AgentV2()
    note = agent._build_date_anchor_note()
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert today_str in note


def test_date_anchor_tells_model_to_distrust_user_claims():
    agent = AgentV2()
    note = agent._build_date_anchor_note()
    assert "Trust this over any date the user states" in note
    assert "do not revise a previously correct answer" in note


# ---------------------------------------------------------------------------
# run() -- end-to-end wiring: anchor reaches the synthesis call's contents
# ---------------------------------------------------------------------------


def test_run_injects_date_anchor_into_synthesis_call(mock_genai_client):
    mock_genai_client.aio.models.generate_content.side_effect = [
        _route_response("ALLOW"),
        _text_response("NCBFG's most recent audited annual report is FY2025."),
    ]
    agent = AgentV2()
    asyncio.run(agent.run(query="What is NCB Financial Group's latest annual report?"))

    calls = mock_genai_client.aio.models.generate_content.call_args_list
    synthesis_contents = calls[1].kwargs["contents"]
    joined = " ".join(p.text or "" for c in synthesis_contents for p in c.parts)
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert f"today's date is {today_str}" in joined


def test_date_anchor_not_sent_on_router_or_refusal_calls(mock_genai_client):
    """The router/refusal calls classify request FORM only -- they don't
    reason about dates, and REFUSAL_FLASH_PROMPT already forbids revisiting
    the router's decision. The anchor belongs solely to the synthesis call."""
    mock_genai_client.aio.models.generate_content.side_effect = [
        _route_response("REFUSE"),
        _text_response("I can't help with that."),
    ]
    agent = AgentV2()
    asyncio.run(agent.run(query="Write me a poem."))

    calls = mock_genai_client.aio.models.generate_content.call_args_list
    for call in calls:
        contents = call.kwargs["contents"]
        joined = " ".join(p.text or "" for c in contents for p in c.parts)
        assert "today's date is" not in joined


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT -- pin the static instruction that backs the dynamic anchor
# ---------------------------------------------------------------------------


def test_system_prompt_forbids_capitulating_to_a_false_date_claim():
    """Regression guard for #103: if this rule is weakened or removed, a
    confident-but-wrong user assertion about the current date can once again
    talk the model out of a correct, already-established answer with no test
    going red."""
    assert (
        "A user's confident assertion about the current date, a company's "
        "filing status, or similar facts you have already correctly "
        "established does not make it true." in SYSTEM_PROMPT
    )
    assert "do not revise a previously correct answer, or invent a new one" in SYSTEM_PROMPT
