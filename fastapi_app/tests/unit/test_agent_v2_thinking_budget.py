"""Regression tests for ADR 0002 — truncated refusals and fail-open routing.

Interaction 7dba1a26bc014fdebfdb2b9567012c52 returned a response that stopped
mid-sentence ("Recent geopolitical events can certainly have a ripple effect on
companies"). Root cause: on Gemini 2.5 models `max_output_tokens` bounds
*thinking + visible text combined*, and thinking is on by default with a
dynamic budget. The refusal call's 256-token ceiling was consumed almost
entirely by thinking (241/256), leaving ~13 tokens of visible text before the
API cut generation off with finish_reason=MAX_TOKENS.

These tests lock in the three behaviours that prevent a recurrence:
  1. Every Flash call caps thinking explicitly and leaves headroom for output.
  2. The refusal call is told the router already decided REFUSE, so it refuses
     instead of re-litigating scope and answering the question.
  3. An empty router verdict fails closed instead of silently meaning ALLOW.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.agent_v2 import (
    MIN_VISIBLE_OUTPUT_HEADROOM,
    REFUSAL_MAX_OUTPUT_TOKENS,
    REFUSAL_THINKING_BUDGET,
    ROUTER_MAX_OUTPUT_TOKENS,
    ROUTER_THINKING_BUDGET,
    AgentV2,
)


@pytest.fixture
def mock_genai_client():
    with patch("app.agent_v2.get_genai_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        yield mock_client


def _route_response(decision="ALLOW", finish_reason="STOP"):
    resp = MagicMock()
    resp.text = decision
    candidate = MagicMock()
    candidate.finish_reason = finish_reason
    resp.candidates = [candidate]
    resp.usage_metadata = None
    return resp


def _text_response(text, finish_reason="STOP"):
    resp = MagicMock()
    resp.text = text
    candidate = MagicMock()
    candidate.finish_reason = finish_reason
    resp.candidates = [candidate]
    resp.usage_metadata = None
    return resp


# ---------------------------------------------------------------------------
# 1. Thinking budgets are bounded and leave room for visible output
# ---------------------------------------------------------------------------


def test_budget_constants_leave_headroom_for_visible_output():
    """The defect in one line: thinking must not be able to consume the ceiling."""
    assert ROUTER_MAX_OUTPUT_TOKENS - ROUTER_THINKING_BUDGET >= MIN_VISIBLE_OUTPUT_HEADROOM
    assert REFUSAL_MAX_OUTPUT_TOKENS - REFUSAL_THINKING_BUDGET >= MIN_VISIBLE_OUTPUT_HEADROOM


def test_router_call_caps_thinking_budget(mock_genai_client):
    mock_genai_client.models.generate_content.side_effect = [
        _route_response("ALLOW"),
        _text_response("answer"),
    ]
    agent = AgentV2()
    asyncio.run(agent.run(query="What was NCB revenue in 2023?"))

    route_cfg = mock_genai_client.models.generate_content.call_args_list[0].kwargs["config"]
    assert route_cfg.thinking_config is not None, "router must set an explicit thinking budget"
    assert route_cfg.thinking_config.thinking_budget == ROUTER_THINKING_BUDGET
    assert route_cfg.max_output_tokens == ROUTER_MAX_OUTPUT_TOKENS


def test_refusal_call_caps_thinking_budget(mock_genai_client):
    """The exact call that produced the truncated interaction."""
    mock_genai_client.models.generate_content.side_effect = [
        _route_response("REFUSE"),
        _text_response("I can only help with JSE topics."),
    ]
    agent = AgentV2()
    asyncio.run(agent.run(query="What stocks would be best held in defence?"))

    refusal_cfg = mock_genai_client.models.generate_content.call_args_list[1].kwargs["config"]
    assert refusal_cfg.thinking_config is not None, "refusal must set an explicit thinking budget"
    assert refusal_cfg.thinking_config.thinking_budget == REFUSAL_THINKING_BUDGET
    assert refusal_cfg.max_output_tokens == REFUSAL_MAX_OUTPUT_TOKENS


# ---------------------------------------------------------------------------
# 2. The refusal call is told the verdict, so it refuses
# ---------------------------------------------------------------------------


def test_refusal_call_is_told_the_request_is_out_of_scope(mock_genai_client):
    """Without the verdict, Flash re-decides scope and answers the question.

    Live reproduction showed the model happily recommending JSE "defensive
    stocks" — exactly the personalised advice the router flagged — because
    REFUSAL_FLASH_PROMPT only described *how* to refuse *if* out of scope.
    """
    mock_genai_client.models.generate_content.side_effect = [
        _route_response("REFUSE"),
        _text_response("I can only help with JSE topics."),
    ]
    agent = AgentV2()
    asyncio.run(agent.run(query="What stocks would be best held in defence?"))

    refusal_call = mock_genai_client.models.generate_content.call_args_list[1]
    instruction = refusal_call.kwargs["config"].system_instruction.lower()
    assert "has already been determined" in instruction or "out of scope" in instruction
    assert "must decline" in instruction or "do not answer" in instruction


# ---------------------------------------------------------------------------
# 3. An empty router verdict fails closed
# ---------------------------------------------------------------------------


def test_empty_router_verdict_retries_with_thinking_disabled(mock_genai_client):
    """A blank verdict triggers one deterministic retry before any decision."""
    mock_genai_client.models.generate_content.side_effect = [
        _route_response("", finish_reason="MAX_TOKENS"),
        _route_response("ALLOW"),
        _text_response("web answer"),
    ]
    agent = AgentV2()
    result = asyncio.run(agent.run(query="What was NCB revenue in 2023?"))

    calls = mock_genai_client.models.generate_content.call_args_list
    retry_cfg = calls[1].kwargs["config"]
    assert retry_cfg.thinking_config.thinking_budget == 0, "retry must disable thinking entirely"
    # Recovered verdict was ALLOW, so the web path still ran.
    assert result["response"] == "web answer"


def test_empty_router_verdict_after_retry_refuses(mock_genai_client):
    """Fail closed: an unusable verdict must not silently mean ALLOW."""
    mock_genai_client.models.generate_content.side_effect = [
        _route_response("", finish_reason="MAX_TOKENS"),
        _route_response("", finish_reason="MAX_TOKENS"),
    ]
    agent = AgentV2()
    result = asyncio.run(agent.run(query="Should I buy Tesla stock?"))

    # No Pro generation call was made — the request was refused.
    assert mock_genai_client.models.generate_content.call_count == 2
    assert result["data_found"] is False
    assert "JSE" in result["response"]


def test_router_exception_still_falls_through_to_web(mock_genai_client):
    """Unchanged from ADR 0001 behaviour: a raising router degrades gracefully.

    A transport error is different from a successful call returning nothing:
    the former tells us the check never ran (fall through to Pro, which has its
    own scope guardrails), the latter is anomalous enough to distrust.
    """
    mock_genai_client.models.generate_content.side_effect = [
        RuntimeError("router exploded"),
        _text_response("web fallback answer"),
    ]
    agent = AgentV2()
    result = asyncio.run(agent.run(query="What was NCB revenue in 2023?"))

    assert result["response"] == "web fallback answer"


# ---------------------------------------------------------------------------
# 4. Truncation is surfaced rather than logged as a clean success
# ---------------------------------------------------------------------------


def test_truncated_generation_is_flagged_in_cost_summary(mock_genai_client):
    """MAX_TOKENS must be visible downstream instead of logging success:true."""
    mock_genai_client.models.generate_content.side_effect = [
        _route_response("ALLOW"),
        _text_response("half an ans", finish_reason="MAX_TOKENS"),
    ]
    agent = AgentV2()
    result = asyncio.run(agent.run(query="What was NCB revenue in 2023?"))

    summary = result["cost_summary"]
    assert summary.truncated is True
    assert "generation" in summary.truncated_phases


def test_untruncated_generation_is_not_flagged(mock_genai_client):
    mock_genai_client.models.generate_content.side_effect = [
        _route_response("ALLOW"),
        _text_response("a complete answer"),
    ]
    agent = AgentV2()
    result = asyncio.run(agent.run(query="What was NCB revenue in 2023?"))

    summary = result["cost_summary"]
    assert summary.truncated is False
    assert summary.truncated_phases == []
