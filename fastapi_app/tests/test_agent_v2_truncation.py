"""Tests for silent-truncation visibility and thinking-budget control on AgentV2.

Background (issue #72): prod answers were breaking off mid-word while being
recorded as `success = true`. Reproducing the refusal call against the live API
showed `finish_reason = MAX_TOKENS` with 217-248 *thinking* tokens consuming a
256-token `max_output_tokens` budget, leaving 4-11 visible tokens.

Two things had to be true for that to reach a user unnoticed:
  1. Nothing read `finish_reason` off the response, so an abnormal stop was
     indistinguishable from a normal one.
  2. Thinking was left at the model default on calls with a tight output cap.

These tests pin both behaviours.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent_v2 import AgentV2, extract_finish_reason, resolve_router_model


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


def _text_response(text, finish_reason=None):
    """A generation response, optionally carrying a finish_reason on candidate 0."""
    resp = MagicMock()
    resp.text = text
    resp.usage_metadata = None
    if finish_reason is None:
        resp.candidates = []
    else:
        candidate = MagicMock()
        candidate.finish_reason = finish_reason
        resp.candidates = [candidate]
    return resp


class _FinishReasonEnum:
    """Stands in for the SDK's FinishReason enum, whose str() is prefixed."""

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"FinishReason.{self.name}"


# ---------------------------------------------------------------------------
# extract_finish_reason
# ---------------------------------------------------------------------------


def test_extract_finish_reason_unwraps_sdk_enum():
    """The SDK yields an enum whose str() is 'FinishReason.MAX_TOKENS' — we want the bare name."""
    resp = _text_response("truncated text", finish_reason=_FinishReasonEnum("MAX_TOKENS"))
    assert extract_finish_reason(resp) == "MAX_TOKENS"


def test_extract_finish_reason_accepts_plain_string():
    resp = _text_response("done", finish_reason="STOP")
    assert extract_finish_reason(resp) == "STOP"


def test_extract_finish_reason_none_when_no_candidates():
    assert extract_finish_reason(_text_response("done")) is None


def test_extract_finish_reason_none_on_garbage():
    """A malformed response must not raise — logging is never allowed to break a request."""
    assert extract_finish_reason(None) is None
    assert extract_finish_reason(object()) is None


# ---------------------------------------------------------------------------
# Thinking budget — the actual fix for issue #72
# ---------------------------------------------------------------------------


def test_router_call_disables_thinking(mock_genai_client):
    """The router only emits one word; thinking must not eat its 256-token budget."""
    mock_genai_client.aio.models.generate_content.side_effect = [
        _route_response("REFUSE"),
        _text_response("JSE topics only."),
    ]
    asyncio.run(AgentV2().run(query="What is bitcoin?"))

    route_cfg = mock_genai_client.aio.models.generate_content.call_args_list[0].kwargs["config"]
    assert route_cfg.thinking_config is not None
    assert route_cfg.thinking_config.thinking_budget == 0


def test_refusal_call_disables_thinking(mock_genai_client):
    """The refusal call is where prod truncated: thinking consumed the whole cap."""
    mock_genai_client.aio.models.generate_content.side_effect = [
        _route_response("REFUSE"),
        _text_response("JSE topics only."),
    ]
    asyncio.run(AgentV2().run(query="Should I buy Tesla?"))

    refusal_cfg = mock_genai_client.aio.models.generate_content.call_args_list[1].kwargs["config"]
    assert refusal_cfg.thinking_config is not None
    assert refusal_cfg.thinking_config.thinking_budget == 0


def test_refusal_output_cap_leaves_headroom(mock_genai_client):
    """256 left no margin. A refusal is 1-2 sentences; the cap should not be the binding limit."""
    mock_genai_client.aio.models.generate_content.side_effect = [
        _route_response("REFUSE"),
        _text_response("JSE topics only."),
    ]
    asyncio.run(AgentV2().run(query="Should I buy Tesla?"))

    refusal_cfg = mock_genai_client.aio.models.generate_content.call_args_list[1].kwargs["config"]
    assert refusal_cfg.max_output_tokens >= 512


# ---------------------------------------------------------------------------
# finish_reason propagation
# ---------------------------------------------------------------------------


def test_refuse_path_surfaces_finish_reason(mock_genai_client):
    mock_genai_client.aio.models.generate_content.side_effect = [
        _route_response("REFUSE"),
        _text_response(
            "As a financial analyst, I cannot",
            finish_reason=_FinishReasonEnum("MAX_TOKENS"),
        ),
    ]
    result = asyncio.run(AgentV2().run(query="top 3 stocks to buy today"))
    assert result["finish_reason"] == "MAX_TOKENS"


def test_refuse_path_surfaces_normal_stop(mock_genai_client):
    mock_genai_client.aio.models.generate_content.side_effect = [
        _route_response("REFUSE"),
        _text_response("JSE topics only.", finish_reason=_FinishReasonEnum("STOP")),
    ]
    result = asyncio.run(AgentV2().run(query="What is bitcoin?"))
    assert result["finish_reason"] == "STOP"


def test_web_path_surfaces_finish_reason(mock_genai_client):
    """Truncation is not unique to the refusal path — the synthesis path reports too."""
    mock_genai_client.aio.models.generate_content.side_effect = [
        _route_response("ALLOW"),
        _text_response("GraceKennedy reported", finish_reason=_FinishReasonEnum("MAX_TOKENS")),
    ]
    result = asyncio.run(AgentV2().run(query="GraceKennedy revenue 2024", enable_web_search=True))
    assert result["finish_reason"] == "MAX_TOKENS"


def test_error_path_finish_reason_is_none(mock_genai_client):
    """A thrown request has no generation outcome to report."""
    mock_genai_client.aio.models.generate_content.side_effect = RuntimeError("boom")
    result = asyncio.run(AgentV2().run(query="anything"))
    assert result["finish_reason"] is None


# ---------------------------------------------------------------------------
# Router model pinning
# ---------------------------------------------------------------------------


def test_router_model_defaults_to_pinned_flash(monkeypatch):
    monkeypatch.delenv("ROUTER_MODEL_NAME", raising=False)
    assert resolve_router_model() == "gemini-2.5-flash"


def test_router_model_overridable_by_env(monkeypatch):
    """Lets a candidate model be trialled in one environment without a code change."""
    monkeypatch.setenv("ROUTER_MODEL_NAME", "gemini-3.7-flash")
    assert resolve_router_model() == "gemini-3.7-flash"


def test_router_model_blank_env_falls_back(monkeypatch):
    monkeypatch.setenv("ROUTER_MODEL_NAME", "   ")
    assert resolve_router_model() == "gemini-2.5-flash"


def test_cost_summary_totals_thinking_tokens(mock_genai_client):
    """Thinking is billed as output; the summary must expose it for logging."""
    route = _route_response("REFUSE")
    refusal = _text_response("JSE topics only.", finish_reason=_FinishReasonEnum("STOP"))
    for resp, thinking in ((route, 0), (refusal, 243)):
        resp.usage_metadata = MagicMock()
        resp.usage_metadata.prompt_token_count = 100
        resp.usage_metadata.candidates_token_count = 10
        resp.usage_metadata.cached_content_token_count = 0
        resp.usage_metadata.total_token_count = 110 + thinking
        resp.usage_metadata.thoughts_token_count = thinking

    mock_genai_client.aio.models.generate_content.side_effect = [route, refusal]
    result = asyncio.run(AgentV2().run(query="Should I buy Tesla?"))
    assert result["cost_summary"].total_thinking_tokens == 243
