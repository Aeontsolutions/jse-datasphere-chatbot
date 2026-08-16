"""Tests for the AgentV2 router (AEO-23 REFUSE safety pre-check).

/chat/stream always answers via Gemini 2.5 Pro + Google Search grounding. The
only job of the cheap Flash router is to catch out-of-scope/unsafe requests
(REFUSE) before they reach Pro; everything else (ALLOW) falls through to the
web/plain path. There is no financial-data/BigQuery branch: a prior version
routed FINANCIAL questions to BigQuery directly, but a query-parsing bug there
(out-of-range years silently dropped) caused it to synthesize answers from the
wrong years' data instead of falling through to web search, so that branch was
removed.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent_v2 import ROUTER_MODEL, AgentV2


@pytest.fixture
def mock_genai_client():
    with patch("app.agent_v2.get_genai_client") as mock_get_client:
        mock_client = MagicMock()
        # AgentV2 calls the SDK's native async client (client.aio.models.
        # generate_content), so this leaf has to be awaitable — a plain
        # MagicMock would return a non-awaitable and fail at the await.
        mock_client.aio.models.generate_content = AsyncMock()
        mock_get_client.return_value = mock_client
        yield mock_client


def test_router_model_is_flash():
    assert ROUTER_MODEL == "gemini-2.5-flash"


# NOTE: every builder sets usage_metadata = None so the real _track_cost takes
# the zero-cost branch instead of choking on MagicMock token counts. This is
# REQUIRED — TokenUsage.from_response treats any truthy usage_metadata as real
# counts, and a bare MagicMock attribute would crash PhaseCost validation.


def _route_response(decision="ALLOW"):
    """Step-1 ROUTE response: plain-text REFUSE / ALLOW, no function call."""
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


def test_route_allow_falls_through_to_web(mock_genai_client):
    # route=ALLOW -> no refusal call; web path runs (route + web = 2 calls).
    mock_genai_client.aio.models.generate_content.side_effect = [
        _route_response("ALLOW"),
        _text_response("NCB's 2023 revenue was J$123.5M, per public filings."),
    ]
    agent = AgentV2()
    result = asyncio.run(agent.run(query="What was NCB revenue in 2023?"))

    assert result["response"] == "NCB's 2023 revenue was J$123.5M, per public filings."
    assert "query_financial_data" not in (result.get("tools_executed") or [])
    assert mock_genai_client.aio.models.generate_content.call_count == 2


def test_route_call_uses_flash_and_no_tools(mock_genai_client):
    # The ROUTE call is a plain-text flash classification — no tools attached.
    mock_genai_client.aio.models.generate_content.side_effect = [
        _route_response("ALLOW"),
        _text_response("answer"),
    ]
    agent = AgentV2()
    asyncio.run(agent.run(query="NCB revenue 2023?"))

    calls = mock_genai_client.aio.models.generate_content.call_args_list
    assert len(calls) == 2
    route_cfg = calls[0].kwargs["config"]
    assert calls[0].kwargs["model"] == "gemini-2.5-flash"
    assert route_cfg.tools is None


def test_router_error_falls_through_to_web(mock_genai_client):
    # If the ROUTE call itself raises, _fast_path swallows it and run() still
    # reaches the web/plain path (graceful degradation).
    mock_genai_client.aio.models.generate_content.side_effect = [
        RuntimeError("router exploded"),
        _text_response("web fallback answer"),
    ]
    agent = AgentV2()
    result = asyncio.run(agent.run(query="What was NCB revenue in 2023?"))

    assert result["response"] == "web fallback answer"
    assert mock_genai_client.aio.models.generate_content.call_count == 2


def test_run_uses_cached_content_when_cache_hits(mock_genai_client):
    """AgentV2.run() passes cached_content= instead of system_instruction= on cache hit."""
    resp = MagicMock()
    resp.text = "hello"
    resp.candidates = []
    resp.usage_metadata = None
    mock_genai_client.aio.models.generate_content.return_value = resp

    with (
        patch("app.agent_v2._SYSTEM_PROMPT_CACHE") as mock_cache,
        patch("app.agent_v2._SYSTEM_PROMPT_NO_SEARCH_CACHE"),
    ):
        mock_cache.get_cache_name.return_value = "cachedContents/xyz"

        agent = AgentV2()
        asyncio.run(agent.run("What is the JSE?", enable_web_search=True))

    config = mock_genai_client.aio.models.generate_content.call_args.kwargs["config"]
    assert config.cached_content == "cachedContents/xyz"
    # system_instruction must be absent (None or not set) when cached_content is used
    assert not getattr(config, "system_instruction", None)


def test_run_falls_back_to_system_instruction_when_cache_returns_none(mock_genai_client):
    """AgentV2.run() uses system_instruction= when cache returns None."""
    resp = MagicMock()
    resp.text = "hello"
    resp.candidates = []
    resp.usage_metadata = None
    mock_genai_client.aio.models.generate_content.return_value = resp

    with (
        patch("app.agent_v2._SYSTEM_PROMPT_CACHE") as mock_cache,
        patch("app.agent_v2._SYSTEM_PROMPT_NO_SEARCH_CACHE"),
    ):
        mock_cache.get_cache_name.return_value = None

        agent = AgentV2()
        asyncio.run(agent.run("What is the JSE?", enable_web_search=True))

    config = mock_genai_client.aio.models.generate_content.call_args.kwargs["config"]
    assert config.system_instruction is not None
    assert not getattr(config, "cached_content", None)


# ---------------------------------------------------------------------------
# REFUSE path (AEO-23)
# ---------------------------------------------------------------------------


def test_refuse_path_uses_flash_only_no_pro(mock_genai_client):
    """REFUSE route: 2 Flash calls (route + refusal), zero Pro calls."""
    mock_genai_client.aio.models.generate_content.side_effect = [
        _route_response("REFUSE"),
        _text_response("Sorry, I can only help with JSE topics."),
    ]
    agent = AgentV2()
    result = asyncio.run(agent.run(query="Write me a poem about mangoes."))

    assert result["response"] == "Sorry, I can only help with JSE topics."
    assert result["record_count"] == 0
    assert result["data_found"] is False
    # Both calls must use Flash, not Pro
    calls = mock_genai_client.aio.models.generate_content.call_args_list
    assert len(calls) == 2
    assert all(c.kwargs["model"] == "gemini-2.5-flash" for c in calls)


def test_refuse_path_no_tools_on_refusal_call(mock_genai_client):
    """The refusal Flash call must have no tools attached (no grounding)."""
    mock_genai_client.aio.models.generate_content.side_effect = [
        _route_response("REFUSE"),
        _text_response("I can only help with JSE-related topics."),
    ]
    agent = AgentV2()
    asyncio.run(agent.run(query="What is bitcoin?"))

    calls = mock_genai_client.aio.models.generate_content.call_args_list
    # calls[0] = route call, calls[1] = refusal generation call
    refusal_cfg = calls[1].kwargs["config"]
    assert refusal_cfg.tools is None


def test_refuse_path_conversation_history_none(mock_genai_client):
    """REFUSE path correctly handles conversation_history=None (no crash, history built fresh)."""
    mock_genai_client.aio.models.generate_content.side_effect = [
        _route_response("REFUSE"),
        _text_response("JSE topics only."),
    ]
    agent = AgentV2()
    result = asyncio.run(agent.run(query="Buy me Tesla stock.", conversation_history=None))
    assert result["conversation_history"] == [
        {"role": "user", "content": "Buy me Tesla stock."},
        {"role": "assistant", "content": "JSE topics only."},
    ]
