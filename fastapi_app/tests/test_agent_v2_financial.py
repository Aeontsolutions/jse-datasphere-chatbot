"""Tests for the AgentV2 financial-data path (ATS-334).

The financial path is two-step ("route then force"): a cheap flash ROUTE call
classifies the question FINANCIAL vs OTHER (plain text, no tools); only when
FINANCIAL does a second flash call force a query_financial_data extraction
(mode=ANY). This fixes flash's reluctance to call the tool in a single AUTO step
(see the ATS-334 eval finding). Non-financial questions fall through to the
unchanged web/plain path.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.agent_v2 import FINANCIAL_DECISION_MODEL, AgentV2
from app.models import FinancialDataRecord


@pytest.fixture
def mock_genai_client():
    with patch("app.agent_v2.get_genai_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        yield mock_client


def test_constructor_stores_financial_manager(mock_genai_client):
    mgr = MagicMock()
    agent = AgentV2(financial_manager=mgr)
    assert agent.financial_manager is mgr


def test_constructor_defaults_financial_manager_none(mock_genai_client):
    agent = AgentV2()
    assert agent.financial_manager is None


def test_metadata_context_includes_symbols_and_years(mock_genai_client):
    mgr = MagicMock()
    mgr.metadata = {"symbols": ["NCB", "GK"], "years": ["2022", "2023"]}
    agent = AgentV2(financial_manager=mgr)
    ctx = agent._get_metadata_context()
    assert "NCB" in ctx and "2023" in ctx


def test_metadata_context_empty_when_no_manager(mock_genai_client):
    agent = AgentV2()
    assert agent._get_metadata_context() == ""


def test_decision_model_is_flash():
    assert FINANCIAL_DECISION_MODEL == "gemini-2.5-flash"


def _record():
    return FinancialDataRecord(
        company="NCB Financial Group",
        symbol="NCB",
        year="2023",
        standard_item="revenue",
        item=123456789.0,
        unit_multiplier=1,
        formatted_value="123,456,789",
    )


# NOTE: every builder sets usage_metadata = None so the real _track_cost takes
# the zero-cost branch instead of choking on MagicMock token counts. This is
# REQUIRED — TokenUsage.from_response treats any truthy usage_metadata as real
# counts, and a bare MagicMock attribute would crash PhaseCost validation.


def _route_response(decision="FINANCIAL"):
    """Step-1 ROUTE response: plain-text FINANCIAL / OTHER, no function call."""
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


def _filters(symbols=None, years=None, items=None):
    from app.models import FinancialDataFilters

    return FinancialDataFilters(
        companies=[],
        symbols=symbols or [],
        years=years or [],
        standard_items=items or [],
        interpretation="parsed",
        data_availability_note="",
        is_follow_up=False,
        context_used="",
    )


def _mock_manager(records, filters=None):
    # parse_user_query is the proven extractor (used by query_and_run); it returns
    # finished filters. query_data returns the rows for those filters.
    mgr = MagicMock()
    mgr.metadata = {"symbols": ["NCBFG"], "years": ["2023"]}
    mgr.parse_user_query.return_value = filters or _filters(symbols=["NCBFG"], years=["2023"])
    mgr.query_data.return_value = records
    return mgr


def test_financial_path_routes_parses_and_synthesizes(mock_genai_client):
    # route=FINANCIAL -> parse_user_query + query_data -> synthesize.
    # Only TWO generate_content calls now (route + synth); extraction is the
    # manager's parse_user_query, not a Gemini tool call.
    mock_genai_client.models.generate_content.side_effect = [
        _route_response("FINANCIAL"),
        _text_response("NCB's 2023 revenue was J$123.5M."),
    ]
    agent = AgentV2(financial_manager=_mock_manager([_record()]))
    result = asyncio.run(agent.run(query="What was NCB revenue in 2023?"))

    assert result["tools_executed"] == ["query_financial_data"]
    assert result["record_count"] == 1
    assert result["filters_used"].symbols == ["NCBFG"]
    assert result["data_preview"] and len(result["data_preview"]) == 1
    assert "123.5M" in result["response"]
    agent.financial_manager.parse_user_query.assert_called_once()
    assert mock_genai_client.models.generate_content.call_count == 2


def test_route_grounding_skips_parse_and_falls_through_to_web(mock_genai_client):
    # route=GROUNDING -> NO parse_user_query, NO query_data; web path runs (route + web).
    mock_genai_client.models.generate_content.side_effect = [
        _route_response("GROUNDING"),
        _text_response("Here is some recent JSE news."),
    ]
    agent = AgentV2(financial_manager=_mock_manager([]))
    result = asyncio.run(agent.run(query="Any recent JSE news?"))

    assert result["response"] == "Here is some recent JSE news."
    assert "query_financial_data" not in (result.get("tools_executed") or [])
    agent.financial_manager.parse_user_query.assert_not_called()
    agent.financial_manager.query_data.assert_not_called()
    assert mock_genai_client.models.generate_content.call_count == 2


def test_route_call_uses_flash_and_no_tools(mock_genai_client):
    # The ROUTE call is a plain-text flash classification — no tools attached.
    mock_genai_client.models.generate_content.side_effect = [
        _route_response("FINANCIAL"),
        _text_response("answer"),
    ]
    agent = AgentV2(financial_manager=_mock_manager([_record()]))
    asyncio.run(agent.run(query="NCB revenue 2023?"))

    calls = mock_genai_client.models.generate_content.call_args_list
    assert len(calls) == 2
    route_cfg = calls[0].kwargs["config"]
    assert calls[0].kwargs["model"] == "gemini-2.5-flash"
    assert route_cfg.tools is None


def test_enable_financial_false_skips_financial(mock_genai_client):
    # Router always runs (catches refusals). The route call returns GROUNDING so
    # the financial branch is skipped, but the router itself still fires (1 Flash
    # call for route, 1 Pro call for web = 2 total).
    mock_genai_client.models.generate_content.side_effect = [
        _route_response("GROUNDING"),  # router returns GROUNDING
        _text_response("web answer"),  # Pro web path
    ]
    mgr = _mock_manager([_record()])
    agent = AgentV2(financial_manager=mgr)
    result = asyncio.run(agent.run(query="What was NCB revenue?", enable_financial_data=False))

    mgr.parse_user_query.assert_not_called()
    mgr.query_data.assert_not_called()
    # route call + web call = 2
    assert mock_genai_client.models.generate_content.call_count == 2
    assert result["response"] == "web answer"


def test_no_manager_skips_financial(mock_genai_client):
    # No manager: router fires (1 Flash route call), FINANCIAL branch skipped,
    # falls through to Pro web path (1 Pro call). Total = 2 calls.
    mock_genai_client.models.generate_content.side_effect = [
        _route_response("FINANCIAL"),  # router says FINANCIAL but no manager to call
        _text_response("web answer"),  # Pro web path
    ]
    agent = AgentV2()  # financial_manager=None
    result = asyncio.run(agent.run(query="What was NCB revenue?"))
    assert result["response"] == "web answer"
    assert mock_genai_client.models.generate_content.call_count == 2


def test_financial_error_falls_through_to_web(mock_genai_client):
    # route=FINANCIAL, parse succeeds, but the BigQuery query raises ->
    # _try_financial returns None -> run() falls through to the web/plain path
    # (graceful degradation: BigQuery being down must not break /chat/stream).
    mock_genai_client.models.generate_content.side_effect = [
        _route_response("FINANCIAL"),
        _text_response("web fallback answer"),
    ]
    mgr = MagicMock()
    mgr.metadata = {"symbols": ["NCBFG"], "years": ["2023"]}
    mgr.parse_user_query.return_value = _filters(symbols=["NCBFG"], years=["2023"])
    mgr.query_data.side_effect = RuntimeError("BigQuery exploded")
    agent = AgentV2(financial_manager=mgr)
    result = asyncio.run(agent.run(query="What was NCB revenue in 2023?"))

    assert result["response"] == "web fallback answer"
    assert "query_financial_data" not in (result.get("tools_executed") or [])
    # route + web fallback
    assert mock_genai_client.models.generate_content.call_count == 2


def test_financial_no_records_falls_through_to_web(mock_genai_client):
    # route=FINANCIAL, parse succeeds, but query returns no rows -> fall through
    # to web rather than synthesizing an empty-data answer.
    mock_genai_client.models.generate_content.side_effect = [
        _route_response("FINANCIAL"),
        _text_response("web fallback answer"),
    ]
    agent = AgentV2(financial_manager=_mock_manager([]))  # query_data -> []
    result = asyncio.run(agent.run(query="What was NCB revenue in 1850?"))

    assert result["response"] == "web fallback answer"
    assert "query_financial_data" not in (result.get("tools_executed") or [])
    assert mock_genai_client.models.generate_content.call_count == 2


def test_run_uses_cached_content_when_cache_hits(mock_genai_client):
    """AgentV2.run() passes cached_content= instead of system_instruction= on cache hit."""
    resp = MagicMock()
    resp.text = "hello"
    resp.candidates = []
    resp.usage_metadata = None
    mock_genai_client.models.generate_content.return_value = resp

    with (
        patch("app.agent_v2._SYSTEM_PROMPT_CACHE") as mock_cache,
        patch("app.agent_v2._SYSTEM_PROMPT_NO_SEARCH_CACHE"),
    ):
        mock_cache.get_cache_name.return_value = "cachedContents/xyz"

        agent = AgentV2()
        asyncio.run(
            agent.run("What is the JSE?", enable_financial_data=False, enable_web_search=True)
        )

    config = mock_genai_client.models.generate_content.call_args.kwargs["config"]
    assert config.cached_content == "cachedContents/xyz"
    # system_instruction must be absent (None or not set) when cached_content is used
    assert not getattr(config, "system_instruction", None)


def test_run_falls_back_to_system_instruction_when_cache_returns_none(mock_genai_client):
    """AgentV2.run() uses system_instruction= when cache returns None."""
    resp = MagicMock()
    resp.text = "hello"
    resp.candidates = []
    resp.usage_metadata = None
    mock_genai_client.models.generate_content.return_value = resp

    with (
        patch("app.agent_v2._SYSTEM_PROMPT_CACHE") as mock_cache,
        patch("app.agent_v2._SYSTEM_PROMPT_NO_SEARCH_CACHE"),
    ):
        mock_cache.get_cache_name.return_value = None

        agent = AgentV2()
        asyncio.run(
            agent.run("What is the JSE?", enable_financial_data=False, enable_web_search=True)
        )

    config = mock_genai_client.models.generate_content.call_args.kwargs["config"]
    assert config.system_instruction is not None
    assert not getattr(config, "cached_content", None)


# ---------------------------------------------------------------------------
# REFUSE path (AEO-23)
# ---------------------------------------------------------------------------

def test_refuse_path_uses_flash_only_no_pro(mock_genai_client):
    """REFUSE route: 2 Flash calls (route + refusal), zero Pro calls."""
    mock_genai_client.models.generate_content.side_effect = [
        _route_response("REFUSE"),
        _text_response("Sorry, I can only help with JSE topics."),
    ]
    agent = AgentV2(financial_manager=_mock_manager([]))
    result = asyncio.run(agent.run(query="Write me a poem about mangoes."))

    assert result["response"] == "Sorry, I can only help with JSE topics."
    assert result["record_count"] == 0
    assert result["data_found"] is False
    # Both calls must use Flash, not Pro
    calls = mock_genai_client.models.generate_content.call_args_list
    assert len(calls) == 2
    assert all(c.kwargs["model"] == "gemini-2.5-flash" for c in calls)


def test_refuse_path_skips_financial_manager(mock_genai_client):
    """On REFUSE the financial manager is never touched."""
    mock_genai_client.models.generate_content.side_effect = [
        _route_response("REFUSE"),
        _text_response("I only assist with JSE topics."),
    ]
    mgr = _mock_manager([])
    agent = AgentV2(financial_manager=mgr)
    asyncio.run(agent.run(query="Tell me about US stock market."))

    mgr.parse_user_query.assert_not_called()
    mgr.query_data.assert_not_called()


def test_refuse_path_no_tools_on_refusal_call(mock_genai_client):
    """The refusal Flash call must have no tools attached (no grounding)."""
    mock_genai_client.models.generate_content.side_effect = [
        _route_response("REFUSE"),
        _text_response("I can only help with JSE-related topics."),
    ]
    agent = AgentV2(financial_manager=_mock_manager([]))
    asyncio.run(agent.run(query="What is bitcoin?"))

    calls = mock_genai_client.models.generate_content.call_args_list
    # calls[0] = route call, calls[1] = refusal generation call
    refusal_cfg = calls[1].kwargs["config"]
    assert refusal_cfg.tools is None
