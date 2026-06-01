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


def _fc_response(args=None):
    """Step-2 EXTRACT response: forces a query_financial_data function call."""
    fc = MagicMock()
    fc.name = "query_financial_data"
    fc.args = (
        args
        if args is not None
        else {"symbols": ["NCB"], "years": ["2023"], "standard_items": ["revenue"]}
    )
    part = MagicMock()
    part.function_call = fc
    content = MagicMock()
    content.parts = [part]
    candidate = MagicMock()
    candidate.content = content
    resp = MagicMock()
    resp.candidates = [candidate]
    resp.text = ""
    resp.usage_metadata = None
    return resp


def _text_response(text):
    resp = MagicMock()
    resp.text = text
    resp.candidates = []
    resp.usage_metadata = None
    return resp


def _mock_manager(records):
    mgr = MagicMock()
    mgr.metadata = {"symbols": ["NCB"], "years": ["2023"]}
    mgr.query_data.return_value = records
    return mgr


def test_financial_path_routes_extracts_and_synthesizes(mock_genai_client):
    # route=FINANCIAL -> force-extract (tool call) -> synthesize = 3 LLM calls.
    mock_genai_client.models.generate_content.side_effect = [
        _route_response("FINANCIAL"),
        _fc_response(),
        _text_response("NCB's 2023 revenue was J$123.5M."),
    ]
    agent = AgentV2(financial_manager=_mock_manager([_record()]))
    result = asyncio.run(agent.run(query="What was NCB revenue in 2023?"))

    assert result["tools_executed"] == ["query_financial_data"]
    assert result["record_count"] == 1
    assert result["filters_used"].symbols == ["NCB"]
    assert result["data_preview"] and len(result["data_preview"]) == 1
    assert "123.5M" in result["response"]
    assert mock_genai_client.models.generate_content.call_count == 3


def test_route_other_skips_extraction_and_falls_through_to_web(mock_genai_client):
    # route=OTHER -> NO extraction call, NO query_data; web path runs (route + web).
    mock_genai_client.models.generate_content.side_effect = [
        _route_response("OTHER"),
        _text_response("Here is some recent JSE news."),
    ]
    agent = AgentV2(financial_manager=_mock_manager([]))
    result = asyncio.run(agent.run(query="Any recent JSE news?"))

    assert result["response"] == "Here is some recent JSE news."
    assert "query_financial_data" not in (result.get("tools_executed") or [])
    agent.financial_manager.query_data.assert_not_called()
    assert mock_genai_client.models.generate_content.call_count == 2


def test_route_then_force_extraction_uses_mode_any(mock_genai_client):
    # The ROUTE call passes no tools; the EXTRACT call forces the financial tool.
    mock_genai_client.models.generate_content.side_effect = [
        _route_response("FINANCIAL"),
        _fc_response(),
        _text_response("answer"),
    ]
    agent = AgentV2(financial_manager=_mock_manager([_record()]))
    asyncio.run(agent.run(query="NCB revenue 2023?"))

    calls = mock_genai_client.models.generate_content.call_args_list
    assert len(calls) == 3
    route_cfg = calls[0].kwargs["config"]
    extract_cfg = calls[1].kwargs["config"]
    # ROUTE call uses flash and no tools (pure classification).
    assert calls[0].kwargs["model"] == "gemini-2.5-flash"
    assert route_cfg.tools is None
    # EXTRACT call forces the query_financial_data tool with mode=ANY.
    assert extract_cfg.tools is not None
    assert "ANY" in str(extract_cfg.tool_config.function_calling_config.mode)


def test_enable_financial_false_skips_financial(mock_genai_client):
    mock_genai_client.models.generate_content.return_value = _text_response("web answer")
    mgr = _mock_manager([_record()])
    agent = AgentV2(financial_manager=mgr)
    result = asyncio.run(agent.run(query="What was NCB revenue?", enable_financial_data=False))

    mgr.query_data.assert_not_called()
    # No route call either — financial path never entered.
    assert mock_genai_client.models.generate_content.call_count == 1
    assert result["response"] == "web answer"


def test_no_manager_skips_financial(mock_genai_client):
    mock_genai_client.models.generate_content.return_value = _text_response("web answer")
    agent = AgentV2()  # financial_manager=None
    result = asyncio.run(agent.run(query="What was NCB revenue?"))
    assert result["response"] == "web answer"
    assert mock_genai_client.models.generate_content.call_count == 1


def test_financial_error_falls_through_to_web(mock_genai_client):
    # route=FINANCIAL, extraction succeeds, but the BigQuery query raises ->
    # _try_financial returns None -> run() falls through to the web/plain path
    # (graceful degradation: BigQuery being down must not break /chat/stream).
    mock_genai_client.models.generate_content.side_effect = [
        _route_response("FINANCIAL"),
        _fc_response(),
        _text_response("web fallback answer"),
    ]
    mgr = MagicMock()
    mgr.metadata = {"symbols": ["NCB"], "years": ["2023"]}
    mgr.query_data.side_effect = RuntimeError("BigQuery exploded")
    agent = AgentV2(financial_manager=mgr)
    result = asyncio.run(agent.run(query="What was NCB revenue in 2023?"))

    assert result["response"] == "web fallback answer"
    assert "query_financial_data" not in (result.get("tools_executed") or [])
    # route + extract + web fallback
    assert mock_genai_client.models.generate_content.call_count == 3
