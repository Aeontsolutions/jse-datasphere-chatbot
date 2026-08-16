"""Tests for AgentV2's ticker/company grounding-note injection (AEO-25).

/chat/stream answers purely from Gemini 2.5 Pro + Google Search grounding
with no BigQuery data-query backstop (see test_agent_v2_router.py). That
architecture has no defense against the model misremembering which company a
JSE ticker symbol belongs to — verified in prod: asking about "MDS" produced
a fabricated claim that the ticker is shared between Medical Disposables &
Supplies Limited and Mailpac Group Limited (whose real symbol is MAILPAC).

The fix reuses the existing two-stage disambiguation pattern from
app.document_selector (already used by /chat's document auto-loading):
Stage 1 is an LLM call that extracts ticker symbols the query is clearly
asking about (so it doesn't mistake a metric abbreviation like "ROC" for a
ticker mention), Stage 2 is a deterministic lookup against the real
symbol_to_company associations already loaded from BigQuery. If a symbol
resolves, a verified fact block is injected into the Pro call's contents.
No BigQuery query happens per-request — only a dict lookup against metadata
that was already loaded at startup.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent_v2 import AgentV2


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


def _financial_manager(symbol_to_company):
    fm = MagicMock()
    fm.metadata = {
        "companies": sorted({c for cs in symbol_to_company.values() for c in cs if c}),
        "symbols": sorted(symbol_to_company.keys()),
        "associations": {"symbol_to_company": symbol_to_company},
    }
    return fm


# ---------------------------------------------------------------------------
# _build_grounding_note — pure Stage-2 lookup, no mocking needed
# ---------------------------------------------------------------------------


def test_grounding_note_none_when_no_symbols_extracted():
    agent = AgentV2()
    fm = _financial_manager({"MDS": ["Medical Disposables & Supplies Limited"]})
    assert agent._build_grounding_note([], fm) is None


def test_grounding_note_none_when_no_financial_manager():
    agent = AgentV2()
    assert agent._build_grounding_note(["MDS"], None) is None


def test_grounding_note_none_when_symbol_not_in_associations():
    agent = AgentV2()
    fm = _financial_manager({"MDS": ["Medical Disposables & Supplies Limited"]})
    assert agent._build_grounding_note(["NOTREAL"], fm) is None


def test_grounding_note_resolves_verified_company():
    agent = AgentV2()
    fm = _financial_manager(
        {
            "MDS": ["Medical Disposables & Supplies Limited"],
            "MAILPAC": ["Mailpac Group Limited"],
        }
    )
    note = agent._build_grounding_note(["MDS"], fm)
    assert note is not None
    assert "MDS = Medical Disposables & Supplies Limited" in note
    # Must not pull in the distinct MAILPAC company just because both are known.
    assert "Mailpac" not in note


def test_grounding_note_is_case_insensitive():
    agent = AgentV2()
    fm = _financial_manager({"MDS": ["Medical Disposables & Supplies Limited"]})
    note = agent._build_grounding_note(["mds"], fm)
    assert note is not None
    assert "MDS = Medical Disposables & Supplies Limited" in note


def test_grounding_note_filters_empty_string_company_artifacts():
    """BigQuery ARRAY_AGG with NULL Company rows can leave '' in the list;
    those aren't real companies and must never surface as a fact."""
    agent = AgentV2()
    fm = _financial_manager({"GK": ["", "Gracekennedy Limited"]})
    note = agent._build_grounding_note(["GK"], fm)
    assert note == "\n".join(
        [
            "[VERIFIED JSE DATA — ground truth from official JSE records. Use this "
            "for company/ticker identity. Do not describe a ticker symbol as shared "
            "or ambiguous unless multiple companies are listed for it below.]",
            "- GK = Gracekennedy Limited",
        ]
    )


def test_grounding_note_lists_multiple_companies_if_genuinely_shared():
    agent = AgentV2()
    fm = _financial_manager({"XYZ": ["Company A Limited", "Company B Limited"]})
    note = agent._build_grounding_note(["XYZ"], fm)
    assert "Company A Limited / Company B Limited" in note


# ---------------------------------------------------------------------------
# run() — end-to-end wiring: extraction -> lookup -> injected into contents
# ---------------------------------------------------------------------------


def test_run_injects_grounding_note_into_pro_call(mock_genai_client):
    mock_genai_client.aio.models.generate_content.side_effect = [
        _route_response("ALLOW"),
        _text_response("Medical Disposables & Supplies reported J$3.88 billion."),
    ]
    fm = _financial_manager(
        {
            "MDS": ["Medical Disposables & Supplies Limited"],
            "MAILPAC": ["Mailpac Group Limited"],
        }
    )
    with patch("app.agent_v2.extract_companies_from_query_async") as mock_extract:
        mock_extract.return_value = {
            "companies": ["Medical Disposables & Supplies Limited"],
            "symbols": ["MDS"],
        }
        agent = AgentV2()
        result = asyncio.run(
            agent.run(query="What was the revenue for MDS in 2025", financial_manager=fm)
        )

    assert result["response"] == "Medical Disposables & Supplies reported J$3.88 billion."
    calls = mock_genai_client.aio.models.generate_content.call_args_list
    assert len(calls) == 2
    pro_contents = calls[1].kwargs["contents"]
    last_turn_parts = pro_contents[-1].parts
    assert len(last_turn_parts) == 2
    assert "MDS = Medical Disposables & Supplies Limited" in last_turn_parts[0].text
    assert last_turn_parts[1].text == "What was the revenue for MDS in 2025"


def test_run_no_grounding_note_without_financial_manager(mock_genai_client):
    mock_genai_client.aio.models.generate_content.side_effect = [
        _route_response("ALLOW"),
        _text_response("answer"),
    ]
    with patch("app.agent_v2.extract_companies_from_query_async") as mock_extract:
        agent = AgentV2()
        asyncio.run(agent.run(query="What was NCB revenue in 2023?"))
        mock_extract.assert_not_called()

    calls = mock_genai_client.aio.models.generate_content.call_args_list
    pro_contents = calls[1].kwargs["contents"]
    assert len(pro_contents[-1].parts) == 1
    assert pro_contents[-1].parts[0].text == "What was NCB revenue in 2023?"


def test_run_no_grounding_note_when_extraction_finds_nothing(mock_genai_client):
    mock_genai_client.aio.models.generate_content.side_effect = [
        _route_response("ALLOW"),
        _text_response("answer"),
    ]
    fm = _financial_manager({"MDS": ["Medical Disposables & Supplies Limited"]})
    with patch("app.agent_v2.extract_companies_from_query_async") as mock_extract:
        mock_extract.return_value = {"companies": [], "symbols": []}
        agent = AgentV2()
        asyncio.run(agent.run(query="How is the Jamaican economy doing?", financial_manager=fm))

    calls = mock_genai_client.aio.models.generate_content.call_args_list
    pro_contents = calls[1].kwargs["contents"]
    assert len(pro_contents[-1].parts) == 1


def test_run_refuse_path_unaffected_by_grounding(mock_genai_client):
    """REFUSE still short-circuits to 2 Flash calls; extraction (if it ever
    resolves) must not force an extra Pro call or change the refusal text."""
    mock_genai_client.aio.models.generate_content.side_effect = [
        _route_response("REFUSE"),
        _text_response("Sorry, I can only help with JSE topics."),
    ]
    fm = _financial_manager({"MDS": ["Medical Disposables & Supplies Limited"]})
    with patch("app.agent_v2.extract_companies_from_query_async") as mock_extract:
        mock_extract.return_value = {
            "companies": ["Medical Disposables & Supplies Limited"],
            "symbols": ["MDS"],
        }
        agent = AgentV2()
        result = asyncio.run(agent.run(query="Write me a poem about MDS.", financial_manager=fm))

    assert result["response"] == "Sorry, I can only help with JSE topics."
    calls = mock_genai_client.aio.models.generate_content.call_args_list
    assert len(calls) == 2
