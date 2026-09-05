"""Tests for the two experiment flags that strip AgentV2's curated company knowledge.

AgentV2 carries company identity in two independent places:

  1. The BigQuery registry -- an LLM symbol-extraction call whose prompt holds
     the company/symbol lists, plus the deterministic symbol->company block
     injected into the synthesis call (AEO-25, test_agent_v2_grounding.py).
  2. A hardcoded "Key JSE Stocks by Sector" list of ~30 tickers inside
     SYSTEM_PROMPT.

`DISABLE_REGISTRY_GROUNDING` and `DISABLE_PROMPT_COMPANY_LIST` switch those
off so the eval suite can measure what the assistant does on internet grounding
alone. Both default to today's behaviour: absent or falsey changes nothing.

These exist to make an experiment reproducible, not to ship a new mode. The
tests that matter most are the two asserting the defaults are unchanged -- a
flag that silently alters production behaviour would invalidate every eval run
taken since it landed.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.agent_v2 as agent_v2
from app.agent_v2 import AgentV2, resolve_prompt_company_list, resolve_registry_grounding


@pytest.fixture
def mock_genai_client():
    with patch("app.agent_v2.get_genai_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock()
        mock_get_client.return_value = mock_client
        yield mock_client


def _response(text):
    resp = MagicMock()
    resp.text = text
    resp.candidates = []
    resp.usage_metadata = None
    return resp


def _financial_manager():
    return MagicMock(
        metadata={
            "companies": ["Medical Disposables & Supplies Limited"],
            "symbols": ["MDS"],
            "associations": {
                "symbol_to_company": {"MDS": ["Medical Disposables & Supplies Limited"]}
            },
        }
    )


# ---------------------------------------------------------------------------
# Resolvers -- default off, opt in explicitly
# ---------------------------------------------------------------------------


def test_registry_grounding_enabled_by_default(monkeypatch):
    monkeypatch.delenv("DISABLE_REGISTRY_GROUNDING", raising=False)
    assert resolve_registry_grounding() is True


def test_prompt_company_list_enabled_by_default(monkeypatch):
    monkeypatch.delenv("DISABLE_PROMPT_COMPANY_LIST", raising=False)
    assert resolve_prompt_company_list() is True


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "  true  "])
def test_registry_grounding_disabled_by_truthy_values(monkeypatch, value):
    monkeypatch.setenv("DISABLE_REGISTRY_GROUNDING", value)
    assert resolve_registry_grounding() is False


@pytest.mark.parametrize("value", ["false", "0", "no", "", "   "])
def test_registry_grounding_survives_falsey_values(monkeypatch, value):
    """A misspelled or empty value must not silently disable grounding --
    an experiment flag that trips on '' would change prod on a blank env var."""
    monkeypatch.setenv("DISABLE_REGISTRY_GROUNDING", value)
    assert resolve_registry_grounding() is True


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes"])
def test_prompt_company_list_disabled_by_truthy_values(monkeypatch, value):
    monkeypatch.setenv("DISABLE_PROMPT_COMPANY_LIST", value)
    assert resolve_prompt_company_list() is False


# ---------------------------------------------------------------------------
# Prompt variant -- the sector list is the only thing that goes
# ---------------------------------------------------------------------------


def test_default_system_prompt_contains_the_sector_list():
    assert "Key JSE Stocks by Sector" in agent_v2.SYSTEM_PROMPT
    assert "NCBFG" in agent_v2.SYSTEM_PROMPT


def test_stripped_prompt_drops_every_ticker_in_the_sector_list():
    stripped = agent_v2.strip_company_list(agent_v2.SYSTEM_PROMPT)
    assert "Key JSE Stocks by Sector" not in stripped
    for ticker in ("NCBFG", "JMMBGL", "WISYNCO", "GK", "CAR", "SCIJMD"):
        assert ticker not in stripped, f"{ticker} survived the strip"


def test_stripped_prompt_keeps_safety_and_scope_rules():
    """Only the company list goes. Losing a safety rule would make the
    experiment measure prompt damage rather than the missing registry."""
    stripped = agent_v2.strip_company_list(agent_v2.SYSTEM_PROMPT)
    for anchor in (
        "SAFETY RULES",
        "SCOPE RULES",
        "STYLE GUIDELINES",
        "No personalised investment recommendations",
        "Persona integrity",
        "You have web search access for current market data.",
    ):
        assert anchor in stripped, f"stripping removed {anchor!r}"


def test_stripped_prompt_leaves_no_double_blank_lines():
    stripped = agent_v2.strip_company_list(agent_v2.SYSTEM_PROMPT)
    assert "\n\n\n" not in stripped


# ---------------------------------------------------------------------------
# run() honours the registry flag
# ---------------------------------------------------------------------------


def _run_agent(mock_client, monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    mock_client.aio.models.generate_content.side_effect = [
        _response("ALLOW"),
        _response("Medical Disposables & Supplies Limited (MDS) is a JSE-listed distributor."),
    ]
    agent = AgentV2()
    with patch("app.agent_v2.extract_companies_from_query_async", new=AsyncMock()) as extract:
        extract.return_value = {"companies": [], "symbols": ["MDS"]}
        result = asyncio.run(
            agent.run(query="what does MDS do", financial_manager=_financial_manager())
        )
    return result, extract, mock_client.aio.models.generate_content


def _synthesis_text(generate_content):
    """Flatten the text of the synthesis call's contents (the 2nd call)."""
    contents = generate_content.call_args_list[1].kwargs["contents"]
    return " ".join(p.text or "" for c in contents for p in c.parts)


def test_grounding_note_injected_by_default(mock_genai_client, monkeypatch):
    monkeypatch.delenv("DISABLE_REGISTRY_GROUNDING", raising=False)
    _, extract, generate_content = _run_agent(mock_genai_client, monkeypatch)
    extract.assert_awaited_once()
    assert "VERIFIED JSE DATA" in _synthesis_text(generate_content)


def test_grounding_note_absent_when_registry_disabled(mock_genai_client, monkeypatch):
    _, extract, generate_content = _run_agent(
        mock_genai_client, monkeypatch, DISABLE_REGISTRY_GROUNDING="true"
    )
    assert "VERIFIED JSE DATA" not in _synthesis_text(generate_content)


def test_extraction_call_skipped_entirely_when_registry_disabled(mock_genai_client, monkeypatch):
    """Not just unused -- not made. An extraction call whose result is thrown
    away would keep paying latency and cost for an experiment arm that is
    meant to show what removing the registry costs."""
    _, extract, _ = _run_agent(mock_genai_client, monkeypatch, DISABLE_REGISTRY_GROUNDING="true")
    extract.assert_not_awaited()


def test_query_still_answered_with_registry_disabled(mock_genai_client, monkeypatch):
    result, _, _ = _run_agent(mock_genai_client, monkeypatch, DISABLE_REGISTRY_GROUNDING="true")
    assert "Medical Disposables" in result["response"]
    assert result["conversation_history"][-1]["role"] == "assistant"
