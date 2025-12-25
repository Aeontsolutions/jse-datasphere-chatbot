"""Unit tests for Agent module - specifically query routing logic."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent import (
    AgentOrchestrator,
    FINANCIAL_KEYWORDS,
    WEB_SEARCH_KEYWORDS,
    YEAR_PATTERN,
)


@pytest.fixture
def mock_financial_manager():
    """Create a mock financial manager with metadata."""
    manager = MagicMock()
    manager.metadata = {
        "symbols": ["GK", "NCB", "CPJ", "JBG", "MDS", "NCBFG", "PROVEN"],
        "years": ["2021", "2022", "2023", "2024"],
        "standard_items": ["revenue", "net_profit", "eps", "total_assets"],
    }
    return manager


@pytest.fixture
def orchestrator(mock_financial_manager):
    """Create an AgentOrchestrator with mocked dependencies."""
    with patch("app.agent.get_genai_client") as mock_client:
        mock_client.return_value = MagicMock()
        return AgentOrchestrator(
            financial_manager=mock_financial_manager,
            associations=None,
        )


@pytest.mark.unit
class TestQueryRoutingConstants:
    """Test the routing constant definitions."""

    def test_financial_keywords_include_metrics(self):
        """Financial keywords should include common financial metrics."""
        expected = {"revenue", "profit", "eps", "earnings", "margin", "assets"}
        assert expected.issubset(FINANCIAL_KEYWORDS)

    def test_web_search_keywords_include_news_terms(self):
        """Web search keywords should include news-related terms."""
        expected = {"news", "latest", "recent", "today", "update"}
        assert expected.issubset(WEB_SEARCH_KEYWORDS)

    def test_year_pattern_matches_valid_years(self):
        """Year pattern should match 4-digit years starting with 20."""
        assert YEAR_PATTERN.search("2023")
        assert YEAR_PATTERN.search("revenue for 2024")
        assert YEAR_PATTERN.search("Compare 2022 and 2023")

    def test_year_pattern_no_match_invalid(self):
        """Year pattern should not match invalid years."""
        assert not YEAR_PATTERN.search("1999")
        assert not YEAR_PATTERN.search("no year here")


@pytest.mark.unit
@pytest.mark.asyncio
class TestQueryRoutingRuleBased:
    """Test rule-based query routing (no LLM calls)."""

    async def test_route_financial_only_with_symbol(self, orchestrator):
        """Query with stock symbol + financial metric should route to financial only."""
        result = await orchestrator._route_query(
            query="What is GK revenue for 2023?",
            enable_web_search=True,
            enable_financial_data=True,
        )
        assert result["use_financial"] is True
        assert result["use_web_search"] is False
        assert result["routing_method"] == "rule_based"
        assert result["confidence"] == "high"

    async def test_route_financial_only_with_year(self, orchestrator):
        """Query with year + financial keyword should route to financial only."""
        result = await orchestrator._route_query(
            query="Show me profit margins for 2023",
            enable_web_search=True,
            enable_financial_data=True,
        )
        assert result["use_financial"] is True
        assert result["use_web_search"] is False
        assert result["routing_method"] == "rule_based"

    async def test_route_financial_only_comparison(self, orchestrator):
        """Comparison query should route to financial only."""
        result = await orchestrator._route_query(
            query="Compare GK and CPJ revenue for 2023",
            enable_web_search=True,
            enable_financial_data=True,
        )
        assert result["use_financial"] is True
        assert result["use_web_search"] is False
        assert result["routing_method"] == "rule_based"

    async def test_route_web_only_news(self, orchestrator):
        """Query with news keywords should route to web only."""
        result = await orchestrator._route_query(
            query="Latest news about Jamaica Stock Exchange",
            enable_web_search=True,
            enable_financial_data=True,
        )
        assert result["use_financial"] is False
        assert result["use_web_search"] is True
        assert result["routing_method"] == "rule_based"
        assert result["confidence"] == "high"

    async def test_route_web_only_recent(self, orchestrator):
        """Query with 'recent' keyword should route to web only."""
        result = await orchestrator._route_query(
            query="What happened recently in the stock market?",
            enable_web_search=True,
            enable_financial_data=True,
        )
        assert result["use_financial"] is False
        assert result["use_web_search"] is True

    async def test_route_both_tools_mixed_signals(self, orchestrator):
        """Query with both financial and web signals should use both tools."""
        result = await orchestrator._route_query(
            query="GK revenue for 2023 and latest news about them",
            enable_web_search=True,
            enable_financial_data=True,
        )
        assert result["use_financial"] is True
        assert result["use_web_search"] is True
        assert result["routing_method"] == "rule_based"
        assert result["confidence"] == "medium"

    async def test_route_respects_disabled_web_search(self, orchestrator):
        """When web search is disabled, should only use financial."""
        result = await orchestrator._route_query(
            query="Any query here",
            enable_web_search=False,
            enable_financial_data=True,
        )
        assert result["use_financial"] is True
        assert result["use_web_search"] is False
        assert result["routing_method"] == "single_tool"

    async def test_route_respects_disabled_financial(self, orchestrator):
        """When financial is disabled, should only use web search."""
        result = await orchestrator._route_query(
            query="Any query here",
            enable_web_search=True,
            enable_financial_data=False,
        )
        assert result["use_financial"] is False
        assert result["use_web_search"] is True
        assert result["routing_method"] == "single_tool"

    async def test_route_both_disabled(self, orchestrator):
        """When both tools are disabled, should return disabled state."""
        result = await orchestrator._route_query(
            query="Any query here",
            enable_web_search=False,
            enable_financial_data=False,
        )
        assert result["use_financial"] is False
        assert result["use_web_search"] is False
        assert result["routing_method"] == "disabled"


@pytest.mark.unit
@pytest.mark.asyncio
class TestQueryRoutingLLMFallback:
    """Test LLM fallback for ambiguous queries."""

    async def test_route_ambiguous_uses_llm(self, orchestrator):
        """Ambiguous query with no clear signals should trigger LLM fallback."""
        # Mock the LLM classification
        orchestrator._classify_with_llm = AsyncMock(
            return_value={"financial": True, "web_search": True}
        )

        result = await orchestrator._route_query(
            query="Tell me about Jamaica Broilers",  # No clear financial/web signal
            enable_web_search=True,
            enable_financial_data=True,
        )
        assert result["routing_method"] == "llm_fallback"
        orchestrator._classify_with_llm.assert_called_once()

    async def test_route_llm_fallback_failure_defaults_to_both(self, orchestrator):
        """If LLM classification fails, should default to both tools."""
        # Mock the LLM classification to raise an exception
        orchestrator._classify_with_llm = AsyncMock(side_effect=Exception("API error"))

        result = await orchestrator._route_query(
            query="Tell me about Jamaica Broilers",
            enable_web_search=True,
            enable_financial_data=True,
        )
        assert result["use_financial"] is True
        assert result["use_web_search"] is True
        assert result["routing_method"] == "default"
        assert result["confidence"] == "low"


@pytest.mark.unit
@pytest.mark.asyncio
class TestClassifyWithLLM:
    """Test the LLM classification method."""

    async def test_classify_parses_json_response(self, orchestrator):
        """LLM classification should parse valid JSON response."""
        mock_response = MagicMock()
        mock_response.text = '{"financial": true, "web_search": false}'

        orchestrator.client.models.generate_content = MagicMock(return_value=mock_response)

        result = await orchestrator._classify_with_llm("What is NCB revenue?")
        assert result["financial"] is True
        assert result["web_search"] is False

    async def test_classify_handles_markdown_json(self, orchestrator):
        """LLM classification should handle markdown-wrapped JSON."""
        mock_response = MagicMock()
        mock_response.text = '```json\n{"financial": false, "web_search": true}\n```'

        orchestrator.client.models.generate_content = MagicMock(return_value=mock_response)

        result = await orchestrator._classify_with_llm("Latest news")
        assert result["financial"] is False
        assert result["web_search"] is True

    async def test_classify_defaults_on_invalid_json(self, orchestrator):
        """LLM classification should default to both on invalid JSON."""
        mock_response = MagicMock()
        mock_response.text = "I think you need both tools"

        orchestrator.client.models.generate_content = MagicMock(return_value=mock_response)

        result = await orchestrator._classify_with_llm("Some query")
        # Should default to both on parse failure
        assert result["financial"] is True
        assert result["web_search"] is True


@pytest.mark.unit
class TestBuildFinancialContext:
    """Test the financial context building for synthesis."""

    def test_build_context_empty_records(self, orchestrator):
        """Empty records should return appropriate message."""
        from app.models import FinancialDataFilters

        filters = FinancialDataFilters(
            companies=[],
            symbols=[],
            years=[],
            standard_items=[],
            interpretation="",
            data_availability_note="",
            is_follow_up=False,
            context_used="",
        )
        result = orchestrator._build_financial_context([], filters)
        assert "No financial data found" in result

    def test_build_context_with_records(self, orchestrator):
        """Records should be formatted into context string."""
        from app.models import FinancialDataFilters, FinancialDataRecord

        records = [
            FinancialDataRecord(
                company="GraceKennedy",
                symbol="GK",
                year="2023",
                standard_item="revenue",
                item=155000000.0,
                unit_multiplier=1,
                formatted_value="$155.00M",
            )
        ]
        filters = FinancialDataFilters(
            companies=["GraceKennedy"],
            symbols=["GK"],
            years=["2023"],
            standard_items=["revenue"],
            interpretation="",
            data_availability_note="",
            is_follow_up=False,
            context_used="",
        )

        result = orchestrator._build_financial_context(records, filters)
        assert "GraceKennedy" in result
        assert "revenue" in result
        assert "2023" in result
