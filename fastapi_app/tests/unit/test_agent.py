"""Unit tests for Agent module - specifically query routing logic."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent import (
    AgentOrchestrator,
    FINANCIAL_KEYWORDS,
    WEB_SEARCH_KEYWORDS,
    YEAR_PATTERN,
    VAGUE_PERFORMANCE_PATTERNS,
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

    def test_vague_performance_patterns_include_key_phrases(self):
        """Vague performance patterns should include common vague phrases."""
        expected = {"how well did", "how did", "perform", "tell me about", "overview"}
        assert expected.issubset(VAGUE_PERFORMANCE_PATTERNS)


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

    async def test_route_vague_query_with_symbol_uses_both(self, orchestrator):
        """Vague performance query with symbol should route to BOTH tools."""
        result = await orchestrator._route_query(
            query="How well did GK perform in 2023?",
            enable_web_search=True,
            enable_financial_data=True,
        )
        assert result["use_financial"] is True
        assert result["use_web_search"] is True
        assert result["routing_method"] == "rule_based"
        assert result.get("reason") == "vague_performance_query"

    async def test_route_vague_query_how_did_with_symbol(self, orchestrator):
        """'How did X do' pattern with symbol should route to BOTH tools."""
        result = await orchestrator._route_query(
            query="How did NCB do last year?",
            enable_web_search=True,
            enable_financial_data=True,
        )
        assert result["use_financial"] is True
        assert result["use_web_search"] is True
        assert result["routing_method"] == "rule_based"

    async def test_route_vague_query_tell_me_about(self, orchestrator):
        """'Tell me about X' pattern with symbol should route to BOTH tools."""
        result = await orchestrator._route_query(
            query="Tell me about GK performance in 2023",
            enable_web_search=True,
            enable_financial_data=True,
        )
        assert result["use_financial"] is True
        assert result["use_web_search"] is True

    async def test_route_vague_query_overview(self, orchestrator):
        """'Overview of X' pattern with symbol should route to BOTH tools."""
        result = await orchestrator._route_query(
            query="Give me an overview of CPJ for 2023",
            enable_web_search=True,
            enable_financial_data=True,
        )
        assert result["use_financial"] is True
        assert result["use_web_search"] is True

    async def test_route_specific_metric_not_vague(self, orchestrator):
        """Query with specific metric should NOT be treated as vague."""
        # Even though it has "how well", it specifies "revenue" so it's not vague
        result = await orchestrator._route_query(
            query="What is GK revenue for 2023?",
            enable_web_search=True,
            enable_financial_data=True,
        )
        # Should route to financial only since it has a specific metric
        assert result["use_financial"] is True
        assert result["use_web_search"] is False


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


@pytest.mark.unit
@pytest.mark.asyncio
class TestOptimizePrompt:
    """Test the prompt optimization for pronoun resolution."""

    async def test_no_optimization_without_history(self, orchestrator):
        """Query without conversation history should return unchanged."""
        result = await orchestrator._optimize_prompt(
            query="What is GK revenue?",
            conversation_history=None,
        )
        assert result == "What is GK revenue?"

    async def test_no_optimization_without_pronouns(self, orchestrator):
        """Query without pronouns should return unchanged."""
        result = await orchestrator._optimize_prompt(
            query="What is GK revenue for 2023?",
            conversation_history=[
                {"role": "user", "content": "Tell me about NCB"},
                {"role": "assistant", "content": "NCB is a bank..."},
            ],
        )
        assert result == "What is GK revenue for 2023?"

    async def test_optimization_with_pronoun_calls_llm(self, orchestrator):
        """Query with pronouns should call LLM for resolution."""
        mock_response = MagicMock()
        mock_response.text = "What is GraceKennedy's revenue for the last 5 years?"

        orchestrator.client.models.generate_content = MagicMock(return_value=mock_response)

        result = await orchestrator._optimize_prompt(
            query="What was their revenue for the last 5 years?",
            conversation_history=[
                {"role": "user", "content": "Tell me about GraceKennedy"},
                {"role": "assistant", "content": "GK is a conglomerate..."},
            ],
        )

        # LLM should have been called
        orchestrator.client.models.generate_content.assert_called_once()
        assert result == "What is GraceKennedy's revenue for the last 5 years?"

    async def test_optimization_handles_llm_failure(self, orchestrator):
        """LLM failure should return original query."""
        orchestrator.client.models.generate_content = MagicMock(side_effect=Exception("API error"))

        result = await orchestrator._optimize_prompt(
            query="What was their revenue?",
            conversation_history=[
                {"role": "user", "content": "Tell me about GK"},
            ],
        )

        # Should fall back to original query
        assert result == "What was their revenue?"

    async def test_optimization_removes_quotes(self, orchestrator):
        """LLM response with quotes should have quotes stripped."""
        mock_response = MagicMock()
        mock_response.text = '"What is NCB revenue?"'

        orchestrator.client.models.generate_content = MagicMock(return_value=mock_response)

        result = await orchestrator._optimize_prompt(
            query="What is their revenue?",
            conversation_history=[
                {"role": "user", "content": "Tell me about NCB"},
            ],
        )

        assert result == "What is NCB revenue?"


@pytest.mark.unit
class TestExecuteFinancialQueryNullHandling:
    """Test that execute_financial_query handles null/None values from LLM."""

    def test_null_symbols_handled(self, mock_financial_manager):
        """Null symbols in args should not cause error."""
        from app.agent import execute_financial_query
        import asyncio

        # Mock the query_data method
        mock_financial_manager.query_data = MagicMock(return_value=[])
        mock_financial_manager.metadata = {"associations": {}}
        mock_financial_manager._post_process_filters = MagicMock(side_effect=lambda x: x)

        # Args with null values (as LLM might return)
        args = {"symbols": None, "years": ["2023"], "standard_items": ["revenue"]}

        # Should not raise NoneType error
        result = asyncio.get_event_loop().run_until_complete(
            execute_financial_query(mock_financial_manager, args)
        )

        records, filters, chart, sources = result
        assert filters.symbols == []  # Should be empty list, not None

    def test_all_null_args_handled(self, mock_financial_manager):
        """All null args should not cause error."""
        from app.agent import execute_financial_query
        import asyncio

        mock_financial_manager.query_data = MagicMock(return_value=[])
        mock_financial_manager.metadata = {"associations": {}}
        mock_financial_manager._post_process_filters = MagicMock(side_effect=lambda x: x)

        # All null values
        args = {"symbols": None, "years": None, "standard_items": None}

        result = asyncio.get_event_loop().run_until_complete(
            execute_financial_query(mock_financial_manager, args)
        )

        records, filters, chart, sources = result
        assert filters.symbols == []
        assert filters.years == []
        assert filters.standard_items == []
