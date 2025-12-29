"""
Diagnostic Unit Tests for Agent Prompt Optimization.

These tests focus on isolating the root causes of UAT failures:
1. Entity detection in queries
2. Context extraction from conversation history
3. Clarification logic triggering
4. Full prompt optimization flow

UAT Issues Identified:
- Over-aggressive clarification when entities ARE present (NCB, GK, etc.)
- Entity detection failing for valid symbols
- Pronoun resolution not using conversation history
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent import (
    AgentOrchestrator,
    CLARIFICATION_TEMPLATES,
    PRONOUNS_NEEDING_RESOLUTION,
    GENERAL_MARKET_TERMS,
)
from app.models import ClarificationReason, PromptOptimizationResult


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_financial_manager_realistic():
    """Create a mock financial manager with REALISTIC metadata (matching production)."""
    manager = MagicMock()
    manager.metadata = {
        # Note: Production has NCBFG, not NCB - this tests partial matching
        "symbols": ["GK", "NCBFG", "CPJ", "JBG", "MDS", "PROVEN", "ELITE", "DOLLA", "FESCO"],
        "companies": [
            "GraceKennedy",
            "NCB Financial Group Limited",
            "Caribbean Producers Jamaica",
        ],
        "years": ["2020", "2021", "2022", "2023", "2024"],
        "standard_items": ["revenue", "net_profit", "eps"],
        "associations": {
            "symbol_to_company": {
                "GK": ["GraceKennedy"],
                "NCBFG": ["NCB Financial Group Limited"],
            },
            "company_to_symbol": {
                "GraceKennedy": ["GK"],
                "NCB Financial Group Limited": ["NCBFG"],
            },
        },
    }
    return manager


@pytest.fixture
def mock_financial_manager_with_metadata():
    """Create a mock financial manager with comprehensive metadata."""
    manager = MagicMock()
    manager.metadata = {
        "symbols": ["GK", "NCB", "CPJ", "JBG", "MDS", "NCBFG", "PROVEN", "ELITE", "DOLLA", "FESCO"],
        "companies": [
            "GraceKennedy",
            "National Commercial Bank",
            "Caribbean Producers Jamaica",
            "Jamaica Broilers Group",
            "Medical Disposables & Supplies",
            "Elite Diagnostic Limited",
            "Dolla Financial Services Limited",
        ],
        "years": ["2020", "2021", "2022", "2023", "2024"],
        "standard_items": ["revenue", "net_profit", "eps", "total_assets", "gross_profit"],
        "associations": {
            "symbol_to_company": {
                "GK": ["GraceKennedy"],
                "NCB": ["National Commercial Bank"],
                "ELITE": ["Elite Diagnostic Limited"],
            },
            "company_to_symbol": {
                "GraceKennedy": ["GK"],
                "National Commercial Bank": ["NCB"],
            },
        },
    }
    return manager


@pytest.fixture
def mock_financial_manager_empty_metadata():
    """Create a mock financial manager with NO metadata - simulates init failure."""
    manager = MagicMock()
    manager.metadata = None
    return manager


@pytest.fixture
def mock_financial_manager_empty_symbols():
    """Create a mock financial manager with empty symbols list."""
    manager = MagicMock()
    manager.metadata = {
        "symbols": [],  # Empty!
        "companies": [],
        "years": ["2023", "2024"],
        "standard_items": ["revenue", "net_profit"],
    }
    return manager


@pytest.fixture
def orchestrator_realistic(mock_financial_manager_realistic):
    """Create AgentOrchestrator with REALISTIC metadata (NCBFG not NCB)."""
    with patch("app.agent.get_genai_client") as mock_client:
        mock_client.return_value = MagicMock()
        return AgentOrchestrator(
            financial_manager=mock_financial_manager_realistic,
            associations=None,
        )


@pytest.fixture
def orchestrator_with_metadata(mock_financial_manager_with_metadata):
    """Create AgentOrchestrator with properly loaded metadata."""
    with patch("app.agent.get_genai_client") as mock_client:
        mock_client.return_value = MagicMock()
        return AgentOrchestrator(
            financial_manager=mock_financial_manager_with_metadata,
            associations=None,
        )


@pytest.fixture
def orchestrator_no_metadata(mock_financial_manager_empty_metadata):
    """Create AgentOrchestrator with NO metadata."""
    with patch("app.agent.get_genai_client") as mock_client:
        mock_client.return_value = MagicMock()
        return AgentOrchestrator(
            financial_manager=mock_financial_manager_empty_metadata,
            associations=None,
        )


@pytest.fixture
def orchestrator_empty_symbols(mock_financial_manager_empty_symbols):
    """Create AgentOrchestrator with empty symbols list."""
    with patch("app.agent.get_genai_client") as mock_client:
        mock_client.return_value = MagicMock()
        return AgentOrchestrator(
            financial_manager=mock_financial_manager_empty_symbols,
            associations=None,
        )


# =============================================================================
# DIAGNOSTIC TESTS: Entity Detection
# =============================================================================


@pytest.mark.unit
class TestEntityDetectionInQuery:
    """
    DIAGNOSTIC: Test that entities are correctly detected in queries.

    These tests isolate the entity detection logic that's failing in UAT.
    The issue: queries like "What is NCB EPS?" are triggering clarification
    when they should proceed.
    """

    def test_detect_symbol_uppercase(self, orchestrator_with_metadata):
        """Symbol in uppercase should be detected."""
        query = "What is NCB revenue for 2023?"
        query_upper = query.upper()

        symbols = orchestrator_with_metadata.financial_manager.metadata.get("symbols", [])
        detected = [s for s in symbols if s in query_upper]

        assert "NCB" in detected, f"NCB should be detected in '{query}'. Symbols: {symbols}"

    def test_detect_symbol_mixed_case(self, orchestrator_with_metadata):
        """Symbol in mixed case query should be detected."""
        query = "What is Ncb revenue for 2023?"
        query_upper = query.upper()

        symbols = orchestrator_with_metadata.financial_manager.metadata.get("symbols", [])
        detected = [s for s in symbols if s in query_upper]

        assert "NCB" in detected, f"NCB should be detected in '{query}'"

    def test_detect_multiple_symbols(self, orchestrator_with_metadata):
        """Multiple symbols in query should all be detected."""
        query = "Compare GK and NCB revenue for 2023"
        query_upper = query.upper()

        symbols = orchestrator_with_metadata.financial_manager.metadata.get("symbols", [])
        detected = [s for s in symbols if s in query_upper]

        assert "GK" in detected, f"GK should be detected in '{query}'"
        assert "NCB" in detected, f"NCB should be detected in '{query}'"

    def test_detect_symbol_at_word_boundary(self, orchestrator_with_metadata):
        """Symbol should be detected even when followed by 's or punctuation."""
        queries = [
            "What is NCB's revenue?",
            "Show NCB, GK data",
            "NCB and GK comparison",
        ]

        symbols = orchestrator_with_metadata.financial_manager.metadata.get("symbols", [])

        for query in queries:
            query_upper = query.upper()
            detected = [s for s in symbols if s in query_upper]
            assert len(detected) > 0, f"Should detect symbols in '{query}'"

    def test_partial_symbol_matching_ncb_to_ncbfg(self, orchestrator_realistic):
        """NCB in query should match NCBFG symbol (partial prefix matching)."""
        query = "What is NCB revenue for 2023?"
        extracted_context = {"entities": [], "years": [], "metrics": [], "last_focus": None}

        result = orchestrator_realistic._detect_ambiguity(
            query=query,
            extracted_context=extracted_context,
            conversation_history=None,
        )

        assert (
            result is None
        ), f"NCB should match NCBFG via partial matching. Got clarification: {result}"

    def test_comparison_partial_matching(self, orchestrator_realistic):
        """Compare GK and NCB should work even when symbol is NCBFG."""
        query = "Compare GK and NCB revenue for 2023"
        extracted_context = {"entities": [], "years": [], "metrics": [], "last_focus": None}

        result = orchestrator_realistic._detect_ambiguity(
            query=query,
            extracted_context=extracted_context,
            conversation_history=None,
        )

        assert result is None, f"Compare with NCB→NCBFG should work. Got clarification: {result}"

    def test_no_detection_with_empty_metadata(self, orchestrator_no_metadata):
        """With no metadata, entity detection should fail gracefully."""
        query = "What is NCB revenue?"

        # This should not crash
        metadata = orchestrator_no_metadata.financial_manager.metadata
        assert metadata is None

    def test_no_detection_with_empty_symbols(self, orchestrator_empty_symbols):
        """With empty symbols list, no entities should be detected."""
        query = "What is NCB revenue?"
        query_upper = query.upper()

        symbols = orchestrator_empty_symbols.financial_manager.metadata.get("symbols", [])
        detected = [s for s in symbols if s in query_upper]

        assert len(detected) == 0, "No symbols should be detected with empty symbols list"


# =============================================================================
# DIAGNOSTIC TESTS: _detect_ambiguity Method
# =============================================================================


@pytest.mark.unit
class TestDetectAmbiguity:
    """
    DIAGNOSTIC: Test the _detect_ambiguity method.

    This is the core method that's over-triggering clarification.
    """

    def test_no_ambiguity_with_symbol_in_query(self, orchestrator_with_metadata):
        """Query with explicit symbol should NOT trigger clarification."""
        query = "What is NCB revenue for 2023?"
        extracted_context = {"entities": [], "years": [], "metrics": [], "last_focus": None}

        result = orchestrator_with_metadata._detect_ambiguity(
            query=query,
            extracted_context=extracted_context,
            conversation_history=None,
        )

        assert result is None, f"Should NOT need clarification for '{query}'. Got: {result}"

    def test_no_ambiguity_with_multiple_symbols(self, orchestrator_with_metadata):
        """Query with multiple symbols should NOT trigger clarification."""
        query = "Compare GK and NCB revenue for 2023"
        extracted_context = {"entities": [], "years": [], "metrics": [], "last_focus": None}

        result = orchestrator_with_metadata._detect_ambiguity(
            query=query,
            extracted_context=extracted_context,
            conversation_history=None,
        )

        assert result is None, f"Should NOT need clarification for comparison query"

    def test_ambiguity_no_entity_anywhere(self, orchestrator_with_metadata):
        """Query with no entity and no history should trigger clarification."""
        query = "What is the revenue?"
        extracted_context = {"entities": [], "years": [], "metrics": [], "last_focus": None}

        result = orchestrator_with_metadata._detect_ambiguity(
            query=query,
            extracted_context=extracted_context,
            conversation_history=None,
        )

        assert result is not None, "Should need clarification for entity-less query"
        assert result[0] == ClarificationReason.NO_ENTITY

    def test_no_ambiguity_entity_in_context(self, orchestrator_with_metadata):
        """Pronoun with entity in context should NOT need clarification."""
        query = "What is their revenue?"
        extracted_context = {
            "entities": ["NCB"],  # Entity from conversation history
            "years": [],
            "metrics": [],
            "last_focus": "NCB",
        }

        result = orchestrator_with_metadata._detect_ambiguity(
            query=query,
            extracted_context=extracted_context,
            conversation_history=[
                {"role": "user", "content": "Tell me about NCB"},
                {"role": "assistant", "content": "NCB is a bank..."},
            ],
        )

        assert result is None, "Pronoun with context entity should NOT need clarification"

    def test_ambiguity_pronoun_no_context(self, orchestrator_with_metadata):
        """Pronoun with NO entity in context should trigger clarification."""
        query = "What is their profit?"
        extracted_context = {"entities": [], "years": [], "metrics": [], "last_focus": None}

        result = orchestrator_with_metadata._detect_ambiguity(
            query=query,
            extracted_context=extracted_context,
            conversation_history=[],
        )

        assert result is not None, "Pronoun without context should need clarification"
        # Note: NO_ENTITY is checked before UNRESOLVED_PRONOUN in the logic
        assert result[0] == ClarificationReason.NO_ENTITY

    def test_no_ambiguity_general_market_query(self, orchestrator_with_metadata):
        """General market queries should NOT need entity clarification."""
        queries = [
            "What is happening on the JSE market?",
            "Latest stock exchange news",
            "How is the sector performing?",
        ]

        for query in queries:
            extracted_context = {"entities": [], "years": [], "metrics": [], "last_focus": None}

            result = orchestrator_with_metadata._detect_ambiguity(
                query=query,
                extracted_context=extracted_context,
                conversation_history=None,
            )

            assert result is None, f"General query '{query}' should NOT need clarification"

    def test_comparison_needs_two_entities(self, orchestrator_with_metadata):
        """'Compare' with less than 2 entities should trigger clarification."""
        query = "Compare the banks"  # No specific entities
        extracted_context = {"entities": [], "years": [], "metrics": [], "last_focus": None}

        result = orchestrator_with_metadata._detect_ambiguity(
            query=query,
            extracted_context=extracted_context,
            conversation_history=None,
        )

        assert result is not None, "Comparison without entities should need clarification"
        assert result[0] == ClarificationReason.AMBIGUOUS_COMPARISON

    def test_comparison_with_two_symbols_no_clarify(self, orchestrator_with_metadata):
        """'Compare GK and NCB' should NOT need clarification."""
        query = "Compare GK and NCB revenue"
        extracted_context = {"entities": [], "years": [], "metrics": [], "last_focus": None}

        result = orchestrator_with_metadata._detect_ambiguity(
            query=query,
            extracted_context=extracted_context,
            conversation_history=None,
        )

        assert result is None, f"Comparison with two symbols should NOT need clarification"

    def test_one_round_max_clarification(self, orchestrator_with_metadata):
        """After one clarification, should NOT clarify again."""
        query = "What about it?"
        extracted_context = {"entities": [], "years": [], "metrics": [], "last_focus": None}

        # Simulate previous clarification in history
        conversation_history = [
            {"role": "user", "content": "Tell me about stocks"},
            {
                "role": "assistant",
                "content": "I want to make sure I understand your question correctly. Which company?",
            },
            {"role": "user", "content": "Just any"},
        ]

        result = orchestrator_with_metadata._detect_ambiguity(
            query=query,
            extracted_context=extracted_context,
            conversation_history=conversation_history,
        )

        # Should NOT clarify again (1-round max)
        assert result is None, "Should not clarify again after 1 round"


# =============================================================================
# DIAGNOSTIC TESTS: Context Extraction
# =============================================================================


@pytest.mark.unit
class TestExtractContextFromHistory:
    """
    DIAGNOSTIC: Test context extraction from conversation history.

    This is crucial for pronoun resolution.
    """

    def test_extract_symbols_from_history(self, orchestrator_with_metadata):
        """Symbols mentioned in history should be extracted."""
        history = [
            {"role": "user", "content": "Tell me about NCB"},
            {"role": "assistant", "content": "NCB is a major bank in Jamaica..."},
        ]

        result = orchestrator_with_metadata._extract_context_from_history(history)

        assert "NCB" in result["entities"], f"NCB should be in entities. Got: {result}"
        assert result["last_focus"] == "NCB", "last_focus should be NCB"

    def test_extract_multiple_symbols(self, orchestrator_with_metadata):
        """Multiple symbols should be extracted in order."""
        history = [
            {"role": "user", "content": "What is NCB revenue?"},
            {"role": "assistant", "content": "NCB revenue was $5M."},
            {"role": "user", "content": "What about GK?"},
            {"role": "assistant", "content": "GK revenue was $10M."},
        ]

        result = orchestrator_with_metadata._extract_context_from_history(history)

        assert "NCB" in result["entities"]
        assert "GK" in result["entities"]
        assert result["last_focus"] == "GK", "last_focus should be the most recent entity"

    def test_extract_years_from_history(self, orchestrator_with_metadata):
        """Years mentioned in history should be extracted."""
        history = [
            {"role": "user", "content": "Show me NCB data for 2023"},
            {"role": "assistant", "content": "Here's the 2023 data..."},
        ]

        result = orchestrator_with_metadata._extract_context_from_history(history)

        assert "2023" in result["years"], f"2023 should be in years. Got: {result}"

    def test_extract_empty_history(self, orchestrator_with_metadata):
        """Empty history should return empty context."""
        result = orchestrator_with_metadata._extract_context_from_history([])

        assert result["entities"] == []
        assert result["years"] == []
        assert result["last_focus"] is None

    def test_extract_none_history(self, orchestrator_with_metadata):
        """None history should return empty context."""
        result = orchestrator_with_metadata._extract_context_from_history(None)

        assert result["entities"] == []
        assert result["years"] == []
        assert result["last_focus"] is None


# =============================================================================
# DIAGNOSTIC TESTS: Full _optimize_prompt Flow
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
@patch("app.agent.record_ai_cost")
@patch("app.agent.record_ai_request")
class TestOptimizePromptFlow:
    """
    DIAGNOSTIC: Test the full _optimize_prompt method.

    These tests verify the complete flow that's failing in UAT.
    """

    async def test_explicit_query_proceeds(
        self, mock_request, mock_cost, orchestrator_with_metadata
    ):
        """Query with explicit entity should return proceed, not clarify."""
        result = await orchestrator_with_metadata._optimize_prompt(
            query="What is NCB revenue for 2023?",
            conversation_history=None,
        )

        assert isinstance(result, PromptOptimizationResult)
        assert (
            result.needs_clarification is False
        ), f"Explicit query should NOT need clarification. Got: {result.clarification_reason}"
        assert result.optimized_query == "What is NCB revenue for 2023?"

    async def test_entity_less_query_clarifies(
        self, mock_request, mock_cost, orchestrator_with_metadata
    ):
        """Query without entity and no history should need clarification."""
        result = await orchestrator_with_metadata._optimize_prompt(
            query="What is the revenue?",
            conversation_history=None,
        )

        assert isinstance(result, PromptOptimizationResult)
        assert result.needs_clarification is True
        assert result.clarification_reason == ClarificationReason.NO_ENTITY

    async def test_pronoun_with_context_proceeds(
        self, mock_request, mock_cost, orchestrator_with_metadata
    ):
        """Pronoun with entity in history should proceed with LLM resolution."""
        # Mock LLM response with proper usage_metadata
        mock_response = MagicMock()
        mock_response.text = "What is NCB's profit?"
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.cached_content_token_count = 0
        mock_response.usage_metadata.total_token_count = 150
        orchestrator_with_metadata.client.models.generate_content = MagicMock(
            return_value=mock_response
        )

        result = await orchestrator_with_metadata._optimize_prompt(
            query="What about their profit?",
            conversation_history=[
                {"role": "user", "content": "What is NCB revenue?"},
                {"role": "assistant", "content": "NCB revenue was $5M."},
            ],
        )

        assert isinstance(result, PromptOptimizationResult)
        assert (
            result.needs_clarification is False
        ), "Pronoun with context should NOT need clarification"

    async def test_pronoun_without_context_clarifies(
        self, mock_request, mock_cost, orchestrator_with_metadata
    ):
        """Pronoun without entity in history should clarify."""
        result = await orchestrator_with_metadata._optimize_prompt(
            query="What is their profit?",
            conversation_history=[],
        )

        assert isinstance(result, PromptOptimizationResult)
        assert result.needs_clarification is True
        # Note: NO_ENTITY is checked before UNRESOLVED_PRONOUN in the logic
        assert result.clarification_reason == ClarificationReason.NO_ENTITY

    async def test_comparison_with_symbols_proceeds(
        self, mock_request, mock_cost, orchestrator_with_metadata
    ):
        """Comparison query with two symbols should proceed."""
        result = await orchestrator_with_metadata._optimize_prompt(
            query="Compare GK and NCB revenue for 2023",
            conversation_history=None,
        )

        assert isinstance(result, PromptOptimizationResult)
        assert (
            result.needs_clarification is False
        ), "Comparison with two symbols should NOT need clarification"


# =============================================================================
# DIAGNOSTIC TESTS: Metadata Loading Issues
# =============================================================================


@pytest.mark.unit
class TestMetadataIssues:
    """
    DIAGNOSTIC: Test behavior when metadata is missing or malformed.

    These tests check if metadata issues could cause the UAT failures.
    """

    def test_entity_detection_with_none_metadata(self, orchestrator_no_metadata):
        """Entity detection should not crash with None metadata."""
        # This should not raise an exception
        query = "What is NCB revenue?"

        # Simulate the check in _detect_ambiguity
        has_entity_in_query = False
        if (
            orchestrator_no_metadata.financial_manager
            and orchestrator_no_metadata.financial_manager.metadata
        ):
            symbols = orchestrator_no_metadata.financial_manager.metadata.get("symbols", [])
            query_upper = query.upper()
            for symbol in symbols:
                if symbol in query_upper:
                    has_entity_in_query = True
                    break

        # With None metadata, no entity should be detected (but no crash)
        assert has_entity_in_query is False

    def test_entity_detection_with_empty_symbols(self, orchestrator_empty_symbols):
        """Entity detection with empty symbols should detect nothing."""
        query = "What is NCB revenue?"

        has_entity_in_query = False
        if (
            orchestrator_empty_symbols.financial_manager
            and orchestrator_empty_symbols.financial_manager.metadata
        ):
            symbols = orchestrator_empty_symbols.financial_manager.metadata.get("symbols", [])
            query_upper = query.upper()
            for symbol in symbols:
                if symbol in query_upper:
                    has_entity_in_query = True
                    break

        assert has_entity_in_query is False, "Empty symbols should not detect any entity"


# =============================================================================
# INTEGRATION TESTS: Full Agent Run (Mocked)
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
class TestAgentRunClarificationBehavior:
    """
    DIAGNOSTIC: Test full agent.run() clarification behavior.

    These tests simulate the actual UAT scenarios.
    """

    async def test_run_with_explicit_entity_no_clarify(self, orchestrator_with_metadata):
        """agent.run() with explicit entity should NOT return clarification."""
        # Mock the phases to focus on the clarification logic
        orchestrator_with_metadata._run_financial_phase = AsyncMock(
            return_value={
                "records": [],
                "filters": None,
                "chart": None,
                "sources": [],
                "context": "No data found",
            }
        )
        orchestrator_with_metadata._run_web_search_phase = AsyncMock(return_value=None)
        orchestrator_with_metadata._synthesize_response = AsyncMock(return_value="No data found.")

        result = await orchestrator_with_metadata.run(
            query="What is NCB EPS for 2024?",
            conversation_history=[],
            enable_web_search=True,
            enable_financial_data=True,
        )

        # The key assertion: should NOT trigger clarification
        assert (
            result.get("needs_clarification") is not True
        ), f"Explicit entity query should NOT need clarification. Response: {result}"

    async def test_run_with_no_entity_does_clarify(self, orchestrator_with_metadata):
        """agent.run() without entity should return clarification."""
        result = await orchestrator_with_metadata.run(
            query="What is the revenue?",
            conversation_history=[],
            enable_web_search=True,
            enable_financial_data=True,
        )

        assert (
            result.get("needs_clarification") is True
        ), "Query without entity should need clarification"
        assert "which company" in result.get("clarification_question", "").lower()


# =============================================================================
# NEW TESTS: Fixes for UAT Failures (2025-12-27)
# =============================================================================


@pytest.mark.unit
class TestGeneralMarketTermsBypass:
    """
    Test that general market terms bypass pronoun detection.

    Fix: "the stock market" was incorrectly matching "the stock" pronoun pattern.
    """

    def test_stock_market_not_pronoun(self, orchestrator_with_metadata):
        """'stock market' should NOT trigger unresolved pronoun clarification."""
        query = "What happened recently in the stock market?"
        extracted_context = {"entities": [], "years": [], "metrics": [], "last_focus": None}

        result = orchestrator_with_metadata._detect_ambiguity(
            query=query,
            extracted_context=extracted_context,
            conversation_history=None,
        )

        assert result is None, f"General market query should NOT need clarification. Got: {result}"

    def test_the_market_not_pronoun(self, orchestrator_with_metadata):
        """'the market' should NOT trigger pronoun clarification."""
        query = "How is the market doing today?"
        extracted_context = {"entities": [], "years": [], "metrics": [], "last_focus": None}

        result = orchestrator_with_metadata._detect_ambiguity(
            query=query,
            extracted_context=extracted_context,
            conversation_history=None,
        )

        assert result is None, "General market query should NOT need clarification"

    def test_general_market_terms_defined(self):
        """Verify GENERAL_MARKET_TERMS contains expected patterns."""
        assert "stock market" in GENERAL_MARKET_TERMS
        assert "the market" in GENERAL_MARKET_TERMS
        assert "the jse" in GENERAL_MARKET_TERMS


@pytest.mark.unit
class TestPronounWithEntityInQuery:
    """
    Test that pronouns with entity IN THE SAME QUERY proceed without clarification.

    Fix: "GK revenue... about them" should resolve "them" to "GK" from same query.
    """

    def test_pronoun_resolves_to_entity_in_query(self, orchestrator_with_metadata):
        """Pronoun with entity in same query should NOT trigger clarification."""
        query = "GK revenue for 2023 and latest news about them"
        extracted_context = {"entities": [], "years": [], "metrics": [], "last_focus": None}

        result = orchestrator_with_metadata._detect_ambiguity(
            query=query,
            extracted_context=extracted_context,
            conversation_history=None,
        )

        assert (
            result is None
        ), f"Pronoun with entity in query should NOT need clarification. Got: {result}"

    def test_their_with_entity_in_query(self, orchestrator_with_metadata):
        """'their' with company name in query should proceed."""
        query = "NCB and their recent announcements"
        extracted_context = {"entities": [], "years": [], "metrics": [], "last_focus": None}

        result = orchestrator_with_metadata._detect_ambiguity(
            query=query,
            extracted_context=extracted_context,
            conversation_history=None,
        )

        assert result is None, "Pronoun with entity in same query should proceed"


@pytest.mark.unit
class TestRemovedProblematicPronouns:
    """
    Test that problematic pronoun patterns have been removed.

    Fix: "the stock" and "it" were causing too many false positives.
    """

    def test_the_stock_removed(self):
        """'the stock' should NOT be in pronoun list (matches 'stock market')."""
        assert "the stock" not in PRONOUNS_NEEDING_RESOLUTION

    def test_it_removed(self):
        """'it' should NOT be in pronoun list (too many false positives)."""
        assert "it" not in PRONOUNS_NEEDING_RESOLUTION

    def test_valid_pronouns_remain(self):
        """Valid pronoun patterns should still be present."""
        assert "their" in PRONOUNS_NEEDING_RESOLUTION
        assert "they" in PRONOUNS_NEEDING_RESOLUTION
        assert "them" in PRONOUNS_NEEDING_RESOLUTION


def _create_mock_response_with_usage(text: str) -> MagicMock:
    """Create a mock Gemini response with usage_metadata for cost tracking."""
    mock_response = MagicMock()
    mock_response.text = text
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 100
    mock_response.usage_metadata.candidates_token_count = 50
    mock_response.usage_metadata.cached_content_token_count = 0
    mock_response.usage_metadata.total_token_count = 150
    return mock_response


@pytest.mark.unit
@pytest.mark.asyncio
@patch("app.agent.record_ai_cost")
@patch("app.agent.record_ai_request")
class TestVagueQueryExpansion:
    """
    Test LLM-based vague query expansion.

    Fix: Vague queries like "How did GK perform?" should be expanded to specific metrics.
    """

    async def test_vague_query_expansion_called(
        self, mock_request, mock_cost, orchestrator_with_metadata
    ):
        """Vague performance query should trigger LLM expansion."""
        # Mock the LLM response with proper usage_metadata
        mock_response = _create_mock_response_with_usage(
            "What were GK's revenue, net_profit, and eps for 2023?"
        )
        orchestrator_with_metadata.client.models.generate_content = MagicMock(
            return_value=mock_response
        )

        result = await orchestrator_with_metadata._expand_vague_query_with_llm(
            query="How well did GK perform in 2023?"
        )

        assert (
            "revenue" in result.lower() or "net_profit" in result.lower()
        ), f"Expanded query should include specific metrics. Got: {result}"

    async def test_specific_query_unchanged(
        self, mock_request, mock_cost, orchestrator_with_metadata
    ):
        """Already specific query should pass through unchanged."""
        mock_response = _create_mock_response_with_usage("What is NCB's revenue for 2023?")
        orchestrator_with_metadata.client.models.generate_content = MagicMock(
            return_value=mock_response
        )

        result = await orchestrator_with_metadata._expand_vague_query_with_llm(
            query="What is NCB's revenue for 2023?"
        )

        assert result == "What is NCB's revenue for 2023?"


# =============================================================================
# RUN DIAGNOSTIC TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
