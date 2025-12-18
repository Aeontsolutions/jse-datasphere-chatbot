"""
DSPy Modules for Financial Data Query Processing

This module provides declarative LLM pipelines using DSPy for:
1. Parsing natural language queries into structured filters
2. Formatting query results into natural language responses

DSPy enables:
- Declarative Signatures defining input/output schemas
- Composable Modules with ChainOfThought reasoning
- Future optimization via MIPROv2/BootstrapFewShot
- Easy model switching between providers
"""

import os
from typing import Optional

from app.logging_config import get_logger

logger = get_logger(__name__)

# Lazy import dspy to handle cases where it's not installed
_dspy = None
_dspy_configured = False


def _get_dspy():
    """Lazy load dspy module."""
    global _dspy
    if _dspy is None:
        try:
            import dspy

            _dspy = dspy
            logger.info("DSPy module loaded successfully")
        except ImportError:
            logger.warning("DSPy not installed. Install with: pip install dspy>=2.5.0")
            _dspy = None
    return _dspy


def configure_dspy_lm() -> bool:
    """
    Configure DSPy with the Gemini language model.

    Returns:
        bool: True if configuration successful, False otherwise
    """
    global _dspy_configured

    dspy = _get_dspy()
    if dspy is None:
        return False

    if _dspy_configured:
        return True

    try:
        api_key = (
            os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("CHATBOT_API_KEY")
        )

        if not api_key:
            logger.warning("No Google API key found for DSPy configuration")
            return False

        # Configure DSPy with Gemini
        # DSPy uses litellm format: "gemini/model-name" for Google AI Studio
        # See: https://docs.litellm.ai/docs/providers/gemini
        lm = dspy.LM(
            "gemini/gemini-2.0-flash-exp",
            api_key=api_key,
            temperature=0.7,
            max_tokens=4096,
        )
        dspy.configure(lm=lm)

        _dspy_configured = True
        logger.info("DSPy configured successfully with Gemini LM")
        return True

    except Exception as e:
        logger.error(f"Failed to configure DSPy: {e}")
        return False


def is_dspy_available() -> bool:
    """Check if DSPy is available and configured."""
    return _get_dspy() is not None and _dspy_configured


# =============================================================================
# DSPy Signatures
# =============================================================================


def _create_signatures():
    """Create DSPy Signature classes. Must be called after dspy is imported."""
    dspy = _get_dspy()
    if dspy is None:
        return None, None

    class ParseFinancialQuery(dspy.Signature):
        """Parse a natural language financial query into structured filters.

        Given a user's query about financial data, extract the relevant filtering
        parameters. Consider conversation history for follow-up questions.

        CRITICAL PARSING RULES:
        1. Trading symbols (MDS, SOS, JBG) go in symbols list
        2. Full company names go in companies list
        3. Empty list [] means "ALL" - return data for all items
        4. Match symbols case-insensitively
        5. For follow-up questions, use conversation context
        """

        query: str = dspy.InputField(desc="User's natural language query")
        conversation_context: str = dspy.InputField(desc="Recent conversation history for context")
        available_metadata: str = dspy.InputField(
            desc="Available companies, symbols, years, and metrics in the database"
        )

        companies: list[str] = dspy.OutputField(
            desc="Company names to filter (empty list = all companies)"
        )
        symbols: list[str] = dspy.OutputField(
            desc="Stock ticker symbols to filter, uppercase (empty list = all symbols)"
        )
        years: list[str] = dspy.OutputField(
            desc="Years to filter as strings like '2024' (empty list = all years)"
        )
        standard_items: list[str] = dspy.OutputField(
            desc="Financial metrics like revenue, net_profit, eps (empty list = all metrics)"
        )
        interpretation: str = dspy.OutputField(
            desc="Brief explanation of how the query was interpreted"
        )
        is_follow_up: bool = dspy.OutputField(
            desc="True if this references previous conversation context"
        )
        context_used: str = dspy.OutputField(
            desc="What context from conversation history was used, if any"
        )

    class FormatFinancialResponse(dspy.Signature):
        """Generate a natural language response from financial query results.

        Create a helpful, conversational response that:
        1. Confirms what data was found
        2. Highlights key insights or patterns
        3. Formats numbers appropriately (millions, billions)
        4. Suggests relevant follow-up queries about financial data
        5. Maintains conversational continuity for follow-ups

        IMPORTANT: Only suggest follow-up questions about querying financial data.
        Do NOT suggest creating charts, calculations, reports, or other actions.
        """

        query: str = dspy.InputField(desc="Original user query")
        interpretation: str = dspy.InputField(desc="How the query was interpreted")
        data_summary: str = dspy.InputField(desc="JSON summary of financial records found")
        record_count: int = dspy.InputField(desc="Total number of records found")
        is_follow_up: bool = dspy.InputField(desc="Whether this is a follow-up question")
        conversation_context: str = dspy.InputField(desc="Recent conversation for continuity")
        recommend_deep_research: bool = dspy.InputField(
            desc="Whether to suggest Deep Research for more detailed data"
        )

        response: str = dspy.OutputField(
            desc="Natural language response with key insights and formatted numbers"
        )
        follow_up_suggestions: list[str] = dspy.OutputField(
            desc="2-3 relevant follow-up questions about financial data queries only"
        )

    return ParseFinancialQuery, FormatFinancialResponse


# =============================================================================
# DSPy Modules
# =============================================================================


class FinancialQueryParser:
    """
    DSPy Module for parsing natural language queries into structured filters.

    Uses ChainOfThought to reason through the query interpretation process.
    """

    def __init__(self):
        self._module = None
        self._signature = None

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of DSPy module."""
        if self._module is not None:
            return True

        dspy = _get_dspy()
        if dspy is None:
            return False

        ParseFinancialQuery, _ = _create_signatures()
        if ParseFinancialQuery is None:
            return False

        self._signature = ParseFinancialQuery
        self._module = dspy.ChainOfThought(ParseFinancialQuery)
        return True

    def parse(
        self,
        query: str,
        conversation_context: str = "",
        available_metadata: str = "",
    ) -> Optional[dict]:
        """
        Parse a natural language query into structured filters.

        Args:
            query: User's natural language query
            conversation_context: Recent conversation history
            available_metadata: JSON string of available data dimensions

        Returns:
            dict with parsed filters, or None if DSPy unavailable
        """
        if not self._ensure_initialized():
            logger.warning("DSPy not available for query parsing")
            return None

        try:
            result = self._module(
                query=query,
                conversation_context=conversation_context,
                available_metadata=available_metadata,
            )

            # Extract results from DSPy Prediction object
            return {
                "companies": list(result.companies) if result.companies else [],
                "symbols": [s.upper() for s in result.symbols] if result.symbols else [],
                "years": [str(y) for y in result.years] if result.years else [],
                "standard_items": list(result.standard_items) if result.standard_items else [],
                "interpretation": str(result.interpretation),
                "is_follow_up": bool(result.is_follow_up),
                "context_used": str(result.context_used) if result.context_used else "",
                "data_availability_note": "",
            }

        except Exception as e:
            logger.error(f"DSPy query parsing failed: {e}")
            return None


class FinancialResponseFormatter:
    """
    DSPy Module for formatting query results into natural language.

    Uses ChainOfThought to generate insightful, conversational responses.
    """

    def __init__(self):
        self._module = None
        self._signature = None

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of DSPy module."""
        if self._module is not None:
            return True

        dspy = _get_dspy()
        if dspy is None:
            return False

        _, FormatFinancialResponse = _create_signatures()
        if FormatFinancialResponse is None:
            return False

        self._signature = FormatFinancialResponse
        self._module = dspy.ChainOfThought(FormatFinancialResponse)
        return True

    def format(
        self,
        query: str,
        interpretation: str,
        data_summary: str,
        record_count: int,
        is_follow_up: bool = False,
        conversation_context: str = "",
        recommend_deep_research: bool = False,
    ) -> Optional[dict]:
        """
        Format query results into a natural language response.

        Args:
            query: Original user query
            interpretation: How the query was interpreted
            data_summary: JSON string of financial records
            record_count: Number of records found
            is_follow_up: Whether this is a follow-up question
            conversation_context: Recent conversation history
            recommend_deep_research: Whether to suggest Deep Research

        Returns:
            dict with response and suggestions, or None if DSPy unavailable
        """
        if not self._ensure_initialized():
            logger.warning("DSPy not available for response formatting")
            return None

        try:
            result = self._module(
                query=query,
                interpretation=interpretation,
                data_summary=data_summary,
                record_count=record_count,
                is_follow_up=is_follow_up,
                conversation_context=conversation_context,
                recommend_deep_research=recommend_deep_research,
            )

            return {
                "response": str(result.response),
                "follow_up_suggestions": (
                    list(result.follow_up_suggestions) if result.follow_up_suggestions else []
                ),
            }

        except Exception as e:
            logger.error(f"DSPy response formatting failed: {e}")
            return None


# =============================================================================
# Module Instances (Singletons)
# =============================================================================

# Global instances for reuse
_query_parser: Optional[FinancialQueryParser] = None
_response_formatter: Optional[FinancialResponseFormatter] = None


def get_query_parser() -> FinancialQueryParser:
    """Get or create the singleton query parser instance."""
    global _query_parser
    if _query_parser is None:
        _query_parser = FinancialQueryParser()
    return _query_parser


def get_response_formatter() -> FinancialResponseFormatter:
    """Get or create the singleton response formatter instance."""
    global _response_formatter
    if _response_formatter is None:
        _response_formatter = FinancialResponseFormatter()
    return _response_formatter
