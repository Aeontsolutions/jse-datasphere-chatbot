"""
Agent module for orchestrating Gemini 3 with multiple tools.

This module provides the AgentOrchestrator class that combines:
- Google Search (native Gemini tool) for web grounding
- SQL Query Tool (custom) for financial database queries

The agent enforces source citations and provides follow-up suggestions.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from google.genai import types

from app.charting import generate_chart
from app.gemini_client import get_genai_client
from app.logging_config import get_logger
from app.models import (
    ChartSpec,
    ClarificationReason,
    FinancialDataFilters,
    FinancialDataRecord,
    PromptOptimizationResult,
)

logger = get_logger(__name__)


# ==============================================================================
# QUERY ROUTING CONSTANTS
# ==============================================================================

# Financial intent indicators - queries about company financials/metrics
FINANCIAL_KEYWORDS = {
    # Metrics
    "revenue",
    "profit",
    "eps",
    "earnings",
    "margin",
    "assets",
    "liabilities",
    "equity",
    "roe",
    "roa",
    "ratio",
    "dividend",
    "income",
    "cash flow",
    "net profit",
    "gross profit",
    "operating profit",
    "shareholders equity",
    "total assets",
    "total liabilities",
    "current ratio",
    "debt to equity",
    # Actions/contexts
    "financials",
    "financial",
    "performance",
    "compare",
    "comparison",
    "balance sheet",
    "income statement",
    "fiscal",
    "quarterly",
    "annual",
}

# Web search intent indicators - queries about news/current events
WEB_SEARCH_KEYWORDS = {
    "news",
    "latest",
    "recent",
    "recently",
    "announced",
    "announcement",
    "today",
    "yesterday",
    "this week",
    "this month",
    "update",
    "updates",
    "breaking",
    "report",
    "reports",
    "article",
    "press release",
    "what happened",
    "current",
    "now",
    "trending",
    "market news",
    "stock news",
    "industry trends",
    "outlook",
}

# Year pattern for detecting financial year references
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

# Vague performance patterns - queries that mention a company but don't specify
# what metrics they want. These should trigger BOTH tools since the financial
# tool can't extract specific metrics from vague queries.
VAGUE_PERFORMANCE_PATTERNS = {
    "how well did",
    "how did",
    "how has",
    "how is",
    "how are",
    "how was",
    "perform",  # catches "perform", "performed", "performing"
    "doing",
    "fared",
    "going",
    "tell me about",
    "what about",
    "overview",
    "summary",
}


# ==============================================================================
# PROMPT OPTIMIZATION CONSTANTS
# ==============================================================================

# Clarification templates for different ambiguity reasons
CLARIFICATION_TEMPLATES = {
    "no_entity": (
        "Which company or stock would you like information about? "
        "For example, you could ask about NCB, GraceKennedy, or any JSE-listed company."
    ),
    "unresolved_pronoun": (
        "Could you clarify which company you're referring to? "
        "I don't have enough context from our conversation to determine what you mean."
    ),
    "ambiguous_comparison": (
        "I'd be happy to compare companies for you. "
        "Could you specify which ones you'd like me to compare?"
    ),
}

# Default metrics when user asks for vague "performance"
DEFAULT_PERFORMANCE_METRICS = ["revenue", "net_profit", "eps"]

# Pronouns and references that need resolution from context
PRONOUNS_NEEDING_RESOLUTION = {
    "their",
    "they",
    "them",
    "it",
    "its",
    "the company",
    "the stock",
    "this company",
    "that company",
}

# Relative time patterns for resolution
RELATIVE_TIME_PATTERNS = {
    "last_n_years": re.compile(r"(?:last|past)\s+(\d+)\s+years?", re.IGNORECASE),
    "last_year": re.compile(r"\blast\s+year\b", re.IGNORECASE),
    "recently": re.compile(r"\brecent(?:ly)?\b", re.IGNORECASE),
    "this_year": re.compile(r"\bthis\s+year\b", re.IGNORECASE),
}

# Marker to detect previous clarification in history (for 1-round max)
CLARIFICATION_MARKER = "I want to make sure I understand"


# ==============================================================================
# TOOL DECLARATIONS
# ==============================================================================


def get_financial_data_tool_declaration() -> types.FunctionDeclaration:
    """
    Define the SQL financial query tool for Gemini function calling.

    This tool allows Gemini to query financial data from the JSE database
    based on stock symbols, years, and financial metrics.
    """
    return types.FunctionDeclaration(
        name="query_financial_data",
        description="""Query financial data from the Jamaica Stock Exchange (JSE) database.
Use this tool when the user asks about:
- Company financial metrics (revenue, profit, EPS, margins, assets, liabilities)
- Financial comparisons between companies
- Historical financial data for specific years
- Stock symbols and their associated company data

Examples of when to use this tool:
- "What is NCB's revenue?" -> query NCB revenue
- "Compare JBG and GK profit margins" -> query JBG, GK for profit margins
- "Show me MDS financials for 2023" -> query MDS for 2023

Available metrics include: revenue, net_profit, gross_profit, operating_profit,
eps, total_assets, total_liabilities, shareholders_equity, gross_profit_margin,
net_profit_margin, operating_profit_margin, roe, roa, current_ratio, debt_to_equity.
""",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "symbols": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                    description="Stock trading symbols (e.g., ['NCB', 'JBG', 'MDS']). Use uppercase.",
                ),
                "years": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                    description="Years to filter by (e.g., ['2022', '2023', '2024']).",
                ),
                "standard_items": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                    description="Financial metrics to retrieve (e.g., ['revenue', 'net_profit', 'eps']).",
                ),
            },
        ),
    )


# ==============================================================================
# TOOL EXECUTION
# ==============================================================================


async def execute_financial_query(
    financial_manager: Any,
    args: Dict[str, Any],
) -> Tuple[List[FinancialDataRecord], FinancialDataFilters, Optional[Dict], List[Dict]]:
    """
    Execute BigQuery financial data query and return results with source metadata.

    Reuses: financial_manager.query_data(filters)

    Args:
        financial_manager: FinancialDataManager instance
        args: Tool arguments from Gemini (symbols, years, standard_items)

    Returns:
        Tuple of (records, filters, chart_spec, sources)
    """
    start_time = time.time()

    try:
        # Build filters from tool arguments (handle None values from LLM)
        raw_symbols = args.get("symbols") or []
        raw_years = args.get("years") or []
        raw_items = args.get("standard_items") or []

        symbols = [s.upper() for s in raw_symbols if s]
        years = [str(y) for y in raw_years if y]
        standard_items = [item.lower().replace(" ", "_") for item in raw_items if item]

        filters = FinancialDataFilters(
            companies=[],
            symbols=symbols,
            years=years,
            standard_items=standard_items,
            interpretation=f"Agent tool call: symbols={symbols}, years={years}, items={standard_items}",
            data_availability_note="",
            is_follow_up=False,
            context_used="",
        )

        # Post-process filters using associations from metadata
        if financial_manager.metadata and "associations" in financial_manager.metadata:
            filters_dict = filters.model_dump()
            filters_dict = financial_manager._post_process_filters(filters_dict)
            filters = FinancialDataFilters(**filters_dict)

        # Query the data - REUSE existing method
        records = financial_manager.query_data(filters)

        # Generate chart if applicable - REUSE existing function
        chart_spec = None
        if records:
            chart_data = generate_chart(records, "")
            if chart_data:
                chart_spec = chart_data

        # Build source citations with explicit metadata
        symbols_str = ", ".join(filters.symbols) if filters.symbols else "all"
        years_str = ", ".join(filters.years) if filters.years else "all years"
        source_entry = {
            "type": "database",
            "description": f"JSE Financial Database: {symbols_str} ({years_str})",
            "table": "financial_data",
        }
        # Add explicit lists for filtering/display (backwards compatible - new fields)
        if filters.symbols:
            source_entry["symbols"] = filters.symbols
        if filters.years:
            source_entry["years"] = filters.years
        if filters.standard_items:
            source_entry["metrics"] = filters.standard_items
        sources = [source_entry]

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Financial query executed in {duration_ms:.2f}ms, returned {len(records)} records"
        )

        return records, filters, chart_spec, sources

    except Exception as e:
        logger.error(f"Financial query execution failed: {e}", exc_info=True)
        raise


# ==============================================================================
# AGENT ORCHESTRATOR
# ==============================================================================


class AgentOrchestrator:
    """
    Orchestrates multi-tool agent interactions with Gemini.

    Combines:
    - Google Search (native Gemini tool) for web grounding
    - SQL Query Tool (custom function) for financial database queries

    Enforces source citations and provides follow-up suggestions.
    """

    def __init__(
        self,
        financial_manager: Any,
        associations: Optional[Dict] = None,
    ):
        """
        Initialize the agent orchestrator.

        Args:
            financial_manager: FinancialDataManager instance for SQL queries
            associations: Symbol-to-company associations from metadata
        """
        self.financial_manager = financial_manager
        self.associations = associations
        self.client = get_genai_client()
        self.model_name = "gemini-3-pro-preview"

    def _build_tools(
        self,
        enable_web_search: bool = True,
        enable_financial_data: bool = True,
    ) -> List[types.Tool]:
        """
        Build the list of tools based on enabled flags.

        Args:
            enable_web_search: Whether to enable Google Search
            enable_financial_data: Whether to enable SQL queries

        Returns:
            List of Tool objects for Gemini
        """
        tools = []

        # Native Google Search tool
        if enable_web_search:
            tools.append(types.Tool(google_search=types.GoogleSearch()))

        # Custom SQL Query tool
        if enable_financial_data and self.financial_manager:
            tools.append(types.Tool(function_declarations=[get_financial_data_tool_declaration()]))

        return tools

    async def _route_query(
        self,
        query: str,
        enable_web_search: bool,
        enable_financial_data: bool,
    ) -> Dict[str, Any]:
        """
        Determine which tools are needed for this query using hybrid approach.

        Strategy:
        1. Rule-based classification for obvious cases (~0ms overhead)
        2. Gemini 2.5 Flash fallback for ambiguous queries (~200-400ms)

        Args:
            query: User's query (may be optimized with context)
            enable_web_search: Whether web search is enabled by user
            enable_financial_data: Whether financial data is enabled by user

        Returns:
            Dict with keys:
                - use_financial: bool
                - use_web_search: bool
                - routing_method: str ("rule_based", "llm_fallback", "default")
                - confidence: str ("high", "medium", "low")
        """
        start_time = time.time()
        query_lower = query.lower()

        # Default: respect user's enabled flags
        if not enable_web_search and not enable_financial_data:
            return {
                "use_financial": False,
                "use_web_search": False,
                "routing_method": "disabled",
                "confidence": "high",
            }

        # If only one tool is enabled, use it
        if not enable_web_search:
            return {
                "use_financial": True,
                "use_web_search": False,
                "routing_method": "single_tool",
                "confidence": "high",
            }
        if not enable_financial_data:
            return {
                "use_financial": False,
                "use_web_search": True,
                "routing_method": "single_tool",
                "confidence": "high",
            }

        # === RULE-BASED CLASSIFICATION ===
        has_financial_signal = False
        has_web_signal = False
        has_symbol = False
        has_year = False

        # Check for stock symbols in query (from metadata)
        if self.financial_manager and self.financial_manager.metadata:
            symbols = self.financial_manager.metadata.get("symbols", [])
            query_upper = query.upper()
            for symbol in symbols:
                if symbol in query_upper:
                    has_symbol = True
                    has_financial_signal = True
                    break

        # Check for year patterns (strong financial signal)
        if YEAR_PATTERN.search(query):
            has_year = True
            has_financial_signal = True

        # Check for financial keywords
        for keyword in FINANCIAL_KEYWORDS:
            if keyword in query_lower:
                has_financial_signal = True
                break

        # Check for web search keywords
        for keyword in WEB_SEARCH_KEYWORDS:
            if keyword in query_lower:
                has_web_signal = True
                break

        # Check for vague performance patterns - these indicate the user wants
        # general info without specifying metrics, so we need both tools
        has_vague_pattern = False
        for pattern in VAGUE_PERFORMANCE_PATTERNS:
            if pattern in query_lower:
                has_vague_pattern = True
                break

        # === ROUTING DECISION ===

        # Vague query with symbol/year: user asks about company performance without
        # specifying metrics -> use BOTH tools to provide comprehensive response
        if has_vague_pattern and (has_symbol or has_year):
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Query routing: both_tools (vague_performance) in {duration_ms:.2f}ms | "
                f"symbol={has_symbol}, year={has_year}, vague_pattern=True"
            )
            return {
                "use_financial": True,
                "use_web_search": True,
                "routing_method": "rule_based",
                "confidence": "medium",
                "reason": "vague_performance_query",
            }

        # Clear financial-only signal: has symbol/year + financial keywords, no web keywords
        if has_financial_signal and not has_web_signal:
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Query routing: financial_only (rule_based) in {duration_ms:.2f}ms | "
                f"symbol={has_symbol}, year={has_year}"
            )
            return {
                "use_financial": True,
                "use_web_search": False,
                "routing_method": "rule_based",
                "confidence": "high",
            }

        # Clear web-only signal: has web keywords, no financial signals
        if has_web_signal and not has_financial_signal:
            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"Query routing: web_only (rule_based) in {duration_ms:.2f}ms")
            return {
                "use_financial": False,
                "use_web_search": True,
                "routing_method": "rule_based",
                "confidence": "high",
            }

        # Both signals present - likely needs both tools
        if has_financial_signal and has_web_signal:
            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"Query routing: both_tools (rule_based) in {duration_ms:.2f}ms")
            return {
                "use_financial": True,
                "use_web_search": True,
                "routing_method": "rule_based",
                "confidence": "medium",
            }

        # === AMBIGUOUS: No clear signals - use LLM fallback ===
        logger.info("Query routing: ambiguous, using LLM fallback")
        try:
            llm_result = await self._classify_with_llm(query)
            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"Query routing: {llm_result} (llm_fallback) in {duration_ms:.2f}ms")
            return {
                "use_financial": llm_result.get("financial", True),
                "use_web_search": llm_result.get("web_search", True),
                "routing_method": "llm_fallback",
                "confidence": "medium",
            }
        except Exception as e:
            # On LLM failure, default to both tools
            logger.warning(f"LLM routing fallback failed: {e}, defaulting to both tools")
            return {
                "use_financial": True,
                "use_web_search": True,
                "routing_method": "default",
                "confidence": "low",
            }

    async def _classify_with_llm(self, query: str) -> Dict[str, bool]:
        """
        Use Gemini 2.5 Flash for fast query classification.

        Only called when rule-based routing is ambiguous.
        Uses low temperature and minimal tokens for speed.

        Args:
            query: The user's query to classify

        Returns:
            Dict with {"financial": bool, "web_search": bool}
        """
        system_instruction = """You are a query classifier for a Jamaica Stock Exchange (JSE) assistant.
Classify whether the query needs:
1. Financial database data (company financials, metrics, historical data)
2. Web search (news, current events, market trends, general info)

Respond with ONLY a JSON object, no other text:
{"financial": true/false, "web_search": true/false}

Examples:
- "What is NCB's revenue?" -> {"financial": true, "web_search": false}
- "Latest news about Jamaica Stock Exchange" -> {"financial": false, "web_search": true}
- "How does NCB compare to industry trends?" -> {"financial": true, "web_search": true}
- "Tell me about GraceKennedy" -> {"financial": true, "web_search": true}
"""

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=query)],
            )
        ]

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,  # Very low for deterministic classification
            max_output_tokens=64,  # Just need JSON response
        )

        # Use fast model for classification
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config,
        )

        # Parse JSON response
        response_text = response.text.strip() if response.text else ""

        # Handle potential markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        try:
            result = json.loads(response_text)
            return {
                "financial": result.get("financial", True),
                "web_search": result.get("web_search", True),
            }
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM classification response: {response_text}")
            # Default to both on parse failure
            return {"financial": True, "web_search": True}

    async def _optimize_prompt(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> PromptOptimizationResult:
        """
        Optimize the user prompt with full context resolution and ambiguity detection.

        This method:
        1. Extracts context from conversation history
        2. Detects critical ambiguities that require clarification
        3. Applies sensible defaults for minor gaps
        4. Resolves pronouns using LLM if needed

        Args:
            query: User's current query
            conversation_history: Previous conversation messages

        Returns:
            PromptOptimizationResult with optimized query and metadata
        """
        # Step 1: Extract context from history
        extracted_context = self._extract_context_from_history(conversation_history)
        logger.info(f"Extracted context: {extracted_context}")

        # Step 2: Check for critical ambiguities
        ambiguity = self._detect_ambiguity(query, extracted_context, conversation_history)
        if ambiguity:
            reason, question = ambiguity
            logger.info(f"Ambiguity detected: {reason.value}")
            return PromptOptimizationResult(
                optimized_query=query,
                needs_clarification=True,
                clarification_question=question,
                clarification_reason=reason,
                resolved_context=extracted_context,
                defaults_applied=[],
                confidence="low",
            )

        # Step 3: Apply defaults for minor gaps
        enriched_context, defaults_applied = self._apply_defaults(query, extracted_context)
        if defaults_applied:
            logger.info(f"Defaults applied: {defaults_applied}")

        # Step 4: Resolve pronouns with LLM if needed
        optimized_query = query
        needs_resolution = any(word in query.lower() for word in PRONOUNS_NEEDING_RESOLUTION)

        if needs_resolution and conversation_history:
            optimized_query = await self._resolve_pronouns_with_llm(query, conversation_history)

        return PromptOptimizationResult(
            optimized_query=optimized_query,
            needs_clarification=False,
            clarification_question=None,
            clarification_reason=None,
            resolved_context=enriched_context,
            defaults_applied=defaults_applied,
            confidence="high" if not defaults_applied else "medium",
        )

    async def _resolve_pronouns_with_llm(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
    ) -> str:
        """
        Use Gemini 2.5 Flash to resolve pronouns in the query.

        Args:
            query: The query with pronouns
            conversation_history: Previous conversation for context

        Returns:
            Query with resolved pronouns
        """
        # Build context from recent history (last 3 exchanges = 6 messages)
        recent_history = conversation_history[-6:]
        context_parts = []
        for msg in recent_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:500]  # Truncate long messages
            context_parts.append(f"{role}: {content}")

        context = "\n".join(context_parts)

        # Use Gemini 2.5 Flash for fast pronoun resolution
        system_instruction = """You are a query rewriter. Your ONLY job is to resolve pronouns and references.

Given conversation context and a query, rewrite the query by replacing all pronouns and references
(like "their", "they", "it", "the company", "this", "that") with the specific entities they refer to.

IMPORTANT:
- Output ONLY the rewritten query, nothing else
- Keep the same intent and meaning
- If you can't determine what a pronoun refers to, keep the original query
- Do not add explanations or commentary

Examples:
Context: "user: What is NCB's revenue?  assistant: NCB's revenue was $5M."
Query: "What about their profit?"
Output: What is NCB's profit?

Context: "user: Tell me about GraceKennedy  assistant: GK is a conglomerate..."
Query: "How did they perform in 2023?"
Output: How did GraceKennedy perform in 2023?"""

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"Context:\n{context}\n\nQuery: {query}")],
            )
        ]

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,  # Low for deterministic output
            max_output_tokens=256,  # Short output - just the rewritten query
        )

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=config,
            )

            optimized = response.text.strip() if response.text else query
            # Remove quotes if the model wrapped the output
            if optimized.startswith('"') and optimized.endswith('"'):
                optimized = optimized[1:-1]
            if optimized.startswith("'") and optimized.endswith("'"):
                optimized = optimized[1:-1]

            logger.info(f"Pronoun resolution: '{query}' -> '{optimized}'")
            return optimized

        except Exception as e:
            logger.warning(f"Pronoun resolution failed: {e}, using original query")
            return query

    def _get_available_metadata_context(self) -> str:
        """
        Get available metadata for context in the prompt.

        Returns:
            String describing available companies, symbols, years, and metrics
        """
        if not self.financial_manager or not self.financial_manager.metadata:
            return ""

        metadata = self.financial_manager.metadata
        context_parts = []

        if "symbols" in metadata:
            symbols = metadata["symbols"][:20]  # Limit to first 20
            context_parts.append(f"Available stock symbols: {', '.join(symbols)}")

        if "years" in metadata:
            years = sorted(metadata["years"])[-5:]  # Last 5 years
            context_parts.append(f"Available years: {', '.join(years)}")

        if "standard_items" in metadata:
            items = metadata["standard_items"][:15]  # First 15 metrics
            context_parts.append(f"Available metrics: {', '.join(items)}")

        return "\n".join(context_parts)

    def _extract_context_from_history(
        self,
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """
        Extract entities, time periods, and metrics from conversation history.

        Args:
            conversation_history: Previous conversation messages

        Returns:
            Dict with keys: entities, years, metrics, last_focus
        """
        if not conversation_history:
            return {"entities": [], "years": [], "metrics": [], "last_focus": None}

        entities = []
        years = []
        metrics = []
        last_focus = None

        # Get metadata for validation
        valid_symbols = set()
        valid_years = set()
        valid_metrics = set()
        if self.financial_manager and self.financial_manager.metadata:
            valid_symbols = set(self.financial_manager.metadata.get("symbols", []))
            valid_years = set(self.financial_manager.metadata.get("years", []))
            valid_metrics = set(self.financial_manager.metadata.get("standard_items", []))

        # Scan last 6 messages (3 exchanges)
        recent_history = conversation_history[-6:]
        for msg in recent_history:
            content = msg.get("content", "")
            content_upper = content.upper()

            # Extract symbols
            for symbol in valid_symbols:
                if symbol in content_upper:
                    if symbol not in entities:
                        entities.append(symbol)
                    last_focus = symbol

            # Extract years
            year_matches = YEAR_PATTERN.findall(content)
            for year in year_matches:
                if year in valid_years and year not in years:
                    years.append(year)

            # Extract metrics (case-insensitive)
            content_lower = content.lower()
            for metric in valid_metrics:
                if metric.lower() in content_lower and metric not in metrics:
                    metrics.append(metric)

        return {
            "entities": entities,
            "years": years,
            "metrics": metrics,
            "last_focus": last_focus,
        }

    def _has_previous_clarification(
        self,
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> bool:
        """Check if the previous assistant message was a clarification request."""
        if not conversation_history or len(conversation_history) < 2:
            return False

        # Check last assistant message
        for msg in reversed(conversation_history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                return CLARIFICATION_MARKER in content
        return False

    def _detect_ambiguity(
        self,
        query: str,
        extracted_context: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> Optional[Tuple[ClarificationReason, str]]:
        """
        Detect if the query has critical ambiguities requiring clarification.

        Args:
            query: The user's query
            extracted_context: Context extracted from history
            conversation_history: Previous conversation

        Returns:
            None if no clarification needed, or (reason, question) tuple
        """
        # Skip clarification if we already asked (1-round max)
        if self._has_previous_clarification(conversation_history):
            logger.info("Skipping clarification (1-round max reached)")
            return None

        query_lower = query.lower()
        query_upper = query.upper()

        # Check if query has entity
        has_entity_in_query = False
        if self.financial_manager and self.financial_manager.metadata:
            symbols = self.financial_manager.metadata.get("symbols", [])
            for symbol in symbols:
                if symbol in query_upper:
                    has_entity_in_query = True
                    break

        has_entity_in_context = bool(extracted_context.get("entities"))

        # Check 1: No entity anywhere - clarify
        if not has_entity_in_query and not has_entity_in_context:
            # Exception: General market questions don't need entity
            general_patterns = ["market", "jse", "stock exchange", "index", "sector"]
            is_general = any(p in query_lower for p in general_patterns)
            if not is_general:
                return (
                    ClarificationReason.NO_ENTITY,
                    CLARIFICATION_TEMPLATES["no_entity"],
                )

        # Check 2: Unresolved pronouns
        has_pronoun = any(p in query_lower for p in PRONOUNS_NEEDING_RESOLUTION)
        if has_pronoun and not has_entity_in_context:
            return (
                ClarificationReason.UNRESOLVED_PRONOUN,
                CLARIFICATION_TEMPLATES["unresolved_pronoun"],
            )

        # Check 3: Ambiguous comparison
        if "compare" in query_lower:
            # Count entities in query
            entity_count = 0
            if self.financial_manager and self.financial_manager.metadata:
                symbols = self.financial_manager.metadata.get("symbols", [])
                for symbol in symbols:
                    if symbol in query_upper:
                        entity_count += 1
            # Also count entities from context
            entity_count += len(extracted_context.get("entities", []))
            if entity_count < 2:
                return (
                    ClarificationReason.AMBIGUOUS_COMPARISON,
                    CLARIFICATION_TEMPLATES["ambiguous_comparison"],
                )

        return None

    def _resolve_relative_time(self, query: str) -> Tuple[str, List[str]]:
        """
        Resolve relative time expressions to specific years.

        Args:
            query: The user's query

        Returns:
            Tuple of (resolved_years_list, defaults_applied_list)
        """
        resolved_years = []
        defaults_applied = []

        # Get most recent year from metadata
        if not self.financial_manager or not self.financial_manager.metadata:
            return resolved_years, defaults_applied

        available_years = sorted(self.financial_manager.metadata.get("years", []))
        if not available_years:
            return resolved_years, defaults_applied

        most_recent = int(available_years[-1])

        # Check for "last N years"
        match = RELATIVE_TIME_PATTERNS["last_n_years"].search(query)
        if match:
            n = int(match.group(1))
            resolved_years = [str(most_recent - i) for i in range(n)]
            resolved_years = [y for y in resolved_years if y in available_years]
            defaults_applied.append(f"year (last {n} years → {', '.join(resolved_years)})")
            return resolved_years, defaults_applied

        # Check for "last year"
        if RELATIVE_TIME_PATTERNS["last_year"].search(query):
            last_year = str(most_recent - 1)
            if last_year in available_years:
                resolved_years = [last_year]
                defaults_applied.append(f"year (last year → {last_year})")
            return resolved_years, defaults_applied

        # Check for "recently"
        if RELATIVE_TIME_PATTERNS["recently"].search(query):
            recent_years = [str(most_recent), str(most_recent - 1)]
            resolved_years = [y for y in recent_years if y in available_years]
            defaults_applied.append(f"year (recently → {', '.join(resolved_years)})")
            return resolved_years, defaults_applied

        # Check for "this year"
        if RELATIVE_TIME_PATTERNS["this_year"].search(query):
            this_year = str(most_recent)
            if this_year in available_years:
                resolved_years = [this_year]
                defaults_applied.append(f"year (this year → {this_year})")
            return resolved_years, defaults_applied

        return resolved_years, defaults_applied

    def _apply_defaults(
        self,
        query: str,
        extracted_context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Apply sensible defaults for minor gaps.

        Args:
            query: The user's query
            extracted_context: Context extracted from history

        Returns:
            Tuple of (enriched_context, defaults_applied_list)
        """
        defaults_applied = []
        enriched = extracted_context.copy()

        # Check if query has explicit year
        has_explicit_year = bool(YEAR_PATTERN.search(query))

        # Check for relative time expressions
        if not has_explicit_year:
            relative_years, relative_defaults = self._resolve_relative_time(query)
            if relative_years:
                enriched["years"] = relative_years
                defaults_applied.extend(relative_defaults)
            elif not enriched.get("years"):
                # No year specified - default to most recent
                if self.financial_manager and self.financial_manager.metadata:
                    available_years = sorted(self.financial_manager.metadata.get("years", []))
                    if available_years:
                        most_recent = available_years[-1]
                        enriched["years"] = [most_recent]
                        defaults_applied.append(f"year (defaulted to {most_recent})")

        # Check for vague metrics
        query_lower = query.lower()
        vague_metric_patterns = ["performance", "how did", "how has", "doing", "financials"]
        has_vague_metric = any(p in query_lower for p in vague_metric_patterns)
        has_explicit_metric = False
        if self.financial_manager and self.financial_manager.metadata:
            valid_metrics = self.financial_manager.metadata.get("standard_items", [])
            has_explicit_metric = any(m.lower() in query_lower for m in valid_metrics)

        if has_vague_metric and not has_explicit_metric and not enriched.get("metrics"):
            enriched["metrics"] = DEFAULT_PERFORMANCE_METRICS
            defaults_applied.append(
                f"metrics (defaulted to {', '.join(DEFAULT_PERFORMANCE_METRICS)})"
            )

        return enriched, defaults_applied

    def _build_clarification_response(
        self,
        query: str,
        optimization_result: PromptOptimizationResult,
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """
        Build a response requesting user clarification.

        Args:
            query: Original user query
            optimization_result: The optimization result with clarification info
            conversation_history: Previous conversation

        Returns:
            Response dictionary for clarification
        """
        response_text = (
            f"{CLARIFICATION_MARKER} your question correctly before proceeding.\n\n"
            f"{optimization_result.clarification_question}"
        )

        # Update conversation history with clarification
        updated_history = self._update_conversation_history(
            query, response_text, conversation_history
        )

        return {
            "response": response_text,
            "data_found": False,
            "record_count": 0,
            "needs_clarification": True,
            "clarification_question": optimization_result.clarification_question,
            "tools_executed": None,
            "sources": None,
            "filters_used": None,
            "data_preview": None,
            "chart": None,
            "web_search_results": None,
            "suggestions": None,
            "conversation_history": updated_history,
            "warnings": None,
        }

    def _build_system_instruction(self) -> str:
        """
        Build the system instruction for the agent.

        Returns:
            System instruction string
        """
        metadata_context = self._get_available_metadata_context()

        return f"""You are a financial research assistant for the Jamaica Stock Exchange (JSE).

Your capabilities:
1. Query financial data from the JSE database using the query_financial_data tool
2. Search the web for recent news and information using Google Search

IMPORTANT RULES:
1. ALWAYS use the appropriate tools to gather information before responding
2. ONLY respond based on information from the tools - do NOT hallucinate or make up data
3. EVERY factual claim MUST be cited with source references:
   - Financial data: cite as [Database: Company, Year]
   - Web results: cite as [Web: Source Title]
4. If you cannot find the requested information, clearly state what was searched and what wasn't found
5. At the end of your response, suggest 2-3 relevant follow-up questions

{metadata_context}

When the user asks about financial data, use the query_financial_data tool.
When the user asks about recent news or current events, use Google Search.
You can use both tools together for comprehensive answers."""

    async def run(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        enable_web_search: bool = True,
        enable_financial_data: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the agent orchestration pipeline.

        Pipeline (two-phase to work around Gemini 3 limitations):
        1. Phase 1: Query financial database if enabled (function calling)
        2. Phase 2: Web search if enabled (Google Search grounding)
        3. Synthesize response from combined context

        Note: gemini-3-pro-preview doesn't support combining Google Search
        with custom function calling in the same request, so we do them separately.

        Args:
            query: User's query
            conversation_history: Previous conversation messages
            enable_web_search: Whether to enable Google Search
            enable_financial_data: Whether to enable SQL queries

        Returns:
            Dictionary with response, tools_executed, sources, etc.
        """
        start_time = time.time()
        logger.info(f"Agent run starting for query: {query[:100]}...")

        if not enable_web_search and not enable_financial_data:
            return self._build_no_tools_response(query, conversation_history)

        # Step 1: Optimize prompt (resolve pronouns/references, detect ambiguity)
        optimization_result = await self._optimize_prompt(query, conversation_history)

        # Step 1.5: Short-circuit if clarification is needed
        if optimization_result.needs_clarification:
            logger.info(f"Clarification needed: {optimization_result.clarification_reason}")
            return self._build_clarification_response(
                query, optimization_result, conversation_history
            )

        optimized_query = optimization_result.optimized_query
        logger.info(f"Optimized query: '{optimized_query}'")
        if optimization_result.defaults_applied:
            logger.info(f"Defaults applied: {optimization_result.defaults_applied}")

        # Step 2: Route query to determine which tools to use
        routing = await self._route_query(
            query,  # Use original query for routing (not optimized)
            enable_web_search,
            enable_financial_data,
        )
        logger.info(
            f"Query routing result: use_financial={routing['use_financial']}, "
            f"use_web_search={routing['use_web_search']}, "
            f"method={routing['routing_method']}, confidence={routing['confidence']}"
        )

        # Initialize results
        tools_executed = []
        sources = []
        financial_records = []
        filters_used = None
        chart_spec = None
        web_search_results = None
        financial_context = ""
        web_context = ""
        # Note: routing info available in 'routing' dict for debugging

        try:
            # ================================================================
            # PHASE 1: Financial Data Query (if routed)
            # ================================================================
            if routing["use_financial"] and self.financial_manager:
                logger.info("Phase 1: Querying financial database...")
                financial_result = await self._run_financial_phase(optimized_query)

                if financial_result:
                    tools_executed.append("query_financial_data")
                    financial_records = financial_result.get("records", [])
                    filters_used = financial_result.get("filters")
                    chart_spec = financial_result.get("chart")
                    sources.extend(financial_result.get("sources", []))
                    financial_context = financial_result.get("context", "")
            else:
                logger.info("Phase 1: Skipped (routing decision)")

            # ================================================================
            # PHASE 2: Web Search (if routed)
            # ================================================================
            if routing["use_web_search"]:
                logger.info("Phase 2: Performing web search...")
                web_result = await self._run_web_search_phase(optimized_query)

                if web_result:
                    tools_executed.append("google_search")
                    web_search_results = web_result.get("search_results")
                    sources.extend(web_result.get("sources", []))
                    web_context = web_result.get("context", "")
            else:
                logger.info("Phase 2: Skipped (routing decision)")

            # ================================================================
            # PHASE 3: Synthesize Response
            # ================================================================
            logger.info("Phase 3: Synthesizing final response...")
            response_text = await self._synthesize_response(
                query=query,
                optimized_query=optimized_query,
                financial_context=financial_context,
                web_context=web_context,
                financial_records=financial_records,
                conversation_history=conversation_history,
            )

            # Extract follow-up questions
            follow_up_questions = self._extract_follow_up_questions(response_text)
            clean_response = self._clean_response_text(response_text)

            total_time = (time.time() - start_time) * 1000
            logger.info(f"Agent run completed in {total_time:.2f}ms")

            # Build data preview (first 10 records)
            data_preview = financial_records[:10] if financial_records else None
            record_count = len(financial_records)

            return {
                "response": clean_response,
                "data_found": record_count > 0 or bool(web_context),
                "record_count": record_count,
                "data_preview": data_preview,
                "tools_executed": tools_executed,
                "sources": sources,
                "filters_used": filters_used,
                "chart": chart_spec,
                "web_search_results": web_search_results,
                "suggestions": follow_up_questions,
                "conversation_history": conversation_history,
            }

        except Exception as e:
            logger.error(f"Agent run failed: {e}", exc_info=True)
            return self._build_error_response(query, str(e), conversation_history)

    async def _run_financial_phase(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Phase 1: Run financial database query using function calling.

        Args:
            query: Optimized query

        Returns:
            Dictionary with records, filters, chart, sources, and context
        """
        try:
            # Build tool for financial query only
            tool = types.Tool(function_declarations=[get_financial_data_tool_declaration()])

            system_instruction = f"""You are a financial data query assistant.
Your ONLY job is to analyze the query and call the query_financial_data function with appropriate parameters.

{self._get_available_metadata_context()}

Analyze the query and extract:
- Stock symbols mentioned (use uppercase like NCB, JBG, MDS)
- Years mentioned (like 2022, 2023, 2024)
- Financial metrics requested (like revenue, net_profit, eps)

Call the query_financial_data function with these parameters."""

            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=query)],
                )
            ]

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[tool],
                temperature=0.3,
                max_output_tokens=1024,
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )

            # Check for function call and execute
            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        if fc.name == "query_financial_data":
                            args = dict(fc.args) if fc.args else {}
                            logger.info(f"Financial query args: {args}")

                            records, filters, chart, db_sources = await execute_financial_query(
                                self.financial_manager, args
                            )

                            # Build context for synthesis
                            context = self._build_financial_context(records, filters)

                            return {
                                "records": records,
                                "filters": filters,
                                "chart": chart,
                                "sources": db_sources,
                                "context": context,
                            }

            return None

        except Exception as e:
            logger.error(f"Financial phase failed: {e}")
            return None

    def _build_financial_context(
        self,
        records: List[FinancialDataRecord],
        filters: FinancialDataFilters,
    ) -> str:
        """Build context string from financial records for synthesis."""
        if not records:
            return "No financial data found for the query."

        context_parts = [f"Financial Data Results ({len(records)} records):"]

        # Group by company
        by_company = {}
        for record in records[:50]:  # Limit to 50 records
            company = record.company or record.symbol
            if company not in by_company:
                by_company[company] = []
            by_company[company].append(record)

        for company, company_records in by_company.items():
            context_parts.append(f"\n{company}:")
            for r in company_records:
                year = r.year or "N/A"
                metric_name = r.standard_item or "metric"
                value = r.item
                if value is not None:
                    # Format large numbers
                    if abs(value) >= 1_000_000:
                        formatted = f"${value/1_000_000:,.2f}M"
                    elif abs(value) >= 1_000:
                        formatted = f"${value/1_000:,.2f}K"
                    else:
                        formatted = f"{value:,.2f}"
                    context_parts.append(f"  - {metric_name} ({year}): {formatted}")
                else:
                    # Use formatted_value for display when item is None
                    context_parts.append(f"  - {metric_name} ({year}): {r.formatted_value}")

        return "\n".join(context_parts)

    async def _run_web_search_phase(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Phase 2: Run web search using Google Search grounding.

        Args:
            query: Optimized query

        Returns:
            Dictionary with search_results, sources, and context
        """
        try:
            # Build tool for web search only
            tool = types.Tool(google_search=types.GoogleSearch())

            system_instruction = """You are a web search assistant for Jamaica Stock Exchange research.
Search the web for relevant information about the query and provide a summary.
Focus on recent news, market updates, and company information from Jamaica.
Cite your sources."""

            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=query)],
                )
            ]

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[tool],
                temperature=0.7,
                max_output_tokens=2048,
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )

            sources = []
            search_results = {}
            context = response.text if response.text else ""

            # Extract grounding metadata
            if response.candidates and response.candidates[0].grounding_metadata:
                grounding = response.candidates[0].grounding_metadata

                if hasattr(grounding, "search_entry_point") and grounding.search_entry_point:
                    search_results["search_entry_point"] = str(grounding.search_entry_point)

                if hasattr(grounding, "grounding_chunks") and grounding.grounding_chunks:
                    chunks = []
                    for chunk in grounding.grounding_chunks:
                        if hasattr(chunk, "web") and chunk.web:
                            web = chunk.web
                            # Extract all available fields from the web chunk
                            title = getattr(web, "title", None)
                            uri = getattr(web, "uri", None)
                            domain = getattr(web, "domain", None)

                            chunk_info = {
                                "title": title,
                                "uri": uri,
                                "domain": domain,
                            }
                            chunks.append(chunk_info)

                            # Build source citation with complete URL
                            # Only add if we have either a title or URL
                            if title or uri:
                                source_entry = {
                                    "type": "web",
                                    "description": title
                                    or domain
                                    or "Web Source",  # backwards compat
                                    "title": title,  # explicit title field
                                }
                                # Only include url if we have one (avoid empty strings)
                                if uri:
                                    source_entry["url"] = uri
                                # Include domain if available
                                if domain:
                                    source_entry["domain"] = domain
                                sources.append(source_entry)
                    search_results["grounding_chunks"] = chunks

                # Extract grounding supports - maps response text segments to source chunks
                if hasattr(grounding, "grounding_supports") and grounding.grounding_supports:
                    supports = []
                    for support in grounding.grounding_supports:
                        support_entry = {}
                        # Extract segment info (text range in response)
                        if hasattr(support, "segment") and support.segment:
                            segment = support.segment
                            support_entry["text_start"] = getattr(segment, "start_index", None)
                            support_entry["text_end"] = getattr(segment, "end_index", None)
                            support_entry["text"] = getattr(segment, "text", None)
                        # Extract chunk indices (which sources support this text)
                        if hasattr(support, "grounding_chunk_indices"):
                            indices = support.grounding_chunk_indices
                            support_entry["chunk_indices"] = list(indices) if indices else []
                        # Extract confidence scores (may be empty for Gemini 2.5+)
                        if hasattr(support, "confidence_scores"):
                            scores = support.confidence_scores
                            support_entry["confidence_scores"] = list(scores) if scores else []
                        if support_entry:
                            supports.append(support_entry)
                    if supports:
                        search_results["grounding_supports"] = supports

                if hasattr(grounding, "web_search_queries") and grounding.web_search_queries:
                    search_results["search_queries"] = list(grounding.web_search_queries)

            return {
                "search_results": search_results if search_results else None,
                "sources": sources,
                "context": context,
            }

        except Exception as e:
            logger.error(f"Web search phase failed: {e}")
            return None

    async def _synthesize_response(
        self,
        query: str,
        optimized_query: str,
        financial_context: str,
        web_context: str,
        financial_records: List[FinancialDataRecord],
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> str:
        """
        Phase 3: Synthesize a final response from all gathered context.

        Args:
            query: Original user query
            optimized_query: Query with resolved references
            financial_context: Context from financial database
            web_context: Context from web search
            financial_records: Raw financial records
            conversation_history: Previous conversation

        Returns:
            Synthesized response text
        """
        # Build combined context
        context_parts = []

        if financial_context:
            context_parts.append(f"DATABASE RESULTS:\n{financial_context}")

        if web_context:
            context_parts.append(f"WEB SEARCH RESULTS:\n{web_context}")

        if not context_parts:
            return "I couldn't find any relevant information for your query. Please try rephrasing or ask about specific companies, financial metrics, or topics."

        combined_context = "\n\n".join(context_parts)

        system_instruction = """You are a financial research assistant for the Jamaica Stock Exchange (JSE).

IMPORTANT RULES:
1. ONLY respond based on the provided context - do NOT hallucinate or make up data
2. EVERY factual claim MUST be cited:
   - Financial data: cite as [Database: Company, Year]
   - Web results: cite as [Web: Source Title]
3. If information is incomplete, acknowledge what was found and what wasn't
4. At the end, suggest 2-3 relevant follow-up questions

Format your response clearly with the data, analysis, and sources."""

        prompt = f"""Based on the following research results, answer the user's question.

User Question: {query}

{combined_context}

Provide a clear, well-cited response with follow-up suggestions."""

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
            max_output_tokens=4096,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

        return response.text if response.text else ""

    async def _process_response(
        self,
        query: str,
        response: Any,
        conversation_history: Optional[List[Dict[str, str]]],
        enable_web_search: bool,
        enable_financial_data: bool,
    ) -> Dict[str, Any]:
        """
        Process Gemini response and extract structured results.

        Args:
            query: Original user query
            response: Gemini response object
            conversation_history: Previous conversation
            enable_web_search: Whether web search was enabled
            enable_financial_data: Whether financial data was enabled

        Returns:
            Structured result dictionary
        """
        tools_executed = []
        sources = []
        financial_records = []
        filters_used = None
        chart_spec = None
        web_search_results = None

        # Check for function calls in the response
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    if fc.name == "query_financial_data":
                        tools_executed.append("query_financial_data")
                        try:
                            args = dict(fc.args) if fc.args else {}
                            records, filters, chart, db_sources = await execute_financial_query(
                                self.financial_manager, args
                            )
                            financial_records = records
                            filters_used = filters
                            chart_spec = chart
                            sources.extend(db_sources)
                        except Exception as e:
                            logger.error(f"Failed to execute financial query: {e}")

        # Check for Google Search grounding
        if response.candidates and response.candidates[0].grounding_metadata:
            grounding = response.candidates[0].grounding_metadata

            if enable_web_search:
                tools_executed.append("google_search")

            # Extract web search metadata
            web_search_results = {}

            if hasattr(grounding, "search_entry_point") and grounding.search_entry_point:
                web_search_results["search_entry_point"] = str(grounding.search_entry_point)

            if hasattr(grounding, "grounding_chunks") and grounding.grounding_chunks:
                chunks = []
                for chunk in grounding.grounding_chunks:
                    if hasattr(chunk, "web") and chunk.web:
                        web = chunk.web
                        # Extract all available fields from the web chunk
                        title = getattr(web, "title", None)
                        uri = getattr(web, "uri", None)
                        domain = getattr(web, "domain", None)

                        chunk_info = {
                            "title": title,
                            "uri": uri,
                            "domain": domain,
                        }
                        chunks.append(chunk_info)

                        # Build source citation with complete URL
                        if title or uri:
                            source_entry = {
                                "type": "web",
                                "description": title or domain or "Web Source",  # backwards compat
                                "title": title,  # explicit title field
                            }
                            if uri:
                                source_entry["url"] = uri
                            if domain:
                                source_entry["domain"] = domain
                            sources.append(source_entry)
                web_search_results["grounding_chunks"] = chunks

            # Extract grounding supports - maps response text segments to source chunks
            if hasattr(grounding, "grounding_supports") and grounding.grounding_supports:
                supports = []
                for support in grounding.grounding_supports:
                    support_entry = {}
                    if hasattr(support, "segment") and support.segment:
                        segment = support.segment
                        support_entry["text_start"] = getattr(segment, "start_index", None)
                        support_entry["text_end"] = getattr(segment, "end_index", None)
                        support_entry["text"] = getattr(segment, "text", None)
                    if hasattr(support, "grounding_chunk_indices"):
                        indices = support.grounding_chunk_indices
                        support_entry["chunk_indices"] = list(indices) if indices else []
                    if hasattr(support, "confidence_scores"):
                        scores = support.confidence_scores
                        support_entry["confidence_scores"] = list(scores) if scores else []
                    if support_entry:
                        supports.append(support_entry)
                if supports:
                    web_search_results["grounding_supports"] = supports

            if hasattr(grounding, "web_search_queries") and grounding.web_search_queries:
                web_search_results["search_queries"] = list(grounding.web_search_queries)

        # Get the response text
        response_text = response.text if response.text else ""

        # Extract follow-up questions from response
        follow_up_questions = self._extract_follow_up_questions(response_text)

        # Clean response text (remove the follow-up section if we extracted it)
        main_response = self._clean_response_text(response_text)

        # Build conversation history update
        updated_history = self._update_conversation_history(
            query, main_response, conversation_history
        )

        return {
            "response": main_response,
            "data_found": len(financial_records) > 0,
            "record_count": len(financial_records),
            "filters_used": filters_used,
            "data_preview": financial_records if financial_records else None,
            "conversation_history": updated_history,
            "warnings": None,
            "suggestions": follow_up_questions,
            "chart": ChartSpec(**chart_spec) if chart_spec else None,
            "sources": sources if sources else None,
            "web_search_results": web_search_results,
            "tools_executed": tools_executed if tools_executed else None,
        }

    def _extract_follow_up_questions(self, response_text: str) -> Optional[List[str]]:
        """
        Extract follow-up questions from the response text.

        Looks for patterns like:
        - **Suggested Follow-up Questions:**
        - **Follow-up Questions:**
        - Numbered lists at the end

        Args:
            response_text: The full response text

        Returns:
            List of follow-up questions or None
        """
        questions = []

        # Pattern 1: Look for explicit follow-up section
        patterns = [
            r"\*\*Suggested Follow-up Questions?:\*\*\s*(.*?)(?:\n\n|$)",
            r"\*\*Follow-up Questions?:\*\*\s*(.*?)(?:\n\n|$)",
            r"Follow-up questions?:\s*(.*?)(?:\n\n|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if match:
                questions_text = match.group(1)
                # Extract numbered questions
                found = re.findall(r"\d+\.\s*(.+?)(?=\d+\.|$)", questions_text, re.DOTALL)
                if found:
                    questions = [q.strip() for q in found if q.strip()][:3]
                    break

        return questions if questions else None

    def _clean_response_text(self, response_text: str) -> str:
        """
        Clean the response text by removing the follow-up questions section.

        Args:
            response_text: The full response text

        Returns:
            Cleaned response text
        """
        # Remove follow-up questions section
        patterns = [
            r"\*\*Suggested Follow-up Questions?:\*\*.*$",
            r"\*\*Follow-up Questions?:\*\*.*$",
            r"Follow-up questions?:.*$",
        ]

        cleaned = response_text
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

        return cleaned.strip()

    def _update_conversation_history(
        self,
        query: str,
        response: str,
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        """
        Update conversation history with the new exchange.

        Args:
            query: User's query
            response: Agent's response
            conversation_history: Previous conversation

        Returns:
            Updated conversation history (limited to last 20 messages)
        """
        if conversation_history:
            updated = conversation_history.copy()
        else:
            updated = []

        updated.append({"role": "user", "content": query})
        updated.append({"role": "assistant", "content": response})

        # Limit to last 20 messages
        if len(updated) > 20:
            updated = updated[-20:]

        return updated

    def _build_no_tools_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """
        Build a response when no tools are enabled.

        Args:
            query: User's query
            conversation_history: Previous conversation

        Returns:
            Response dictionary
        """
        response_text = (
            "I'm sorry, but I don't have any tools enabled to help answer your question. "
            "Please enable either web search or financial data queries to get relevant information."
        )

        return {
            "response": response_text,
            "data_found": False,
            "record_count": 0,
            "filters_used": None,
            "data_preview": None,
            "conversation_history": self._update_conversation_history(
                query, response_text, conversation_history
            ),
            "warnings": ["No tools were enabled for this query"],
            "suggestions": None,
            "chart": None,
            "sources": None,
            "web_search_results": None,
            "tools_executed": None,
        }

    def _build_error_response(
        self,
        query: str,
        error: str,
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """
        Build a response when an error occurs.

        Args:
            query: User's query
            error: Error message
            conversation_history: Previous conversation

        Returns:
            Response dictionary
        """
        response_text = (
            "I encountered an error while processing your request. "
            "Please try rephrasing your question or try again later. "
            f"Error details: {error}"
        )

        return {
            "response": response_text,
            "data_found": False,
            "record_count": 0,
            "filters_used": None,
            "data_preview": None,
            "conversation_history": self._update_conversation_history(
                query, response_text, conversation_history
            ),
            "warnings": [f"Error occurred: {error}"],
            "suggestions": None,
            "chart": None,
            "sources": None,
            "web_search_results": None,
            "tools_executed": None,
        }
