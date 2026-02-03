"""
Simplified Agent module for orchestrating Gemini with multiple tools.

This module provides the AgentOrchestrator class that combines:
- Google Search (native Gemini tool) for web grounding
- SQL Query Tool (custom) for financial database queries

Architecture: 3-phase pipeline
1. Smart Optimization - Single LLM call for clarification, routing, pronoun resolution
2. Tool Execution - Financial query and/or web search
3. Response Synthesis - Combine results into natural response
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from google.genai import types

from app.charting import generate_chart
from app.gemini_client import get_genai_client
from app.logging_config import get_logger
from app.models import (
    ClarificationReason,
    CostSummary,
    FinancialDataFilters,
    FinancialDataRecord,
    PhaseCost,
)
from app.utils.cost_tracking import calculate_cost_from_response
from app.utils.monitoring import record_ai_cost

logger = get_logger(__name__)


# ==============================================================================
# CONSTANTS (Minimal - trust LLM for intent classification)
# ==============================================================================

# Year pattern for relative time resolution
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

# Relative time patterns for defaults
RELATIVE_TIME_PATTERNS = {
    "last_n_years": re.compile(r"(?:last|past)\s+(\d+)\s+years?", re.IGNORECASE),
    "last_year": re.compile(r"\blast\s+year\b", re.IGNORECASE),
    "recently": re.compile(r"\brecent(?:ly)?\b", re.IGNORECASE),
    "this_year": re.compile(r"\bthis\s+year\b", re.IGNORECASE),
}

# Clarification templates
CLARIFICATION_TEMPLATES = {
    "no_entity": (
        "Which company or stock would you like information about? "
        "For example, you could ask about NCB, GraceKennedy, or any JSE-listed company."
    ),
    "unresolved_pronoun": (
        "Could you please clarify which company you're referring to? "
        "I don't have enough context from our conversation to determine the company you mean."
    ),
    "ambiguous_comparison": (
        "I'd be happy to compare companies for you. "
        "Could you specify which ones you'd like me to compare?"
    ),
}

# Default metrics for vague "performance" queries
DEFAULT_PERFORMANCE_METRICS = ["revenue", "net_profit", "eps"]

# Marker for detecting previous clarification (1-round max rule)
CLARIFICATION_MARKER = "I want to make sure I understand"


# ==============================================================================
# TOOL DECLARATION
# ==============================================================================


def get_financial_data_tool_declaration() -> types.FunctionDeclaration:
    """Define the financial query tool for Gemini function calling."""
    return types.FunctionDeclaration(
        name="query_financial_data",
        description="""Query financial data from the Jamaica Stock Exchange (JSE) database.
Use this tool when the user asks about:
- Company financial metrics (revenue, profit, EPS, margins, assets, liabilities)
- Financial comparisons between companies
- Historical financial data for specific years

Available metrics: revenue, net_profit, gross_profit, operating_profit, eps,
total_assets, total_liabilities, shareholders_equity, gross_profit_margin,
net_profit_margin, operating_profit_margin, roe, roa, current_ratio, debt_to_equity.""",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "symbols": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                    description="Stock trading symbols (e.g., ['NCB', 'JBG', 'GK']). Use uppercase.",
                ),
                "years": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                    description="Years to filter by (e.g., ['2022', '2023', '2024']).",
                ),
                "standard_items": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                    description="Financial metrics to retrieve (e.g., ['revenue', 'net_profit']).",
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
    """Execute financial data query and return results with source metadata."""
    start_time = time.time()

    try:
        # Build filters from tool arguments (handle None values)
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
            interpretation=f"Agent query: symbols={symbols}, years={years}, items={standard_items}",
            data_availability_note="",
            is_follow_up=False,
            context_used="",
        )

        # Post-process filters using associations from metadata
        if financial_manager.metadata and "associations" in financial_manager.metadata:
            filters_dict = filters.model_dump()
            filters_dict = financial_manager._post_process_filters(filters_dict)
            filters = FinancialDataFilters(**filters_dict)

        # Query the data
        records = financial_manager.query_data(filters)

        # Generate chart if applicable
        chart_spec = None
        if records:
            chart_data = generate_chart(records, "")
            if chart_data:
                chart_spec = chart_data

        # Build source citations
        symbols_str = ", ".join(filters.symbols) if filters.symbols else "all"
        years_str = ", ".join(filters.years) if filters.years else "all years"
        source_entry = {
            "type": "database",
            "description": f"JSE Financial Database: {symbols_str} ({years_str})",
            "table": "financial_data",
        }
        if filters.symbols:
            source_entry["symbols"] = filters.symbols
        if filters.years:
            source_entry["years"] = filters.years
        if filters.standard_items:
            source_entry["metrics"] = filters.standard_items
        sources = [source_entry]

        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Financial query: {len(records)} records in {duration_ms:.2f}ms")

        return records, filters, chart_spec, sources

    except Exception as e:
        logger.error(f"Financial query failed: {e}", exc_info=True)
        raise


# ==============================================================================
# AGENT ORCHESTRATOR
# ==============================================================================


class AgentOrchestrator:
    """
    Simplified agent with 3-phase architecture.

    Phase 1: Smart optimization (clarification, routing, pronoun resolution)
    Phase 2: Tool execution (financial and/or web search)
    Phase 3: Response synthesis
    """

    def __init__(self, financial_manager: Any):
        self.financial_manager = financial_manager
        self.client = get_genai_client()
        self.model_name = "gemini-2.5-flash"
        self._phase_costs: List[PhaseCost] = []

    # --------------------------------------------------------------------------
    # Cost Tracking
    # --------------------------------------------------------------------------

    def _reset_cost_tracking(self) -> None:
        """Reset cost tracking for a new request."""
        self._phase_costs = []

    def _add_phase_cost(
        self,
        phase: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        input_cost: float,
        output_cost: float,
        total_cost: float,
    ) -> None:
        """Add a phase cost to tracking."""
        self._phase_costs.append(
            PhaseCost(
                phase=phase,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                input_cost_usd=input_cost,
                output_cost_usd=output_cost,
                total_cost_usd=total_cost,
            )
        )

    def _build_cost_summary(self) -> CostSummary:
        """Build cost summary from accumulated phases."""
        return CostSummary(
            total_input_tokens=sum(p.input_tokens for p in self._phase_costs),
            total_output_tokens=sum(p.output_tokens for p in self._phase_costs),
            total_cached_tokens=sum(p.cached_tokens for p in self._phase_costs),
            total_cost_usd=sum(p.total_cost_usd for p in self._phase_costs),
            phases=self._phase_costs.copy(),
        )

    def _track_cost(self, response: Any, phase: str) -> None:
        """Track cost from a Gemini response."""
        cost = calculate_cost_from_response(self.model_name, response, phase)
        record_ai_cost(
            model=cost.model,
            phase=cost.phase,
            input_tokens=cost.token_usage.input_tokens,
            output_tokens=cost.token_usage.output_tokens,
            input_cost=cost.input_cost,
            output_cost=cost.output_cost,
            total_cost=cost.total_cost,
            cached_tokens=cost.token_usage.cached_tokens,
        )
        self._add_phase_cost(
            phase=phase,
            model=self.model_name,
            input_tokens=cost.token_usage.input_tokens,
            output_tokens=cost.token_usage.output_tokens,
            cached_tokens=cost.token_usage.cached_tokens,
            input_cost=cost.input_cost,
            output_cost=cost.output_cost,
            total_cost=cost.total_cost,
        )

    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------

    def _format_conversation(self, history: Optional[List[Dict[str, str]]]) -> str:
        """Format conversation history for LLM context."""
        if not history:
            return "(No previous conversation)"
        recent = history[-10:]  # Last 10 messages for context
        return "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent])

    def _has_previous_clarification(self, history: Optional[List[Dict[str, str]]]) -> bool:
        """Check if we already asked for clarification (1-round max rule)."""
        if not history:
            return False
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                return CLARIFICATION_MARKER in msg.get("content", "")
        return False

    def _get_metadata_context(self) -> str:
        """Get available symbols and metadata for LLM context."""
        if not self.financial_manager or not self.financial_manager.metadata:
            return ""

        metadata = self.financial_manager.metadata
        parts = []

        if "symbols" in metadata:
            symbols = metadata["symbols"][:30]
            parts.append(f"Stock symbols: {', '.join(symbols)}")

        if "associations" in metadata:
            s2c = metadata["associations"].get("symbol_to_company", {})
            if s2c:
                mappings = [f"{sym}: {', '.join(cos)}" for sym, cos in list(s2c.items())[:15]]
                parts.append("Symbol-Company: " + "; ".join(mappings))

        if "years" in metadata:
            years = sorted(metadata["years"])[-5:]
            parts.append(f"Years: {', '.join(years)}")

        return "\n".join(parts)

    def _resolve_relative_time(self, query: str) -> Tuple[List[str], List[str]]:
        """Resolve relative time expressions to specific years."""
        if not self.financial_manager or not self.financial_manager.metadata:
            return [], []

        available_years = sorted(self.financial_manager.metadata.get("years", []))
        if not available_years:
            return [], []

        most_recent = int(available_years[-1])
        resolved_years = []
        defaults_applied = []

        # Check "last N years"
        match = RELATIVE_TIME_PATTERNS["last_n_years"].search(query)
        if match:
            n = int(match.group(1))
            resolved_years = [str(most_recent - i) for i in range(n)]
            resolved_years = [y for y in resolved_years if y in available_years]
            defaults_applied.append(f"year (last {n} years → {', '.join(resolved_years)})")
            return resolved_years, defaults_applied

        # Check "last year"
        if RELATIVE_TIME_PATTERNS["last_year"].search(query):
            last_year = str(most_recent - 1)
            if last_year in available_years:
                return [last_year], [f"year (last year → {last_year})"]

        # Check "recently"
        if RELATIVE_TIME_PATTERNS["recently"].search(query):
            recent = [str(most_recent), str(most_recent - 1)]
            recent = [y for y in recent if y in available_years]
            return recent, [f"year (recently → {', '.join(recent)})"]

        # Check "this year"
        if RELATIVE_TIME_PATTERNS["this_year"].search(query):
            if str(most_recent) in available_years:
                return [str(most_recent)], [f"year (this year → {most_recent})"]

        return [], []

    # --------------------------------------------------------------------------
    # Phase 1: Smart Optimization (Single LLM Call)
    # --------------------------------------------------------------------------

    async def _smart_optimize(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]],
        enable_web_search: bool,
        enable_financial_data: bool,
    ) -> Dict[str, Any]:
        """
        Single LLM call for: clarification detection, pronoun resolution, tool routing.

        Returns dict with:
            - needs_clarification: bool
            - clarification_reason: ClarificationReason or None
            - clarification_question: str or None
            - optimized_query: str
            - routing: {"use_financial": bool, "use_web_search": bool}
            - defaults_applied: List[str]
        """
        # Fast path: Skip if we already asked for clarification (1-round max)
        if self._has_previous_clarification(conversation_history):
            logger.info("Skipping clarification (1-round max reached)")
            return {
                "needs_clarification": False,
                "clarification_reason": None,
                "clarification_question": None,
                "optimized_query": query,
                "routing": {
                    "use_financial": enable_financial_data,
                    "use_web_search": enable_web_search,
                },
                "defaults_applied": [],
            }

        # Resolve relative time expressions
        resolved_years, time_defaults = self._resolve_relative_time(query)

        # Build the unified prompt
        metadata_context = self._get_metadata_context()
        conversation_str = self._format_conversation(conversation_history)

        prompt = f"""You are a query analyzer for a Jamaica Stock Exchange (JSE) chatbot.

AVAILABLE DATA:
{metadata_context}

CONVERSATION HISTORY:
{conversation_str}

CURRENT QUERY: "{query}"

YOUR TASK: Route the query to the correct tool(s) and resolve any pronouns.

TOOL ROUTING RULES (CRITICAL - FOLLOW EXACTLY):

FINANCIAL ONLY - Any query asking for specific financial metrics:
- "What is X's revenue" → FINANCIAL (always)
- "What is X's profit" → FINANCIAL (always)
- "What is X's EPS" → FINANCIAL (always)
- Any query with: revenue, profit, EPS, assets, liabilities, margins, ratios, current ratio, debt to equity
- Financial comparisons ("compare NCB and GK revenue")
- "Show me X's financials" → FINANCIAL
- Year-specific financial data: "for 2023", "last 5 years", "this year", "last year" WITH financial metric

WEB ONLY - Current events WITHOUT financial metrics:
- "latest news about X" (no financial metric mentioned)
- "announcements", "today" (no financial metric mentioned)
- "what is happening on the JSE/market"
- "when was X founded" or historical facts
- Companies NOT in our database (Evolve Loan Company, etc.)
- "recently" WITHOUT a specific financial metric

BOTH TOOLS - Vague performance or general overviews:
- "How did X perform" or "How did X do" (no specific metric)
- "Tell me about X" (general company overview)
- "overview of X"
- Mixed requests: financial metric AND "news" in same query

ENTITY RESOLUTION:
- "The company", "the stock" → resolve from history
- "their", "its", "they", "them" → resolve to entity from history
- "the other company" → resolve from history
- "Compare them" → use BOTH entities from history
- "tell me more" → continue with entity from history
- Company name like "Dolla Financial Services" → use as entity (even without symbol)

OUTPUT FORMAT - Choose ONE:
  PROCEED|FINANCIAL: <query with resolved entity names>
  PROCEED|WEB: <query with resolved entity names>
  PROCEED|BOTH: <query with resolved entity names>
  CLARIFY:NO_ENTITY (no entity anywhere)
  CLARIFY:UNRESOLVED_PRONOUN (pronoun with no referent)
  CLARIFY:AMBIGUOUS_COMPARISON ("compare" with <2 identifiable entities)

EXAMPLES:
- "What is NCB's revenue for 2023?" → PROCEED|FINANCIAL: What is NCB's revenue for 2023?
- "What is NCB's revenue?" → PROCEED|FINANCIAL: What is NCB's revenue?
- "What is NCB's revenue this year?" → PROCEED|FINANCIAL: What is NCB's revenue this year?
- "Show NCB's revenue for the last 5 years" → PROCEED|FINANCIAL: Show NCB's revenue for the last 5 years
- "What was NCB's profit last year?" → PROCEED|FINANCIAL: What was NCB's profit last year?
- "What is NCB EPS for 2024?" → PROCEED|FINANCIAL: What is NCB EPS for 2024?
- "Show me GK's financials for 2023" → PROCEED|FINANCIAL: Show me GK's financials for 2023
- "Compare GK and NCB revenue for 2023" → PROCEED|FINANCIAL: Compare GK and NCB revenue for 2023
- "How has GK performed recently?" → PROCEED|BOTH: How has GK performed recently?
- "Latest news about Jamaica Stock Exchange" → PROCEED|WEB: Latest news about Jamaica Stock Exchange
- "What happened recently in the stock market?" → PROCEED|WEB: What happened recently in the stock market?
- "Did NCB make any announcements today?" → PROCEED|WEB: Did NCB make any announcements today?
- "GK revenue for 2023 and latest news" → PROCEED|BOTH: GK revenue for 2023 and latest news
- "How well did GK perform in 2023?" → PROCEED|BOTH: How well did GK perform in 2023?
- "How did NCB do last year?" → PROCEED|BOTH: How did NCB do last year?
- "Tell me about Jamaica Broilers" → PROCEED|BOTH: Tell me about Jamaica Broilers
- "Give me an overview of CPJ for 2023" → PROCEED|BOTH: Give me an overview of CPJ for 2023
- "Compare NCB with the other company" [GK in history] → PROCEED|FINANCIAL: Compare NCB with GK
- "tell me more" [NCB in history] → PROCEED|BOTH: tell me more about NCB
- "What is their profit?" [NCB in history] → PROCEED|FINANCIAL: What is NCB's profit?
- "Compare them" [NCB and GK in history] → PROCEED|FINANCIAL: Compare NCB and GK
- "What is the company's profit margin?" [JBG in history] → PROCEED|FINANCIAL: What is Jamaica Broilers' profit margin?
- "what is their current ratio?" [Dolla Financial Services in history] → PROCEED|FINANCIAL: What is Dolla's current ratio?
- "what about Evolve Loan Company?" → PROCEED|WEB: what about Evolve Loan Company?
- "What is the revenue?" (empty history) → CLARIFY:NO_ENTITY
- "Compare the banks" (empty history) → CLARIFY:AMBIGUOUS_COMPARISON"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=512),
            )
            self._track_cost(response, "smart_optimization")

            result_text = (response.text or "").strip()
            logger.info(f"Smart optimization result: {result_text[:100]}")

            # Parse response
            result_upper = result_text.upper()

            if "CLARIFY:NO_ENTITY" in result_upper:
                return {
                    "needs_clarification": True,
                    "clarification_reason": ClarificationReason.NO_ENTITY,
                    "clarification_question": CLARIFICATION_TEMPLATES["no_entity"],
                    "optimized_query": query,
                    "routing": {"use_financial": False, "use_web_search": False},
                    "defaults_applied": time_defaults,
                }
            elif "CLARIFY:UNRESOLVED_PRONOUN" in result_upper:
                return {
                    "needs_clarification": True,
                    "clarification_reason": ClarificationReason.UNRESOLVED_PRONOUN,
                    "clarification_question": CLARIFICATION_TEMPLATES["unresolved_pronoun"],
                    "optimized_query": query,
                    "routing": {"use_financial": False, "use_web_search": False},
                    "defaults_applied": time_defaults,
                }
            elif "CLARIFY:AMBIGUOUS_COMPARISON" in result_upper:
                return {
                    "needs_clarification": True,
                    "clarification_reason": ClarificationReason.AMBIGUOUS_COMPARISON,
                    "clarification_question": CLARIFICATION_TEMPLATES["ambiguous_comparison"],
                    "optimized_query": query,
                    "routing": {"use_financial": False, "use_web_search": False},
                    "defaults_applied": time_defaults,
                }
            elif "PROCEED|" in result_upper:
                # Parse PROCEED|TOOL: query
                proceed_idx = result_text.upper().find("PROCEED|")
                after_proceed = result_text[proceed_idx + 8 :]

                tool_type = "BOTH"
                optimized_query = query

                if after_proceed.upper().startswith("FINANCIAL:"):
                    tool_type = "FINANCIAL"
                    optimized_query = after_proceed[10:].strip()
                elif after_proceed.upper().startswith("WEB:"):
                    tool_type = "WEB"
                    optimized_query = after_proceed[4:].strip()
                elif after_proceed.upper().startswith("BOTH:"):
                    tool_type = "BOTH"
                    optimized_query = after_proceed[5:].strip()
                elif ":" in after_proceed:
                    colon_idx = after_proceed.find(":")
                    tool_part = after_proceed[:colon_idx].strip().upper()
                    optimized_query = after_proceed[colon_idx + 1 :].strip()
                    if "FINANCIAL" in tool_part:
                        tool_type = "FINANCIAL"
                    elif "WEB" in tool_part:
                        tool_type = "WEB"
                    else:
                        tool_type = "BOTH"

                # Clean quotes
                if optimized_query.startswith('"') and optimized_query.endswith('"'):
                    optimized_query = optimized_query[1:-1]
                if not optimized_query:
                    optimized_query = query

                # Apply enable flags
                routing = {
                    "use_financial": tool_type in ("FINANCIAL", "BOTH") and enable_financial_data,
                    "use_web_search": tool_type in ("WEB", "BOTH") and enable_web_search,
                }

                return {
                    "needs_clarification": False,
                    "clarification_reason": None,
                    "clarification_question": None,
                    "optimized_query": optimized_query,
                    "routing": routing,
                    "defaults_applied": time_defaults,
                    "resolved_years": resolved_years,
                }
            else:
                # Default: proceed with both tools
                logger.warning(
                    f"Unclear optimization result: {result_text[:50]}, defaulting to BOTH"
                )
                return {
                    "needs_clarification": False,
                    "clarification_reason": None,
                    "clarification_question": None,
                    "optimized_query": query,
                    "routing": {
                        "use_financial": enable_financial_data,
                        "use_web_search": enable_web_search,
                    },
                    "defaults_applied": time_defaults,
                }

        except Exception as e:
            logger.error(f"Smart optimization failed: {e}, proceeding with defaults")
            return {
                "needs_clarification": False,
                "clarification_reason": None,
                "clarification_question": None,
                "optimized_query": query,
                "routing": {
                    "use_financial": enable_financial_data,
                    "use_web_search": enable_web_search,
                },
                "defaults_applied": time_defaults,
            }

    # --------------------------------------------------------------------------
    # Phase 2: Tool Execution
    # --------------------------------------------------------------------------

    async def _execute_financial(
        self,
        query: str,
        resolved_years: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute financial database query using function calling."""
        try:
            tool = types.Tool(function_declarations=[get_financial_data_tool_declaration()])

            system_prompt = f"""Extract financial query parameters from the user query.
{self._get_metadata_context()}

Call query_financial_data with: symbols (uppercase), years, standard_items (metrics)."""

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=query)])],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[tool],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="ANY", allowed_function_names=["query_financial_data"]
                        )
                    ),
                    temperature=0.3,
                    max_output_tokens=512,
                ),
            )
            self._track_cost(response, "financial_extraction")

            # Extract function call
            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts or []:
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        if fc.name == "query_financial_data":
                            args = dict(fc.args) if fc.args else {}

                            # Apply resolved years if LLM didn't extract any
                            if not args.get("years") and resolved_years:
                                args["years"] = resolved_years
                                logger.info(f"Applied resolved years: {resolved_years}")

                            records, filters, chart, sources = await execute_financial_query(
                                self.financial_manager, args
                            )

                            context = self._build_financial_context(records)
                            return {
                                "records": records,
                                "filters": filters,
                                "chart": chart,
                                "sources": sources,
                                "context": context,
                            }

            logger.warning("Financial phase: No function call in response")
            return None

        except Exception as e:
            logger.error(f"Financial execution failed: {e}")
            return None

    def _build_financial_context(self, records: List[FinancialDataRecord]) -> str:
        """Build context string from financial records."""
        if not records:
            return "No financial data found."

        lines = [f"Financial Data ({len(records)} records):"]
        by_company: Dict[str, List[FinancialDataRecord]] = {}

        for record in records[:50]:
            company = record.company or record.symbol
            if company not in by_company:
                by_company[company] = []
            by_company[company].append(record)

        for company, company_records in by_company.items():
            lines.append(f"\n{company}:")
            for r in company_records:
                year = r.year or "N/A"
                metric = r.standard_item or "metric"
                if r.item is not None:
                    if abs(r.item) >= 1_000_000:
                        formatted = f"${r.item/1_000_000:,.2f}M"
                    elif abs(r.item) >= 1_000:
                        formatted = f"${r.item/1_000:,.2f}K"
                    else:
                        formatted = f"{r.item:,.2f}"
                    lines.append(f"  - {metric} ({year}): {formatted}")
                else:
                    lines.append(f"  - {metric} ({year}): {r.formatted_value}")

        return "\n".join(lines)

    async def _execute_web_search(self, query: str) -> Optional[Dict[str, Any]]:
        """Execute web search using Google Search grounding."""
        try:
            tool = types.Tool(google_search=types.GoogleSearch())

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=query)])],
                config=types.GenerateContentConfig(
                    system_instruction="Search for Jamaica Stock Exchange information. Cite sources.",
                    tools=[tool],
                    temperature=0.7,
                    max_output_tokens=2048,
                ),
            )
            self._track_cost(response, "web_search")

            # Extract grounding metadata
            sources = []
            search_results = {}

            if response.candidates:
                candidate = response.candidates[0]
                grounding = getattr(candidate, "grounding_metadata", None)

                if grounding:
                    # Extract search entry point
                    if hasattr(grounding, "search_entry_point"):
                        sep = grounding.search_entry_point
                        if hasattr(sep, "rendered_content"):
                            search_results["search_entry_point"] = sep.rendered_content

                    # Extract grounding chunks (web sources)
                    chunks = getattr(grounding, "grounding_chunks", []) or []
                    grounding_chunks = []
                    for chunk in chunks:
                        if hasattr(chunk, "web") and chunk.web:
                            web = chunk.web
                            chunk_info = {
                                "title": getattr(web, "title", ""),
                                "uri": getattr(web, "uri", ""),
                            }
                            grounding_chunks.append(chunk_info)
                            # Add to sources
                            if chunk_info["uri"]:
                                sources.append(
                                    {
                                        "type": "web",
                                        "title": chunk_info["title"],
                                        "url": chunk_info["uri"],
                                    }
                                )
                    if grounding_chunks:
                        search_results["grounding_chunks"] = grounding_chunks

                    # Extract web search queries
                    queries = getattr(grounding, "web_search_queries", []) or []
                    if queries:
                        search_results["queries"] = list(queries)

            context = response.text if response.text else ""

            return {
                "search_results": search_results if search_results else None,
                "sources": sources,
                "context": context,
            }

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return None

    # --------------------------------------------------------------------------
    # Phase 3: Response Synthesis
    # --------------------------------------------------------------------------

    async def _synthesize(
        self,
        query: str,
        financial_context: str,
        web_context: str,
        original_query: str = "",
    ) -> str:
        """Synthesize final response from tool results."""
        # Combine contexts
        combined_context = ""
        if financial_context and financial_context != "No financial data found.":
            combined_context += f"DATABASE RESULTS:\n{financial_context}\n\n"
        if web_context:
            combined_context += f"WEB SEARCH RESULTS:\n{web_context}\n\n"

        if not combined_context.strip():
            return "I couldn't find specific information for your query. Could you try rephrasing or being more specific about what you're looking for?"

        system_prompt = """You are Jacie, a knowledgeable JSE assistant.
Be professional, concise, and helpful. Present data naturally.
CRITICAL: You MUST mention the company name and/or stock symbol in your response.
For web sources, mention where info came from. End with 1-2 follow-up suggestions when helpful."""

        # Extract entity from query for emphasis
        entity_hint = ""
        entities_to_check = [
            "NCB",
            "GK",
            "GraceKennedy",
            "Grace Kennedy",
            "JBG",
            "Jamaica Broilers",
            "DOLLA",
            "Dolla",
            "CPJ",
            "MDS",
            "JMMB",
            "Supreme Ventures",
            "SVL",
        ]
        for entity in entities_to_check:
            if entity.lower() in query.lower():
                entity_hint = f"Entity being discussed: {entity}\n"
                break

        # Use optimized query which has resolved entity names
        user_prompt = f"""User's question: {original_query if original_query else query}
Resolved query: {query}
{entity_hint}
{combined_context}

Provide a helpful response. CRITICAL REQUIREMENT: Start your response by mentioning the company name or stock symbol from the resolved query (e.g., "GK's market cap..." or "Jamaica Broilers' profit margin...")."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                    max_output_tokens=4096,
                ),
            )
            self._track_cost(response, "synthesis")
            return (
                response.text if response.text else "I encountered an issue generating a response."
            )

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return f"I encountered an error: {str(e)}"

    # --------------------------------------------------------------------------
    # Response Builders
    # --------------------------------------------------------------------------

    def _build_clarification_response(
        self,
        query: str,
        optimization: Dict[str, Any],
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """Build response requesting user clarification."""
        response_text = (
            f"{CLARIFICATION_MARKER} your question correctly before proceeding.\n\n"
            f"{optimization['clarification_question']}"
        )

        updated_history = list(history) if history else []
        updated_history.append({"role": "user", "content": query})
        updated_history.append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "data_found": False,
            "record_count": 0,
            "needs_clarification": True,
            "clarification_question": optimization["clarification_question"],
            "tools_executed": None,
            "sources": None,
            "filters_used": None,
            "data_preview": None,
            "chart": None,
            "web_search_results": None,
            "suggestions": None,
            "conversation_history": updated_history[-20:],
            "warnings": None,
            "cost_summary": self._build_cost_summary(),
        }

    def _build_no_tools_response(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """Build response when no tools are enabled."""
        response_text = (
            "I don't have any tools enabled to help with your query. "
            "Please enable financial data or web search to get information."
        )

        updated_history = list(history) if history else []
        updated_history.append({"role": "user", "content": query})
        updated_history.append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "data_found": False,
            "record_count": 0,
            "needs_clarification": False,
            "clarification_question": None,
            "tools_executed": [],
            "sources": None,
            "filters_used": None,
            "data_preview": None,
            "chart": None,
            "web_search_results": None,
            "suggestions": None,
            "conversation_history": updated_history[-20:],
            "warnings": ["No tools were enabled for this query"],
            "cost_summary": self._build_cost_summary(),
        }

    def _build_error_response(
        self,
        query: str,
        error: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """Build error response."""
        response_text = f"I encountered an error processing your request: {error}"

        updated_history = list(history) if history else []
        updated_history.append({"role": "user", "content": query})
        updated_history.append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "data_found": False,
            "record_count": 0,
            "needs_clarification": False,
            "clarification_question": None,
            "tools_executed": None,
            "sources": None,
            "filters_used": None,
            "data_preview": None,
            "chart": None,
            "web_search_results": None,
            "suggestions": None,
            "conversation_history": updated_history[-20:],
            "warnings": [f"Error: {error}"],
            "cost_summary": self._build_cost_summary(),
        }

    def _extract_follow_up_questions(self, response: str) -> Optional[List[str]]:
        """Extract follow-up questions from response text."""
        patterns = [
            r"\*\*(?:Suggested )?Follow-up Questions?:\*\*",
            r"Follow-up [Qq]uestions?:",
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                after_header = response[match.end() :]
                questions = re.findall(r"\d+\.\s*(.+?)(?:\n|$)", after_header)
                if questions:
                    return questions[:3]
        return None

    def _clean_response_text(self, response: str) -> str:
        """Remove follow-up questions section from response."""
        patterns = [
            r"\n*\*\*(?:Suggested )?Follow-up Questions?:\*\*.*$",
            r"\n*Follow-up [Qq]uestions?:.*$",
        ]

        cleaned = response
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)
        return cleaned.strip()

    # --------------------------------------------------------------------------
    # Main Entry Point
    # --------------------------------------------------------------------------

    async def run(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        enable_web_search: bool = True,
        enable_financial_data: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the 3-phase agent pipeline.

        Phase 1: Smart optimization (clarification, routing, pronoun resolution)
        Phase 2: Tool execution (financial and/or web search)
        Phase 3: Response synthesis
        """
        start_time = time.time()
        logger.info(f"Agent run: {query[:100]}...")

        self._reset_cost_tracking()

        # Early exit if no tools enabled
        if not enable_web_search and not enable_financial_data:
            return self._build_no_tools_response(query, conversation_history)

        try:
            # Phase 1: Smart Optimization
            optimization = await self._smart_optimize(
                query, conversation_history, enable_web_search, enable_financial_data
            )

            # Short-circuit if clarification needed
            if optimization["needs_clarification"]:
                logger.info(f"Clarification needed: {optimization['clarification_reason']}")
                return self._build_clarification_response(query, optimization, conversation_history)

            optimized_query = optimization["optimized_query"]
            routing = optimization["routing"]
            resolved_years = optimization.get("resolved_years", [])

            logger.info(
                f"Routing: financial={routing['use_financial']}, web={routing['use_web_search']}"
            )

            # Phase 2: Tool Execution
            tools_executed = []
            sources = []
            financial_records = []
            filters_used = None
            chart_spec = None
            web_search_results = None
            financial_context = ""
            web_context = ""

            if routing["use_financial"] and self.financial_manager:
                financial_result = await self._execute_financial(optimized_query, resolved_years)
                if financial_result:
                    tools_executed.append("query_financial_data")
                    financial_records = financial_result.get("records", [])
                    filters_used = financial_result.get("filters")
                    chart_spec = financial_result.get("chart")
                    sources.extend(financial_result.get("sources", []))
                    financial_context = financial_result.get("context", "")

            if routing["use_web_search"]:
                web_result = await self._execute_web_search(optimized_query)
                if web_result:
                    tools_executed.append("google_search")
                    web_search_results = web_result.get("search_results")
                    sources.extend(web_result.get("sources", []))
                    web_context = web_result.get("context", "")

            # Phase 3: Synthesis (pass optimized query with entity names, and original query)
            response_text = await self._synthesize(
                optimized_query, financial_context, web_context, original_query=query
            )
            follow_up = self._extract_follow_up_questions(response_text)
            clean_response = self._clean_response_text(response_text)

            total_time = (time.time() - start_time) * 1000
            logger.info(f"Agent completed in {total_time:.2f}ms")

            return {
                "response": clean_response,
                "data_found": len(financial_records) > 0 or bool(web_context),
                "record_count": len(financial_records),
                "data_preview": financial_records[:10] if financial_records else None,
                "tools_executed": tools_executed,
                "sources": sources,
                "filters_used": filters_used,
                "chart": chart_spec,
                "web_search_results": web_search_results,
                "suggestions": follow_up,
                "conversation_history": conversation_history,
                "cost_summary": self._build_cost_summary(),
            }

        except Exception as e:
            logger.error(f"Agent run failed: {e}", exc_info=True)
            return self._build_error_response(query, str(e), conversation_history)
