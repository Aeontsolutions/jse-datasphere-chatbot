"""
Simplified Agent V2 - Google Search Grounding Only.

This module provides a streamlined agent that uses Gemini 2.5 Pro with native
Google Search grounding for JSE financial research. Unlike the original agent,
this uses a single generate_content call with the GoogleSearch tool.

Architecture:
- Single LLM call with GoogleSearch tool for web grounding
- Simple conversation history management
- JSE-focused system prompt
- Compatible with AgentChatRequest/AgentChatResponse contract
"""

from typing import Any, Dict, List, Optional

from google.genai import types

from app.gemini_client import get_genai_client
from app.logging_config import get_logger
from app.models import CostSummary, PhaseCost
from app.utils.cost_tracking import calculate_cost_from_response
from app.utils.monitoring import record_ai_cost

logger = get_logger(__name__)

# ==============================================================================
# SYSTEM PROMPT - JSE Financial Analyst
# ==============================================================================

SYSTEM_PROMPT = """You are JSE Financial Analyst, an expert AI assistant for the Jamaica Stock Exchange (JSE) and Jamaican financial sector.

## Your Expertise:
- JSE listed companies, stock performance, market trends, IPOs
- Jamaican economy: GDP, inflation, BOJ monetary policy
- Key sectors: Banking (NCB, JMMB, Sagicor), Manufacturing (Wisynco, Seprod), Conglomerates (GraceKennedy)
- Investment analysis: P/E ratios, dividend yields, ROE comparisons

## Key JSE Stocks by Sector:
**Finance**: NCBFG, JMMBGL, SJ, SGJ, BIL, PROVEN, JSE, MGL, VMIL, EPLY, PJX, SELECTF, SCIJMD
**Manufacturing**: WISYNCO, SEP, JBG, CCC, SALF, BRG, LASM, WIPT
**Conglomerates**: GK, PJAM, JP
**Retail**: CAR, CPJ, LASD

## Guidelines:
- Use J$ as primary currency with USD conversions where helpful
- Cite sources from web searches
- Present balanced views with risks and opportunities
- Always include investment disclaimers when providing financial advice

You have web search access for current market data."""


# ==============================================================================
# AGENT V2 - Simple Google Search Grounding
# ==============================================================================


class AgentV2:
    """
    Simplified agent using Gemini 2.5 Pro with native Google Search grounding.

    This follows the pattern from the example script - a single generate_content
    call with the GoogleSearch tool for web grounding.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        """
        Initialize the agent.

        Args:
            model_name: Gemini model to use. Defaults to gemini-2.5-pro.
        """
        self.client = get_genai_client()
        self.model_name = model_name
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

    def _build_contents(
        self, messages: Optional[List[Dict[str, str]]], new_message: str
    ) -> List[types.Content]:
        """
        Build Gemini Content objects from conversation history.

        Args:
            messages: Previous conversation history
            new_message: The new user message

        Returns:
            List of Content objects for Gemini
        """
        contents = []

        if messages:
            for msg in messages:
                # Map "assistant" to "model" for Gemini
                role = "user" if msg.get("role") == "user" else "model"
                content = msg.get("content", "")
                if content:
                    contents.append(
                        types.Content(role=role, parts=[types.Part.from_text(text=content)])
                    )

        # Add the new user message
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=new_message)]))

        return contents

    def _extract_grounding_metadata(self, response: Any) -> Dict[str, Any]:
        """
        Extract grounding metadata from Gemini response.

        Args:
            response: Gemini response object

        Returns:
            Dictionary with sources and search results
        """
        sources = []
        search_results = {}

        if not response.candidates:
            return {"sources": sources, "search_results": search_results}

        candidate = response.candidates[0]
        grounding = getattr(candidate, "grounding_metadata", None)

        if not grounding:
            return {"sources": sources, "search_results": search_results}

        # Extract search entry point (rendered HTML)
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

                # Add to sources list
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

        # Extract web search queries used
        queries = getattr(grounding, "web_search_queries", []) or []
        if queries:
            search_results["queries"] = list(queries)

        return {"sources": sources, "search_results": search_results}

    # --------------------------------------------------------------------------
    # Main Entry Point
    # --------------------------------------------------------------------------

    async def run(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Run the agent with Google Search grounding.

        This makes a single generate_content call with the GoogleSearch tool
        for web grounding, following the simpler approach from the example script.

        Args:
            query: User's question
            conversation_history: Previous conversation messages

        Returns:
            Dictionary compatible with AgentChatResponse
        """
        logger.info(f"AgentV2 run: {query[:100]}...")

        self._reset_cost_tracking()

        try:
            # Build conversation contents
            contents = self._build_contents(conversation_history, query)

            # Google Search tool for grounding
            google_search_tool = types.Tool(google_search=types.GoogleSearch())

            # Single generate_content call with grounding
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.7,
                    top_p=0.95,
                    max_output_tokens=8192,
                    tools=[google_search_tool],
                ),
            )

            # Track cost
            self._track_cost(response, "generation")

            # Extract response text
            response_text = response.text if response.text else ""

            if not response_text:
                response_text = "I encountered an issue generating a response. Please try again."

            # Extract grounding metadata
            grounding_data = self._extract_grounding_metadata(response)
            sources = grounding_data["sources"]
            search_results = grounding_data.get("search_results")

            # Build updated conversation history
            updated_history = list(conversation_history) if conversation_history else []
            updated_history.append({"role": "user", "content": query})
            updated_history.append({"role": "assistant", "content": response_text})

            # Keep last 20 messages
            if len(updated_history) > 20:
                updated_history = updated_history[-20:]

            logger.info(
                f"AgentV2 completed. Response length: {len(response_text)}, "
                f"Sources: {len(sources)}"
            )

            return {
                "response": response_text,
                "data_found": bool(response_text and len(response_text) > 50),
                "record_count": 0,  # No financial records in this version
                "needs_clarification": False,
                "clarification_question": None,
                "tools_executed": ["google_search"] if sources else [],
                "sources": sources if sources else None,
                "filters_used": None,  # No financial filters
                "data_preview": None,  # No financial data
                "chart": None,  # No charts
                "web_search_results": search_results if search_results else None,
                "suggestions": None,  # Could be extracted from response if needed
                "conversation_history": updated_history,
                "warnings": None,
                "cost_summary": self._build_cost_summary(),
            }

        except Exception as e:
            logger.error(f"AgentV2 run failed: {e}", exc_info=True)

            # Build error response
            error_message = f"I encountered an error: {str(e)}"

            updated_history = list(conversation_history) if conversation_history else []
            updated_history.append({"role": "user", "content": query})
            updated_history.append({"role": "assistant", "content": error_message})

            return {
                "response": error_message,
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
                "warnings": [f"Error: {str(e)}"],
                "cost_summary": self._build_cost_summary(),
            }
