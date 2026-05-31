"""
query_financial_data tool primitives for Gemini function calling.

Lifted from the archived AgentOrchestrator (see _archive/agent_orchestrator.py)
into a live, reusable module so AgentV2 can call the JSE BigQuery financial
data on /chat/stream (ATS-334, Option 2).

Contents:
- get_financial_data_tool_declaration() / get_financial_tool(): the tool schema
- execute_financial_query(): build filters, query BigQuery (off-thread), chart, sources
- build_financial_context(): compact text summary of records for synthesis
- extract_query_financial_data_call(): pull the function call from a response
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from google.genai import types

from app.charting import generate_chart
from app.logging_config import get_logger
from app.models import FinancialDataFilters, FinancialDataRecord

logger = get_logger(__name__)


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


def get_financial_tool() -> types.Tool:
    """Wrap the financial data declaration in a Gemini Tool."""
    return types.Tool(function_declarations=[get_financial_data_tool_declaration()])


def extract_query_financial_data_call(response: Any) -> Optional[Any]:
    """Return the first query_financial_data function_call part, or None."""
    if not getattr(response, "candidates", None):
        return None
    candidate = response.candidates[0]
    content = getattr(candidate, "content", None)
    if not content:
        return None
    for part in getattr(content, "parts", None) or []:
        fc = getattr(part, "function_call", None)
        if fc and getattr(fc, "name", None) == "query_financial_data":
            return fc
    return None
