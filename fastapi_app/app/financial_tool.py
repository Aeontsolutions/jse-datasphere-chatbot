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


def _resolve_to_canonical_symbols(symbols: List[str], financial_manager: Any) -> List[str]:
    """Map colloquial company tokens to canonical JSE trading symbols.

    Tokens already matching a known symbol (e.g. "GK", "NCBFG") are kept as-is.
    An unrecognized token (e.g. "NCB") is matched as a fragment against company
    NAMES and mapped to that company's symbol via company_to_symbol (e.g.
    "NCB" -> "NCB Financial Group Limited" -> "NCBFG"). Empty-string company
    keys in the metadata are ignored. If metadata/associations are unavailable
    or no match is found, the original token is preserved unchanged.
    """
    metadata = getattr(financial_manager, "metadata", None) or {}
    known_symbols = {s.upper() for s in metadata.get("symbols", []) if s}
    if not known_symbols:
        return symbols
    associations = metadata.get("associations") or {}
    company_to_symbol = associations.get("company_to_symbol") or {}

    resolved: List[str] = []
    for token in symbols:
        if token in known_symbols:
            resolved.append(token)
            continue
        # Unrecognized: fragment-match against company names (skip empty keys).
        matched_symbols: List[str] = []
        for company, syms in company_to_symbol.items():
            if company and token.lower() in company.lower():
                matched_symbols.extend(s.upper() for s in syms if s)
        if matched_symbols:
            resolved.extend(matched_symbols)
        else:
            resolved.append(token)  # preserve; let downstream handle/limit it

    # De-dup while preserving order.
    seen: set = set()
    deduped = []
    for s in resolved:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


async def execute_financial_query(
    financial_manager: Any,
    args: Dict[str, Any],
) -> Tuple[List[FinancialDataRecord], FinancialDataFilters, Optional[Dict], List[Dict]]:
    """Execute financial data query and return results with source metadata.

    The BigQuery call (financial_manager.query_data) is synchronous/blocking, so
    it is run via asyncio.to_thread to avoid blocking the event loop.
    """
    try:
        raw_symbols = args.get("symbols") or []
        raw_years = args.get("years") or []
        raw_items = args.get("standard_items") or []

        symbols = [s.upper() for s in raw_symbols if s]
        years = [str(y) for y in raw_years if y]
        standard_items = [item.lower().replace(" ", "_") for item in raw_items if item]

        # The query_financial_data tool only exposes `symbols`, but the model
        # often emits a colloquial company token ("NCB") that is not a canonical
        # JSE trading symbol ("NCBFG"). _post_process_filters matches `symbols` by
        # EXACT match, so an unrecognized token is dropped -> empty filter ->
        # WHERE 1=1 over all companies. (Routing it to `companies` instead is
        # worse: that path's fragment matcher collapses "NCB" onto an empty-string
        # company entry in the metadata and yields garbage symbols.) The reliable
        # path is symbols: so resolve any unrecognized token to its canonical
        # symbol here via company_to_symbol (fragment match on the company NAME),
        # and keep everything in `symbols`. Gated on metadata, so behavior is
        # unchanged when it is absent (no regression).
        symbols = _resolve_to_canonical_symbols(symbols, financial_manager)

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

        # Post-process filters using associations from metadata (e.g. symbol->company)
        if financial_manager.metadata and "associations" in financial_manager.metadata:
            filters_dict = filters.model_dump()
            filters_dict = financial_manager._post_process_filters(filters_dict)
            filters = FinancialDataFilters(**filters_dict)

        return await run_financial_filters(financial_manager, filters)

    except Exception as e:
        logger.error(f"Financial query failed: {e}", exc_info=True)
        raise


async def query_and_run(
    financial_manager: Any,
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[List[FinancialDataRecord], FinancialDataFilters, Optional[Dict], List[Dict]]:
    """Parse a natural-language query into filters via the proven extractor, then run it.

    Uses FinancialDataManager.parse_user_query — the same extractor /fast_chat_v2
    relies on — instead of a hand-rolled tool-arg parse. It handles full company
    names, symbol normalization, metric synonyms, follow-up/pronoun resolution,
    and runs _post_process_filters internally, returning a finished
    FinancialDataFilters. parse_user_query is synchronous (it makes a blocking
    Gemini call), so it is run off-thread.
    """
    filters = await asyncio.to_thread(
        financial_manager.parse_user_query, query, conversation_history
    )
    return await run_financial_filters(financial_manager, filters)


async def run_financial_filters(
    financial_manager: Any,
    filters: FinancialDataFilters,
) -> Tuple[List[FinancialDataRecord], FinancialDataFilters, Optional[Dict], List[Dict]]:
    """Run an already-built FinancialDataFilters against BigQuery (off-thread).

    Returns (records, filters, chart_spec, sources). The blocking query_data call
    is wrapped in asyncio.to_thread to avoid blocking the event loop.
    """
    start_time = time.time()

    # Query the data (blocking BigQuery call -> off-thread)
    records = await asyncio.to_thread(financial_manager.query_data, filters)

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


def build_financial_context(records: List[FinancialDataRecord]) -> str:
    """Build a compact context string from financial records (caps at 50)."""
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
