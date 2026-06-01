"""Unit tests for app.financial_tool — the query_financial_data primitives."""

import asyncio
from unittest.mock import MagicMock

import pytest

from app.financial_tool import (
    build_financial_context,
    execute_financial_query,
    extract_query_financial_data_call,
    get_financial_data_tool_declaration,
    get_financial_tool,
)
from app.models import FinancialDataRecord


def _fc_response(name="query_financial_data", args=None):
    """Build a Gemini-style response carrying a single function_call part."""
    fc = MagicMock()
    fc.name = name
    fc.args = args if args is not None else {"symbols": ["NCB"]}
    part = MagicMock()
    part.function_call = fc
    content = MagicMock()
    content.parts = [part]
    candidate = MagicMock()
    candidate.content = content
    response = MagicMock()
    response.candidates = [candidate]
    return response


def _record(
    company="NCB Financial Group", symbol="NCB", year="2023", item_name="revenue", item=123456789.0
):
    return FinancialDataRecord(
        company=company,
        symbol=symbol,
        year=year,
        standard_item=item_name,
        item=item,
        unit_multiplier=1,
        formatted_value=f"{item:,.0f}",
    )


class TestToolDeclaration:
    def test_declaration_name_and_params(self):
        decl = get_financial_data_tool_declaration()
        assert decl.name == "query_financial_data"
        props = decl.parameters.properties
        assert set(props.keys()) == {"symbols", "years", "standard_items"}

    def test_get_financial_tool_wraps_declaration(self):
        tool = get_financial_tool()
        assert tool.function_declarations[0].name == "query_financial_data"


class TestExtractCall:
    def test_extracts_matching_call(self):
        fc = extract_query_financial_data_call(
            _fc_response(args={"symbols": ["NCB"], "years": ["2023"]})
        )
        assert fc is not None
        assert fc.name == "query_financial_data"

    def test_returns_none_when_no_candidates(self):
        response = MagicMock()
        response.candidates = []
        assert extract_query_financial_data_call(response) is None

    def test_returns_none_for_other_function(self):
        assert extract_query_financial_data_call(_fc_response(name="something_else")) is None


class TestExtractCallEdgeCases:
    def test_returns_none_when_content_missing(self):
        candidate = MagicMock()
        candidate.content = None
        response = MagicMock()
        response.candidates = [candidate]
        assert extract_query_financial_data_call(response) is None

    def test_returns_none_when_parts_empty(self):
        content = MagicMock()
        content.parts = []
        candidate = MagicMock()
        candidate.content = content
        response = MagicMock()
        response.candidates = [candidate]
        assert extract_query_financial_data_call(response) is None


class TestExecuteFinancialQuery:
    def test_builds_filters_and_calls_query_data(self):
        manager = MagicMock()
        manager.metadata = {}  # no associations -> skip post-processing
        manager.query_data.return_value = [_record()]

        records, filters, chart, sources = asyncio.run(
            execute_financial_query(
                manager,
                {"symbols": ["ncb"], "years": [2023], "standard_items": ["Net Profit"]},
            )
        )

        manager.query_data.assert_called_once()
        passed = manager.query_data.call_args.args[0]
        assert passed.symbols == ["NCB"]  # uppercased
        assert passed.years == ["2023"]  # stringified
        assert passed.standard_items == ["net_profit"]  # lower + underscored
        assert len(records) == 1
        assert sources[0]["type"] == "database"

    def test_empty_results_no_chart(self):
        manager = MagicMock()
        manager.metadata = {}
        manager.query_data.return_value = []
        records, filters, chart, sources = asyncio.run(
            execute_financial_query(manager, {"symbols": ["NCB"]})
        )
        assert records == []
        assert chart is None

    def test_non_canonical_symbol_routed_to_companies(self):
        # The tool only exposes `symbols`, but the model often emits a colloquial
        # company token ("NCB") that is not a canonical JSE symbol ("NCBFG").
        # When metadata + associations are present, such tokens must be routed to
        # the `companies` bucket (fragment-matched by _post_process_filters), NOT
        # left in `symbols` (exact-matched -> dropped -> WHERE 1=1 over all rows).
        manager = MagicMock()
        manager.metadata = {
            "symbols": ["NCBFG", "GK"],
            "associations": {"symbol_to_company": {}},
        }
        captured = {}
        manager._post_process_filters.side_effect = lambda d: (captured.update(d) or d)
        manager.query_data.return_value = []

        asyncio.run(
            execute_financial_query(manager, {"symbols": ["NCB"], "standard_items": ["revenue"]})
        )

        assert captured["symbols"] == []  # NCB is not a known symbol
        assert captured["companies"] == ["NCB"]  # routed here for fragment match

    def test_canonical_symbol_kept_in_symbols(self):
        # A real trading symbol (GK) must stay in `symbols`, not get moved.
        manager = MagicMock()
        manager.metadata = {
            "symbols": ["NCBFG", "GK"],
            "associations": {"symbol_to_company": {}},
        }
        captured = {}
        manager._post_process_filters.side_effect = lambda d: (captured.update(d) or d)
        manager.query_data.return_value = []

        asyncio.run(execute_financial_query(manager, {"symbols": ["GK"]}))

        assert captured["symbols"] == ["GK"]
        assert captured["companies"] == []

    def test_no_metadata_keeps_tokens_in_symbols(self):
        # No metadata/associations -> cannot resolve companies, so DO NOT reroute;
        # preserve the original behavior (token stays in symbols). No regression.
        manager = MagicMock()
        manager.metadata = {}
        manager.query_data.return_value = []

        _, filters, _, _ = asyncio.run(execute_financial_query(manager, {"symbols": ["NCB"]}))

        assert filters.symbols == ["NCB"]
        assert filters.companies == []


class TestBuildFinancialContext:
    def test_empty(self):
        assert build_financial_context([]) == "No financial data found."

    def test_groups_by_company_and_formats(self):
        ctx = build_financial_context([_record(item=5_000_000.0)])
        assert "NCB Financial Group" in ctx
        assert "revenue" in ctx
        assert "$5.00M" in ctx
