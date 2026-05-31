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


def _record(company="NCB Financial Group", symbol="NCB", year="2023",
            item_name="revenue", item=123456789.0):
    return FinancialDataRecord(
        company=company, symbol=symbol, year=year, standard_item=item_name,
        item=item, unit_multiplier=1, formatted_value=f"{item:,.0f}",
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
        assert passed.symbols == ["NCB"]              # uppercased
        assert passed.years == ["2023"]               # stringified
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


class TestBuildFinancialContext:
    def test_empty(self):
        assert build_financial_context([]) == "No financial data found."

    def test_groups_by_company_and_formats(self):
        ctx = build_financial_context([_record(item=5_000_000.0)])
        assert "NCB Financial Group" in ctx
        assert "revenue" in ctx
        assert "$5.00M" in ctx
