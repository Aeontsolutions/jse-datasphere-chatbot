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
    query_and_run,
    run_financial_filters,
)
from app.models import FinancialDataFilters, FinancialDataRecord


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

    # Faithful slice of the REAL BigQuery metadata: note the empty-string company
    # entry ('') — routing "NCB" into the companies bucket collapses onto it and
    # yields garbage symbols, which is why we resolve to canonical symbols instead.
    _REAL_META = {
        "symbols": ["NCBFG", "GK", "EFRESH", "KLE"],
        "companies": ["", "NCB Financial Group Limited", "Gracekennedy Limited"],
        "associations": {
            "company_to_symbol": {
                "": ["EFRESH", "KLE"],
                "NCB Financial Group Limited": ["NCBFG"],
                "Gracekennedy Limited": ["GK"],
            },
            "symbol_to_company": {
                "NCBFG": ["NCB Financial Group Limited"],
                "GK": ["Gracekennedy Limited"],
            },
        },
    }

    def test_non_canonical_token_resolved_to_canonical_symbol(self):
        # "NCB" is not a real symbol; it must resolve to NCBFG via the company
        # name, and stay in `symbols` (NOT be routed to the broken companies path).
        manager = MagicMock()
        manager.metadata = self._REAL_META
        captured = {}
        manager._post_process_filters.side_effect = lambda d: (captured.update(d) or d)
        manager.query_data.return_value = []

        _, filters, _, _ = asyncio.run(
            execute_financial_query(manager, {"symbols": ["NCB"], "standard_items": ["revenue"]})
        )

        # Resolved before post-processing: symbols=[NCBFG], companies stays empty.
        assert captured["symbols"] == ["NCBFG"]
        assert captured["companies"] == []
        # And NOT the all-companies-leak symbols the old routing produced.
        assert "EFRESH" not in captured["symbols"]

    def test_canonical_symbol_passed_through(self):
        # A real trading symbol (GK) is kept exactly, no rewrite.
        manager = MagicMock()
        manager.metadata = self._REAL_META
        captured = {}
        manager._post_process_filters.side_effect = lambda d: (captured.update(d) or d)
        manager.query_data.return_value = []

        asyncio.run(execute_financial_query(manager, {"symbols": ["GK"]}))

        assert captured["symbols"] == ["GK"]
        assert captured["companies"] == []

    def test_unresolvable_token_preserved(self):
        # A token matching no symbol and no company name is preserved unchanged
        # (downstream exact-match will simply find nothing — no all-rows leak here).
        manager = MagicMock()
        manager.metadata = self._REAL_META
        captured = {}
        manager._post_process_filters.side_effect = lambda d: (captured.update(d) or d)
        manager.query_data.return_value = []

        asyncio.run(execute_financial_query(manager, {"symbols": ["ZZZZ"]}))

        assert captured["symbols"] == ["ZZZZ"]

    def test_no_metadata_keeps_tokens_in_symbols(self):
        # No metadata -> cannot resolve, so preserve the original token in symbols
        # (no regression to the metadata-less path).
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


def _filters(symbols=None, companies=None, years=None, items=None):
    return FinancialDataFilters(
        companies=companies or [],
        symbols=symbols or [],
        years=years or [],
        standard_items=items or [],
        interpretation="test",
        data_availability_note="",
        is_follow_up=False,
        context_used="",
    )


class TestRunFinancialFilters:
    def test_runs_query_offthread_and_builds_sources(self):
        manager = MagicMock()
        manager.query_data.return_value = [_record()]
        records, filters, chart, sources = asyncio.run(
            run_financial_filters(manager, _filters(symbols=["NCBFG"], years=["2023"]))
        )
        manager.query_data.assert_called_once()
        passed = manager.query_data.call_args.args[0]
        assert passed.symbols == ["NCBFG"]
        assert len(records) == 1
        assert sources[0]["type"] == "database"
        assert sources[0]["symbols"] == ["NCBFG"]

    def test_empty_results_no_chart(self):
        manager = MagicMock()
        manager.query_data.return_value = []
        records, _, chart, _ = asyncio.run(
            run_financial_filters(manager, _filters(symbols=["NCBFG"]))
        )
        assert records == []
        assert chart is None


class TestQueryAndRun:
    def test_uses_parse_user_query_then_runs(self):
        # parse_user_query is the proven /fast_chat_v2 extractor — it returns a
        # fully-formed, post-processed FinancialDataFilters. query_and_run must
        # use it (NOT re-parse args by hand) and pass its result to query_data.
        manager = MagicMock()
        parsed = _filters(
            symbols=["NCBFG", "GK"], companies=["NCB Financial Group Limited"], years=["2022"]
        )
        manager.parse_user_query.return_value = parsed
        manager.query_data.return_value = [_record()]

        records, filters, _, _ = asyncio.run(
            query_and_run(manager, "Compare NCB and GK 2022", conversation_history=None)
        )

        manager.parse_user_query.assert_called_once()
        # filters came from parse_user_query, unchanged
        passed = manager.query_data.call_args.args[0]
        assert passed.symbols == ["NCBFG", "GK"]
        assert filters.symbols == ["NCBFG", "GK"]
        assert len(records) == 1

    def test_passes_conversation_history_to_parser(self):
        manager = MagicMock()
        manager.parse_user_query.return_value = _filters(symbols=["NCBFG"])
        manager.query_data.return_value = []
        history = [{"role": "user", "content": "earlier"}]

        asyncio.run(query_and_run(manager, "what about 2022?", conversation_history=history))

        # the parser receives the query and the history (for pronoun/follow-up resolution)
        call = manager.parse_user_query.call_args
        assert call.args[0] == "what about 2022?"
        assert history in (list(call.args) + list(call.kwargs.values()))
