"""Unit tests for app.financial_tool — the query_financial_data primitives."""

from unittest.mock import MagicMock

import pytest

from app.financial_tool import (
    extract_query_financial_data_call,
    get_financial_data_tool_declaration,
    get_financial_tool,
)


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
