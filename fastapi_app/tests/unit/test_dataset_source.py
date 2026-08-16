"""Unit tests for dataset provenance on /fast_chat_v2.

A financial figure's honest provenance is the table it came from plus the
filters applied — not a fabricated per-number URL. See the known limitation
in the design doc: the table has no column pointing back to the statement
PDF, so a dataset source deliberately stops at the query.
"""

from unittest.mock import patch

import pytest

from app.financial_utils import FinancialDataManager
from app.models import FinancialDataFilters, SourceType


@pytest.fixture
def manager():
    with patch.object(FinancialDataManager, "__init__", lambda self: None):
        mgr = FinancialDataManager()
    mgr.project_id = "jse-proj"
    mgr.dataset = "financials"
    mgr.table = "statements"
    return mgr


@pytest.mark.unit
class TestDescribeSource:
    def test_source_is_typed_dataset(self, manager):
        source = manager.describe_source(FinancialDataFilters(), 0)
        assert source.type == SourceType.DATASET

    def test_table_is_fully_qualified(self, manager):
        source = manager.describe_source(FinancialDataFilters(), 3)
        assert source.table == "jse-proj.financials.statements"

    def test_record_count_is_carried(self, manager):
        source = manager.describe_source(FinancialDataFilters(), 42)
        assert source.record_count == 42

    def test_only_query_shaping_filters_are_included(self, manager):
        """Conversational fields describe interpretation, not what was read."""
        filters = FinancialDataFilters(
            companies=["NCB Financial Group"],
            symbols=["NCBFG"],
            years=["2023"],
            standard_items=["net_profit"],
            interpretation="user wants NCB net profit",
            is_follow_up=True,
            context_used="previous turn",
            data_availability_note="note",
            unrecognized_items=["ebitda"],
        )
        source = manager.describe_source(filters, 4)
        assert source.filters == {
            "companies": ["NCB Financial Group"],
            "symbols": ["NCBFG"],
            "years": ["2023"],
            "standard_items": ["net_profit"],
        }

    def test_title_names_the_dataset_readably(self, manager):
        source = manager.describe_source(FinancialDataFilters(), 1)
        assert source.title
        assert "jse-proj" not in source.title  # a title, not a table id

    def test_retrieved_at_is_populated(self, manager):
        source = manager.describe_source(FinancialDataFilters(), 1)
        assert source.retrieved_at is not None


@pytest.mark.unit
class TestSourceCacheRoundTrip:
    def test_source_survives_json_round_trip(self, manager):
        """Sources go through Redis as JSON; nothing in them may be unserializable."""
        import json

        from app.main import _jsonable
        from app.models import Source

        source = manager.describe_source(FinancialDataFilters(companies=["NCB"], years=["2023"]), 5)
        payload = json.loads(json.dumps(_jsonable([source])))
        restored = [Source(**item) for item in payload]
        assert restored[0].table == source.table
        assert restored[0].filters == source.filters
        assert restored[0].record_count == 5
