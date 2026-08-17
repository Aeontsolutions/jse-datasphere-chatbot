"""Unit tests for dataset provenance on /fast_chat_v2.

A financial figure's honest provenance is the table it came from plus a
description of the rows read — not a fabricated per-number URL. See the known
limitation in the design doc: the table has no column pointing back to the
statement PDF, so a dataset source deliberately stops at the query.

`filters` describes the rows actually returned rather than the parser's
filters. Those diverge badly in practice: `_post_process_filters` expands a
company into its symbols, and the source table holds thousands of rows with a
blank Company, so a blank value matches every symbol in the dataset. A
single-company query reports ~100 symbols in `filters_used` — accurate as a
record of the SQL, misleading as a citation.
"""

from unittest.mock import patch

import pytest

from app.financial_utils import FinancialDataManager
from app.models import FinancialDataRecord, SourceType


@pytest.fixture
def manager():
    with patch.object(FinancialDataManager, "__init__", lambda self: None):
        mgr = FinancialDataManager()
    mgr.project_id = "jse-proj"
    mgr.dataset = "financials"
    mgr.table = "statements"
    return mgr


def _record(
    company="NCB Financial Group Limited",
    symbol="NCBFG",
    year="2023",
    item_name="net_profit",
    value=2507191000.0,
):
    return FinancialDataRecord(
        company=company,
        symbol=symbol,
        year=year,
        standard_item=item_name,
        item=value,
        unit_multiplier=1,
        formatted_value="2.51B",
    )


@pytest.mark.unit
class TestDescribeSource:
    def test_source_is_typed_dataset(self, manager):
        assert manager.describe_source([]).type == SourceType.DATASET

    def test_table_is_fully_qualified(self, manager):
        assert manager.describe_source([_record()]).table == "jse-proj.financials.statements"

    def test_record_count_matches_rows_read(self, manager):
        assert manager.describe_source([_record(), _record(year="2022")]).record_count == 2

    def test_filters_describe_the_rows_returned(self, manager):
        source = manager.describe_source([_record()])
        assert source.filters == {
            "companies": ["NCB Financial Group Limited"],
            "symbols": ["NCBFG"],
            "years": ["2023"],
            "standard_items": ["net_profit"],
        }

    def test_values_are_deduped_and_sorted(self, manager):
        source = manager.describe_source(
            [
                _record(year="2023"),
                _record(year="2022"),
                _record(year="2023"),
                _record(company="GraceKennedy Limited", symbol="GK", year="2024"),
            ]
        )
        assert source.filters["years"] == ["2022", "2023", "2024"]
        assert source.filters["symbols"] == ["GK", "NCBFG"]
        assert source.filters["companies"] == [
            "GraceKennedy Limited",
            "NCB Financial Group Limited",
        ]

    def test_blank_values_are_excluded(self, manager):
        """The source table holds thousands of blank-Company rows; a blank is
        never meaningful provenance."""
        source = manager.describe_source([_record(), _record(company="", symbol="   ")])
        assert source.filters["companies"] == ["NCB Financial Group Limited"]
        assert source.filters["symbols"] == ["NCBFG"]

    def test_does_not_report_the_whole_symbol_universe(self, manager):
        """Regression guard for the bug this replaced: a one-company answer
        must not claim ~100 symbols just because the parser expanded them."""
        source = manager.describe_source([_record()])
        assert source.filters["symbols"] == ["NCBFG"]
        assert len(source.filters["symbols"]) == 1

    def test_empty_results_produce_empty_filters(self, manager):
        source = manager.describe_source([])
        assert source.record_count == 0
        assert source.filters == {
            "companies": [],
            "symbols": [],
            "years": [],
            "standard_items": [],
        }

    def test_title_names_the_dataset_readably(self, manager):
        source = manager.describe_source([_record()])
        assert source.title
        assert "jse-proj" not in source.title  # a title, not a table id

    def test_retrieved_at_is_populated(self, manager):
        assert manager.describe_source([_record()]).retrieved_at is not None


@pytest.mark.unit
class TestSourceCacheRoundTrip:
    def test_source_survives_json_round_trip(self, manager):
        """Sources go through Redis as JSON; nothing in them may be unserializable."""
        import json

        from app.main import _jsonable
        from app.models import Source

        source = manager.describe_source([_record(), _record(year="2022")])
        payload = json.loads(json.dumps(_jsonable([source])))
        restored = [Source(**item) for item in payload]
        assert restored[0].table == source.table
        assert restored[0].filters == source.filters
        assert restored[0].record_count == 2
