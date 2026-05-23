import csv
import logging
import os
from collections import defaultdict

import pytest

from app.financial_utils import FinancialDataManager
from app.models import FinancialDataFilters

# Helper to load CSV as list of dicts
TEST_CSV_PATH = os.path.join(os.path.dirname(__file__), "bq_test_data.csv")


def load_bq_test_data():
    with open(TEST_CSV_PATH, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)


# Helper to simulate aggregation results
class MockRow:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# Fixture to patch BigQuery client and return CSV data or aggregation results
@pytest.fixture
def mock_bq_client(monkeypatch):
    test_data = load_bq_test_data()

    class MockQueryResult:
        def __init__(self, rows):
            self._rows = rows

        def result(self):
            for row in self._rows:
                yield row

    class MockBQClient:
        def query(self, query, job_config=None):
            q = query.lower()
            # Simulate aggregation queries for metadata
            if "array_agg" in q and "group by" in q:
                if "company, array_agg(distinct symbol) as symbols" in q:
                    # company_to_symbol
                    agg = defaultdict(set)
                    for row in test_data:
                        agg[row["company"]].add(row["symbol"])
                    rows = [MockRow(Company=k, symbols=list(v)) for k, v in agg.items()]
                    return MockQueryResult(rows)
                elif "symbol, array_agg(distinct company) as companies" in q:
                    # symbol_to_company
                    agg = defaultdict(set)
                    for row in test_data:
                        agg[row["symbol"]].add(row["company"])
                    rows = [MockRow(Symbol=k, companies=list(v)) for k, v in agg.items()]
                    return MockQueryResult(rows)
                elif "company, array_agg(distinct cast(year as string)) as years" in q:
                    # company_to_years
                    agg = defaultdict(set)
                    for row in test_data:
                        agg[row["company"]].add(str(row["year"]))
                    rows = [MockRow(Company=k, years=list(v)) for k, v in agg.items()]
                    return MockQueryResult(rows)
                elif "company, array_agg(distinct standard_item) as items" in q:
                    # company_to_items
                    agg = defaultdict(set)
                    for row in test_data:
                        agg[row["company"]].add(row["standard_item"])
                    rows = [MockRow(Company=k, items=list(v)) for k, v in agg.items()]
                    return MockQueryResult(rows)
                elif "cast(year as string) as year, array_agg(distinct company) as companies" in q:
                    # year_to_companies
                    agg = defaultdict(set)
                    for row in test_data:
                        agg[str(row["year"])].add(row["company"])
                    rows = [MockRow(Year=k, companies=list(v)) for k, v in agg.items()]
                    return MockQueryResult(rows)
                elif "standard_item, array_agg(distinct company) as companies" in q:
                    # item_to_companies
                    agg = defaultdict(set)
                    for row in test_data:
                        agg[row["standard_item"]].add(row["company"])
                    rows = [MockRow(standard_item=k, companies=list(v)) for k, v in agg.items()]
                    return MockQueryResult(rows)
                elif (
                    "company, cast(year as string) as year, array_agg(distinct standard_item) as items"
                    in q
                ):
                    # company_year_to_items
                    agg = defaultdict(lambda: defaultdict(set))
                    for row in test_data:
                        agg[row["company"]][str(row["year"])].add(row["standard_item"])
                    rows = []
                    for company, years in agg.items():
                        for year, items in years.items():
                            rows.append(MockRow(Company=company, Year=year, items=list(items)))
                    return MockQueryResult(rows)
                elif (
                    "symbol, cast(year as string) as year, array_agg(distinct standard_item) as items"
                    in q
                ):
                    # symbol_year_to_items
                    agg = defaultdict(lambda: defaultdict(set))
                    for row in test_data:
                        agg[row["symbol"]][str(row["year"])].add(row["standard_item"])
                    rows = []
                    for symbol, years in agg.items():
                        for year, items in years.items():
                            rows.append(MockRow(Symbol=symbol, Year=year, items=list(items)))
                    return MockQueryResult(rows)
                elif (
                    "cast(year as string) as year, array_agg(distinct standard_item) as items" in q
                ):
                    # year_to_items
                    agg = defaultdict(set)
                    for row in test_data:
                        agg[str(row["year"])].add(row["standard_item"])
                    rows = [MockRow(Year=k, items=list(v)) for k, v in agg.items()]
                    return MockQueryResult(rows)
            # Simple distinct queries
            elif "select distinct company" in q:
                companies = {row["company"] for row in test_data}
                rows = [MockRow(Company=c) for c in companies]
                return MockQueryResult(rows)
            elif "select distinct symbol" in q:
                symbols = {row["symbol"] for row in test_data}
                rows = [MockRow(Symbol=s) for s in symbols]
                return MockQueryResult(rows)
            elif "select distinct cast(year as string) as year" in q:
                years = {str(row["year"]) for row in test_data}
                rows = [MockRow(Year=y) for y in years]
                return MockQueryResult(rows)
            elif "select distinct standard_item" in q:
                items = {row["standard_item"] for row in test_data}
                rows = [MockRow(standard_item=i) for i in items]
                return MockQueryResult(rows)
            # Otherwise, return filtered rows for data queries
            else:
                rows = [MockRow(**dict(row.items())) for row in test_data]
                # Simulate parameterized filtering
                if (
                    job_config
                    and hasattr(job_config, "query_parameters")
                    and job_config.query_parameters
                ):
                    filters = {}
                    for param in job_config.query_parameters:
                        # param.name: companies, symbols, years, items
                        # param.array_type: STRING
                        # param.value: list of values
                        filters[param.name] = {str(v) for v in param.value}

                    def row_matches(row):
                        # company
                        if "companies" in filters and row.company not in filters["companies"]:
                            return False
                        if "symbols" in filters and row.symbol not in filters["symbols"]:
                            return False
                        if "years" in filters and str(row.year) not in filters["years"]:
                            return False
                        if "items" in filters and row.standard_item not in filters["items"]:
                            return False
                        return True

                    rows = [row for row in rows if row_matches(row)]
                return MockQueryResult(rows)

    monkeypatch.setattr(
        "fastapi_app.app.financial_utils.bigquery.Client", lambda *args, **kwargs: MockBQClient()
    )
    return MockBQClient()


def test_initialization_and_metadata_loading(mock_bq_client):
    manager = FinancialDataManager()
    assert manager.metadata is not None
    assert "companies" in manager.metadata
    assert len(manager.metadata["companies"]) > 0


def test_query_data_with_filters(mock_bq_client):
    manager = FinancialDataManager()
    filters = FinancialDataFilters(
        companies=["Elite Diagnostic Limited"],
        symbols=["ELITE"],
        years=["2024"],
        standard_items=["net_profit"],
    )
    results = manager.query_data(filters)
    assert isinstance(results, list)
    assert all(r.company == "Elite Diagnostic Limited" for r in results)
    assert all(r.symbol == "ELITE" for r in results)
    assert all(r.year == "2024" for r in results)
    assert all(r.standard_item == "net_profit" for r in results)


def test_query_data_no_filters_returns_all(mock_bq_client):
    manager = FinancialDataManager()
    filters = FinancialDataFilters()
    results = manager.query_data(filters)
    assert isinstance(results, list)
    assert len(results) > 0


def test_bigquery_client_init_error(monkeypatch, caplog):
    """Test that FinancialDataManager logs and handles BigQuery client init errors."""

    def raise_init(*args, **kwargs):
        raise Exception("BigQuery init failed!")

    monkeypatch.setattr("fastapi_app.app.financial_utils.bigquery.Client", raise_init)
    with caplog.at_level(logging.ERROR):
        manager = FinancialDataManager()
        assert manager.bq_client is None or not hasattr(manager, "bq_client")
        assert any(
            "BigQuery" in r.message and "failed" in r.message.lower() for r in caplog.records
        )


def test_query_data_query_error(monkeypatch, caplog):
    """Test that query_data logs and handles query errors."""

    class MockBQClient:
        def query(self, *a, **k):
            raise Exception("Query failed!")

    monkeypatch.setattr(
        "fastapi_app.app.financial_utils.bigquery.Client", lambda *a, **k: MockBQClient()
    )
    manager = FinancialDataManager()
    with caplog.at_level(logging.ERROR):
        filters = FinancialDataFilters()
        results = manager.query_data(filters)
        assert results == []
        assert any("Error querying BigQuery" in r.message for r in caplog.records)


def test_query_data_missing_data(monkeypatch, caplog):
    """Test that query_data returns empty list and logs info when no data is found."""

    class MockQueryResult:
        def result(self):
            return iter([])

    class MockBQClient:
        def query(self, *a, **k):
            return MockQueryResult()

    monkeypatch.setattr(
        "fastapi_app.app.financial_utils.bigquery.Client", lambda *a, **k: MockBQClient()
    )
    manager = FinancialDataManager()
    with caplog.at_level(logging.INFO):
        filters = FinancialDataFilters()
        results = manager.query_data(filters)
        assert results == []
        # Not always an error, but should log info about 0 records
        assert any(
            "returned 0 records" in r.message.lower() or "no data" in r.message.lower()
            for r in caplog.records
        )


# --- SECURITY TEST: Only SELECT and Parameterized Queries ---
def test_only_select_and_parameterized_queries(monkeypatch):
    """Test that only SELECT queries are executed and all queries are parameterized (no SQL injection risk)."""
    select_queries = []
    param_checks = []

    class MockJobConfig:
        def __init__(self, query_parameters=None):
            self.query_parameters = query_parameters or []

    class MockQueryResult:
        def result(self):
            return []

    class MockBQClient:
        def query(self, query, job_config=None):
            # Assert only SELECT queries are executed
            assert query.strip().lower().startswith("select"), f"Non-SELECT query executed: {query}"
            select_queries.append(query)
            # If filters are present, ensure parameterization is used
            if any(word in query for word in ["@companies", "@symbols", "@years", "@items"]):
                assert (
                    job_config is not None
                ), "Query with filters must use job_config for parameters"
                assert hasattr(
                    job_config, "query_parameters"
                ), "job_config must have query_parameters"
                # Ensure no user input is interpolated directly
                for param in job_config.query_parameters:
                    assert isinstance(param.name, str)
                    assert isinstance(param.value, list)
                param_checks.append(True)
            return MockQueryResult()

    monkeypatch.setattr(
        "fastapi_app.app.financial_utils.bigquery.Client", lambda *a, **k: MockBQClient()
    )
    monkeypatch.setattr(
        "fastapi_app.app.financial_utils.bigquery.ArrayQueryParameter",
        lambda name, typ, value: type("P", (), {"name": name, "type": typ, "value": value})(),
    )
    monkeypatch.setattr("fastapi_app.app.financial_utils.bigquery.QueryJobConfig", MockJobConfig)

    from fastapi_app.app.financial_utils import FinancialDataManager
    from fastapi_app.app.models import FinancialDataFilters

    manager = FinancialDataManager()
    # Test with filters (should use parameterized query)
    filters = FinancialDataFilters(
        companies=["TestCo"], symbols=["TCO"], years=["2024"], standard_items=["revenue"]
    )
    manager.query_data(filters)
    # Test with no filters (should still be SELECT)
    filters = FinancialDataFilters()
    manager.query_data(filters)
    # Ensure at least one parameterized query was checked
    assert param_checks, "No parameterized queries were checked."
    # Ensure all queries were SELECT
    assert all(q.strip().lower().startswith("select") for q in select_queries)


# --- LLM PARSING TEST: Dedicated test for parse_user_query ---
def test_parse_user_query_llm(monkeypatch, mock_bq_client):
    """Test that parse_user_query correctly parses a user query into FinancialDataFilters using the LLM (mocked)."""
    from fastapi_app.app.financial_utils import FinancialDataManager
    from fastapi_app.app.models import FinancialDataFilters

    # Prepare a mock LLM model
    class MockLLM:
        def generate_content(self, prompt):
            class Response:
                # Simulate a realistic LLM output for a query about Elite Diagnostic Limited's revenue in 2024
                text = '{"companies": ["Elite Diagnostic Limited"], "symbols": ["ELITE"], "years": ["2024"], "standard_items": ["revenue"], "interpretation": "Elite Diagnostic Limited revenue for 2024", "data_availability_note": "", "is_follow_up": false, "context_used": ""}'

            return Response()

    # Patch the model initialization to use the mock
    monkeypatch.setattr(
        FinancialDataManager, "_initialize_ai_model", lambda self: setattr(self, "model", MockLLM())
    )

    # Instantiate manager (will use mock LLM and mock metadata)
    manager = FinancialDataManager()
    manager.model = MockLLM()  # Ensure model is set

    # Simulate metadata as would be loaded from BigQuery
    manager.metadata = {
        "companies": [
            "Elite Diagnostic Limited",
            "Dolla Financial Services Limited",
            "Future Energy Source Company Limited",
        ],
        "symbols": ["ELITE", "DOLLA", "FESCO"],
        "years": ["2024", "2023", "2022"],
        "standard_items": ["revenue", "net_profit", "gross_profit"],
        "associations": {
            "company_to_symbol": {"Elite Diagnostic Limited": ["ELITE"]},
            "symbol_to_company": {"ELITE": ["Elite Diagnostic Limited"]},
        },
    }

    # Test a realistic user query
    query = "Show me Elite Diagnostic Limited revenue for 2024"
    filters = manager.parse_user_query(query)

    # Assert the parsed filters match expected output
    assert isinstance(filters, FinancialDataFilters)
    assert filters.companies == ["Elite Diagnostic Limited"]
    assert filters.symbols == ["ELITE"]
    assert filters.years == ["2024"]
    assert filters.standard_items == ["revenue"]
    assert filters.is_follow_up is False
    assert "Elite Diagnostic Limited" in filters.interpretation


# --- LLM PARSING TEST: Multiple companies and symbols ---
def test_parse_user_query_llm_multiple_companies_symbols(monkeypatch, mock_bq_client):
    """Test that parse_user_query correctly parses a query with multiple companies and symbols."""
    from fastapi_app.app.financial_utils import FinancialDataManager
    from fastapi_app.app.models import FinancialDataFilters

    # Prepare a mock LLM model
    class MockLLM:
        def generate_content(self, prompt):
            class Response:
                # Simulate LLM output for a query about two companies, two symbols, two years, and revenue
                text = '{"companies": ["Elite Diagnostic Limited", "Dolla Financial Services Limited"], "symbols": ["ELITE", "DOLLA"], "years": ["2023", "2024"], "standard_items": ["revenue"], "interpretation": "Compare Elite Diagnostic Limited and Dolla Financial Services Limited revenue for 2023 and 2024", "data_availability_note": "", "is_follow_up": false, "context_used": ""}'

            return Response()

    monkeypatch.setattr(
        FinancialDataManager, "_initialize_ai_model", lambda self: setattr(self, "model", MockLLM())
    )

    manager = FinancialDataManager()
    manager.model = MockLLM()
    manager.metadata = {
        "companies": [
            "Elite Diagnostic Limited",
            "Dolla Financial Services Limited",
            "Future Energy Source Company Limited",
        ],
        "symbols": ["ELITE", "DOLLA", "FESCO"],
        "years": ["2024", "2023", "2022"],
        "standard_items": ["revenue", "net_profit", "gross_profit"],
        "associations": {
            "company_to_symbol": {
                "Elite Diagnostic Limited": ["ELITE"],
                "Dolla Financial Services Limited": ["DOLLA"],
            },
            "symbol_to_company": {
                "ELITE": ["Elite Diagnostic Limited"],
                "DOLLA": ["Dolla Financial Services Limited"],
            },
        },
    }

    query = "Compare Elite Diagnostic Limited and Dolla Financial Services Limited revenue for 2023 and 2024"
    filters = manager.parse_user_query(query)

    assert isinstance(filters, FinancialDataFilters)
    assert set(filters.companies) == {
        "Elite Diagnostic Limited",
        "Dolla Financial Services Limited",
    }
    assert set(filters.symbols) == {"ELITE", "DOLLA"}
    assert set(filters.years) == {"2023", "2024"}
    assert filters.standard_items == ["revenue"]
    assert filters.is_follow_up is False
    assert "Elite Diagnostic Limited" in filters.interpretation
    assert "Dolla Financial Services Limited" in filters.interpretation


# --- LLM PARSING TEST: Symbol-only query (should infer company) ---
def test_parse_user_query_llm_symbol_only(monkeypatch, mock_bq_client):
    """Test that parse_user_query correctly infers the company from a symbol-only query."""
    from fastapi_app.app.financial_utils import FinancialDataManager
    from fastapi_app.app.models import FinancialDataFilters

    # Prepare a mock LLM model
    class MockLLM:
        def generate_content(self, prompt):
            class Response:
                # Simulate LLM output for a query about ELITE's revenue for 2024, only symbol provided
                text = '{"companies": [], "symbols": ["ELITE"], "years": ["2024"], "standard_items": ["revenue"], "interpretation": "ELITE revenue for 2024", "data_availability_note": "", "is_follow_up": false, "context_used": ""}'

            return Response()

    monkeypatch.setattr(
        FinancialDataManager, "_initialize_ai_model", lambda self: setattr(self, "model", MockLLM())
    )

    manager = FinancialDataManager()
    manager.model = MockLLM()
    manager.metadata = {
        "companies": [
            "Elite Diagnostic Limited",
            "Dolla Financial Services Limited",
            "Future Energy Source Company Limited",
        ],
        "symbols": ["ELITE", "DOLLA", "FESCO"],
        "years": ["2024", "2023", "2022"],
        "standard_items": ["revenue", "net_profit", "gross_profit"],
        "associations": {
            "company_to_symbol": {"Elite Diagnostic Limited": ["ELITE"]},
            "symbol_to_company": {"ELITE": ["Elite Diagnostic Limited"]},
        },
    }

    query = "Show me ELITE revenue for 2024"
    filters = manager.parse_user_query(query)

    assert isinstance(filters, FinancialDataFilters)
    assert filters.symbols == ["ELITE"]
    # Company should be inferred from symbol
    assert filters.companies == ["Elite Diagnostic Limited"]
    assert filters.years == ["2024"]
    assert filters.standard_items == ["revenue"]
    assert filters.is_follow_up is False
    assert "ELITE" in filters.interpretation


# ── ATS-325: ratio / dividend unit formatting tests ────────────────────────────


class _RatioRow:
    """Minimal BQ row stub for formatting tests."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _manager_with_ratio_rows(monkeypatch, rows):
    """Return a FinancialDataManager whose BQ client yields the given rows."""

    class _Result:
        def result(self):
            return iter(rows)

    class _Client:
        def query(self, *a, **kw):
            return _Result()

    client = _Client()

    def _fake_init_bq(self):
        self.bq_client = client

    monkeypatch.setattr(FinancialDataManager, "_initialize_bigquery_client", _fake_init_bq)
    monkeypatch.setattr(FinancialDataManager, "load_metadata_from_bigquery", lambda self: None)
    manager = FinancialDataManager()
    manager.metadata = {
        "companies": [],
        "symbols": [],
        "years": [],
        "standard_items": [],
        "associations": {},
    }
    return manager


def _ratio_row(**overrides):
    defaults = {
        "Company": "Test Company",
        "Symbol": "TST",
        "Year": 2023,
        "standard_item": "dividend_per_share",
        "item": 1.0,
        "unit_multiplier": 1.0,
        "item_type": "ratio",
        "item_name": "test_item",
    }
    defaults.update(overrides)
    return _RatioRow(**defaults)


def test_dividend_per_share_formats_as_currency(monkeypatch):
    """CCC dividend_per_share 1.8976 → J$1.90, not 1.90% (ATS-325)."""
    row = _ratio_row(
        Company="Caribbean Cement Company Limited",
        Symbol="CCC",
        standard_item="dividend_per_share",
        item=1.8976,
    )
    manager = _manager_with_ratio_rows(monkeypatch, [row])
    results = manager.query_data(FinancialDataFilters())
    assert len(results) == 1
    assert results[0].formatted_value == "J$1.90", results[0].formatted_value


def test_eps_formats_as_currency(monkeypatch):
    """EPS 2.85 → J$2.85 (ATS-325)."""
    row = _ratio_row(standard_item="eps", item=2.85)
    manager = _manager_with_ratio_rows(monkeypatch, [row])
    results = manager.query_data(FinancialDataFilters())
    assert results[0].formatted_value == "J$2.85", results[0].formatted_value


def test_dividend_payout_ratio_fraction_multiplied(monkeypatch):
    """GK dividend_payout_ratio 0.343 (fraction) → 34.30%, not 0.34% (ATS-325)."""
    row = _ratio_row(
        Company="Gracekennedy Limited",
        Symbol="GK",
        standard_item="dividend_payout_ratio",
        item=0.343,
    )
    manager = _manager_with_ratio_rows(monkeypatch, [row])
    results = manager.query_data(FinancialDataFilters())
    assert results[0].formatted_value == "34.30%", results[0].formatted_value


def test_dividend_payout_ratio_already_percent(monkeypatch):
    """BPOW dividend_payout_ratio 21.41 (already %) → 21.41%, not 2141% (ATS-325)."""
    row = _ratio_row(
        Company="Blue Power Group Limited",
        Symbol="BPOW",
        standard_item="dividend_payout_ratio",
        item=21.41,
    )
    manager = _manager_with_ratio_rows(monkeypatch, [row])
    results = manager.query_data(FinancialDataFilters())
    assert results[0].formatted_value == "21.41%", results[0].formatted_value


def test_current_ratio_no_percent_suffix(monkeypatch):
    """current_ratio 1.11 → '1.11' (plain decimal, no %) (ATS-325)."""
    row = _ratio_row(standard_item="current_ratio", item=1.11)
    manager = _manager_with_ratio_rows(monkeypatch, [row])
    results = manager.query_data(FinancialDataFilters())
    assert results[0].formatted_value == "1.11", results[0].formatted_value


def test_debt_to_equity_ratio_no_percent_suffix(monkeypatch):
    """debt_to_equity_ratio 1.06 → '1.06' (plain decimal, no %) (ATS-325)."""
    row = _ratio_row(standard_item="debt_to_equity_ratio", item=1.06)
    manager = _manager_with_ratio_rows(monkeypatch, [row])
    results = manager.query_data(FinancialDataFilters())
    assert results[0].formatted_value == "1.06", results[0].formatted_value


def test_unknown_ratio_keeps_percent_suffix(monkeypatch):
    """Unknown ratio items still get % appended (default behaviour preserved)."""
    row = _ratio_row(standard_item="some_other_ratio", item=0.75)
    manager = _manager_with_ratio_rows(monkeypatch, [row])
    results = manager.query_data(FinancialDataFilters())
    assert results[0].formatted_value == "0.75%", results[0].formatted_value


def test_non_ratio_item_type_unaffected(monkeypatch):
    """Items with item_type != 'ratio' are not touched by the ratio branch."""
    row = _ratio_row(
        standard_item="revenue", item=1000.0, unit_multiplier=1000000.0, item_type="currency"
    )
    manager = _manager_with_ratio_rows(monkeypatch, [row])
    results = manager.query_data(FinancialDataFilters())
    # 1000 * 1_000_000 = 1_000_000_000 → "1.00B" (hits the >= 1e9 branch)
    assert "B" in results[0].formatted_value, results[0].formatted_value
