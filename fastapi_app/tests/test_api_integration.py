import pytest
from fastapi.testclient import TestClient
from fastapi_app.app.main import app
from fastapi_app.app.financial_utils import FinancialDataManager
from fastapi_app.app.models import FinancialDataRequest
import os
import csv
from collections import defaultdict

# --- Mock BigQuery client and data (reuse logic from test_financial_utils.py) ---
TEST_CSV_PATH = os.path.join(os.path.dirname(__file__), 'bq_test_data.csv')
def load_bq_test_data():
    with open(TEST_CSV_PATH, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)
class MockRow:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def __getitem__(self, key):
        return getattr(self, key)

def get_agg_field(rows, field):
    # Helper for aggregation fields
    return list(sorted(set(row[field] for row in rows if row[field])))

class MockQueryResult:
    def __init__(self, rows):
        self._rows = rows
    def result(self):
        for row in self._rows:
            yield row

class MockBQClient:
    def query(self, query, job_config=None):
        q = query.lower()
        test_data = load_bq_test_data()
        # Simulate aggregation queries for metadata
        if 'array_agg' in q and 'group by' in q:
            if 'company, array_agg(distinct symbol) as symbols' in q:
                agg = defaultdict(set)
                for row in test_data:
                    agg[row['company']].add(row['symbol'])
                rows = [MockRow(Company=k, symbols=list(v)) for k, v in agg.items()]
                return MockQueryResult(rows)
            elif 'symbol, array_agg(distinct company) as companies' in q:
                agg = defaultdict(set)
                for row in test_data:
                    agg[row['symbol']].add(row['company'])
                rows = [MockRow(Symbol=k, companies=list(v)) for k, v in agg.items()]
                return MockQueryResult(rows)
            elif 'company, array_agg(distinct cast(year as string)) as years' in q:
                agg = defaultdict(set)
                for row in test_data:
                    agg[row['company']].add(str(row['year']))
                rows = [MockRow(Company=k, years=list(v)) for k, v in agg.items()]
                return MockQueryResult(rows)
            elif 'company, array_agg(distinct standard_item) as items' in q:
                agg = defaultdict(set)
                for row in test_data:
                    agg[row['company']].add(row['standard_item'])
                rows = [MockRow(Company=k, items=list(v)) for k, v in agg.items()]
                return MockQueryResult(rows)
            elif 'cast(year as string) as year, array_agg(distinct company) as companies' in q:
                agg = defaultdict(set)
                for row in test_data:
                    agg[str(row['year'])].add(row['company'])
                rows = [MockRow(Year=k, companies=list(v)) for k, v in agg.items()]
                return MockQueryResult(rows)
            elif 'standard_item, array_agg(distinct company) as companies' in q:
                agg = defaultdict(set)
                for row in test_data:
                    agg[row['standard_item']].add(row['company'])
                rows = [MockRow(standard_item=k, companies=list(v)) for k, v in agg.items()]
                return MockQueryResult(rows)
            elif 'company, cast(year as string) as year, array_agg(distinct standard_item) as items' in q:
                agg = defaultdict(lambda: defaultdict(set))
                for row in test_data:
                    agg[row['company']][str(row['year'])].add(row['standard_item'])
                rows = []
                for company, years in agg.items():
                    for year, items in years.items():
                        rows.append(MockRow(Company=company, Year=year, items=list(items)))
                return MockQueryResult(rows)
            elif 'symbol, cast(year as string) as year, array_agg(distinct standard_item) as items' in q:
                agg = defaultdict(lambda: defaultdict(set))
                for row in test_data:
                    agg[row['symbol']][str(row['year'])].add(row['standard_item'])
                rows = []
                for symbol, years in agg.items():
                    for year, items in years.items():
                        rows.append(MockRow(Symbol=symbol, Year=year, items=list(items)))
                return MockQueryResult(rows)
            elif 'cast(year as string) as year, array_agg(distinct standard_item) as items' in q:
                agg = defaultdict(set)
                for row in test_data:
                    agg[str(row['year'])].add(row['standard_item'])
                rows = [MockRow(Year=k, items=list(v)) for k, v in agg.items()]
                return MockQueryResult(rows)
        # Simple distinct queries
        elif 'select distinct company' in q:
            companies = set(row['company'] for row in test_data)
            rows = [MockRow(Company=c) for c in companies]
            return MockQueryResult(rows)
        elif 'select distinct symbol' in q:
            symbols = set(row['symbol'] for row in test_data)
            rows = [MockRow(Symbol=s) for s in symbols]
            return MockQueryResult(rows)
        elif 'select distinct cast(year as string) as year' in q:
            years = set(str(row['year']) for row in test_data)
            rows = [MockRow(Year=y) for y in years]
            return MockQueryResult(rows)
        elif 'select distinct standard_item' in q:
            items = set(row['standard_item'] for row in test_data)
            rows = [MockRow(standard_item=i) for i in items]
            return MockQueryResult(rows)
        # Otherwise, return filtered rows for data queries
        else:
            rows = [MockRow(**{k: v for k, v in row.items()}) for row in test_data]
            # Simulate parameterized filtering
            if job_config and hasattr(job_config, 'query_parameters') and job_config.query_parameters:
                filters = {}
                for param in job_config.query_parameters:
                    # Handle both ArrayQueryParameter and scalar
                    if hasattr(param, 'value') and isinstance(param.value, (list, set, tuple)):
                        filters[param.name] = set(str(v) for v in param.value)
                    elif hasattr(param, 'parameter_value') and hasattr(param.parameter_value, 'value') and isinstance(param.parameter_value.value, (list, set, tuple)):
                        filters[param.name] = set(str(v) for v in param.parameter_value.value)
                    elif hasattr(param, 'value'):
                        filters[param.name] = {str(param.value)}
                    elif hasattr(param, 'parameter_value') and hasattr(param.parameter_value, 'value'):
                        filters[param.name] = {str(param.parameter_value.value)}
                def row_matches(row):
                    if 'companies' in filters and getattr(row, 'company', None) not in filters['companies']:
                        return False
                    if 'symbols' in filters and getattr(row, 'symbol', None) not in filters['symbols']:
                        return False
                    if 'years' in filters and str(getattr(row, 'year', None)) not in filters['years']:
                        return False
                    if 'items' in filters and getattr(row, 'standard_item', None) not in filters['items']:
                        return False
                    return True
                rows = [row for row in rows if row_matches(row)]
            return MockQueryResult(rows)
        # Default: return empty result set for any unrecognized query
        return MockQueryResult([])

@pytest.fixture(autouse=True)
def patch_bq_client(monkeypatch):
    monkeypatch.setattr('fastapi_app.app.financial_utils.bigquery.Client', lambda *a, **kw: MockBQClient())
    # Re-initialize the app's FinancialDataManager after patching
    from fastapi_app.app.main import app as fastapi_app_instance
    fastapi_app_instance.state.financial_manager = FinancialDataManager()
    yield

client = TestClient(app)

def test_fast_chat_v2_valid_query():
    payload = {
        "query": "Show me Elite Diagnostic Limited net profit for 2024",
        "memory_enabled": False,
        "conversation_history": []
    }
    response = client.post("/fast_chat_v2", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["data_found"], bool)
    assert data["record_count"] > 0

def test_financial_metadata_endpoint():
    response = client.get("/financial/metadata")
    assert response.status_code == 200
    data = response.json()
    # Check for company_to_items in the associations
    associations = data["metadata"]["associations"]
    assert "company_to_items" in associations 