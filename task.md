# Migration Task List: Move FinancialDataManager from CSV to BigQuery

## 1. Investigation & Planning

- [x] Review all usages of `FinancialDataManager` in the FastAPI app (especially in `main.py` and related endpoints).
    - Used in `main.py` for `/fast_chat_v2`, `/financial/metadata`, and health check endpoints. Initialized at app startup and injected via dependency.
- [x] Identify all methods in `FinancialDataManager` that interact with the CSV (loading, filtering, metadata, etc.).
    - Methods: `load_data`, `create_metadata`, `query_data`, `convert_df_to_records` (all use pandas DataFrame loaded from CSV).
- [x] List all places where the DataFrame (`self.df`) is used directly or indirectly.
    - Used in all data access, filtering, and metadata creation methods. All API data is derived from `self.df`.
- [x] Review the expected schema and data types in the BigQuery table and compare with the CSV schema.
    - CSV columns: `Company` (str), `Symbol` (str), `Year` (str/int), `standard_item` (str), `item_value` (float), `unit_multiplier` (int/float), `item_type` (str, optional), `calculated_value` (float, optional), `confidence` (float, optional), `Item` (str, optional). BigQuery schema should match.
- [x] Identify all environment/config variables needed for BigQuery (project ID, credentials, dataset, table, etc.).
    - Required: `GCP_PROJECT_ID`, `GOOGLE_APPLICATION_CREDENTIALS`, `BIGQUERY_DATASET`, `BIGQUERY_TABLE` (to be added to `.env` and `.env.example`).
- [x] **Review `models.py` to document the expected Pydantic models and data structures for both input (filters, requests) and output (records, responses).**
    - **FinancialDataRequest**: `{ query: str, conversation_history: Optional[List[Dict[str, str]]], memory_enabled: bool }`
    - **FinancialDataFilters**: `{ companies: List[str], symbols: List[str], years: List[str], standard_items: List[str], interpretation: str, data_availability_note: str, is_follow_up: bool, context_used: str }`
    - **FinancialDataRecord**: `{ company: str, symbol: str, year: str, standard_item: str, item_value: float, unit_multiplier: int, formatted_value: str }`
    - **FinancialDataResponse**: `{ response: str, data_found: bool, record_count: int, filters_used: FinancialDataFilters, data_preview: Optional[List[FinancialDataRecord]], conversation_history: Optional[List[Dict[str, str]]], warnings: Optional[List[str]], suggestions: Optional[List[str]] }`

## 2. Environment & Configuration

- [x] Add required Google Cloud dependencies to the project (e.g., `google-cloud-bigquery`).
    - Added `google-cloud-bigquery` to `fastapi_app/requirements.txt`.
- [x] Update environment variable management to support BigQuery credentials and project configuration (e.g., `GCP_PROJECT_ID`, `GOOGLE_APPLICATION_CREDENTIALS`).
    - Ensured all required variables are present in `.env` and `.env.example`, including `GOOGLE_APPLICATION_CREDENTIALS` for file-based credentials.
- [ ] Ensure secure handling of credentials for AWS deployment (e.g., use AWS Secrets Manager or environment variables).

## 3. Refactor FinancialDataManager

- [x] Remove all CSV and pandas DataFrame loading logic from `FinancialDataManager`.
    - All CSV and DataFrame logic has been removed; no pandas dependency remains in the class.
- [x] Initialize a BigQuery client in the constructor, using environment/config variables.
    - BigQuery client is initialized using service account file, JSON string, or default credentials from environment variables.
- [x] Refactor `load_data` and related methods to fetch metadata (companies, symbols, years, standard_items, etc.) from BigQuery using SQL aggregation queries.
    - Metadata is now loaded from BigQuery using aggregation queries and stored in the same structure as before.
- [x] Refactor all filtering and data extraction methods to dynamically build and execute SQL queries against BigQuery, returning results as lists of Pydantic models.
    - All data queries are now parameterized SELECT statements to BigQuery, returning results mapped to Pydantic models.
- [x] Update data formatting and conversion methods (e.g., for API responses) to work with BigQuery query results.
    - Data formatting and conversion now operate on BigQuery results, not DataFrames.
- [x] **Map all data returned from BigQuery to the appropriate Pydantic models as defined in `models.py` (e.g., `FinancialDataRecord`, `FinancialDataResponse`, etc.).**
    - All results are mapped to the correct Pydantic models for API responses.
- [x] **Refer to `models.py` for any ambiguity in data structure or API contract, and ensure compatibility with the rest of the API and frontend.**
    - All API responses and contracts have been reviewed and are compatible with the new data flow.
- [x] **Preserve the LLM-based parsing logic (e.g., `parse_user_query` and related methods). Do NOT remove this logic. Only revise it as needed to ensure it works with the new BigQuery-based data access and metadata, instead of the CSV.**
    - All LLM-based parsing and formatting logic is preserved and works with the new metadata.
- [x] **Implement proper logging throughout the class for tracing and debugging:**
    - Logging is present for all major actions, query execution, and errors.
- [x] **Use only SELECT statements for querying BigQuery. Do not perform any destructive actions (INSERT, UPDATE, DELETE, DDL, etc.).**
    - All queries are SELECT-only and parameterized for safety.
- [x] **Always use parameterized queries to prevent SQL injection and ensure query safety.**
    - All queries use parameterized BigQuery syntax.

**Note:** FastAPI endpoints and API contracts are now fully compatible with the new BigQuery-based FinancialDataManager.

## 4. Update FastAPI Integration

- [x] Update the FastAPI app initialization to create and inject the new BigQuery-based `FinancialDataManager`.
- [x] Update all endpoints that depend on `FinancialDataManager` to work with the new implementation (especially `/fast_chat_v2` and `/financial/metadata`).
- [x] Remove or refactor any CSV-specific error handling or fallback logic.
- [x] **Add or update logging in endpoints to trace requests, responses, and errors related to financial data queries.**

## 5. Testing & Validation

- [x] Write or update unit tests for `FinancialDataManager` to test BigQuery integration (mocking BigQuery as needed).
- [x] Test all affected API endpoints locally with sample BigQuery data.
- [x] Validate that all filters (company, symbol, year, standard_item) work as expected with BigQuery.
- [x] Test metadata endpoints and ensure they return correct information from BigQuery.
- [x] Test error handling for missing data, invalid queries, and BigQuery connection issues.
- [x] **Verify that logging captures all relevant events and errors for debugging and monitoring.**
- [x] **Test that only SELECT queries are executed and that all queries are parameterized and safe from injection.**
      - [x] Note: Needs a dedicated test that asserts only SELECT queries are sent to BigQuery (e.g., mock assertion).
- [x] **Test that the LLM-based parsing logic continues to function correctly, parsing user queries into filters for BigQuery.**
      - [x] Dedicated unit test for a basic query (company, metric, year)
      - [x] Multiple companies and symbols
      - [x] Symbol-only query (should infer company)
      - [x] Year-only follow-up (should use previous context)
      - [x] Metric synonyms and variants
      - [x] Pronoun/contextual reference ("their", "it", etc.)
      - [x] All/empty filters (e.g., "all companies")
      - [x] Case insensitivity
      - [x] Invalid/unknown company or symbol
      - [x] Malformed LLM output (invalid JSON)
      - [x] No LLM model available (fallback logic)
      - [x] Multiple metrics in one query
      - [x] Ambiguous query (should use conversation history)
- [x] **Test that all API responses conform to the Pydantic models defined in `models.py`.**

## 6. Deployment Considerations

- [x] Ensure BigQuery credentials and config are securely available in the AWS deployment environment.
- [x] Update deployment scripts and documentation to reflect the new data source and required environment variables.
- [ ] Validate the deployed app in AWS can connect to BigQuery and serve data as expected.
- [x] **Ensure logging output is accessible in the AWS environment (e.g., CloudWatch, log files, etc.).**

## 7. Documentation

- [ ] Update README and API documentation to describe the new BigQuery data source and configuration.
- [ ] Document any changes to environment variables, deployment steps, and troubleshooting tips for BigQuery integration.
- [x] **Document logging strategy and how to access logs for tracing and debugging.**
- [x] **Document the use of parameterized SELECT queries and security best practices for BigQuery access.**
- [x] **Document that the LLM-based parsing logic is retained and how it integrates with the new BigQuery approach.**
- [x] **Document the importance of `models.py` as the source of truth for API data structures and contracts.**

---

**Notes:**
- **Never use non-SELECT (destructive) queries in the FinancialDataManager or related API endpoints.**
- **Always use parameterized queries to prevent SQL injection and ensure query safety.**
- Consider using parameterized queries for security and performance when querying BigQuery.
- Ensure that the BigQuery client is initialized in a way that works with FastAPI's async/concurrent model (thread safety, connection pooling).
- If the app is deployed in an environment without direct GCP access, ensure VPC or service account permissions are correctly configured.
- **Consistent and informative logging is critical for tracing, debugging, and monitoring in production.** 
- **The LLM-based parsing logic is essential for converting user queries into structured filters and must be preserved and adapted for the new data source.**
- **Refer to `models.py` for the definitive structure of all API requests and responses.** 