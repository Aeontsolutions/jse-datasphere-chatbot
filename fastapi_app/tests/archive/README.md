# Archived Tests

Tests for code that was archived from `fastapi_app/app/` on 2026-08-14 (see
`../../app/_archive/README.md`). Kept for reference; will not pass as-is since
the modules they import now live under `app._archive.*`.

- **test_streaming_units.py**, **test_streaming.py**, **test_streaming_flow.py**,
  **run_streaming_tests.py**, **README_STREAMING_TESTS.md** - unit/integration
  coverage for `app.streaming_chat` / `app.streaming_financial_chat`
  (`/chat/stream/v0` and `/fast_chat_v2/stream`, both removed).
- **validate_fast_chat_v2.sh** - dependency-check script asserting on
  `financial_data.csv` / `metadata_for_fast_chat_v2.json`, a local-file data
  architecture that predates the current BigQuery-backed
  `app.financial_utils.FinancialDataManager`. Stale independent of the
  streaming-endpoint archival.
- **smoke_test_summarizer.py** - standalone S3→Gemini→Chroma summarizer
  pipeline. No Chroma vector store is wired into `main.py`; unrelated to any
  registered route.
