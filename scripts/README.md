# Scripts

Utility scripts for development, testing, and maintenance.

## Files

- **run_server.py** - Runs the app locally (`uvicorn app.main:app`)
- **run_eval.py** - Runs the persona-driven eval suite against `/chat/stream` and `/fast_chat_v2`
- **check_eval_gate.py** - Gates a deploy on an eval run vs. a checked-in baseline (see `docs/CI_CD_DEPLOY_PIPELINE.md`)
- **rebuild_metadata.py** - Rebuilds document metadata from S3 bucket
- **run_metadata_rebuild.py** - Runner script for metadata rebuild
- **find_unmapped_codes.py** - Identifies unmapped company codes
- **push_copilot_secrets.py** - Pushes `.env` values to AWS SSM for the Copilot deploy
- **create_interactions_table.py** - One-time (idempotent) creation of the BigQuery `interactions` table used for permanent interaction logging (see `docs/CACHING_LOGGING_TRACING.md`)

See `archive/` for manual test clients that targeted endpoints removed in the
2026-08-14 prod-cleanup pass (`/chroma/*`, `/chat/stream/v0`'s async-job polling).

## Usage

Most scripts require environment variables to be set. Copy `.env.example` to `.env` and configure:

```bash
cd fastapi_app
cp .env.example .env
# Edit .env with your credentials

# Run scripts from project root
python scripts/rebuild_metadata.py
```
