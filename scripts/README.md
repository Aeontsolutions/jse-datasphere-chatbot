# Scripts

Utility scripts for development, testing, and maintenance.

## Files

- **rebuild_metadata.py** - Rebuilds document metadata from S3 bucket
- **run_metadata_rebuild.py** - Runner script for metadata rebuild
- **find_unmapped_codes.py** - Identifies unmapped company codes
- **test_client.py** - Manual test client for API endpoints
- **streaming_test_client.html** - Browser-based streaming test client

## Usage

Most scripts require environment variables to be set. Copy `.env.example` to `.env` and configure:

```bash
cd fastapi_app
cp .env.example .env
# Edit .env with your credentials

# Run scripts from project root
python scripts/rebuild_metadata.py
```
