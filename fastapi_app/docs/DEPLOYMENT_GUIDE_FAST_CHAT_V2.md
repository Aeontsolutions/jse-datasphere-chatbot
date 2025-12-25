# Fast Chat V2 Deployment Guide

This guide covers deploying the FastAPI application with the new `fast_chat_v2` endpoint for financial data queries.

## Overview

The `fast_chat_v2` endpoint provides natural language querying of financial data using AI-powered query parsing. It requires specific data files and dependencies to function properly.

## Required Files

Ensure these files are present in the `fastapi_app/` directory before building:

### Data Files
- `financial_data.csv` - The main financial dataset (2101+ records)
- `metadata_for_fast_chat_v2.json` - Financial data metadata for AI processing
- `companies.json` - Company information (existing file)

### Configuration Files
- `.env` - Environment variables including API keys
- `requirements.txt` - Updated with pandas dependency

## Environment Variables

The following environment variables are required for `fast_chat_v2`:

```bash
# Google AI Configuration (for query parsing)
GOOGLE_API_KEY=your_google_api_key_here
# OR
GCP_SERVICE_ACCOUNT_INFO=your_gcp_service_account_json

# Standard app configuration
CHROMA_HOST=http://chroma:8000
LOG_LEVEL=INFO

# AWS Configuration (for other endpoints)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_DEFAULT_REGION=your_region
DOCUMENT_METADATA_S3_BUCKET=your_bucket
```

## Deployment Steps

### 1. Pre-deployment Validation

Run the validation script to check all dependencies:

```bash
cd fastapi_app
./validate_fast_chat_v2.sh
```

### 2. Build and Deploy

```bash
# Build the containers
docker-compose build

# Start the services
docker-compose up -d

# Check logs
docker-compose logs -f api
```

### 3. Test the Endpoint

```bash
# Test basic functionality
python test_fast_chat_v2.py

# Or use curl to test
curl -X POST "http://localhost:8000/fast_chat_v2" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me MDS revenue for 2024",
    "memory_enabled": true
  }'
```

## Troubleshooting

### Common Issues

1. **"Financial data service is not available"**
   - Check that `financial_data.csv` exists in the container
   - Verify pandas is installed: `docker-compose exec api python -c "import pandas"`

2. **"AI parsing failed, using fallback"**
   - Check Google AI credentials in environment variables
   - Verify `metadata_for_fast_chat_v2.json` is accessible

3. **Container build fails**
   - Ensure `pandas>=2.0.0` is in `requirements.txt`
   - Check that data files are not excluded by `.dockerignore`

### Validation Commands

Inside the container:
```bash
# Check files are present
docker-compose exec api ls -la financial_data.csv metadata_for_fast_chat_v2.json

# Test financial manager import
docker-compose exec api python -c "from app.financial_utils import FinancialDataManager; print('OK')"

# Check pandas
docker-compose exec api python -c "import pandas; print(f'Pandas {pandas.__version__} available')"
```

## Health Check

The `/health` endpoint now includes financial data status:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "s3_client": "available",
  "metadata": "available",
  "financial_data": {
    "status": "available",
    "records": 2101
  }
}
```

## Performance Notes

- The `fast_chat_v2` endpoint loads the entire CSV into memory on startup
- First query may be slower due to AI model initialization
- Consider increasing container memory for large datasets
- Response times: ~2-5 seconds for AI-parsed queries, ~500ms for cached queries

## Security Considerations

- Ensure `.env` file contains valid API keys
- The `financial_data.csv` is included in the container - ensure it doesn't contain sensitive data
- Consider using Docker secrets for production deployments
- Restrict access to financial endpoints as needed

## Monitoring

Monitor these logs for issues:
```bash
# Application logs
docker-compose logs -f api | grep "fast_chat_v2"

# ChromaDB logs
docker-compose logs -f chroma

# Check financial data manager initialization
docker-compose logs api | grep "Financial"
```
