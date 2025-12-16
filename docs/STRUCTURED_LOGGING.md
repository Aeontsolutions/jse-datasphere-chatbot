# Structured Logging with structlog

This document describes the structured logging implementation in the JSE DataSphere Chatbot application.

## Overview

The application uses `structlog` for structured logging with JSON output, automatic request ID tracking, and consistent log formatting across all modules.

## Features

- **JSON Output**: All logs are formatted as JSON for easy parsing and filtering
- **Request ID Tracking**: Every HTTP request gets a unique ID that's automatically included in all logs
- **Timestamp**: ISO-8601 formatted timestamps on all log entries
- **Structured Fields**: Use keyword arguments instead of string concatenation
- **Logger Names**: Automatic tracking of which module generated each log entry
- **Log Levels**: Standard Python logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Usage

### Getting a Logger

```python
from app.logging_config import get_logger

logger = get_logger(__name__)
```

### Logging Events

Use keyword arguments to add structured fields to your logs:

```python
# Info level
logger.info("user_login",
    user_id="user123",
    login_method="oauth",
    ip_address="192.168.1.1"
)

# Error level
logger.error("database_query_failed",
    query="SELECT * FROM users",
    error=str(e),
    error_type=type(e).__name__,
    duration_ms=150.5
)

# Warning level
logger.warning("rate_limit_approaching",
    user_id="user456",
    current_requests=95,
    limit=100
)
```

### Example JSON Output

```json
{
  "event": "s3_download_failed",
  "bucket": "my-bucket",
  "key": "document.pdf",
  "error": "NoSuchKey",
  "error_code": "404",
  "level": "error",
  "logger": "app.s3_client",
  "timestamp": "2025-12-16T13:45:00.123456Z",
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

## Request ID Middleware

The `RequestIDMiddleware` automatically:
- Generates a unique ID for each HTTP request
- Binds it to the structlog context (appears in all logs for that request)
- Adds it to the request state (`request.state.request_id`)
- Returns it in response headers (`X-Request-ID`)

This enables request correlation across distributed systems and makes debugging easier.

## Log Event Naming Conventions

Use descriptive, underscore-separated event names:

- **Good**: `s3_download_failed`, `user_authentication_success`, `cache_miss`
- **Avoid**: `error`, `success`, `failed` (too generic)

## Best Practices

### 1. Event-First Logging
```python
# Good - event name comes first
logger.error("payment_processing_failed",
    order_id="12345",
    amount=99.99,
    error=str(e)
)

# Avoid - using f-strings for structured data
logger.error(f"Payment processing failed for order {order_id}")
```

### 2. Include Context
```python
# Good - includes relevant context
logger.info("document_processed",
    document_id="doc123",
    page_count=42,
    processing_time_ms=1250.5,
    extracted_chars=15000
)
```

### 3. Error Handling
```python
try:
    result = some_operation()
except Exception as e:
    logger.error("operation_failed",
        operation="some_operation",
        error=str(e),
        error_type=type(e).__name__,
        # Add any relevant context
        input_params={"param1": value1}
    )
    raise
```

### 4. Performance Monitoring
```python
import time

start_time = time.time()
# ... do work ...
duration = (time.time() - start_time) * 1000

logger.info("api_request_completed",
    method="POST",
    path="/api/chat",
    status_code=200,
    duration_ms=duration,
    response_size=len(response_data)
)
```

## Log Levels

- **DEBUG**: Detailed diagnostic information (not shown in production by default)
- **INFO**: General informational messages about application flow
- **WARNING**: Warning messages about potential issues
- **ERROR**: Error messages for failures that need attention
- **CRITICAL**: Critical errors that may cause application failure

## Configuration

Logging is configured in `app/logging_config.py`:

```python
from app.logging_config import configure_logging, get_logger

# Configure at application startup
configure_logging(log_level="INFO")  # or "DEBUG", "WARNING", "ERROR"
```

The log level can be controlled via the `LOG_LEVEL` environment variable (configured in `app/config.py`).

## Querying JSON Logs

### Using jq
```bash
# Filter by event type
cat logs.json | jq 'select(.event == "s3_download_failed")'

# Filter by log level
cat logs.json | jq 'select(.level == "error")'

# Filter by request ID
cat logs.json | jq 'select(.request_id == "abc123")'

# Extract specific fields
cat logs.json | jq '{timestamp, event, error, request_id}'
```

### Using grep
```bash
# Find all errors
grep '"level":"error"' logs.json

# Find specific event
grep '"event":"s3_download_failed"' logs.json

# Find by request ID
grep '"request_id":"abc123"' logs.json
```

## Migration Notes

### Before (Phase 2.2)
```python
import logging
logger = logging.getLogger(__name__)

logger.error("s3_download_failed", extra={
    "bucket": bucket_name,
    "key": key,
    "error": str(e)
})
```

### After (Phase 2.3)
```python
from app.logging_config import get_logger
logger = get_logger(__name__)

logger.error("s3_download_failed",
    bucket=bucket_name,
    key=key,
    error=str(e)
)
```

## Files Modified

- `fastapi_app/requirements.txt` - Added `structlog>=24.1.0`
- `fastapi_app/app/logging_config.py` - New module for structlog configuration
- `fastapi_app/app/middleware/request_id.py` - New middleware for request ID tracking
- `fastapi_app/app/main.py` - Updated to use structlog and add middleware
- `fastapi_app/app/s3_client.py` - Converted to structlog
- `fastapi_app/app/streaming_chat.py` - Converted to structlog
- `fastapi_app/app/gemini_client.py` - Converted to structlog
- `fastapi_app/app/metadata_loader.py` - Converted to structlog
- `fastapi_app/app/document_selector.py` - Converted to structlog
- `fastapi_app/app/pdf_utils.py` - Converted to structlog
- `fastapi_app/app/streaming_financial_chat.py` - Converted to structlog
- `fastapi_app/app/financial_utils.py` - Converted to structlog
- `fastapi_app/app/progress_tracker.py` - Converted to structlog
- `fastapi_app/app/redis_job_store.py` - Converted to structlog

## Testing

Run the application and observe JSON-formatted logs in stdout/stderr:

```bash
cd fastapi_app
uvicorn app.main:app --reload
```

All logs will now be in JSON format with automatic request ID correlation.
