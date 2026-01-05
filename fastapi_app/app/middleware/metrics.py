"""
Prometheus metrics collection and middleware.

This module provides comprehensive metrics collection for the FastAPI application
including HTTP requests, document operations, AI requests, and BigQuery queries.
"""

import time
from typing import Callable

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# HTTP METRICS
# =============================================================================

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    labelnames=["method", "endpoint", "status_code"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    labelnames=["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
)

active_requests = Gauge(
    "active_requests",
    "Number of active HTTP requests currently being processed",
)


# =============================================================================
# DOCUMENT OPERATIONS METRICS
# =============================================================================

document_loads_total = Counter(
    "document_loads_total",
    "Total document loads",
    labelnames=["source"],  # source: s3, cache, error
)

document_load_duration_seconds = Histogram(
    "document_load_duration_seconds",
    "Document load duration in seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

document_selection_duration_seconds = Histogram(
    "document_selection_duration_seconds",
    "Document selection duration in seconds (LLM-based)",
    buckets=(0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0),
)


# =============================================================================
# AI REQUEST METRICS
# =============================================================================

ai_requests_total = Counter(
    "ai_requests_total",
    "Total AI model requests",
    labelnames=["model", "status"],  # status: success, error
)

ai_request_duration_seconds = Histogram(
    "ai_request_duration_seconds",
    "AI request duration in seconds",
    labelnames=["model"],
    buckets=(0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 60.0),
)

ai_tokens_total = Counter(
    "ai_tokens_total",
    "Total AI tokens consumed",
    labelnames=["model", "token_type"],  # token_type: input, output
)

ai_cost_dollars_total = Counter(
    "ai_cost_dollars_total",
    "Total AI API costs in USD",
    labelnames=["model", "phase"],  # phase: classification, synthesis, etc.
)


# =============================================================================
# BIGQUERY METRICS
# =============================================================================

bigquery_queries_total = Counter(
    "bigquery_queries_total",
    "Total BigQuery queries",
    labelnames=["status"],  # status: success, error
)

bigquery_query_duration_seconds = Histogram(
    "bigquery_query_duration_seconds",
    "BigQuery query duration in seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

bigquery_rows_returned = Histogram(
    "bigquery_rows_returned",
    "Number of rows returned by BigQuery queries",
    buckets=(1, 10, 50, 100, 500, 1000, 5000, 10000),
)


# =============================================================================
# CACHE METRICS
# =============================================================================

cache_operations_total = Counter(
    "cache_operations_total",
    "Total cache operations",
    labelnames=["operation", "status"],  # operation: hit, miss, set, evict
)


# =============================================================================
# METRICS MIDDLEWARE
# =============================================================================


class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware to automatically track HTTP request metrics.

    Tracks:
    - Request count by method, endpoint, and status code
    - Request duration by method and endpoint
    - Active request count (in-flight requests)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip metrics collection for the /metrics endpoint itself to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)

        # Extract endpoint path (normalize path parameters)
        endpoint = self._normalize_endpoint(request.url.path)
        method = request.method

        # Track active requests
        active_requests.inc()

        # Start timing
        start_time = time.time()

        try:
            # Process request
            response = await call_next(request)
            status_code = response.status_code

            # Record metrics
            http_requests_total.labels(
                method=method, endpoint=endpoint, status_code=status_code
            ).inc()

            return response

        except Exception:
            # Record error metrics
            http_requests_total.labels(method=method, endpoint=endpoint, status_code=500).inc()
            raise

        finally:
            # Record duration
            duration = time.time() - start_time
            http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)

            # Decrement active requests
            active_requests.dec()

    def _normalize_endpoint(self, path: str) -> str:
        """
        Normalize endpoint path to avoid cardinality explosion.

        Converts dynamic path parameters to placeholders:
        - /jobs/abc123 -> /jobs/{job_id}
        - /financial/metadata -> /financial/metadata
        """
        # Common patterns to normalize
        if path.startswith("/jobs/") and len(path.split("/")) == 3:
            return "/jobs/{job_id}"

        # Return as-is for static endpoints
        return path


# =============================================================================
# METRICS ENDPOINT
# =============================================================================


def get_metrics() -> bytes:
    """
    Generate Prometheus metrics in text format.

    Returns:
        bytes: Metrics in Prometheus exposition format
    """
    return generate_latest()


def get_metrics_content_type() -> str:
    """
    Get the content type for Prometheus metrics.

    Returns:
        str: Content type header value
    """
    return CONTENT_TYPE_LATEST
