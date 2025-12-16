"""
Performance monitoring utilities and decorators.

This module provides convenient utilities for tracking operation performance
and recording business metrics using Prometheus.
"""

import functools
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Callable, Optional

from app.logging_config import get_logger
from app.middleware.metrics import (
    ai_request_duration_seconds,
    ai_requests_total,
    ai_tokens_total,
    bigquery_queries_total,
    bigquery_query_duration_seconds,
    bigquery_rows_returned,
    cache_operations_total,
    document_load_duration_seconds,
    document_loads_total,
    document_selection_duration_seconds,
)

logger = get_logger(__name__)


# =============================================================================
# TIMING CONTEXT MANAGERS
# =============================================================================


@contextmanager
def track_time(operation_name: str):
    """
    Context manager for tracking operation duration.

    Logs the operation duration and can be used with metrics recording.

    Args:
        operation_name: Name of the operation being tracked

    Yields:
        dict: Dictionary with 'duration' key that will be populated on exit

    Example:
        >>> with track_time("my_operation") as timing:
        ...     # do work
        ...     pass
        >>> print(f"Operation took {timing['duration']:.2f}s")
    """
    result = {"duration": 0.0}
    start_time = time.time()

    try:
        yield result
    finally:
        duration = time.time() - start_time
        result["duration"] = duration
        logger.debug(f"{operation_name} completed in {duration:.3f}s")


@asynccontextmanager
async def track_time_async(operation_name: str):
    """
    Async context manager for tracking operation duration.

    Args:
        operation_name: Name of the operation being tracked

    Yields:
        dict: Dictionary with 'duration' key that will be populated on exit

    Example:
        >>> async with track_time_async("my_async_operation") as timing:
        ...     await async_work()
        >>> print(f"Operation took {timing['duration']:.2f}s")
    """
    result = {"duration": 0.0}
    start_time = time.time()

    try:
        yield result
    finally:
        duration = time.time() - start_time
        result["duration"] = duration
        logger.debug(f"{operation_name} completed in {duration:.3f}s")


# =============================================================================
# OPERATION TRACKING DECORATORS
# =============================================================================


def track_operation(operation_type: str, **metric_labels):
    """
    Decorator for tracking operation performance with automatic metric recording.

    This is a generic decorator that can be customized with metric labels.
    For specific operations, use the specialized decorators below.

    Args:
        operation_type: Type of operation (e.g., "document_load", "ai_request")
        **metric_labels: Additional labels to attach to metrics

    Example:
        >>> @track_operation("custom_operation", source="api")
        ... def my_function():
        ...     # do work
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = False

            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            finally:
                duration = time.time() - start_time
                logger.debug(
                    f"{operation_type} operation",
                    duration=duration,
                    success=success,
                    **metric_labels,
                )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = False

            try:
                result = func(*args, **kwargs)
                success = True
                return result
            finally:
                duration = time.time() - start_time
                logger.debug(
                    f"{operation_type} operation",
                    duration=duration,
                    success=success,
                    **metric_labels,
                )

        # Return appropriate wrapper based on function type
        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# =============================================================================
# DOCUMENT OPERATION TRACKING
# =============================================================================


def record_document_load(source: str, duration: float, success: bool = True):
    """
    Record metrics for a document load operation.

    Args:
        source: Source of the document (s3, cache, error)
        duration: Duration of the operation in seconds
        success: Whether the operation was successful
    """
    document_loads_total.labels(source=source if success else "error").inc()
    if success:
        document_load_duration_seconds.observe(duration)

    logger.info(
        "document_load_recorded",
        source=source,
        duration=duration,
        success=success,
    )


def record_document_selection(duration: float, num_documents: int = 0):
    """
    Record metrics for document selection operation.

    Args:
        duration: Duration of the selection operation in seconds
        num_documents: Number of documents selected
    """
    document_selection_duration_seconds.observe(duration)

    logger.info(
        "document_selection_recorded",
        duration=duration,
        num_documents=num_documents,
    )


# =============================================================================
# AI REQUEST TRACKING
# =============================================================================


def record_ai_request(
    model: str,
    duration: float,
    success: bool = True,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
):
    """
    Record metrics for an AI model request.

    Args:
        model: Model name (e.g., "gemini-2.0-flash-exp")
        duration: Duration of the request in seconds
        success: Whether the request was successful
        input_tokens: Number of input tokens consumed (optional)
        output_tokens: Number of output tokens generated (optional)
    """
    status = "success" if success else "error"
    ai_requests_total.labels(model=model, status=status).inc()

    if success:
        ai_request_duration_seconds.labels(model=model).observe(duration)

        # Record token usage if available
        if input_tokens is not None:
            ai_tokens_total.labels(model=model, token_type="input").inc(input_tokens)
        if output_tokens is not None:
            ai_tokens_total.labels(model=model, token_type="output").inc(output_tokens)

    logger.info(
        "ai_request_recorded",
        model=model,
        duration=duration,
        success=success,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


# =============================================================================
# BIGQUERY OPERATION TRACKING
# =============================================================================


def record_bigquery_query(duration: float, rows_returned: int = 0, success: bool = True):
    """
    Record metrics for a BigQuery query operation.

    Args:
        duration: Duration of the query in seconds
        rows_returned: Number of rows returned by the query
        success: Whether the query was successful
    """
    status = "success" if success else "error"
    bigquery_queries_total.labels(status=status).inc()

    if success:
        bigquery_query_duration_seconds.observe(duration)
        if rows_returned > 0:
            bigquery_rows_returned.observe(rows_returned)

    logger.info(
        "bigquery_query_recorded",
        duration=duration,
        rows_returned=rows_returned,
        success=success,
    )


# =============================================================================
# CACHE OPERATION TRACKING
# =============================================================================


def record_cache_operation(operation: str, success: bool = True):
    """
    Record metrics for cache operations.

    Args:
        operation: Type of cache operation (hit, miss, set, evict)
        success: Whether the operation was successful
    """
    status = "success" if success else "error"
    cache_operations_total.labels(operation=operation, status=status).inc()

    logger.debug(
        "cache_operation_recorded",
        operation=operation,
        success=success,
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def increment_counter(counter_name: str, labels: Optional[dict] = None):
    """
    Generic function to increment any counter metric.

    Args:
        counter_name: Name of the counter to increment
        labels: Dictionary of label key-value pairs
    """
    # This is a placeholder for future extensibility
    # Specific metrics should use their dedicated recording functions
    logger.debug(f"Counter incremented: {counter_name}", labels=labels)


def observe_histogram(histogram_name: str, value: float, labels: Optional[dict] = None):
    """
    Generic function to observe a value in any histogram metric.

    Args:
        histogram_name: Name of the histogram
        value: Value to observe
        labels: Dictionary of label key-value pairs
    """
    # This is a placeholder for future extensibility
    # Specific metrics should use their dedicated recording functions
    logger.debug(f"Histogram observed: {histogram_name}", value=value, labels=labels)
