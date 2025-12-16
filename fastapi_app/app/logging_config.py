"""
Structured logging configuration using structlog.

Provides JSON-formatted structured logging for production environments
with request ID tracking and consistent log formatting.
"""

import logging
import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog for the application.

    Sets up:
    - JSON formatting for machine-readable logs
    - Request ID processor for correlation
    - Timestamp formatting
    - Log level filtering
    """

    # Configure structlog processors
    structlog.configure(
        processors=[
            # Add context from contextvars
            structlog.contextvars.merge_contextvars,
            # Add log level
            structlog.stdlib.add_log_level,
            # Add logger name
            structlog.stdlib.add_logger_name,
            # Add timestamp
            structlog.processors.TimeStamper(fmt="iso"),
            # Add stack info for exceptions
            structlog.processors.StackInfoRenderer(),
            # Format exceptions
            structlog.processors.format_exc_info,
            # Unwrap context
            structlog.processors.UnicodeDecoder(),
            # Render as JSON for production
            structlog.processors.JSONRenderer(),
        ],
        # Wrapper class
        wrapper_class=structlog.stdlib.BoundLogger,
        # Context class
        context_class=dict,
        # Logger factory
        logger_factory=structlog.stdlib.LoggerFactory(),
        # Cache loggers
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger instance."""
    return structlog.get_logger(name)
