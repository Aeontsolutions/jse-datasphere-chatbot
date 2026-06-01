"""Retry helper for transient HTTP errors in eval clients."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

_RETRYABLE_STATUS: frozenset[int] = frozenset({502, 503, 504})
_MAX_RETRIES = 2
_BASE_DELAY_S = 2.0
_BACKOFF_FACTOR = 3.0

T = TypeVar("T")


async def send_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    label: str = "",
    max_retries: int = _MAX_RETRIES,
) -> T:
    """Call fn(), retrying up to max_retries times on transient errors.

    Retries: httpx.ReadTimeout, httpx.ConnectError, HTTP 502/503/504.
    Never retries 4xx or non-gateway 5xx (e.g. 500).
    """
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except (httpx.ReadTimeout, httpx.ConnectError) as exc:
            if attempt == max_retries:
                raise
            _log_retry(attempt, max_retries, label, type(exc).__name__)
            await asyncio.sleep(_backoff(attempt))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in _RETRYABLE_STATUS or attempt == max_retries:
                raise
            _log_retry(attempt, max_retries, label, f"HTTP {exc.response.status_code}")
            await asyncio.sleep(_backoff(attempt))
    raise AssertionError("unreachable")  # pragma: no cover


def _backoff(attempt: int) -> float:
    return _BASE_DELAY_S * (_BACKOFF_FACTOR**attempt) + random.uniform(0, 1)


def _log_retry(attempt: int, max_retries: int, label: str, reason: str) -> None:
    tag = f"[{label}] " if label else ""
    print(f"  {tag}retry {attempt + 1}/{max_retries}: {reason}")
