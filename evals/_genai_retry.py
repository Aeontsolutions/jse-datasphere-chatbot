"""Retry helper for transient Gemini SDK errors.

Sibling of `evals/client/_retry.py`, which does the same job for the httpx
calls against the chatbot. This one covers the `google.genai` SDK, whose
exceptions are a different family entirely -- hence a second helper rather
than a predicate bolted onto the first.

Why it exists: on 2026-08-31 the CI eval gate failed with
`2 judge_failed_count (max allowed 0)` on a run where every quality dimension
improved. Both failures were `ServerError: 503 UNAVAILABLE` -- "This model is
currently experiencing high demand. Spikes in demand are usually temporary.
Please try again later." Gemini asked to be retried; nothing retried it,
because judge.py called generate_content outside its try block. An unjudged
conversation is dropped from the scored sample, so a transient 503 quietly
shrinks the evidence a release decision rests on.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from google.genai import errors

# 429 is the only client error worth retrying: the request is fine, we are
# just early. Every other 4xx (bad prompt, bad key, bad model name) will fail
# identically on the next attempt, so retrying burns budget and delays the
# real error.
_RETRYABLE_CLIENT_STATUS: frozenset[int] = frozenset({429})
_MAX_RETRIES = 2
_BASE_DELAY_S = 2.0
_BACKOFF_FACTOR = 3.0

T = TypeVar("T")


def _is_retryable(exc: errors.APIError) -> bool:
    # ServerError is the SDK's 5xx family -- always transient by definition.
    if isinstance(exc, errors.ServerError):
        return True
    return getattr(exc, "code", None) in _RETRYABLE_CLIENT_STATUS


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    label: str = "",
    max_retries: int = _MAX_RETRIES,
) -> T:
    """Call fn(), retrying transient Gemini errors with exponential backoff.

    Retries 5xx (`ServerError`) and 429 rate limits. Re-raises everything else
    immediately, including our own bugs -- a ValueError from this codebase is
    not something a second attempt will fix.
    """
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except errors.APIError as exc:
            if not _is_retryable(exc) or attempt == max_retries:
                raise
            _log_retry(attempt, max_retries, label, f"{getattr(exc, 'code', '?')}")
            await asyncio.sleep(_backoff(attempt))
    raise AssertionError("unreachable")  # pragma: no cover


def _backoff(attempt: int) -> float:
    return _BASE_DELAY_S * (_BACKOFF_FACTOR**attempt) + random.uniform(0, 1)


def _log_retry(attempt: int, max_retries: int, label: str, reason: str) -> None:
    tag = f"[{label}] " if label else ""
    print(f"  {tag}gemini retry {attempt + 1}/{max_retries}: {reason}")
