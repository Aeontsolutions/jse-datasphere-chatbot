"""Manages a Gemini cached content entry with hash-based invalidation."""

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

from google.genai import types

from app.gemini_client import get_genai_client
from app.logging_config import get_logger

logger = get_logger(__name__)


class PromptCache:
    """Wraps a single Google Gemini CachedContent with hash-based invalidation.

    Thread-safety: good enough for CPython — attribute reads/writes are atomic
    under the GIL; the worst-case race is two concurrent requests both creating
    a cache entry, which is wasteful but not incorrect.
    """

    def __init__(self, model_name: str, display_name: str, ttl_seconds: int = 3600) -> None:
        self._model_name = model_name
        self._display_name = display_name
        self._ttl_seconds = ttl_seconds
        self._cache = None
        self._cache_hash: Optional[str] = None
        self._expires_at: Optional[datetime] = None

    def _content_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _is_valid(self, content_hash: str) -> bool:
        return (
            self._cache is not None
            and self._cache_hash == content_hash
            and self._expires_at is not None
            and datetime.now(timezone.utc) < self._expires_at
        )

    def get_cache_name(self, system_instruction: str) -> Optional[str]:
        """Return a cache name for the given system instruction.

        Creates or renews the underlying CachedContent if needed. Returns None
        on failure — callers must fall back to passing system_instruction directly.
        """
        content_hash = self._content_hash(system_instruction)
        if self._is_valid(content_hash):
            return self._cache.name

        try:
            client = get_genai_client()
            cache = client.caches.create(
                model=self._model_name,
                config=types.CreateCachedContentConfig(
                    system_instruction=system_instruction,
                    display_name=self._display_name,
                    ttl=f"{self._ttl_seconds}s",
                ),
            )
            self._cache = cache
            self._cache_hash = content_hash
            # Expire 60 s before real TTL so we renew before the server drops it
            self._expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=max(self._ttl_seconds - 60, 0)
            )
            logger.info(
                "prompt_cache_created",
                extra={"display_name": self._display_name, "cache_name": cache.name},
            )
            return cache.name
        except Exception as exc:
            logger.warning(
                "prompt_cache_create_failed",
                extra={"display_name": self._display_name, "error": str(exc)},
            )
            return None
