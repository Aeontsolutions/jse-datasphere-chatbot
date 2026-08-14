"""
Redis-backed response cache for exact/normalized-match prompt caching.

Phase 1: normalized-string matching only (no semantic similarity). Caching is
skipped entirely by callers whenever conversation history is present, since a
follow-up's correct answer depends on context this cache doesn't key on.

No-ops safely whenever Redis isn't configured or unreachable, so callers never
need to branch on availability.
"""

import hashlib
import json
from typing import Any, Optional

import redis.asyncio as redis

from app.logging_config import get_logger

logger = get_logger(__name__)


class ResponseCache:
    """Thin async wrapper around Redis for caching endpoint response payloads."""

    def __init__(self, url: Optional[str], ttl_seconds: int = 3600, enabled: bool = True):
        self._ttl = ttl_seconds
        self._client: Optional["redis.Redis"] = None
        self._enabled = bool(enabled and url)
        if self._enabled:
            try:
                self._client = redis.from_url(
                    url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
            except Exception as exc:
                logger.warning("response_cache_init_failed", extra={"error": str(exc)})
                self._enabled = False
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._enabled and self._client is not None

    @staticmethod
    def build_key(namespace: str, *parts: Any) -> str:
        """Build a deterministic cache key from a namespace and key parts.

        Namespacing by endpoint prevents cross-endpoint collisions since
        fast_chat_v2 and chat_stream have different response shapes.
        """
        normalized = "|".join(str(p) for p in parts)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"cache:v1:{namespace}:{digest}"

    @staticmethod
    def normalize_query(query: str) -> str:
        """Lowercase + collapse whitespace. No punctuation/synonym handling (phase 1)."""
        return " ".join(query.strip().lower().split())

    async def get(self, key: str) -> Optional[dict]:
        if not self.enabled:
            return None
        try:
            raw = await self._client.get(key)
            return json.loads(raw) if raw is not None else None
        except Exception as exc:
            logger.warning("response_cache_get_failed", extra={"key": key, "error": str(exc)})
            return None

    async def set(self, key: str, value: dict, ttl_seconds: Optional[int] = None) -> None:
        if not self.enabled:
            return
        try:
            await self._client.set(key, json.dumps(value, default=str), ex=ttl_seconds or self._ttl)
        except Exception as exc:
            logger.warning("response_cache_set_failed", extra={"key": key, "error": str(exc)})

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as exc:
                logger.warning("response_cache_close_failed", extra={"error": str(exc)})


def build_response_cache() -> ResponseCache:
    """Construct a ResponseCache from the current app config."""
    from app.config import get_config

    cfg = get_config()
    return ResponseCache(
        url=cfg.redis.url,
        ttl_seconds=cfg.redis.response_cache_ttl_seconds,
        enabled=cfg.redis.response_cache_enabled,
    )
