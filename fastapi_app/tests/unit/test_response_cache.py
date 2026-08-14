"""Unit tests for ResponseCache."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.response_cache import ResponseCache, build_response_cache


def test_disabled_when_url_missing():
    cache = ResponseCache(url=None, ttl_seconds=60, enabled=True)
    assert cache.enabled is False


def test_disabled_when_flag_off():
    cache = ResponseCache(url="redis://localhost:6379/0", ttl_seconds=60, enabled=False)
    assert cache.enabled is False


def test_disabled_on_construction_failure():
    with patch("app.response_cache.redis.from_url", side_effect=Exception("boom")):
        cache = ResponseCache(url="redis://localhost:6379/0", ttl_seconds=60, enabled=True)
    assert cache.enabled is False


@pytest.mark.asyncio
async def test_get_set_noop_when_disabled():
    cache = ResponseCache(url=None, ttl_seconds=60, enabled=True)
    await cache.set("key", {"a": 1})
    assert await cache.get("key") is None


def test_build_key_deterministic():
    k1 = ResponseCache.build_key("fast_chat_v2", "revenue for nyc holdings")
    k2 = ResponseCache.build_key("fast_chat_v2", "revenue for nyc holdings")
    assert k1 == k2


def test_build_key_differs_by_namespace_and_parts():
    base = ResponseCache.build_key("fast_chat_v2", "same query")
    other_ns = ResponseCache.build_key("chat_stream", "same query")
    other_parts = ResponseCache.build_key("fast_chat_v2", "same query", True)
    assert base != other_ns
    assert base != other_parts


def test_normalize_query_collapses_whitespace_and_case():
    assert (
        ResponseCache.normalize_query("  Revenue   For   NYC Holdings  ")
        == "revenue for nyc holdings"
    )


@pytest.mark.asyncio
async def test_get_set_roundtrip_with_mocked_client():
    with patch("app.response_cache.redis.from_url") as mock_from_url:
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=None)
        mock_client.set = AsyncMock()
        mock_from_url.return_value = mock_client

        cache = ResponseCache(url="redis://localhost:6379/0", ttl_seconds=60, enabled=True)
        assert cache.enabled is True

        await cache.set("key", {"response": "hi"})
        mock_client.set.assert_called_once()
        args, kwargs = mock_client.set.call_args
        assert args[0] == "key"
        assert kwargs["ex"] == 60

        mock_client.get = AsyncMock(return_value='{"response": "hi"}')
        result = await cache.get("key")
        assert result == {"response": "hi"}


@pytest.mark.asyncio
async def test_get_set_swallow_client_errors():
    with patch("app.response_cache.redis.from_url") as mock_from_url:
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=Exception("connection lost"))
        mock_client.set = AsyncMock(side_effect=Exception("connection lost"))
        mock_from_url.return_value = mock_client

        cache = ResponseCache(url="redis://localhost:6379/0", ttl_seconds=60, enabled=True)
        assert await cache.get("key") is None
        await cache.set("key", {"a": 1})  # must not raise


def test_build_response_cache_uses_config(monkeypatch):
    mock_config = MagicMock()
    mock_config.redis.url = None
    mock_config.redis.response_cache_ttl_seconds = 3600
    mock_config.redis.response_cache_enabled = True
    with patch("app.config.get_config", return_value=mock_config):
        cache = build_response_cache()
    assert cache.enabled is False
