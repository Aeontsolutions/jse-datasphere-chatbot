"""Unit tests for PromptCache."""
import hashlib
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.utils.prompt_cache import PromptCache


@pytest.fixture
def mock_client():
    client = MagicMock()
    mock_cache = MagicMock()
    mock_cache.name = "cachedContents/abc123"
    client.caches.create.return_value = mock_cache
    return client, mock_cache


def test_get_cache_name_creates_on_first_call(mock_client):
    client, mock_cache = mock_client
    with patch("app.utils.prompt_cache.get_genai_client", return_value=client):
        pc = PromptCache(model_name="gemini-2.5-pro", display_name="test-cache")
        name = pc.get_cache_name("stable system instruction")
    assert name == "cachedContents/abc123"
    client.caches.create.assert_called_once()


def test_get_cache_name_reuses_on_second_call(mock_client):
    client, mock_cache = mock_client
    with patch("app.utils.prompt_cache.get_genai_client", return_value=client):
        pc = PromptCache(model_name="gemini-2.5-pro", display_name="test-cache")
        pc.get_cache_name("stable system instruction")
        pc.get_cache_name("stable system instruction")
    assert client.caches.create.call_count == 1


def test_get_cache_name_recreates_on_content_change(mock_client):
    client, mock_cache = mock_client
    with patch("app.utils.prompt_cache.get_genai_client", return_value=client):
        pc = PromptCache(model_name="gemini-2.5-pro", display_name="test-cache")
        pc.get_cache_name("prompt version 1")
        pc.get_cache_name("prompt version 2")
    assert client.caches.create.call_count == 2


def test_get_cache_name_recreates_when_expired(mock_client):
    client, mock_cache = mock_client
    with patch("app.utils.prompt_cache.get_genai_client", return_value=client):
        pc = PromptCache(model_name="gemini-2.5-pro", display_name="test-cache", ttl_seconds=3600)
        pc.get_cache_name("stable")
        # Manually expire
        pc._expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        pc.get_cache_name("stable")
    assert client.caches.create.call_count == 2


def test_get_cache_name_returns_none_on_exception(mock_client):
    client, _ = mock_client
    client.caches.create.side_effect = Exception("API error")
    with patch("app.utils.prompt_cache.get_genai_client", return_value=client):
        pc = PromptCache(model_name="gemini-2.5-pro", display_name="test-cache")
        result = pc.get_cache_name("stable")
    assert result is None


def test_get_cache_name_propagates_http_exception_from_client_init():
    """HTTPException raised by get_genai_client() must not be swallowed."""
    pc = PromptCache(model_name="gemini-2.5-pro", display_name="test-cache")
    with patch(
        "app.utils.prompt_cache.get_genai_client",
        side_effect=HTTPException(status_code=503, detail="no key"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            pc.get_cache_name("stable system instruction")
    assert exc_info.value.status_code == 503


def test_get_cache_name_retries_after_create_failure(mock_client):
    """After caches.create() fails, the next call retries the API (state not corrupted)."""
    client, mock_cache = mock_client
    # First call to create raises; second call succeeds
    client.caches.create.side_effect = [Exception("transient error"), mock_cache]
    with patch("app.utils.prompt_cache.get_genai_client", return_value=client):
        pc = PromptCache(model_name="gemini-2.5-pro", display_name="test-cache")
        first = pc.get_cache_name("stable system instruction")
        second = pc.get_cache_name("stable system instruction")
    assert first is None
    assert second == "cachedContents/abc123"
    assert client.caches.create.call_count == 2
