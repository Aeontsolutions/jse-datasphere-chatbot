"""Unit tests for PromptCache."""
import hashlib
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

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
        pc._expires_at = datetime.utcnow() - timedelta(seconds=1)
        pc.get_cache_name("stable")
    assert client.caches.create.call_count == 2


def test_get_cache_name_returns_none_on_exception(mock_client):
    client, _ = mock_client
    client.caches.create.side_effect = Exception("API error")
    with patch("app.utils.prompt_cache.get_genai_client", return_value=client):
        pc = PromptCache(model_name="gemini-2.5-pro", display_name="test-cache")
        result = pc.get_cache_name("stable")
    assert result is None
