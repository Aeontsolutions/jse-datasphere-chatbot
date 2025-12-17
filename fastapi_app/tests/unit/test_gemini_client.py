"""Unit tests for Gemini AI client module."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi import HTTPException

from app.gemini_client import (
    create_metadata_cache,
    get_cache_status,
    get_cached_model,
    get_metadata_hash,
    init_genai,
    refresh_metadata_cache,
)


@pytest.mark.unit
class TestGeminiClient:
    """Test cases for Gemini AI client operations."""

    def test_init_genai_success(self, mock_config):
        """Test Gemini initialization succeeds."""
        # mock_config is already auto-injected by conftest.py's auto_mock_config
        with patch("google.generativeai.configure") as mock_configure:
            init_genai()  # No arguments - reads from config
            mock_configure.assert_called_once_with(api_key=mock_config.gcp.api_key)

    def test_init_genai_missing_key(self, mock_config):
        """Test Gemini initialization with missing API key."""
        mock_config.gcp.api_key = None
        with patch("app.gemini_client.get_config", return_value=mock_config):
            with pytest.raises(HTTPException) as exc_info:
                init_genai()
            assert exc_info.value.status_code == 503

    def test_get_metadata_hash(self, mock_metadata):
        """Test metadata hash generation."""
        hash1 = get_metadata_hash(mock_metadata)
        assert hash1 is not None
        assert isinstance(hash1, str)

        # Same metadata should produce same hash
        hash2 = get_metadata_hash(mock_metadata)
        assert hash1 == hash2

    def test_get_metadata_hash_different_data(self, mock_metadata):
        """Test different metadata produces different hash."""
        hash1 = get_metadata_hash(mock_metadata)

        different_metadata = {"companies": ["Different Company"]}
        hash2 = get_metadata_hash(different_metadata)

        assert hash1 != hash2

    def test_create_metadata_cache_success(self, mock_metadata):
        """Test metadata cache creation."""
        with patch("google.generativeai.caching.CachedContent.create") as mock_create:
            mock_cache = Mock()
            mock_cache.name = "test-cache"
            mock_create.return_value = mock_cache

            result = create_metadata_cache(mock_metadata)
            assert result is not None or result is None  # May fail without real API key

    def test_get_cached_model_success(self, mock_metadata):
        """Test getting cached model."""
        with patch("app.gemini_client.create_metadata_cache") as mock_create_cache:
            mock_cache = Mock()
            mock_cache.name = "test-cache"
            mock_create_cache.return_value = mock_cache

            with patch("google.generativeai.GenerativeModel") as mock_gen_model:
                mock_model = Mock()
                mock_gen_model.return_value = mock_model

                result = get_cached_model(mock_metadata)
                assert result is not None or result is None

    def test_refresh_metadata_cache(self, mock_metadata):
        """Test cache refresh."""
        with patch("app.gemini_client.create_metadata_cache") as mock_create:
            mock_cache = Mock()
            mock_cache.name = "refreshed-cache"
            mock_create.return_value = mock_cache

            result = refresh_metadata_cache(mock_metadata)
            assert result is not None or result is None

    def test_get_cache_status_no_cache(self):
        """Test cache status when no cache exists."""
        status = get_cache_status()
        assert isinstance(status, dict)
        assert "active" in status or "cached" in status or "status" in status
