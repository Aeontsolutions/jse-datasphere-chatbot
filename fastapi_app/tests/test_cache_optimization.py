from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from app.document_selector import semantic_document_selection
from app.gemini_client import (
    create_metadata_cache,
    get_cache_status,
    get_cached_model,
    get_metadata_hash,
    init_genai,
    refresh_metadata_cache,
)

# Sample metadata for testing
SAMPLE_METADATA = {
    "documents": [
        {
            "company": "Access Bank",
            "document_type": "Annual Report",
            "year": "2023",
            "filename": "access_bank_2023.pdf",
            "document_link": "https://example.com/access_bank_2023.pdf",
        },
        {
            "company": "NCB Financial Group",
            "document_type": "Quarterly Report",
            "year": "2023",
            "filename": "ncb_q3_2023.pdf",
            "document_link": "https://example.com/ncb_q3_2023.pdf",
        },
    ]
}


class TestCacheOptimization:
    """Test suite for Google Gemini context caching optimization"""

    def test_get_metadata_hash(self):
        """Test metadata hash generation for cache invalidation"""
        hash1 = get_metadata_hash(SAMPLE_METADATA)
        hash2 = get_metadata_hash(SAMPLE_METADATA)

        # Same metadata should produce same hash
        assert hash1 == hash2

        # Different metadata should produce different hash
        modified_metadata = SAMPLE_METADATA.copy()
        modified_metadata["documents"].append(
            {
                "company": "Test Corp",
                "document_type": "Test Report",
                "year": "2023",
                "filename": "test.pdf",
                "document_link": "https://example.com/test.pdf",
            }
        )

        hash3 = get_metadata_hash(modified_metadata)
        assert hash1 != hash3

    @patch("app.utils.genai")
    @patch("app.utils.init_genai")
    def test_create_metadata_cache_success(self, mock_init_genai, mock_genai):
        """Test successful metadata cache creation"""
        # Mock the cache creation
        mock_cache = Mock()
        mock_cache.name = "test-cache-123"
        mock_genai.caches.create.return_value = mock_cache

        # Test cache creation
        cache = create_metadata_cache(SAMPLE_METADATA)

        # Verify cache was created
        assert cache is not None
        assert cache.name == "test-cache-123"
        mock_init_genai.assert_called_once()
        mock_genai.caches.create.assert_called_once()

    @patch("app.utils.genai")
    @patch("app.utils.init_genai")
    def test_create_metadata_cache_failure(self, mock_init_genai, mock_genai):
        """Test cache creation failure handling"""
        # Mock cache creation failure
        mock_genai.caches.create.side_effect = Exception("API Error")

        # Test cache creation
        cache = create_metadata_cache(SAMPLE_METADATA)

        # Verify graceful failure
        assert cache is None

    @patch("app.utils.create_metadata_cache")
    @patch("app.utils.genai")
    def test_get_cached_model_success(self, mock_genai, mock_create_cache):
        """Test successful cached model creation"""
        # Mock cache and model
        mock_cache = Mock()
        mock_cache.name = "test-cache-123"
        mock_create_cache.return_value = mock_cache

        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model

        # Test getting cached model
        model, using_cache = get_cached_model(SAMPLE_METADATA)

        # Verify success
        assert model is not None
        assert using_cache is True
        mock_genai.GenerativeModel.assert_called_once_with(
            model_name="gemini-1.5-pro-001", cached_content=mock_cache
        )

    @patch("app.utils.create_metadata_cache")
    def test_get_cached_model_failure(self, mock_create_cache):
        """Test cached model creation failure handling"""
        # Mock cache creation failure
        mock_create_cache.return_value = None

        # Test getting cached model
        model, using_cache = get_cached_model(SAMPLE_METADATA)

        # Verify graceful failure
        assert model is None
        assert using_cache is False

    @patch("app.utils.create_metadata_cache")
    def test_refresh_metadata_cache(self, mock_create_cache):
        """Test metadata cache refresh functionality"""
        # Mock successful cache creation
        mock_cache = Mock()
        mock_create_cache.return_value = mock_cache

        # Test cache refresh
        success = refresh_metadata_cache(SAMPLE_METADATA)

        # Verify success
        assert success is True
        mock_create_cache.assert_called_once_with(SAMPLE_METADATA)

    def test_get_cache_status_no_cache(self):
        """Test cache status when no cache exists"""
        # Clear any existing cache
        import app.utils

        app.utils._metadata_cache = None
        app.utils._cache_expiry = None
        app.utils._cache_hash = None

        status = get_cache_status()

        assert status["status"] == "no_cache"
        assert status["cache_name"] is None
        assert status["expires_at"] is None
        assert status["hash"] is None

    def test_get_cache_status_active_cache(self):
        """Test cache status with active cache"""
        # Mock active cache
        import app.utils

        mock_cache = Mock()
        mock_cache.name = "test-cache-123"
        app.utils._metadata_cache = mock_cache
        app.utils._cache_expiry = datetime.now() + timedelta(hours=1)
        app.utils._cache_hash = "test-hash"

        status = get_cache_status()

        assert status["status"] == "active"
        assert status["cache_name"] == "test-cache-123"
        assert status["hash"] == "test-hash"
        assert status["expires_at"] is not None

    def test_get_cache_status_expired_cache(self):
        """Test cache status with expired cache"""
        # Mock expired cache
        import app.utils

        mock_cache = Mock()
        mock_cache.name = "test-cache-123"
        app.utils._metadata_cache = mock_cache
        app.utils._cache_expiry = datetime.now() - timedelta(hours=1)  # Expired
        app.utils._cache_hash = "test-hash"

        status = get_cache_status()

        assert status["status"] == "expired"
        assert status["cache_name"] == "test-cache-123"

    @patch("app.utils.get_cached_model")
    @patch("app.utils.GenerativeModel")
    def test_semantic_document_selection_with_cache(self, mock_vertex_model, mock_get_cached_model):
        """Test LLM selection using cached context for improved performance"""
        # Mock cached model
        mock_cached_model = Mock()
        mock_response = Mock()
        mock_response.text = '{"companies_mentioned": ["Access Bank"], "documents_to_load": [{"company": "Access Bank", "document_link": "https://example.com/access_bank_2023.pdf", "filename": "access_bank_2023.pdf", "reason": "test"}]}'
        mock_cached_model.generate_content.return_value = mock_response
        mock_get_cached_model.return_value = (mock_cached_model, True)

        # Test LLM selection with cache
        result = semantic_document_selection(
            query="Show me Access Bank financials",
            metadata=SAMPLE_METADATA,
            conversation_history=[],
        )

        # Verify cache was used
        mock_get_cached_model.assert_called_once_with(SAMPLE_METADATA)
        mock_cached_model.generate_content.assert_called_once()
        # Vertex AI model should not be used when cache is available
        mock_vertex_model.assert_not_called()

        # Verify result
        assert result is not None
        assert "companies_mentioned" in result
        assert "documents_to_load" in result

    @patch("app.utils.get_cached_model")
    @patch("app.utils.GenerativeModel")
    def test_semantic_document_selection_without_cache(
        self, mock_vertex_model, mock_get_cached_model
    ):
        """Test LLM selection without cache (traditional approach)"""
        # Mock cache failure
        mock_get_cached_model.return_value = (None, False)

        # Mock Vertex AI model
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = '{"companies_mentioned": ["Access Bank"], "documents_to_load": [{"company": "Access Bank", "document_link": "https://example.com/access_bank_2023.pdf", "filename": "access_bank_2023.pdf", "reason": "test"}]}'
        mock_model.generate_content.return_value = mock_response
        mock_vertex_model.return_value = mock_model

        # Test LLM selection without cache
        result = semantic_document_selection(
            query="Show me Access Bank financials",
            metadata=SAMPLE_METADATA,
            conversation_history=[],
        )

        # Verify fallback to traditional approach
        mock_get_cached_model.assert_called_once_with(SAMPLE_METADATA)
        mock_vertex_model.assert_called_once_with("gemini-2.0-flash-001")
        mock_model.generate_content.assert_called_once()

        # Verify result
        assert result is not None
        assert "companies_mentioned" in result
        assert "documents_to_load" in result

    @patch("app.utils.os.getenv")
    def test_init_genai_success(self, mock_getenv):
        """Test successful Google GenerativeAI initialization"""
        # Mock environment variable
        mock_getenv.return_value = "test-api-key"

        with patch("app.utils.genai.configure") as mock_configure:
            # Test initialization
            init_genai()

            # Verify configuration
            mock_configure.assert_called_once_with(api_key="test-api-key")

    @patch("app.utils.os.getenv")
    def test_init_genai_failure(self, mock_getenv):
        """Test Google GenerativeAI initialization failure"""
        # Mock missing API key
        mock_getenv.return_value = None

        # Test initialization failure
        with pytest.raises(ValueError, match="SUMMARIZER_API_KEY not found"):
            init_genai()

    @patch("app.utils.get_cached_model")
    def test_llm_fallback_conversation_history_with_cache(self, mock_get_cached_model):
        """Test LLM fallback with conversation history using cached context"""
        # Mock cached model
        mock_cached_model = Mock()
        mock_response = Mock()
        mock_response.text = '{"companies_mentioned": ["Access Bank"], "documents_to_load": [{"company": "Access Bank", "document_link": "https://example.com/access_bank_2023.pdf", "filename": "access_bank_2023.pdf", "reason": "test"}]}'
        mock_cached_model.generate_content.return_value = mock_response
        mock_get_cached_model.return_value = (mock_cached_model, True)

        conversation_history = [
            {"role": "user", "content": "What companies do you have data for?"},
            {
                "role": "assistant",
                "content": "I have data for Access Bank and NCB Financial Group.",
            },
            {"role": "user", "content": "Show me Access Bank details"},
        ]

        # Test with conversation history
        result = semantic_document_selection(
            query="What about their 2023 performance?",
            metadata=SAMPLE_METADATA,
            conversation_history=conversation_history,
        )

        # Verify the prompt included conversation history
        mock_cached_model.generate_content.assert_called_once()
        call_args = mock_cached_model.generate_content.call_args[0][0]
        assert "Previous conversation history:" in call_args
        assert "Access Bank details" in call_args

        # Verify result
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__])
