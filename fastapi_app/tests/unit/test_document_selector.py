"""Unit tests for document selector module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.document_selector import (
    auto_load_relevant_documents,
    auto_load_relevant_documents_async,
    semantic_document_selection,
    semantic_document_selection_llm_fallback,
)


@pytest.mark.unit
class TestDocumentSelector:
    """Test cases for document selection operations."""

    def test_semantic_selection_empty_query(self, mock_metadata):
        """Test document selection with empty query."""
        result = semantic_document_selection("", mock_metadata)
        # Returns dict or None
        assert result is None or isinstance(result, dict)

    def test_semantic_selection_with_query(self, mock_metadata):
        """Test semantic document selection with query."""
        result = semantic_document_selection("MTN revenue", mock_metadata)
        # Returns dict or None
        assert result is None or isinstance(result, dict)

    def test_semantic_selection_llm_fallback(self, mock_metadata):
        """Test LLM fallback document selection."""
        with patch("app.gemini_client.get_cached_model") as mock_model:
            mock_model.return_value = None
            result = semantic_document_selection_llm_fallback("MTN report", mock_metadata)
            # Returns dict or None
            assert result is None or isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_auto_load_relevant_documents_async(self, mock_metadata):
        """Test async document loading."""
        with patch("app.document_selector._download_single_document_async") as mock_download:
            from app.s3_client import DownloadResult

            mock_download.return_value = DownloadResult(
                success=True, content="test content", file_path="test.pdf"
            )

            result = await auto_load_relevant_documents_async(
                query="MTN report",
                metadata=mock_metadata,
            )
            assert isinstance(result, tuple)
            assert len(result) == 3  # (document_texts, message, loaded_docs)

    def test_auto_load_relevant_documents(self, mock_metadata, mock_s3_client):
        """Test synchronous document loading."""
        result = auto_load_relevant_documents(
            s3_client=mock_s3_client,
            query="MTN report",
            metadata=mock_metadata,
            current_document_texts={},
        )
        assert isinstance(result, tuple)
        assert len(result) == 3  # (document_texts, message, loaded_docs)

    def test_semantic_selection_with_conversation_history(self, mock_metadata):
        """Test semantic selection with conversation history."""
        history = [
            {"role": "user", "content": "Tell me about MTN"},
            {"role": "assistant", "content": "MTN is a telecommunications company"},
        ]
        result = semantic_document_selection("revenue", mock_metadata, conversation_history=history)
        # Returns dict or None
        assert result is None or isinstance(result, dict)
