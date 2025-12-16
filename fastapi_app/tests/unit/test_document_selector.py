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
        assert isinstance(result, (list, tuple))

    def test_semantic_selection_with_query(self, mock_metadata):
        """Test semantic document selection with query."""
        result = semantic_document_selection("MTN revenue", mock_metadata)
        assert isinstance(result, (list, tuple))

    def test_semantic_selection_llm_fallback(self, mock_metadata):
        """Test LLM fallback document selection."""
        with patch("app.gemini_client.get_cached_model") as mock_model:
            mock_model.return_value = None
            result = semantic_document_selection_llm_fallback("MTN report", mock_metadata)
            assert isinstance(result, (list, tuple))

    @pytest.mark.asyncio
    async def test_auto_load_relevant_documents_async(self, mock_metadata, mock_s3_client):
        """Test async document loading."""
        with patch("app.document_selector._download_single_document_async") as mock_download:
            mock_download.return_value = ("test content", "test.pdf")

            result = await auto_load_relevant_documents_async(
                query="MTN report",
                s3_client=mock_s3_client,
                metadata=mock_metadata,
                max_documents=1,
            )
            assert isinstance(result, tuple)

    def test_auto_load_relevant_documents(self, mock_metadata, mock_s3_client):
        """Test synchronous document loading."""
        result = auto_load_relevant_documents(
            query="MTN report", s3_client=mock_s3_client, metadata=mock_metadata, max_documents=1
        )
        assert isinstance(result, tuple)

    def test_semantic_selection_with_conversation_history(self, mock_metadata):
        """Test semantic selection with conversation history."""
        history = [
            {"role": "user", "content": "Tell me about MTN"},
            {"role": "assistant", "content": "MTN is a telecommunications company"},
        ]
        result = semantic_document_selection("revenue", mock_metadata, conversation_history=history)
        assert isinstance(result, (list, tuple))
