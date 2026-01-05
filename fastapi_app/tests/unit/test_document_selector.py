"""Unit tests for document selector module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.document_selector import (
    auto_load_relevant_documents,
    auto_load_relevant_documents_async,
    resolve_companies,
    semantic_document_selection,
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

    def test_semantic_selection_llm_based(self, mock_metadata):
        """Test LLM-based document selection."""
        with patch("app.gemini_client.get_cached_model") as mock_model:
            mock_model.return_value = None
            result = semantic_document_selection("MTN report", mock_metadata)
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

    def test_resolve_companies_deduplication(self):
        """Test that resolve_companies properly deduplicates by lowercase key."""
        available_companies = [
            "Barita Investments Limited",
            "TRANSJAMAICAN HIGHWAY LIMITED",
            "NCB Financial Group Limited",
        ]
        symbol_to_company = {
            "TJH": ["TRANSJAMAICAN HIGHWAY LIMITED"],
            "BARI": ["Barita Investments Limited"],
        }

        # Simulate LLM extracting duplicates with different casing
        extracted = {
            "companies": ["barita investments limited", "Barita Investments Limited", "BARITA"],
            "symbols": ["TJH", "tjh"],  # Same symbol different case
        }

        result = resolve_companies(extracted, available_companies, symbol_to_company)

        # Should deduplicate and return only unique companies with metadata casing
        assert len(result) == 2  # Only Barita and TransJamaican
        assert "Barita Investments Limited" in result
        assert "TRANSJAMAICAN HIGHWAY LIMITED" in result
        # Should NOT contain lowercase or raw extracted values
        assert "barita investments limited" not in result
        assert "tjh" not in result

    def test_resolve_companies_partial_match(self):
        """Test that resolve_companies handles partial matches correctly."""
        available_companies = [
            "Barita Investments Limited",
            "NCB Financial Group Limited",
        ]

        # Partial match - "Barita" should match "Barita Investments Limited"
        extracted = {"companies": ["Barita"], "symbols": []}

        result = resolve_companies(extracted, available_companies)

        assert len(result) == 1
        assert "Barita Investments Limited" in result

    def test_resolve_companies_no_match(self):
        """Test that resolve_companies handles unresolvable companies."""
        available_companies = ["Barita Investments Limited"]

        extracted = {"companies": ["Unknown Company XYZ"], "symbols": []}

        result = resolve_companies(extracted, available_companies)

        assert len(result) == 0
