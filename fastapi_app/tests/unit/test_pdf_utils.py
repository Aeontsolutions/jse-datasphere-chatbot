"""Unit tests for PDF utilities module."""

from io import BytesIO
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException

from app.pdf_utils import extract_text_from_pdf


@pytest.mark.unit
class TestPDFUtils:
    """Test cases for PDF utility functions."""

    def test_extract_text_from_valid_pdf_bytes(self, mock_pdf_bytes):
        """Test text extraction from valid PDF bytes."""
        with patch("pypdf.PdfReader") as mock_reader:
            mock_page = Mock()
            mock_page.extract_text.return_value = "Test PDF content"
            mock_reader.return_value.pages = [mock_page]

            result = extract_text_from_pdf(BytesIO(mock_pdf_bytes))
            assert isinstance(result, str)
            assert len(result) > 0

    def test_extract_text_from_invalid_pdf_bytes(self):
        """Test text extraction from invalid PDF bytes raises HTTPException."""
        invalid_bytes = b"not a pdf"
        with pytest.raises(HTTPException) as exc_info:
            extract_text_from_pdf(BytesIO(invalid_bytes))
        assert exc_info.value.status_code == 400
        assert "corrupted" in str(exc_info.value.detail).lower()

    def test_extract_text_empty_pdf(self):
        """Test text extraction from empty PDF."""
        with patch("pypdf.PdfReader") as mock_reader:
            mock_reader.return_value.pages = []

            result = extract_text_from_pdf(BytesIO(b"%PDF-1.4\n%%EOF"))
            assert result == ""

    def test_extract_text_multiple_pages(self):
        """Test text extraction from multi-page PDF."""
        with patch("pypdf.PdfReader") as mock_reader:
            mock_page1 = Mock()
            mock_page1.extract_text.return_value = "Page 1 content"
            mock_page2 = Mock()
            mock_page2.extract_text.return_value = "Page 2 content"
            mock_reader.return_value.pages = [mock_page1, mock_page2]

            result = extract_text_from_pdf(BytesIO(b"%PDF-1.4\n%%EOF"))
            assert "Page 1 content" in result
            assert "Page 2 content" in result

    def test_extract_text_exception_handling(self):
        """Test that exceptions are raised as HTTPException."""
        with patch("pypdf.PdfReader", side_effect=Exception("Read error")):
            with pytest.raises(HTTPException) as exc_info:
                extract_text_from_pdf(BytesIO(b"%PDF-1.4\n%%EOF"))
            assert exc_info.value.status_code == 500
