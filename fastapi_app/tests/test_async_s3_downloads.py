#!/usr/bin/env python3
"""
Unit tests for async S3 download functionality.
Tests the robust asynchronous document download capabilities including
concurrent downloads, retry logic, timeout handling, and progress tracking.
"""

import asyncio
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Import the modules to test
from app.config import S3DownloadConfig
from app.document_selector import (
    _download_with_semaphore,
    auto_load_relevant_documents_async,
)
from app.metadata_loader import (
    download_metadata_from_s3_async,
    load_metadata_from_s3_async,
)
from app.pdf_utils import extract_text_from_pdf_bytes as _extract_text_from_pdf_bytes
from app.s3_client import (
    DownloadResult,
    download_and_extract_from_s3_async,
    init_async_s3_client,
)

# Mock data for testing
MOCK_PDF_CONTENT = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
MOCK_METADATA_JSON = '{"companies": [{"name": "TestCorp", "documents": []}]}'
MOCK_EXTRACTED_TEXT = "This is test content from a PDF document."


class TestS3DownloadConfig:
    """Test the S3DownloadConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = S3DownloadConfig()
        assert config.max_retries == 2
        assert config.retry_delay == 0.5
        assert config.max_retry_delay == 10.0
        assert config.timeout == 300.0
        assert config.chunk_size == 8192
        assert config.concurrent_downloads == 5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = S3DownloadConfig(
            max_retries=5, retry_delay=2.0, timeout=600.0, concurrent_downloads=10
        )
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.timeout == 600.0
        assert config.concurrent_downloads == 10


class TestDownloadResult:
    """Test the DownloadResult class."""

    def test_success_result(self):
        """Test successful download result."""
        result = DownloadResult(
            success=True, content="test content", download_time=1.5, retry_count=0
        )
        assert result.success is True
        assert result.content == "test content"
        assert result.error is None
        assert result.download_time == 1.5
        assert result.retry_count == 0

    def test_failure_result(self):
        """Test failed download result."""
        result = DownloadResult(
            success=False, error="Download failed", download_time=2.0, retry_count=2
        )
        assert result.success is False
        assert result.content is None
        assert result.error == "Download failed"
        assert result.download_time == 2.0
        assert result.retry_count == 2


class TestAsyncS3Client:
    """Test async S3 client initialization."""

    @pytest.mark.asyncio
    @patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "test_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret",
            "AWS_DEFAULT_REGION": "us-east-1",
        },
    )
    @patch("app.utils.aioboto3.Session")
    async def test_init_async_s3_client_success(self, mock_session_class):
        """Test successful async S3 client initialization."""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        session = await init_async_s3_client()

        assert session == mock_session
        mock_session_class.assert_called_once_with(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            region_name="us-east-1",
        )

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    async def test_init_async_s3_client_missing_credentials(self):
        """Test async S3 client initialization with missing credentials."""
        with pytest.raises(ValueError, match="AWS credentials not found"):
            await init_async_s3_client()


class TestAsyncDocumentDownload:
    """Test async document download functionality."""

    @pytest.fixture
    def mock_progress_callback(self):
        """Create a mock progress callback."""
        return AsyncMock()

    @pytest.fixture
    def download_config(self):
        """Create a test download configuration."""
        return S3DownloadConfig(
            max_retries=2, retry_delay=0.1, timeout=10.0, concurrent_downloads=2
        )

    @pytest.mark.asyncio
    @patch("app.utils.init_async_s3_client")
    @patch("app.utils._extract_text_from_pdf_bytes")
    async def test_download_and_extract_success(
        self, mock_extract_text, mock_init_client, download_config, mock_progress_callback
    ):
        """Test successful document download and text extraction."""
        # Setup mocks
        mock_extract_text.return_value = MOCK_EXTRACTED_TEXT
        mock_s3_client = AsyncMock()
        mock_session = Mock()

        # Create an async context manager mock
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__.return_value = mock_s3_client
        async_context_manager.__aexit__.return_value = None
        mock_session.client.return_value = async_context_manager
        mock_init_client.return_value = mock_session

        # Mock S3 get_object response
        mock_response = {"Body": AsyncMock()}
        mock_response["Body"].__aiter__.return_value = [MOCK_PDF_CONTENT]
        mock_s3_client.get_object.return_value = mock_response

        # Test download
        result = await download_and_extract_from_s3_async(
            "s3://test-bucket/test-file.pdf",
            config=download_config,
            progress_callback=mock_progress_callback,
        )

        # Assertions
        assert result.success is True
        assert result.content == MOCK_EXTRACTED_TEXT
        assert result.error is None
        assert result.retry_count == 0

        # Check progress callbacks were called
        assert mock_progress_callback.call_count >= 3
        mock_progress_callback.assert_any_call(
            "download_start", "Starting download of test-file.pdf"
        )
        mock_progress_callback.assert_any_call(
            "extraction_complete", f"Extracted {len(MOCK_EXTRACTED_TEXT)} characters"
        )

    @pytest.mark.asyncio
    async def test_download_invalid_s3_path(self, download_config, mock_progress_callback):
        """Test download with invalid S3 path."""
        result = await download_and_extract_from_s3_async(
            "invalid-path", config=download_config, progress_callback=mock_progress_callback
        )

        assert result.success is False
        assert "Invalid S3 path format" in result.error

    @pytest.mark.asyncio
    @patch("app.utils.init_async_s3_client")
    async def test_download_with_retry(
        self, mock_init_client, download_config, mock_progress_callback
    ):
        """Test download with retry logic."""
        # Setup client to fail first time, succeed second time
        mock_s3_client = AsyncMock()
        mock_session = Mock()

        # Create an async context manager mock
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__.return_value = mock_s3_client
        async_context_manager.__aexit__.return_value = None
        mock_session.client.return_value = async_context_manager
        mock_init_client.return_value = mock_session

        # First call fails, second succeeds
        mock_response = {"Body": AsyncMock()}
        mock_response["Body"].__aiter__.return_value = [MOCK_PDF_CONTENT]

        mock_s3_client.get_object.side_effect = [Exception("Network error"), mock_response]

        with patch("app.utils._extract_text_from_pdf_bytes", return_value=MOCK_EXTRACTED_TEXT):
            result = await download_and_extract_from_s3_async(
                "s3://test-bucket/test-file.pdf",
                config=download_config,
                progress_callback=mock_progress_callback,
            )

        assert result.success is True
        assert result.retry_count == 1
        assert mock_s3_client.get_object.call_count == 2

    @pytest.mark.asyncio
    @patch("app.utils.init_async_s3_client")
    async def test_download_max_retries_exceeded(
        self, mock_init_client, download_config, mock_progress_callback
    ):
        """Test download when max retries are exceeded."""
        mock_s3_client = AsyncMock()
        mock_session = Mock()

        # Create an async context manager mock
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__.return_value = mock_s3_client
        async_context_manager.__aexit__.return_value = None
        mock_session.client.return_value = async_context_manager
        mock_init_client.return_value = mock_session

        # Always fail
        mock_s3_client.get_object.side_effect = Exception("Persistent error")

        result = await download_and_extract_from_s3_async(
            "s3://test-bucket/test-file.pdf",
            config=download_config,
            progress_callback=mock_progress_callback,
        )

        assert result.success is False
        assert "after 2 attempts" in result.error
        assert result.retry_count == 1  # 0-indexed, so 1 means 2 attempts
        assert mock_s3_client.get_object.call_count == 2


class TestAsyncMetadataDownload:
    """Test async metadata download functionality."""

    @pytest.fixture
    def mock_progress_callback(self):
        """Create a mock progress callback."""
        return AsyncMock()

    @pytest.fixture
    def download_config(self):
        """Create a test download configuration."""
        return S3DownloadConfig(max_retries=2, retry_delay=0.1, timeout=10.0)

    @pytest.mark.asyncio
    @patch("app.utils.init_async_s3_client")
    async def test_download_metadata_success(
        self, mock_init_client, download_config, mock_progress_callback
    ):
        """Test successful metadata download."""
        mock_s3_client = AsyncMock()
        mock_session = Mock()

        # Create an async context manager mock
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__.return_value = mock_s3_client
        async_context_manager.__aexit__.return_value = None
        mock_session.client.return_value = async_context_manager
        mock_init_client.return_value = mock_session

        # Mock S3 response
        mock_response = {"Body": AsyncMock()}
        mock_response["Body"].__aiter__.return_value = [MOCK_METADATA_JSON.encode("utf-8")]
        mock_s3_client.get_object.return_value = mock_response

        result = await download_metadata_from_s3_async(
            "test-bucket",
            "metadata.json",
            config=download_config,
            progress_callback=mock_progress_callback,
        )

        assert result.success is True
        assert result.content == MOCK_METADATA_JSON
        assert result.error is None

        # Check progress callbacks
        mock_progress_callback.assert_any_call(
            "metadata_download_start", "Downloading metadata: metadata.json"
        )
        mock_progress_callback.assert_any_call(
            "metadata_download_complete", f"Downloaded metadata ({len(MOCK_METADATA_JSON)} bytes)"
        )

    @pytest.mark.asyncio
    @patch("app.utils.init_async_s3_client")
    async def test_download_metadata_invalid_json(
        self, mock_init_client, download_config, mock_progress_callback
    ):
        """Test metadata download with invalid JSON."""
        mock_s3_client = AsyncMock()
        mock_session = Mock()

        # Create an async context manager mock
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__.return_value = mock_s3_client
        async_context_manager.__aexit__.return_value = None
        mock_session.client.return_value = async_context_manager
        mock_init_client.return_value = mock_session

        # Mock S3 response with invalid JSON
        invalid_json = b"{ invalid json content"
        mock_response = {"Body": AsyncMock()}
        mock_response["Body"].__aiter__.return_value = [invalid_json]
        mock_s3_client.get_object.return_value = mock_response

        result = await download_metadata_from_s3_async(
            "test-bucket",
            "metadata.json",
            config=download_config,
            progress_callback=mock_progress_callback,
        )

        assert result.success is False
        assert "Invalid JSON metadata" in result.error


class TestConcurrentDocumentLoading:
    """Test concurrent document loading functionality."""

    @pytest.fixture
    def mock_progress_callback(self):
        """Create a mock progress callback."""
        return AsyncMock()

    @pytest.fixture
    def download_config(self):
        """Create a test download configuration."""
        return S3DownloadConfig(max_retries=1, retry_delay=0.1, timeout=5.0, concurrent_downloads=2)

    @pytest.fixture
    def mock_metadata(self):
        """Create mock metadata for testing."""
        return {
            "TestCorp": {
                "documents": [
                    {
                        "filename": "test1.pdf",
                        "document_link": "s3://test-bucket/test1.pdf",
                        "type": "financial",
                    },
                    {
                        "filename": "test2.pdf",
                        "document_link": "s3://test-bucket/test2.pdf",
                        "type": "financial",
                    },
                ]
            }
        }

    @pytest.mark.asyncio
    @patch("app.utils.semantic_document_selection")
    @patch("app.utils.download_and_extract_from_s3_async")
    async def test_concurrent_document_loading_success(
        self, mock_download, mock_selection, mock_metadata, download_config, mock_progress_callback
    ):
        """Test successful concurrent document loading."""
        # Mock document selection
        mock_selection.return_value = {
            "companies_mentioned": ["TestCorp"],
            "documents_to_load": [
                {
                    "filename": "test1.pdf",
                    "document_link": "s3://test-bucket/test1.pdf",
                    "reason": "Financial report",
                },
                {
                    "filename": "test2.pdf",
                    "document_link": "s3://test-bucket/test2.pdf",
                    "reason": "Annual report",
                },
            ],
        }

        # Mock successful downloads
        mock_download.side_effect = [
            DownloadResult(success=True, content="Content 1", download_time=1.0),
            DownloadResult(success=True, content="Content 2", download_time=1.5),
        ]

        document_texts, message, loaded_docs = await auto_load_relevant_documents_async(
            query="Tell me about TestCorp",
            metadata=mock_metadata,
            config=download_config,
            progress_callback=mock_progress_callback,
        )

        assert len(document_texts) == 2
        assert "test1.pdf" in document_texts
        assert "test2.pdf" in document_texts
        assert document_texts["test1.pdf"] == "Content 1"
        assert document_texts["test2.pdf"] == "Content 2"
        assert len(loaded_docs) == 2
        assert "Successfully loaded 2 documents concurrently" in message

        # Verify concurrent execution (both downloads should be called)
        assert mock_download.call_count == 2

    @pytest.mark.asyncio
    @patch("app.utils.semantic_document_selection")
    @patch("app.utils.download_and_extract_from_s3_async")
    async def test_concurrent_loading_partial_failure(
        self, mock_download, mock_selection, mock_metadata, download_config, mock_progress_callback
    ):
        """Test concurrent loading with partial failures."""
        # Mock document selection
        mock_selection.return_value = {
            "companies_mentioned": ["TestCorp"],
            "documents_to_load": [
                {
                    "filename": "test1.pdf",
                    "document_link": "s3://test-bucket/test1.pdf",
                    "reason": "Financial report",
                },
                {
                    "filename": "test2.pdf",
                    "document_link": "s3://test-bucket/test2.pdf",
                    "reason": "Annual report",
                },
            ],
        }

        # Mock one success, one failure
        mock_download.side_effect = [
            DownloadResult(success=True, content="Content 1", download_time=1.0),
            DownloadResult(success=False, error="Download failed", download_time=2.0),
        ]

        document_texts, message, loaded_docs = await auto_load_relevant_documents_async(
            query="Tell me about TestCorp",
            metadata=mock_metadata,
            config=download_config,
            progress_callback=mock_progress_callback,
        )

        assert len(document_texts) == 1
        assert "test1.pdf" in document_texts
        assert "test2.pdf" not in document_texts
        assert len(loaded_docs) == 1
        assert "Successfully loaded 1 documents concurrently (1 failed)" in message

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Test that semaphore properly limits concurrent downloads."""
        # Track concurrent executions
        concurrent_count = 0
        max_concurrent = 0

        async def mock_download_task():
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.1)  # Simulate download time
            concurrent_count -= 1
            return DownloadResult(success=True, content="test")

        # Create semaphore with limit of 2
        semaphore = asyncio.Semaphore(2)

        # Run 5 tasks concurrently
        tasks = [_download_with_semaphore(semaphore, mock_download_task()) for _ in range(5)]

        results = await asyncio.gather(*tasks)

        # Should have limited to 2 concurrent executions
        assert max_concurrent <= 2
        assert len(results) == 5
        assert all(r.success for r in results)


class TestPDFTextExtraction:
    """Test PDF text extraction functionality."""

    def test_extract_text_from_valid_pdf_bytes(self):
        """Test text extraction from valid PDF bytes."""
        # This is a very basic test - in reality you'd need a proper PDF
        # For now, just test that the function handles the call properly
        with patch("app.utils.pypdf.PdfReader") as mock_reader_class:
            mock_reader = Mock()
            mock_page = Mock()
            mock_page.extract_text.return_value = "Test page content"
            mock_reader.pages = [mock_page]
            mock_reader_class.return_value = mock_reader

            result = _extract_text_from_pdf_bytes(MOCK_PDF_CONTENT)

            assert result == "Test page content\n\n"
            mock_reader_class.assert_called_once()

    def test_extract_text_from_invalid_pdf_bytes(self):
        """Test text extraction from invalid PDF bytes."""
        with patch("app.utils.pypdf.PdfReader", side_effect=Exception("Invalid PDF")):
            result = _extract_text_from_pdf_bytes(b"invalid pdf content")

            assert result is None


class TestLoadMetadataFromS3Async:
    """Test async metadata loading from S3."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"DOCUMENT_METADATA_S3_BUCKET": "test-bucket"})
    @patch("app.utils.download_metadata_from_s3_async")
    @patch("app.utils.parse_metadata_file")
    async def test_load_metadata_success(self, mock_parse, mock_download):
        """Test successful metadata loading."""
        mock_progress_callback = AsyncMock()

        # Mock successful download
        mock_download.return_value = DownloadResult(success=True, content=MOCK_METADATA_JSON)

        # Mock successful parsing
        mock_parsed_metadata = {"companies": ["TestCorp"]}
        mock_parse.return_value = mock_parsed_metadata

        result = await load_metadata_from_s3_async(progress_callback=mock_progress_callback)

        assert result == mock_parsed_metadata
        mock_download.assert_called_once_with(
            "test-bucket", "metadata.json", None, mock_progress_callback
        )
        mock_parse.assert_called_once_with(MOCK_METADATA_JSON)

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    async def test_load_metadata_missing_bucket_env(self):
        """Test metadata loading with missing bucket environment variable."""
        mock_progress_callback = AsyncMock()

        result = await load_metadata_from_s3_async(progress_callback=mock_progress_callback)

        assert result is None
        mock_progress_callback.assert_called_with(
            "metadata_error", "DOCUMENT_METADATA_S3_BUCKET not found in environment variables"
        )
