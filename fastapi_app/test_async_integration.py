#!/usr/bin/env python3
"""
Integration test script to verify async S3 download functionality
This script tests the async functionality with mock data to ensure everything works end-to-end
"""

import asyncio
import logging
import os
from unittest.mock import Mock, AsyncMock, patch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock environment for testing
os.environ.update(
    {
        "AWS_ACCESS_KEY_ID": "test_key_id",
        "AWS_SECRET_ACCESS_KEY": "test_secret_key",
        "AWS_DEFAULT_REGION": "us-east-1",
        "DOCUMENT_METADATA_S3_BUCKET": "test-bucket",
    }
)

# Import our async functions
from app.config import S3DownloadConfig
from app.s3_client import download_and_extract_from_s3_async
from app.metadata_loader import load_metadata_from_s3_async
from app.document_selector import auto_load_relevant_documents_async

# Mock data
MOCK_PDF_CONTENT = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
MOCK_METADATA = {
    "TestCorp": {
        "documents": [
            {
                "filename": "test_report.pdf",
                "document_link": "s3://test-bucket/test_report.pdf",
                "type": "financial",
                "period": "2024",
            }
        ]
    }
}


async def test_async_download_with_progress():
    """Test async download with progress tracking"""
    logger.info("Testing async download with progress tracking...")

    progress_messages = []

    async def progress_callback(step: str, message: str):
        progress_messages.append(f"{step}: {message}")
        logger.info(f"Progress: {step} - {message}")

    config = S3DownloadConfig(max_retries=2, timeout=30.0)

    # Mock the entire download chain
    with (
        patch("app.utils.init_async_s3_client") as mock_init_client,
        patch("app.utils._extract_text_from_pdf_bytes") as mock_extract,
    ):

        # Setup mock client
        mock_s3_client = AsyncMock()
        mock_session = Mock()
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__.return_value = mock_s3_client
        async_context_manager.__aexit__.return_value = None
        mock_session.client.return_value = async_context_manager
        mock_init_client.return_value = mock_session

        # Mock S3 response
        mock_response = {"Body": AsyncMock()}
        mock_response["Body"].__aiter__.return_value = [MOCK_PDF_CONTENT]
        mock_s3_client.get_object.return_value = mock_response

        # Mock text extraction
        mock_extract.return_value = "This is extracted text from the PDF."

        # Test the download
        result = await download_and_extract_from_s3_async(
            "s3://test-bucket/test-document.pdf", config=config, progress_callback=progress_callback
        )

        # Verify result
        assert result.success, f"Download failed: {result.error}"
        assert result.content == "This is extracted text from the PDF."
        assert (
            len(progress_messages) >= 3
        ), f"Expected at least 3 progress messages, got {len(progress_messages)}"

        logger.info("✓ Async download with progress tracking: PASSED")
        return True


async def test_concurrent_downloads():
    """Test concurrent document downloads"""
    logger.info("Testing concurrent document downloads...")

    progress_messages = []

    async def progress_callback(step: str, message: str):
        progress_messages.append(f"{step}: {message}")
        logger.info(f"Progress: {step} - {message}")

    config = S3DownloadConfig(max_retries=1, concurrent_downloads=3)

    # Mock semantic document selection
    with (
        patch("app.utils.semantic_document_selection") as mock_selection,
        patch("app.utils.download_and_extract_from_s3_async") as mock_download,
    ):

        # Mock document selection response
        mock_selection.return_value = {
            "companies_mentioned": ["TestCorp"],
            "documents_to_load": [
                {
                    "filename": "doc1.pdf",
                    "document_link": "s3://test-bucket/doc1.pdf",
                    "reason": "Financial report",
                },
                {
                    "filename": "doc2.pdf",
                    "document_link": "s3://test-bucket/doc2.pdf",
                    "reason": "Annual report",
                },
                {
                    "filename": "doc3.pdf",
                    "document_link": "s3://test-bucket/doc3.pdf",
                    "reason": "Quarterly report",
                },
            ],
        }

        # Mock download results with different timing
        from app.s3_client import DownloadResult

        mock_download.side_effect = [
            DownloadResult(success=True, content="Content 1", download_time=1.0),
            DownloadResult(success=True, content="Content 2", download_time=1.5),
            DownloadResult(success=True, content="Content 3", download_time=0.8),
        ]

        # Test concurrent loading
        document_texts, message, loaded_docs = await auto_load_relevant_documents_async(
            query="Tell me about TestCorp financials",
            metadata=MOCK_METADATA,
            config=config,
            progress_callback=progress_callback,
        )

        # Verify results
        assert len(document_texts) == 3, f"Expected 3 documents, got {len(document_texts)}"
        assert len(loaded_docs) == 3, f"Expected 3 loaded docs, got {len(loaded_docs)}"
        assert "Successfully loaded 3 documents concurrently" in message

        # Verify all downloads were called (concurrent execution)
        assert mock_download.call_count == 3

        logger.info("✓ Concurrent document downloads: PASSED")
        return True


async def test_retry_logic():
    """Test retry logic with failures"""
    logger.info("Testing retry logic with failures...")

    progress_messages = []

    async def progress_callback(step: str, message: str):
        progress_messages.append(f"{step}: {message}")
        logger.info(f"Progress: {step} - {message}")

    config = S3DownloadConfig(max_retries=3, retry_delay=0.1, timeout=10.0)

    with (
        patch("app.utils.init_async_s3_client") as mock_init_client,
        patch("app.utils._extract_text_from_pdf_bytes") as mock_extract,
    ):

        # Setup mock client
        mock_s3_client = AsyncMock()
        mock_session = Mock()
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__.return_value = mock_s3_client
        async_context_manager.__aexit__.return_value = None
        mock_session.client.return_value = async_context_manager
        mock_init_client.return_value = mock_session

        # Mock S3 response (fail twice, then succeed)
        mock_response = {"Body": AsyncMock()}
        mock_response["Body"].__aiter__.return_value = [MOCK_PDF_CONTENT]

        mock_s3_client.get_object.side_effect = [
            Exception("Network timeout"),
            Exception("Connection refused"),
            mock_response,  # Success on third try
        ]

        # Mock text extraction
        mock_extract.return_value = "Text extracted after retries."

        # Test the download with retries
        result = await download_and_extract_from_s3_async(
            "s3://test-bucket/retry-test.pdf", config=config, progress_callback=progress_callback
        )

        # Verify result
        assert result.success, f"Download should succeed after retries: {result.error}"
        assert result.retry_count == 2, f"Expected 2 retries, got {result.retry_count}"
        assert result.content == "Text extracted after retries."

        # Verify retry attempts
        assert mock_s3_client.get_object.call_count == 3

        logger.info("✓ Retry logic with failures: PASSED")
        return True


async def test_metadata_loading():
    """Test async metadata loading"""
    logger.info("Testing async metadata loading...")

    progress_messages = []

    async def progress_callback(step: str, message: str):
        progress_messages.append(f"{step}: {message}")
        logger.info(f"Progress: {step} - {message}")

    with (
        patch("app.utils.download_metadata_from_s3_async") as mock_download,
        patch("app.utils.parse_metadata_file") as mock_parse,
    ):

        # Mock successful metadata download
        from app.s3_client import DownloadResult

        mock_download.return_value = DownloadResult(
            success=True, content='{"companies": ["TestCorp"]}'
        )

        # Mock successful parsing
        mock_parse.return_value = {"companies": ["TestCorp"]}

        # Test metadata loading
        metadata = await load_metadata_from_s3_async(progress_callback=progress_callback)

        # Verify result
        assert metadata is not None, "Metadata should not be None"
        assert "companies" in metadata, "Metadata should contain companies"
        assert metadata["companies"] == ["TestCorp"]

        # Verify function calls
        mock_download.assert_called_once()
        mock_parse.assert_called_once_with('{"companies": ["TestCorp"]}')

        logger.info("✓ Async metadata loading: PASSED")
        return True


async def main():
    """Run all integration tests"""
    logger.info("Starting async S3 download integration tests...")

    tests = [
        test_async_download_with_progress,
        test_concurrent_downloads,
        test_retry_logic,
        test_metadata_loading,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            logger.error(f"✗ {test.__name__}: FAILED - {str(e)}")
            failed += 1

    logger.info(f"\nIntegration tests completed: {passed} passed, {failed} failed")

    if failed == 0:
        logger.info("🎉 All async S3 download integration tests PASSED!")
        return True
    else:
        logger.error("❌ Some integration tests FAILED!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
