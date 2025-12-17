"""Unit tests for S3 client module."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aioboto3
import pytest
from botocore.exceptions import ClientError
from fastapi import HTTPException

from app.s3_client import (
    download_and_extract_from_s3,
    init_async_s3_client,
    init_s3_client,
)


@pytest.mark.unit
class TestS3Client:
    """Test cases for S3 client operations."""

    def test_init_s3_client_success(self, mock_config):
        """Test S3 client initialization succeeds."""
        with patch("boto3.client") as mock_boto:
            mock_boto.return_value = Mock()
            from app.s3_client import init_s3_client

            client = init_s3_client()
            assert client is not None

    def test_download_and_extract_invalid_path(self):
        """Test download fails with invalid S3 path."""
        mock_client = Mock()
        with pytest.raises(HTTPException) as exc_info:
            download_and_extract_from_s3(mock_client, "invalid-path")
        assert exc_info.value.status_code == 400
        assert "Invalid S3 path" in str(exc_info.value.detail)

    def test_download_and_extract_not_found(self):
        """Test download fails when file not found."""
        mock_client = Mock()
        # Simulate ClientError on download_fileobj (the actual method used)
        mock_client.download_fileobj.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject"
        )

        with pytest.raises(HTTPException) as exc_info:
            download_and_extract_from_s3(mock_client, "s3://bucket/missing.pdf")
        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value.detail).lower()

    def test_download_and_extract_success(self, mock_pdf_bytes):
        """Test successful download and extraction."""
        mock_client = Mock()

        # Mock the download_fileobj to write to the temp file
        def mock_download(bucket, key, file_obj):
            file_obj.write(mock_pdf_bytes)

        mock_client.download_fileobj = Mock(side_effect=mock_download)

        # Mock PDF extraction
        with patch("app.s3_client.extract_text_from_pdf", return_value="Test PDF content"):
            result = download_and_extract_from_s3(mock_client, "s3://bucket/test.pdf")
            assert result is not None
            assert isinstance(result, str)
            assert "Test PDF content" in result

    @pytest.mark.asyncio
    async def test_init_async_s3_client_success(self, aws_credentials):
        """Test async S3 client initialization."""
        # init_async_s3_client returns an aioboto3.Session, not a context manager
        session = await init_async_s3_client()
        assert session is not None
        # Verify it's an aioboto3.Session
        assert hasattr(session, "client")

    def test_download_and_extract_client_error(self):
        """Test download handles general client errors."""
        mock_client = Mock()
        # AccessDenied raises 503, not 500
        mock_client.download_fileobj.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access denied"}}, "GetObject"
        )

        with pytest.raises(HTTPException) as exc_info:
            download_and_extract_from_s3(mock_client, "s3://bucket/test.pdf")
        assert exc_info.value.status_code == 503  # Service unavailable for client errors
