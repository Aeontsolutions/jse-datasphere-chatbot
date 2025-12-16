"""Unit tests for S3 client module."""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from botocore.exceptions import ClientError
from fastapi import HTTPException
import aioboto3

from app.s3_client import (
    init_s3_client,
    init_async_s3_client,
    download_and_extract_from_s3,
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
        mock_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject"
        )

        with pytest.raises(HTTPException) as exc_info:
            download_and_extract_from_s3(mock_client, "s3://bucket/missing.pdf")
        assert exc_info.value.status_code == 404

    def test_download_and_extract_success(self, mock_pdf_bytes):
        """Test successful download and extraction."""
        mock_client = Mock()
        mock_body = Mock()
        mock_body.read.return_value = mock_pdf_bytes
        mock_client.get_object.return_value = {"Body": mock_body}

        result = download_and_extract_from_s3(mock_client, "s3://bucket/test.pdf")
        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_init_async_s3_client_success(self, aws_credentials):
        """Test async S3 client initialization."""
        with patch("aioboto3.Session") as mock_session:
            mock_client = AsyncMock()
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_client
            mock_session.return_value.client.return_value = mock_context

            from app.s3_client import init_async_s3_client

            async with init_async_s3_client() as client:
                assert client is not None

    def test_download_and_extract_client_error(self):
        """Test download handles general client errors."""
        mock_client = Mock()
        mock_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access denied"}}, "GetObject"
        )

        with pytest.raises(HTTPException) as exc_info:
            download_and_extract_from_s3(mock_client, "s3://bucket/test.pdf")
        assert exc_info.value.status_code == 500
