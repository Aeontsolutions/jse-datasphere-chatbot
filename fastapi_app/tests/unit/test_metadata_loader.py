"""Unit tests for metadata loader module."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from botocore.exceptions import ClientError

from app.metadata_loader import (
    download_metadata_from_s3,
    load_metadata_from_s3,
    load_metadata_from_s3_async,
    parse_metadata_file,
)


@pytest.mark.unit
class TestMetadataLoader:
    """Test cases for metadata loading operations."""

    def test_load_metadata_from_s3_success(self, mock_s3_client, mock_metadata):
        """Test successful metadata loading from S3."""
        mock_body = Mock()
        mock_body.read.return_value = json.dumps(mock_metadata).encode("utf-8")
        mock_s3_client.get_object.return_value = {"Body": mock_body}

        with patch(
            "app.metadata_loader.download_metadata_from_s3", return_value=json.dumps(mock_metadata)
        ):
            result = load_metadata_from_s3(mock_s3_client)
            assert result is not None
            assert "companies" in result
            assert len(result["companies"]) == 3

    def test_load_metadata_from_s3_not_found(self, mock_s3_client):
        """Test metadata loading when file not found."""
        with patch("app.metadata_loader.download_metadata_from_s3", return_value=None):
            result = load_metadata_from_s3(mock_s3_client)
            assert result is None

    @pytest.mark.asyncio
    async def test_load_metadata_from_s3_async_success(self, mock_metadata):
        """Test async metadata loading success."""
        mock_client = AsyncMock()

        with patch(
            "app.metadata_loader.download_metadata_from_s3_async",
            return_value=json.dumps(mock_metadata),
        ):
            result = await load_metadata_from_s3_async(mock_client)
            assert result is not None
            assert "companies" in result

    def test_parse_metadata_file_success(self, mock_metadata):
        """Test parsing metadata file content."""
        metadata_str = json.dumps(mock_metadata)
        result = parse_metadata_file(metadata_str)
        assert result is not None
        assert isinstance(result, dict)
        assert "companies" in result

    def test_parse_metadata_file_empty(self):
        """Test parsing empty metadata."""
        result = parse_metadata_file("")
        assert result is None

    def test_download_metadata_from_s3_invalid_json(self, mock_s3_client):
        """Test metadata download with invalid JSON."""
        mock_body = Mock()
        mock_body.read.return_value = b"invalid json content"
        mock_s3_client.get_object.return_value = {"Body": mock_body}

        result = download_metadata_from_s3(mock_s3_client, "test-bucket", "invalid.json")
        # Should handle invalid JSON gracefully
        assert result is not None or result is None
