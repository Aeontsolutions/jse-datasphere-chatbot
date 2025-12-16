#!/usr/bin/env python3
"""
Comprehensive test fixtures for JSE DataSphere Chatbot.

Provides mocked dependencies and test data for unit and integration tests.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from app.models import StreamingChatRequest, ProgressUpdate
import boto3
from io import BytesIO


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_streaming_request():
    """Create a mock StreamingChatRequest for testing."""
    return StreamingChatRequest(
        query="Test query about financial data",
        auto_load_documents=True,
        memory_enabled=True,
        conversation_history=[
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ],
    )


@pytest.fixture
def mock_config():
    """Mock application configuration."""
    config = Mock()
    config.aws.region = "us-east-1"
    config.aws.s3_bucket = "test-bucket"
    config.aws.access_key_id = "test-key"
    config.aws.secret_access_key = "test-secret"
    config.gcp.project_id = "test-project"
    config.gcp.api_key = "test-api-key"
    config.bigquery.dataset = "test-dataset"
    config.bigquery.table = "test-table"
    config.redis.url = None
    config.log_level = "INFO"
    return config


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client for testing."""
    client = Mock()
    client.download_file = AsyncMock()
    client.head_object = AsyncMock()
    client.download_fileobj = Mock()
    client.get_object = Mock(return_value={"Body": Mock(read=Mock(return_value=b"test content"))})
    return client


@pytest.fixture
def mock_metadata():
    """Create mock metadata for testing."""
    return {
        "companies": ["MTN", "VOD", "JBG"],
        "documents": [
            {
                "filename": "mtn_annual_report_2023.pdf",
                "company": "MTN",
                "year": "2023",
                "type": "annual_report",
            },
            {
                "filename": "vod_financial_statements_2023.pdf",
                "company": "VOD",
                "year": "2023",
                "type": "financial_statements",
            },
        ],
    }


@pytest.fixture
def mock_bigquery_client():
    """Mock BigQuery client for testing."""
    client = Mock()
    client.query = Mock(return_value=Mock(result=Mock(return_value=[])))
    return client


@pytest.fixture
def mock_gemini_model():
    """Mock Gemini AI model for testing."""
    model = Mock()
    model.generate_content = Mock(return_value=Mock(text='{"answer": "test response"}'))
    return model


@pytest.fixture
def test_client():
    """FastAPI test client."""
    from app.main import app

    return TestClient(app)


@pytest.fixture
async def async_test_client():
    """Async FastAPI test client."""
    from httpx import AsyncClient
    from app.main import app

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_pdf_bytes():
    """Mock PDF file bytes for testing."""
    # Simple PDF header (not a real PDF, but enough for testing)
    return b"%PDF-1.4\n%Test PDF content\n%%EOF"


@pytest.fixture
def aws_credentials(monkeypatch):
    """Mocked AWS Credentials for boto3."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture
def mock_chroma_collection():
    """Create a mock ChromaDB collection for testing."""
    collection = Mock()
    collection.query = AsyncMock(
        return_value={
            "documents": [["Test document content"]],
            "metadatas": [[{"filename": "test.pdf"}]],
            "distances": [[0.1]],
        }
    )
    return collection


@pytest.fixture
def mock_meta_collection():
    """Create a mock metadata collection for testing."""
    collection = Mock()
    collection.query = AsyncMock(
        return_value={
            "documents": [["Test metadata"]],
            "metadatas": [[{"company": "MTN"}]],
            "distances": [[0.1]],
        }
    )
    return collection


@pytest.fixture
def mock_financial_manager():
    """Create a mock financial manager for testing."""
    manager = Mock()

    # Mock the filters object
    mock_filters = Mock()
    mock_filters.companies = ["MTN"]
    mock_filters.years = ["2023"]
    mock_filters.standard_items = ["revenue"]
    mock_filters.interpretation = "Query about MTN revenue in 2023"
    mock_filters.is_follow_up = False

    manager.parse_user_query = AsyncMock(return_value=mock_filters)
    manager.validate_data_availability = AsyncMock(return_value={"warnings": [], "suggestions": []})
    manager.query_data = AsyncMock(
        return_value=[{"company": "MTN", "year": "2023", "revenue": 1000000}]
    )
    manager.format_response = AsyncMock(return_value="MTN revenue in 2023 was 1,000,000")

    return manager


@pytest.fixture
def sample_progress_update():
    """Create a sample ProgressUpdate for testing."""
    return ProgressUpdate(
        step="test_step",
        message="Test progress message",
        progress=50.0,
        timestamp="2024-01-01T12:00:00Z",
        details={"documents_loaded": 2, "companies": ["MTN"]},
    )


@pytest.fixture
def sample_sse_message():
    """Create a sample SSE message for testing."""
    return 'event: progress\ndata: {"step": "test", "progress": 50, "message": "Test"}\n\n'


@pytest.fixture
def sample_sse_stream():
    """Create a sample SSE stream with multiple events."""
    return (
        'event: progress\ndata: {"step": "start", "progress": 5, "message": "Starting..."}\n\n'
        'event: progress\ndata: {"step": "processing", "progress": 50, "message": "Processing..."}\n\n'
        'event: result\ndata: {"response": "Final result", "documents_loaded": ["doc1.pdf"]}\n\n'
    )


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


# Async test utilities
class AsyncTestCase:
    """Base class for async test cases."""

    @pytest.fixture(autouse=True)
    def setup_event_loop(self, event_loop):
        """Set up the event loop for async tests."""
        self.loop = event_loop
        asyncio.set_event_loop(event_loop)

    async def assert_async_raises(self, exception_class, coro):
        """Assert that a coroutine raises a specific exception."""
        with pytest.raises(exception_class):
            await coro

    async def assert_async_not_raises(self, coro):
        """Assert that a coroutine does not raise any exception."""
        try:
            await coro
        except Exception as e:
            pytest.fail(f"Expected no exception, but got {type(e).__name__}: {e}")


# Helper functions for testing
def parse_sse_message(message):
    """Parse an SSE message and return event type and data."""
    lines = message.strip().split("\n")
    event_type = ""
    data = {}

    for line in lines:
        if line.startswith("event: "):
            event_type = line[7:]
        elif line.startswith("data: "):
            import json

            data_str = line[6:]
            if data_str and data_str != "{}":
                data = json.loads(data_str)

    return event_type, data


def create_mock_response(data, status_code=200):
    """Create a mock response object for testing."""
    response = Mock()
    response.status_code = status_code
    response.json.return_value = data
    response.raise_for_status = Mock()
    return response


def create_mock_streaming_response(events):
    """Create a mock streaming response with SSE events."""
    response = Mock()
    response.status_code = 200
    response.iter_content.return_value = [event.encode("utf-8") for event in events]
    response.raise_for_status = Mock()
    return response
