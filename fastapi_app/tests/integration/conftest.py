"""Integration test configuration and fixtures."""

import os
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient


# Set mock environment variables before any imports
@pytest.fixture(scope="session", autouse=True)
def mock_environment():
    """Mock environment variables for all integration tests."""
    env_vars = {
        # AWS Config
        "AWS_ACCESS_KEY_ID": "test-access-key-id",
        "AWS_SECRET_ACCESS_KEY": "test-secret-access-key",
        "AWS_REGION": "us-east-1",
        "DOCUMENT_METADATA_S3_BUCKET": "test-bucket",
        # GCP Config
        "GEMINI_API_KEY": "test-gemini-api-key",
        "VERTEX_AI_PROJECT_ID": "test-project-id",
        "VERTEX_AI_LOCATION": "us-central1",
        # Redis Config - Set empty to use in-memory job store
        "REDIS_URL": "",
        # App Config
        "LOG_LEVEL": "INFO",
    }

    # Set all environment variables
    original_env = {}
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    yield

    # Restore original environment
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture(autouse=True)
def mock_external_services():
    """Auto-mock external services for all integration tests."""
    with patch("google.generativeai.configure"):
        with patch("google.generativeai.GenerativeModel"):
            with patch("app.main.init_vertex_ai"):  # Mock Vertex AI init
                # Mock S3 client initialization
                with patch("app.main.init_s3_client") as mock_init_s3:
                    mock_s3 = Mock()
                    mock_init_s3.return_value = mock_s3
                    # Mock metadata loading
                    with patch("app.main.load_metadata_from_s3") as mock_load_metadata:
                        mock_load_metadata.return_value = {
                            "companies": [
                                {
                                    "name": "MTN",
                                    "documents": [{"title": "Test Doc", "s3_path": "s3://test"}],
                                }
                            ]
                        }
                        # Mock FinancialDataManager
                        with patch("app.main.FinancialDataManager") as mock_fdm:
                            mock_fdm_instance = Mock()
                            mock_fdm_instance.metadata = {"test": "data"}
                            mock_fdm.return_value = mock_fdm_instance
                            yield


@pytest.fixture
def test_client():
    """Create test client with lifespan context."""
    from app.main import app

    # Use TestClient with lifespan context
    with TestClient(app) as client:
        yield client
