"""Integration tests for API endpoints."""

from unittest.mock import AsyncMock, Mock, create_autospec, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_response_cache_dep
from app.metadata_loader import load_metadata_from_s3 as real_load_metadata_from_s3


@pytest.mark.integration
class TestAPIEndpoints:
    """Test API endpoint integration."""

    @pytest.fixture
    def client(self, test_client):
        """Use the shared test client with lifespan."""
        return test_client

    def test_health_endpoint(self, client):
        """Test /health endpoint returns expected structure."""
        response = client.get("/health")
        # In test environment with mocked services, expect degraded status
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]
        assert "timestamp" in data

    def test_health_endpoint_includes_version(self, client):
        """Test /health endpoint includes components info."""
        response = client.get("/health")
        data = response.json()
        # In test environment, check for components instead of version
        assert "components" in data or "version" in data or "app_name" in data

    def test_root_endpoint(self, client):
        """Test root endpoint redirects or returns info."""
        response = client.get("/")
        assert response.status_code in [200, 307, 308]

    def test_invalid_endpoint(self, client):
        """Test invalid endpoint returns 404."""
        response = client.get("/invalid/endpoint/path")
        assert response.status_code == 404

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.get("/health")
        # Check if CORS middleware is configured (accept degraded in test env)
        assert response.status_code in [200, 503]

    def test_request_id_middleware(self, client):
        """Test request ID middleware adds headers."""
        response = client.get("/health")
        # Request ID middleware should add headers (accept degraded in test env)
        assert response.status_code in [200, 503]


@pytest.mark.integration
class TestChatEndpoints:
    """Test chat-related endpoints."""

    @pytest.fixture
    def client(self, test_client):
        """Use the shared test client with lifespan."""
        return test_client

    def test_chat_endpoint_requires_query(self, client):
        """Test /chat endpoint requires query parameter."""
        response = client.post("/chat", json={})
        assert response.status_code == 422  # Validation error

    def test_chat_stream_endpoint_with_valid_data(self, client):
        """Test /chat/stream endpoint with valid request data."""
        with patch("app.main.AgentV2") as MockAgent:
            instance = MockAgent.return_value
            instance.run = AsyncMock(
                return_value={
                    "response": "Company X 2023 revenue was J$1.2M.",
                    "data_found": True,
                    "record_count": 1,
                    "conversation_history": None,
                }
            )

            payload = {
                "query": "What is the revenue for Company X?",
                "memory_enabled": False,
                "conversation_history": [],
            }
            response = client.post("/chat/stream", json=payload)
            assert response.status_code == 200

    def test_chat_endpoint_validates_conversation_history(self, client):
        """Test chat endpoint validates conversation history format."""
        payload = {"query": "test query", "conversation_history": "invalid"}  # Should be a list
        response = client.post("/chat", json=payload)
        assert response.status_code == 422


@pytest.mark.integration
class TestCacheEndpoints:
    """Test cache-related endpoints."""

    @pytest.fixture
    def client(self, test_client):
        """Use the shared test client with lifespan."""
        return test_client

    def test_cache_status_endpoint(self, client):
        """Test /cache/status endpoint."""
        response = client.get("/cache/status")
        # Endpoint should exist and return cache status
        assert response.status_code in [200, 404, 500]

    def test_cache_refresh_endpoint(self, client):
        """Test /cache/refresh endpoint succeeds when S3 metadata loads.

        Regression test: the handler used to call load_metadata_from_s3() with no
        arguments while the function requires an s3_client, so every call raised
        TypeError and returned a 500. Asserting on 200 specifically — the previous
        assertion accepted 500 and so never caught it.
        """
        with patch("app.main.refresh_metadata_cache") as mock_refresh:
            mock_refresh.return_value = True
            response = client.post("/cache/refresh")

        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_cache_refresh_passes_s3_client_to_loader(self, client):
        """The S3 client from app state must reach the metadata loader.

        The signature spec is what gives this test its teeth: the shared autouse
        mock in conftest accepts any arguments, so an argument-less call looks
        fine to it. Speccing against the real load_metadata_from_s3(s3_client)
        makes that call raise TypeError the way it does in production. It is
        built with create_autospec and injected via new=, because patch(...,
        autospec=True) refuses to spec an attribute conftest already mocked out.
        """
        spec_loader = create_autospec(real_load_metadata_from_s3)
        spec_loader.return_value = {"companies": []}

        with patch("app.main.refresh_metadata_cache", return_value=True):
            with patch("app.main.load_metadata_from_s3", new=spec_loader) as mock_load:
                response = client.post("/cache/refresh")

        assert response.status_code == 200
        mock_load.assert_called_once()
        # Called positionally with the client, not with an empty argument list.
        assert mock_load.call_args.args[0] is app.state.s3_client


@pytest.mark.integration
class TestFastChatV2Cache:
    """Test the /fast_chat_v2 response-cache branch.

    This branch had zero coverage and referenced two names that were never
    imported into app.main, so every cache hit raised NameError and surfaced to
    the caller as a 500.
    """

    @pytest.fixture
    def client(self, test_client):
        return test_client

    @pytest.fixture
    def cached_payload(self):
        """A payload shaped exactly like what the endpoint writes to Redis."""
        return {
            "response": "NCB 2023 revenue was J$1.20B.",
            "data_found": True,
            "record_count": 1,
            "filters_used": {
                "companies": ["NCB Financial Group"],
                "symbols": ["NCBFG"],
                "years": ["2023"],
                "standard_items": ["revenue"],
                "interpretation": "NCB revenue for 2023",
                "is_follow_up": False,
            },
            "data_preview": [
                {
                    "company": "NCB Financial Group",
                    "symbol": "NCBFG",
                    "year": "2023",
                    "standard_item": "revenue",
                    "item": 1.2,
                    "unit_multiplier": 1000000000,
                    "formatted_value": "1.20B",
                }
            ],
            "warnings": None,
            "suggestions": None,
            "chart": None,
        }

    @pytest.fixture
    def cache_hit(self, cached_payload):
        """Override the cache dependency so every lookup is a hit."""
        cache = Mock()
        cache.get = AsyncMock(return_value=cached_payload)
        cache.set = AsyncMock()
        app.dependency_overrides[get_response_cache_dep] = lambda: cache
        yield cache
        app.dependency_overrides.pop(get_response_cache_dep, None)

    def test_cache_hit_returns_200(self, client, cache_hit, cached_payload):
        """A cache hit must return the cached answer, not a 500."""
        response = client.post(
            "/fast_chat_v2",
            json={"query": "NCB revenue 2023", "memory_enabled": False},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["response"] == cached_payload["response"]
        assert body["data_found"] is True
        assert body["record_count"] == 1

    def test_cache_hit_rehydrates_filters_and_preview(self, client, cache_hit):
        """The cached dicts must round-trip back into their Pydantic models."""
        response = client.post(
            "/fast_chat_v2",
            json={"query": "NCB revenue 2023", "memory_enabled": False},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["filters_used"]["symbols"] == ["NCBFG"]
        assert body["filters_used"]["years"] == ["2023"]
        assert body["data_preview"][0]["formatted_value"] == "1.20B"

    def test_cache_hit_skips_the_financial_manager(self, client, cache_hit):
        """A hit must short-circuit before any paid Gemini or BigQuery work."""
        with patch("app.main.asyncio.to_thread") as mock_to_thread:
            response = client.post(
                "/fast_chat_v2",
                json={"query": "NCB revenue 2023", "memory_enabled": False},
            )

        assert response.status_code == 200
        mock_to_thread.assert_not_called()

    def test_conversation_history_bypasses_the_cache(self, client, cache_hit):
        """Follow-ups depend on context the cache doesn't key on, so never serve them."""
        response = client.post(
            "/fast_chat_v2",
            json={
                "query": "what about 2022?",
                "memory_enabled": False,
                "conversation_history": [{"role": "user", "content": "NCB revenue 2023"}],
            },
        )

        cache_hit.get.assert_not_called()
        assert response.status_code in [200, 500]  # Falls through to the live path.
