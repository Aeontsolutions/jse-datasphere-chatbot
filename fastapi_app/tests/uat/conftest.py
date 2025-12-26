"""
UAT-specific pytest fixtures and configuration.

These fixtures support running UAT tests against a live API endpoint,
either locally or deployed.
"""

import json
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

import httpx
import pytest
import pytest_asyncio

from .uat_runner import UATTestRunner


# Path to test cases JSON
TEST_CASES_PATH = Path(__file__).parent / "prompt_optimization_test_cases.json"


def pytest_addoption(parser):
    """Add custom pytest options for UAT tests."""
    parser.addoption(
        "--uat-url",
        action="store",
        default="http://localhost:8000",
        help="Base URL for UAT tests (default: http://localhost:8000)",
    )
    parser.addoption(
        "--uat-timeout",
        action="store",
        default="60",
        help="Timeout in seconds for each test request (default: 60)",
    )
    parser.addoption(
        "--uat-retries",
        action="store",
        default="0",
        help="Number of retry attempts for failed requests (default: 0)",
    )
    parser.addoption(
        "--uat-concurrency",
        action="store",
        default="1",
        help="Max concurrent requests for parallel mode (default: 1)",
    )


@pytest.fixture(scope="session")
def uat_base_url(request) -> str:
    """Get the UAT base URL from pytest options."""
    return request.config.getoption("--uat-url")


@pytest.fixture(scope="session")
def uat_timeout(request) -> float:
    """Get the UAT timeout from pytest options."""
    return float(request.config.getoption("--uat-timeout"))


@pytest.fixture(scope="session")
def uat_retries(request) -> int:
    """Get the UAT retries from pytest options."""
    return int(request.config.getoption("--uat-retries"))


@pytest.fixture(scope="session")
def uat_concurrency(request) -> int:
    """Get the UAT concurrency from pytest options."""
    return int(request.config.getoption("--uat-concurrency"))


@pytest.fixture(scope="session")
def test_cases() -> Dict[str, List[Dict[str, Any]]]:
    """Load all test cases from JSON."""
    with open(TEST_CASES_PATH, "r") as f:
        data = json.load(f)

    return {
        "tool_routing": data.get("tool_routing_test_cases", []),
        "prompt_optimization": data.get("prompt_optimization_test_cases", []),
        "clarification": data.get("clarification_test_cases", []),
        "defaults": data.get("defaults_test_cases", []),
        "edge_cases": data.get("edge_cases", []),
    }


@pytest.fixture(scope="session")
def uat_runner(uat_base_url, uat_timeout, uat_retries, uat_concurrency) -> UATTestRunner:
    """Create a UAT runner instance configured from pytest options."""
    runner = UATTestRunner(
        base_url=uat_base_url,
        timeout=uat_timeout,
        max_retries=uat_retries,
        concurrency=uat_concurrency,
    )
    runner.load_test_cases()
    return runner


@pytest_asyncio.fixture(scope="session")
async def uat_client(uat_base_url, uat_timeout) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Shared async HTTP client for UAT tests."""
    async with httpx.AsyncClient(base_url=uat_base_url, timeout=uat_timeout) as client:
        yield client


@pytest_asyncio.fixture(scope="function")
async def health_check(uat_client: httpx.AsyncClient):
    """Ensure the target API is healthy before running tests."""
    try:
        response = await uat_client.get("/health")
        if response.status_code not in [200, 503]:
            pytest.skip(f"API health check returned unexpected status: {response.status_code}")
    except httpx.ConnectError:
        pytest.skip("API not reachable - ensure the server is running")
    except httpx.TimeoutException:
        pytest.skip("API health check timed out")


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "uat: marks tests as UAT (User Acceptance Tests)")
    config.addinivalue_line("markers", "tool_routing: tool routing tests")
    config.addinivalue_line("markers", "prompt_optimization: prompt optimization tests")
    config.addinivalue_line("markers", "clarification: clarification flow tests")
    config.addinivalue_line("markers", "defaults: defaults application tests")
    config.addinivalue_line("markers", "edge_cases: edge case tests")
