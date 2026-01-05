# Test Suite

Comprehensive test suite for JSE DataSphere Chatbot FastAPI application.

## Overview

This test suite provides unit and integration tests for all major components of the application, including:
- S3 client operations
- Metadata loading
- PDF processing
- Document selection
- Gemini AI client
- API endpoints
- Streaming chat functionality
- Financial data queries

## Test Organization

```
tests/
├── unit/                    # Unit tests for individual modules
│   ├── test_s3_client.py
│   ├── test_metadata_loader.py
│   ├── test_pdf_utils.py
│   ├── test_document_selector.py
│   └── test_gemini_client.py
├── integration/             # Integration tests
│   ├── test_api_endpoints.py
│   └── test_streaming_flow.py
├── conftest.py             # Shared fixtures and test configuration
└── README.md               # This file
```

## Running Tests

### Prerequisites

Install development dependencies:
```bash
cd fastapi_app
pip install -e ".[dev]"
```

### All Tests

Run the complete test suite:
```bash
pytest
```

### With Coverage

Generate coverage report:
```bash
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

View coverage in terminal:
```bash
pytest --cov=app --cov-report=term-missing
```

### Unit Tests Only

```bash
pytest tests/unit/
```

### Integration Tests Only

```bash
pytest tests/integration/
```

### Specific Test File

```bash
pytest tests/unit/test_s3_client.py
```

### Specific Test Class or Function

```bash
pytest tests/unit/test_s3_client.py::TestS3Client::test_init_s3_client_success
```

### Run Tests in Parallel

Speed up test execution using pytest-xdist:
```bash
pytest -n auto
```

### Watch Mode

Run tests automatically when files change:
```bash
pip install pytest-watch
ptw
```

## Test Markers

Tests are organized using pytest markers:

- **@pytest.mark.unit** - Unit tests for individual functions/classes
- **@pytest.mark.integration** - Integration tests across multiple components
- **@pytest.mark.slow** - Slow-running tests (skip with `-m "not slow"`)
- **@pytest.mark.asyncio** - Asynchronous tests

### Running Specific Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run unit and integration tests
pytest -m "unit or integration"
```

## Coverage Requirements

The project maintains a **70% code coverage** threshold:

```bash
# Check coverage against threshold
pytest --cov=app --cov-report=term --cov-fail-under=70
```

Coverage configuration is in `pyproject.toml`:
```toml
[tool.coverage.report]
fail_under = 70
```

## Writing Tests

### Test Structure (AAA Pattern)

Follow the Arrange-Act-Assert pattern:

```python
def test_example(mock_fixture):
    # Arrange: Set up test data and mocks
    test_data = {"key": "value"}

    # Act: Execute the code under test
    result = function_under_test(test_data)

    # Assert: Verify the results
    assert result == expected_value
```

### Unit Test Example

```python
import pytest
from unittest.mock import Mock, patch

@pytest.mark.unit
class TestMyModule:
    """Test cases for MyModule."""

    def test_function_success(self, mock_fixture):
        """Test successful function execution."""
        # Arrange
        mock_client = Mock()
        mock_client.method.return_value = "expected"

        # Act
        result = my_function(mock_client)

        # Assert
        assert result == "expected"
        mock_client.method.assert_called_once()
```

### Integration Test Example

```python
import pytest
from fastapi.testclient import TestClient

@pytest.mark.integration
class TestAPIIntegration:
    """Test API endpoint integration."""

    def test_endpoint(self, client):
        """Test endpoint returns expected response."""
        response = client.get("/endpoint")
        assert response.status_code == 200
        assert "expected_key" in response.json()
```

### Async Test Example

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await async_function()
    assert result is not None
```

## Fixtures

Common fixtures are defined in `conftest.py`:

- **mock_config** - Mock application configuration
- **mock_s3_client** - Mock S3 client
- **mock_metadata** - Mock metadata dictionary
- **mock_bigquery_client** - Mock BigQuery client
- **mock_gemini_model** - Mock Gemini AI model
- **test_client** - FastAPI TestClient
- **async_test_client** - Async HTTP client
- **mock_pdf_bytes** - Mock PDF file bytes
- **aws_credentials** - Mock AWS credentials

### Using Fixtures

```python
def test_with_fixture(mock_s3_client):
    """Test using a fixture."""
    result = function_using_s3(mock_s3_client)
    assert result is not None
```

## Mocking Best Practices

### Mock External Services

Always mock external services (S3, BigQuery, Gemini AI):

```python
from unittest.mock import patch, Mock

def test_with_mocked_service():
    with patch("app.module.external_service") as mock_service:
        mock_service.return_value = "mocked_response"
        result = function_calling_service()
        assert result == "expected"
```

### Mock Async Functions

Use `AsyncMock` for async functions:

```python
from unittest.mock import AsyncMock

async def test_async_mock():
    mock_func = AsyncMock(return_value="result")
    result = await mock_func()
    assert result == "result"
```

## Debugging Tests

### Run with Verbose Output

```bash
pytest -v
```

### Show Print Statements

```bash
pytest -s
```

### Stop at First Failure

```bash
pytest -x
```

### Drop into Debugger on Failure

```bash
pytest --pdb
```

### Run Last Failed Tests

```bash
pytest --lf
```

## CI/CD Integration

Tests run automatically on:
- Push to `main` or `refactor` branches
- Pull requests to `main` or `refactor`

GitHub Actions workflows:
- **test.yml** - Runs full test suite with coverage
- **lint.yml** - Runs code quality checks (Black, Ruff, MyPy)
- **pre-commit.yml** - Runs pre-commit hooks

## Continuous Testing

### Pre-commit Hooks

Install pre-commit hooks:
```bash
pre-commit install
```

Hooks run automatically before each commit:
- Code formatting (Black)
- Linting (Ruff)
- Test execution (fast tests only)

### Manual Pre-commit Run

```bash
pre-commit run --all-files
```

## Test Data

Test data files are located in:
- `tests/bq_test_data.csv` - Mock BigQuery test data

## Troubleshooting

### Import Errors

Ensure the app directory is in Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

Or use editable install:
```bash
pip install -e .
```

### Async Test Issues

If async tests hang, check event loop configuration in `conftest.py`:
```python
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
```

### Coverage Not Updating

Clear coverage cache:
```bash
rm -rf .coverage htmlcov/
pytest --cov=app --cov-report=html
```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)

## Contributing

When adding new features:

1. Write tests first (TDD)
2. Ensure tests pass locally
3. Maintain or improve coverage (≥70%)
4. Follow existing test patterns
5. Update this README if adding new test categories

## Questions?

For questions about the test suite, please refer to:
- This README
- `conftest.py` for fixture documentation
- Existing tests for examples
