# Streaming Functionality Tests

This directory contains comprehensive tests for the SSE (Server-Sent Events) streaming functionality in the Jacie chatbot.

## Overview

The streaming tests cover:

1. **Unit Tests** (`test_streaming_units.py`) - Test individual components
2. **Integration Tests** (`test_streaming.py`) - Test end-to-end functionality
3. **Test Configuration** (`conftest.py`) - Common fixtures and utilities
4. **Test Runner** (`run_streaming_tests.py`) - Easy test execution

## Test Coverage

### Unit Tests (`test_streaming_units.py`)

#### ProgressTracker Tests
- ✅ `test_emit_progress` - Test progress update emission
- ✅ `test_emit_final_result` - Test final result emission
- ✅ `test_emit_error` - Test error message emission
- ✅ `test_stream_updates_progress` - Test SSE progress event streaming
- ✅ `test_stream_updates_result` - Test SSE result event streaming
- ✅ `test_stream_updates_error` - Test SSE error event streaming
- ✅ `test_stream_updates_heartbeat` - Test heartbeat functionality
- ✅ `test_stream_updates_multiple_events` - Test multiple events in sequence

#### Streaming Chat Tests
- ✅ `test_process_streaming_chat_creates_tracker` - Test tracker creation
- ✅ `test_process_streaming_chat_starts_background_task` - Test background task creation
- ✅ `test_process_chat_async_with_valid_metadata` - Test valid metadata handling
- ✅ `test_process_chat_async_without_metadata` - Test missing metadata error handling
- ✅ `test_process_chat_async_fast_mode` - Test fast mode processing

#### Streaming Financial Chat Tests
- ✅ `test_process_streaming_financial_chat_creates_tracker` - Test financial tracker creation
- ✅ `test_process_streaming_financial_chat_starts_background_task` - Test financial background task

#### SSE Event Handling Tests
- ✅ `test_format_sse_message` - Test SSE message formatting
- ✅ `test_sse_message_parsing` - Test SSE message parsing
- ✅ `test_multiple_sse_messages` - Test multiple SSE message parsing

#### ProgressUpdate Model Tests
- ✅ `test_progress_update_creation` - Test ProgressUpdate model creation
- ✅ `test_progress_update_serialization` - Test JSON serialization

#### Integration Tests
- ✅ `test_full_streaming_flow` - Test complete streaming flow

### Integration Tests (`test_streaming.py`)

The integration tests test the actual API endpoints:

- ✅ `/chat/stream` - Traditional chat streaming
- ✅ `/fast_chat/stream` - Vector DB chat streaming
- ✅ `/fast_chat_v2/stream` - Financial data streaming

## Running the Tests

### Quick Start

```bash
# Run all streaming tests
cd fastapi_app
python tests/run_streaming_tests.py

# Run with verbose output
python tests/run_streaming_tests.py --verbose

# Run with coverage report
python tests/run_streaming_tests.py --coverage
```

### Test Types

```bash
# Run only unit tests
python tests/run_streaming_tests.py --type unit

# Run only integration tests
python tests/run_streaming_tests.py --type integration

# Run all tests (default)
python tests/run_streaming_tests.py --type all
```

### Specific Tests

```bash
# List all available tests
python tests/run_streaming_tests.py --list

# Run a specific test
python tests/run_streaming_tests.py --test-name test_emit_progress

# Run tests matching a pattern
python tests/run_streaming_tests.py --test-name "test_stream_updates"
```

### Advanced Options

```bash
# Run tests in parallel (requires pytest-xdist)
python tests/run_streaming_tests.py --parallel

# Run with coverage and generate HTML report
python tests/run_streaming_tests.py --coverage

# Combine options
python tests/run_streaming_tests.py --type unit --verbose --coverage
```

### Direct Pytest Usage

You can also run tests directly with pytest:

```bash
# Run unit tests
pytest tests/test_streaming_units.py -v

# Run specific test class
pytest tests/test_streaming_units.py::TestProgressTracker -v

# Run specific test method
pytest tests/test_streaming_units.py::TestProgressTracker::test_emit_progress -v

# Run with coverage
pytest tests/test_streaming_units.py --cov=app --cov-report=html
```

## Test Dependencies

The tests require the following packages:

```bash
pip install pytest pytest-asyncio pytest-cov pytest-xdist
```

## Test Structure

### Fixtures (`conftest.py`)

Common test fixtures include:

- `mock_streaming_request` - Mock StreamingChatRequest
- `mock_s3_client` - Mock S3 client
- `mock_metadata` - Mock document metadata
- `mock_chroma_collection` - Mock ChromaDB collection
- `mock_financial_manager` - Mock financial data manager
- `sample_progress_update` - Sample ProgressUpdate
- `sample_sse_message` - Sample SSE message
- `sample_sse_stream` - Sample SSE stream

### Helper Functions

- `parse_sse_message()` - Parse SSE messages
- `create_mock_response()` - Create mock HTTP responses
- `create_mock_streaming_response()` - Create mock streaming responses

## What the Tests Verify

### ProgressTracker
- ✅ Progress updates are properly queued and emitted
- ✅ Final results are correctly formatted and sent
- ✅ Error messages are properly handled
- ✅ Heartbeat functionality works when no updates are available
- ✅ Multiple events are processed in correct sequence
- ✅ SSE message format is correct

### Streaming Chat Processing
- ✅ Background tasks are properly created
- ✅ ProgressTracker instances are correctly initialized
- ✅ Error handling works for missing metadata
- ✅ Both traditional and fast modes work correctly
- ✅ Async processing doesn't block the event loop

### SSE Event Handling
- ✅ SSE messages are properly formatted
- ✅ Event parsing works correctly
- ✅ Multiple events in a stream are handled properly
- ✅ JSON serialization/deserialization works

### Integration
- ✅ Complete streaming flows work end-to-end
- ✅ Real-time progress updates are visible
- ✅ Final results are delivered correctly
- ✅ Error conditions are handled gracefully

## Debugging Tests

### Verbose Output
```bash
python tests/run_streaming_tests.py --verbose
```

### Specific Test Debugging
```bash
# Run with maximum verbosity
pytest tests/test_streaming_units.py::TestProgressTracker::test_emit_progress -vvv -s

# Run with print statements visible
pytest tests/test_streaming_units.py -s
```

### Coverage Analysis
```bash
# Generate HTML coverage report
python tests/run_streaming_tests.py --coverage

# View coverage in browser
open htmlcov/index.html
```

## Adding New Tests

When adding new streaming functionality:

1. **Unit Tests**: Add to `test_streaming_units.py`
2. **Integration Tests**: Add to `test_streaming.py`
3. **Fixtures**: Add to `conftest.py` if reusable
4. **Documentation**: Update this README

### Test Naming Convention

- Unit tests: `test_<function_name>_<scenario>`
- Integration tests: `test_<endpoint>_<scenario>`
- Classes: `Test<ComponentName>`

### Example Test Structure

```python
@pytest.mark.asyncio
async def test_new_functionality(self, mock_request, mock_s3_client):
    """Test description."""
    # Arrange
    tracker = ProgressTracker()

    # Act
    await tracker.emit_progress("test", "message", 50.0)

    # Assert
    update = await tracker.updates_queue.get()
    assert update.step == "test"
    assert update.progress == 50.0
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions step
- name: Run streaming tests
  run: |
    cd fastapi_app
    python tests/run_streaming_tests.py --type unit --coverage
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're in the `fastapi_app` directory
2. **Async Test Failures**: Check that `pytest-asyncio` is installed
3. **Mock Issues**: Verify mock objects are properly configured
4. **SSE Parsing Errors**: Check message format matches expected structure

### Test Environment

```bash
# Set up test environment
cd fastapi_app
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov pytest-xdist
```

## Performance Considerations

- Unit tests should run quickly (< 1 second each)
- Integration tests may take longer due to real API calls
- Use `--parallel` for faster execution on multi-core systems
- Monitor test execution time to catch performance regressions
