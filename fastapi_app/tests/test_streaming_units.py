#!/usr/bin/env python3
"""
Unit tests for the SSE streaming functionality.
Tests the ProgressTracker, streaming chat processing, and SSE event handling.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch

# Import the modules to test
from app.progress_tracker import ProgressTracker, format_sse_message
from app.models import ProgressUpdate, StreamingChatRequest
from app.streaming_chat import process_streaming_chat, _process_chat_async
from app.streaming_financial_chat import process_streaming_financial_chat


class TestProgressTracker:
    """Test the ProgressTracker class functionality."""

    @pytest.fixture
    def tracker(self):
        """Create a fresh ProgressTracker instance for each test."""
        return ProgressTracker()

    @pytest.mark.asyncio
    async def test_emit_progress(self, tracker):
        """Test emitting progress updates."""
        await tracker.emit_progress("test_step", "Test message", 50.0, {"key": "value"})

        # Check that the update was queued
        update = await tracker.updates_queue.get()
        assert isinstance(update, ProgressUpdate)
        assert update.step == "test_step"
        assert update.message == "Test message"
        assert update.progress == 50.0
        assert update.details == {"key": "value"}
        assert tracker.current_step == "test_step"
        assert tracker.current_progress == 50.0

    @pytest.mark.asyncio
    async def test_emit_final_result(self, tracker):
        """Test emitting final result."""
        result_data = {"response": "Test response", "documents": ["doc1.pdf"]}
        await tracker.emit_final_result(result_data)

        # Check that the result was queued
        update = await tracker.updates_queue.get()
        assert isinstance(update, dict)
        assert update["type"] == "result"
        assert update["data"] == result_data

    @pytest.mark.asyncio
    async def test_emit_error(self, tracker):
        """Test emitting error messages."""
        error_msg = "Test error message"
        await tracker.emit_error(error_msg)

        # Check that the error was queued
        update = await tracker.updates_queue.get()
        assert isinstance(update, dict)
        assert update["type"] == "error"
        assert update["error"] == error_msg

    @pytest.mark.asyncio
    async def test_stream_updates_progress(self, tracker):
        """Test streaming progress updates."""
        # Emit a progress update
        await tracker.emit_progress("test_step", "Test message", 25.0)

        # Get the stream generator
        stream = tracker.stream_updates()

        # Get the first update
        update = await anext(stream)

        # Parse the SSE format
        lines = update.strip().split("\n")
        assert lines[0] == "event: progress"

        # Parse the data
        data_line = lines[1]
        assert data_line.startswith("data: ")
        data_json = data_line[6:]  # Remove "data: " prefix
        data = json.loads(data_json)

        assert data["step"] == "test_step"
        assert data["message"] == "Test message"
        assert data["progress"] == 25.0

    @pytest.mark.asyncio
    async def test_stream_updates_result(self, tracker):
        """Test streaming final result."""
        result_data = {"response": "Test response"}
        await tracker.emit_final_result(result_data)

        # Get the stream generator
        stream = tracker.stream_updates()

        # Get the result update
        update = await anext(stream)

        # Parse the SSE format
        lines = update.strip().split("\n")
        assert lines[0] == "event: result"

        # Parse the data
        data_line = lines[1]
        assert data_line.startswith("data: ")
        data_json = data_line[6:]
        data = json.loads(data_json)

        assert data == result_data

    @pytest.mark.asyncio
    async def test_stream_updates_error(self, tracker):
        """Test streaming error messages."""
        error_msg = "Test error"
        await tracker.emit_error(error_msg)

        # Get the stream generator
        stream = tracker.stream_updates()

        # Get the error update
        update = await anext(stream)

        # Parse the SSE format
        lines = update.strip().split("\n")
        assert lines[0] == "event: error"

        # Parse the data
        data_line = lines[1]
        assert data_line.startswith("data: ")
        data_json = data_line[6:]
        data = json.loads(data_json)

        assert data["error"] == error_msg

    @pytest.mark.asyncio
    async def test_stream_updates_heartbeat(self, tracker):
        """Test heartbeat functionality when no updates are available."""
        stream = tracker.stream_updates()

        # Wait for heartbeat (should be sent after 1 second timeout)
        update = await asyncio.wait_for(anext(stream), timeout=1.5)

        # Parse the SSE format
        lines = update.strip().split("\n")
        assert lines[0] == "event: heartbeat"
        assert lines[1] == "data: {}"

    @pytest.mark.asyncio
    async def test_stream_updates_multiple_events(self, tracker):
        """Test streaming multiple events in sequence."""
        # Emit multiple updates
        await tracker.emit_progress("step1", "First step", 25.0)
        await tracker.emit_progress("step2", "Second step", 50.0)
        await tracker.emit_final_result({"response": "Final result"})

        # Get the stream generator
        stream = tracker.stream_updates()

        # Get all updates
        updates = []
        async for update in stream:
            updates.append(update)
            if "event: result" in update:
                break

        # Should have 3 updates: 2 progress + 1 result
        assert len(updates) == 3

        # Check first progress update
        lines1 = updates[0].strip().split("\n")
        assert lines1[0] == "event: progress"
        data1 = json.loads(lines1[1][6:])
        assert data1["step"] == "step1"
        assert data1["progress"] == 25.0

        # Check second progress update
        lines2 = updates[1].strip().split("\n")
        assert lines2[0] == "event: progress"
        data2 = json.loads(lines2[1][6:])
        assert data2["step"] == "step2"
        assert data2["progress"] == 50.0

        # Check final result
        lines3 = updates[2].strip().split("\n")
        assert lines3[0] == "event: result"
        data3 = json.loads(lines3[1][6:])
        assert data3["response"] == "Final result"


class TestStreamingChat:
    """Test the streaming chat processing functionality."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock StreamingChatRequest."""
        return StreamingChatRequest(
            query="Test query",
            auto_load_documents=True,
            memory_enabled=True,
            conversation_history=[],
        )

    @pytest.fixture
    def mock_s3_client(self):
        """Create a mock S3 client."""
        return Mock()

    @pytest.fixture
    def mock_metadata(self):
        """Create mock metadata."""
        return {
            "companies": ["Test Company"],
            "documents": [{"filename": "test_doc.pdf", "company": "Test Company"}],
        }

    @pytest.mark.asyncio
    async def test_process_streaming_chat_creates_tracker(
        self, mock_streaming_request, mock_s3_client, mock_metadata
    ):
        """Test that process_streaming_chat creates and returns a ProgressTracker."""
        tracker = await process_streaming_chat(
            request=mock_streaming_request,
            s3_client=mock_s3_client,
            metadata=mock_metadata,
            use_fast_mode=False,
        )

        assert isinstance(tracker, ProgressTracker)

    @pytest.mark.asyncio
    async def test_process_streaming_chat_starts_background_task(
        self, mock_streaming_request, mock_s3_client, mock_metadata
    ):
        """Test that process_streaming_chat starts a background task."""
        with patch("asyncio.create_task") as mock_create_task:
            tracker = await process_streaming_chat(
                request=mock_streaming_request,
                s3_client=mock_s3_client,
                metadata=mock_metadata,
                use_fast_mode=False,
            )

            # Verify that create_task was called
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_chat_async_with_valid_metadata(
        self, mock_streaming_request, mock_s3_client, mock_metadata
    ):
        """Test _process_chat_async with valid metadata."""
        tracker = ProgressTracker()

        with patch("app.streaming_chat._process_traditional_chat") as mock_traditional:
            await _process_chat_async(
                request=mock_streaming_request,
                s3_client=mock_s3_client,
                metadata=mock_metadata,
                tracker=tracker,
                use_fast_mode=False,
            )

            # Verify that traditional chat was called
            mock_traditional.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_chat_async_without_metadata(
        self, mock_streaming_request, mock_s3_client
    ):
        """Test _process_chat_async without metadata (should emit error)."""
        tracker = ProgressTracker()

        await _process_chat_async(
            request=mock_streaming_request,
            s3_client=mock_s3_client,
            metadata=None,
            tracker=tracker,
            use_fast_mode=False,
        )

        # Check that progress was emitted first, then error check, then error
        progress_update = await tracker.updates_queue.get()
        assert isinstance(progress_update, ProgressUpdate)
        assert progress_update.step == "start"

        error_check_update = await tracker.updates_queue.get()
        assert isinstance(error_check_update, ProgressUpdate)
        assert error_check_update.step == "error_check"

        error_update = await tracker.updates_queue.get()
        assert isinstance(error_update, dict)
        assert error_update["type"] == "error"
        assert "Metadata not available" in error_update["error"]

    @pytest.mark.asyncio
    async def test_process_chat_async_fast_mode(
        self, mock_streaming_request, mock_s3_client, mock_metadata
    ):
        """Test _process_chat_async in fast mode."""
        tracker = ProgressTracker()
        mock_collection = Mock()
        mock_meta_collection = Mock()

        with patch("app.streaming_chat._process_fast_chat") as mock_fast:
            await _process_chat_async(
                request=mock_streaming_request,
                s3_client=mock_s3_client,
                metadata=mock_metadata,
                tracker=tracker,
                collection=mock_collection,
                meta_collection=mock_meta_collection,
                use_fast_mode=True,
            )

            # Verify that fast chat was called
            mock_fast.assert_called_once()


class TestStreamingFinancialChat:
    """Test the streaming financial chat functionality."""

    @pytest.fixture
    def mock_financial_manager(self):
        """Create a mock financial manager."""
        manager = Mock()
        manager.parse_user_query = AsyncMock(return_value=Mock())
        manager.validate_data_availability = AsyncMock(return_value={})
        manager.query_data = AsyncMock(return_value=[])
        manager.format_response = AsyncMock(return_value="Test response")
        return manager

    @pytest.mark.asyncio
    async def test_process_streaming_financial_chat_creates_tracker(
        self, mock_streaming_request, mock_financial_manager
    ):
        """Test that process_streaming_financial_chat creates and returns a ProgressTracker."""
        tracker = await process_streaming_financial_chat(
            request=mock_streaming_request, financial_manager=mock_financial_manager
        )

        assert isinstance(tracker, ProgressTracker)

    @pytest.mark.asyncio
    async def test_process_streaming_financial_chat_starts_background_task(
        self, mock_streaming_request, mock_financial_manager
    ):
        """Test that process_streaming_financial_chat starts a background task."""
        with patch("asyncio.create_task") as mock_create_task:
            tracker = await process_streaming_financial_chat(
                request=mock_streaming_request, financial_manager=mock_financial_manager
            )

            # Verify that create_task was called
            mock_create_task.assert_called_once()


class TestSSEEventHandling:
    """Test SSE event handling and parsing."""

    def test_format_sse_message(self):
        """Test the format_sse_message utility function."""
        event = "test_event"
        data = '{"key": "value"}'

        result = format_sse_message(event, data)

        expected = 'event: test_event\ndata: {"key": "value"}\n\n'
        assert result == expected

    def test_sse_message_parsing(self):
        """Test parsing SSE messages."""
        # Simulate an SSE message
        sse_message = 'event: progress\ndata: {"step": "test", "progress": 50}\n\n'

        lines = sse_message.strip().split("\n")
        current_event = ""
        parsed_data = None

        for line in lines:
            if line.startswith("event: "):
                current_event = line[7:]
            elif line.startswith("data: "):
                data_json = line[6:]
                parsed_data = json.loads(data_json)

        assert current_event == "progress"
        assert parsed_data["step"] == "test"
        assert parsed_data["progress"] == 50

    def test_multiple_sse_messages(self):
        """Test parsing multiple SSE messages."""
        # Simulate multiple SSE messages
        sse_data = (
            'event: progress\ndata: {"step": "step1", "progress": 25}\n\n'
            'event: progress\ndata: {"step": "step2", "progress": 50}\n\n'
            'event: result\ndata: {"response": "final"}\n\n'
        )

        events = []
        lines = sse_data.strip().split("\n")
        current_event = ""

        for line in lines:
            if line.startswith("event: "):
                current_event = line[7:]
            elif line.startswith("data: "):
                data_json = line[6:]
                data = json.loads(data_json)
                events.append((current_event, data))

        assert len(events) == 3
        assert events[0][0] == "progress"
        assert events[0][1]["step"] == "step1"
        assert events[1][0] == "progress"
        assert events[1][1]["step"] == "step2"
        assert events[2][0] == "result"
        assert events[2][1]["response"] == "final"


class TestProgressUpdateModel:
    """Test the ProgressUpdate model."""

    def test_progress_update_creation(self):
        """Test creating a ProgressUpdate instance."""
        update = ProgressUpdate(
            step="test_step",
            message="Test message",
            progress=75.0,
            timestamp="2024-01-01T12:00:00Z",
            details={"key": "value"},
        )

        assert update.step == "test_step"
        assert update.message == "Test message"
        assert update.progress == 75.0
        assert update.timestamp == "2024-01-01T12:00:00Z"
        assert update.details == {"key": "value"}

    def test_progress_update_serialization(self):
        """Test ProgressUpdate JSON serialization."""
        update = ProgressUpdate(
            step="test_step",
            message="Test message",
            progress=50.0,
            timestamp="2024-01-01T12:00:00Z",
            details={"count": 5},
        )

        json_str = update.model_dump_json()
        data = json.loads(json_str)

        assert data["step"] == "test_step"
        assert data["message"] == "Test message"
        assert data["progress"] == 50.0
        assert data["timestamp"] == "2024-01-01T12:00:00Z"
        assert data["details"]["count"] == 5


class TestStreamingIntegration:
    """Integration tests for the streaming functionality."""

    @pytest.mark.asyncio
    async def test_full_streaming_flow(self, mock_streaming_request, mock_s3_client, mock_metadata):
        """Test a complete streaming flow from start to finish."""
        tracker = await process_streaming_chat(
            request=mock_streaming_request,
            s3_client=mock_s3_client,
            metadata=mock_metadata,
            use_fast_mode=False,
        )

        # Mock the traditional chat processing to emit some events
        async def mock_traditional_chat(request, s3_client, metadata, tracker):
            await tracker.emit_progress("start", "Starting...", 5.0)
            await asyncio.sleep(0.1)
            await tracker.emit_progress("processing", "Processing...", 50.0)
            await asyncio.sleep(0.1)
            await tracker.emit_final_result({"response": "Test response"})

        with patch(
            "app.streaming_chat._process_traditional_chat", side_effect=mock_traditional_chat
        ):
            # Start the background task
            asyncio.create_task(
                _process_chat_async(
                    request=mock_streaming_request,
                    s3_client=mock_s3_client,
                    metadata=mock_metadata,
                    tracker=tracker,
                    use_fast_mode=False,
                )
            )

            # Collect all events from the stream
            events = []
            async for event in tracker.stream_updates():
                events.append(event)
                if "event: result" in event:
                    break

            # Should have at least 3 events: 2 progress + 1 result
            assert len(events) >= 3

            # Check that we have progress events
            progress_events = [e for e in events if "event: progress" in e]
            assert len(progress_events) >= 2

            # Check that we have a result event
            result_events = [e for e in events if "event: result" in e]
            assert len(result_events) == 1


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
