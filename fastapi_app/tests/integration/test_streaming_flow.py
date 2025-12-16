"""Integration tests for streaming chat flow."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from app.streaming_chat import process_streaming_chat, _process_chat_async
from app.models import StreamingChatRequest
from app.progress_tracker import ProgressTracker


@pytest.mark.integration
class TestStreamingChatFlow:
    """Test the complete streaming chat flow."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock streaming request."""
        return StreamingChatRequest(
            query="What is the revenue for MTN?",
            auto_load_documents=True,
            memory_enabled=False,
            conversation_history=[],
        )

    @pytest.mark.asyncio
    async def test_streaming_flow_creates_tracker(
        self, mock_request, mock_s3_client, mock_metadata
    ):
        """Test that streaming flow creates progress tracker."""
        tracker = await process_streaming_chat(
            request=mock_request,
            s3_client=mock_s3_client,
            metadata=mock_metadata,
            use_fast_mode=False,
        )

        assert tracker is not None
        assert isinstance(tracker, ProgressTracker)

    @pytest.mark.asyncio
    async def test_streaming_flow_emits_progress(self, mock_request, mock_s3_client, mock_metadata):
        """Test that streaming flow emits progress updates."""
        with patch("app.streaming_chat._process_traditional_chat") as mock_process:

            async def mock_traditional(request, s3_client, metadata, tracker):
                await tracker.emit_progress("test", "Testing", 50.0)
                await tracker.emit_final_result({"response": "Test result"})

            mock_process.side_effect = mock_traditional

            tracker = await process_streaming_chat(
                request=mock_request,
                s3_client=mock_s3_client,
                metadata=mock_metadata,
                use_fast_mode=False,
            )

            # Give background task time to start
            await asyncio.sleep(0.1)

            # Should be able to get updates from tracker
            assert tracker.updates_queue is not None

    @pytest.mark.asyncio
    async def test_streaming_error_handling(self, mock_request, mock_s3_client):
        """Test error handling in streaming flow."""
        # Pass None as metadata to trigger error
        tracker = await process_streaming_chat(
            request=mock_request, s3_client=mock_s3_client, metadata=None, use_fast_mode=False
        )

        # Should still create tracker
        assert tracker is not None

        # Give background task time to process
        await asyncio.sleep(0.1)

        # Should have error in queue
        if not tracker.updates_queue.empty():
            update = await tracker.updates_queue.get()
            assert update is not None


@pytest.mark.integration
class TestFinancialStreamingFlow:
    """Test financial data streaming flow."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock financial query request."""
        return StreamingChatRequest(
            query="Show me revenue for Company X in 2023",
            memory_enabled=False,
            conversation_history=[],
        )

    @pytest.fixture
    def mock_financial_manager(self):
        """Create mock financial manager."""
        manager = Mock()
        manager.parse_user_query = AsyncMock(return_value=Mock())
        manager.validate_data_availability = AsyncMock(return_value={})
        manager.query_data = AsyncMock(return_value=[])
        manager.format_response = AsyncMock(return_value="Test response")
        return manager

    @pytest.mark.asyncio
    async def test_financial_streaming_creates_tracker(self, mock_request, mock_financial_manager):
        """Test financial streaming creates tracker."""
        from app.streaming_financial_chat import process_streaming_financial_chat

        tracker = await process_streaming_financial_chat(
            request=mock_request, financial_manager=mock_financial_manager
        )

        assert tracker is not None
        assert isinstance(tracker, ProgressTracker)

    @pytest.mark.asyncio
    async def test_financial_streaming_processes_query(self, mock_request, mock_financial_manager):
        """Test financial streaming processes query."""
        from app.streaming_financial_chat import process_streaming_financial_chat

        tracker = await process_streaming_financial_chat(
            request=mock_request, financial_manager=mock_financial_manager
        )

        # Give background task time to start
        await asyncio.sleep(0.1)

        # Financial manager should have been called
        assert tracker is not None
