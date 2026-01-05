import asyncio
from typing import Dict, Optional, Any
from app.progress_tracker import ProgressTracker
from app.models import StreamingChatRequest
from app.gemini_client import generate_chat_response
from app.document_selector import auto_load_relevant_documents_async
from app.config import S3DownloadConfig
from app.logging_config import get_logger

logger = get_logger(__name__)


async def process_streaming_chat(
    request: StreamingChatRequest,
    s3_client: Any,
    metadata: Dict,
    collection: Any = None,
    meta_collection: Any = None,
    use_fast_mode: bool = False,
    tracker: Optional[ProgressTracker] = None,
    associations: Optional[Dict] = None,
) -> ProgressTracker:
    """
    Process a chat request with streaming progress updates

    Args:
        request: The chat request
        s3_client: S3 client for document loading
        metadata: Document metadata
        collection: Deprecated (previously used for ChromaDB)
        meta_collection: Deprecated (previously used for ChromaDB)
        use_fast_mode: Deprecated (fast mode using ChromaDB has been removed)
        tracker: Optional progress tracker
        associations: Optional symbol-to-company mappings for better document selection

    Returns:
        ProgressTracker instance that can be used to stream updates
    """

    tracker = tracker or ProgressTracker()

    # Start the processing in the background
    asyncio.create_task(
        _process_chat_async(
            request,
            s3_client,
            metadata,
            tracker,
            collection,
            meta_collection,
            use_fast_mode,
            associations,
        )
    )

    return tracker


async def _process_chat_async(
    request: StreamingChatRequest,
    s3_client: Any,
    metadata: Dict,
    tracker: ProgressTracker,
    collection: Any = None,
    meta_collection: Any = None,
    use_fast_mode: bool = False,
    associations: Optional[Dict] = None,
):
    """Internal async processing function"""

    try:
        await tracker.emit_progress("start", "Starting chat processing...", 5.0)
        await asyncio.sleep(0.1)  # Give time for the event to be processed

        # Check metadata availability with better error handling
        if not metadata:
            await tracker.emit_progress("error_check", "Checking system configuration...", 10.0)
            await asyncio.sleep(0.1)
            await tracker.emit_error(
                "Metadata not available. Please check S3 access and ensure AWS credentials are configured properly."
            )
            return

        if use_fast_mode:
            # ChromaDB has been deprecated - fast mode is no longer supported
            await tracker.emit_error(
                "Fast mode (ChromaDB) has been deprecated. Please use the /chat/stream endpoint instead."
            )
            return
        else:
            await _process_traditional_chat(request, s3_client, metadata, tracker, associations)

    except Exception as e:
        logger.error(
            "streaming_chat_processing_failed",
            error=str(e),
            error_type=type(e).__name__,
            auto_load=request.auto_load_documents,
            memory_enabled=request.memory_enabled,
        )
        await tracker.emit_error(f"Error generating response: {str(e)}")


async def _process_traditional_chat(
    request: StreamingChatRequest,
    s3_client: Any,
    metadata: Dict,
    tracker: ProgressTracker,
    associations: Optional[Dict] = None,
):
    """Process chat using traditional S3 document loading"""

    await tracker.emit_progress("doc_loading", "Initializing robust document download...", 20.0)
    await asyncio.sleep(0.1)  # Give time for the event to be processed

    # Initialize variables
    document_texts = {}
    document_selection_message = None
    loaded_docs = []

    # Check if metadata is available for async processing
    if not metadata:
        await tracker.emit_progress(
            "doc_loading", "Metadata not available, skipping document loading", 30.0
        )
        document_selection_message = "Metadata not available - no documents loaded"
    else:
        # Auto-load relevant documents if enabled
        if request.auto_load_documents:
            try:
                # Create download configuration for robust downloads
                download_config = S3DownloadConfig(
                    max_retries=2,  # Reduced from 3
                    retry_delay=0.5,  # Reduced from 1.0
                    timeout=120.0,
                    concurrent_downloads=3,  # Limit concurrency for streaming
                )

                # Create progress callback to emit streaming updates
                async def download_progress_callback(step: str, message: str):
                    # Map download steps to progress percentages
                    progress_map = {
                        "document_selection_start": 25.0,
                        "document_selection_complete": 35.0,
                        "concurrent_download_start": 40.0,
                        "single_download_start": 45.0,
                        "download_start": 50.0,
                        "download_complete": 55.0,
                        "text_extraction": 60.0,
                        "extraction_complete": 65.0,
                        "single_download_complete": 70.0,
                        "concurrent_download_complete": 75.0,
                        "documents_already_loaded": 45.0,
                        "download_failed": 40.0,
                        "download_error": 40.0,
                        "metadata_download_start": 30.0,
                        "metadata_download_complete": 35.0,
                    }

                    progress = progress_map.get(step, 50.0)
                    await tracker.emit_progress("doc_loading", message, progress)

                # Use async document loading with progress tracking
                document_texts, document_selection_message, loaded_docs = (
                    await auto_load_relevant_documents_async(
                        query=request.query,
                        metadata=metadata,
                        conversation_history=request.conversation_history,
                        current_document_texts={},  # Start with empty document_texts since this is stateless
                        config=download_config,
                        progress_callback=download_progress_callback,
                        associations=associations,
                    )
                )

                await tracker.emit_progress(
                    "doc_loading",
                    f"Robustly loaded {len(loaded_docs)} documents using async downloads",
                    75.0,
                    {"documents_loaded": len(loaded_docs), "download_method": "async_concurrent"},
                )
                await asyncio.sleep(0.1)  # Give time for the event to be processed

            except Exception as e:
                logger.error(
                    "streaming_async_document_load_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    query=request.query[:100] if request.query else None,
                )
                await tracker.emit_progress("doc_loading", f"Async download error: {str(e)}", 30.0)
                document_selection_message = f"Error in async document loading: {str(e)}"
        else:
            await tracker.emit_progress("doc_loading", "Document auto-loading disabled", 30.0)
            document_selection_message = "Document auto-loading was disabled"

    await tracker.emit_progress("ai_generation", "Generating AI response...", 80.0)
    await asyncio.sleep(0.1)  # Give time for the event to be processed

    # Generate response
    response_text = generate_chat_response(
        request.query, document_texts, request.conversation_history, document_selection_message
    )

    await tracker.emit_progress("finalizing", "Finalizing response...", 95.0)
    await asyncio.sleep(0.1)  # Give time for the event to be processed

    # Update conversation history if memory is enabled
    updated_conversation_history = None
    if request.memory_enabled and request.conversation_history:
        updated_conversation_history = request.conversation_history.copy()
        updated_conversation_history.append({"role": "user", "content": request.query})
        updated_conversation_history.append({"role": "assistant", "content": response_text})
    elif request.memory_enabled:
        updated_conversation_history = [
            {"role": "user", "content": request.query},
            {"role": "assistant", "content": response_text},
        ]

    # Final result
    result = {
        "response": response_text,
        "documents_loaded": loaded_docs if loaded_docs else None,
        "document_selection_message": document_selection_message,
        "conversation_history": updated_conversation_history,
    }

    await tracker.emit_progress("complete", "Response generation complete!", 100.0)
    await asyncio.sleep(0.1)  # Give time for the event to be processed
    await tracker.emit_final_result(result)
