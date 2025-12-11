import asyncio
import logging
from typing import Dict, List, Optional, Any
from app.progress_tracker import ProgressTracker
from app.models import StreamingChatRequest, ChatResponse
from app.utils import (
    auto_load_relevant_documents,
    generate_chat_response,
    semantic_document_selection,
    auto_load_relevant_documents_async,
    S3DownloadConfig,
)
# ChromaDB has been deprecated - fast_mode is no longer supported
# from app.chroma_utils import (
#     query_collection as chroma_query_collection,
#     qa_bot,
# )
import os

logger = logging.getLogger(__name__)

async def process_streaming_chat(
    request: StreamingChatRequest,
    s3_client: Any,
    metadata: Dict,
    collection: Any = None,
    meta_collection: Any = None,
    use_fast_mode: bool = False,
    tracker: Optional[ProgressTracker] = None,
) -> ProgressTracker:
    """
    Process a chat request with streaming progress updates
    
    Args:
        request: The chat request
        s3_client: S3 client for document loading
        metadata: Document metadata
        collection: ChromaDB collection (for fast mode)
        meta_collection: Metadata collection (for fast mode)
        use_fast_mode: Whether to use vector DB mode or traditional mode
    
    Returns:
        ProgressTracker instance that can be used to stream updates
    """
    
    tracker = tracker or ProgressTracker()
    
    # Start the processing in the background
    asyncio.create_task(_process_chat_async(
        request, s3_client, metadata, tracker, collection, meta_collection, use_fast_mode
    ))
    
    return tracker

async def _process_chat_async(
    request: StreamingChatRequest,
    s3_client: Any,
    metadata: Dict,
    tracker: ProgressTracker,
    collection: Any = None,
    meta_collection: Any = None,
    use_fast_mode: bool = False
):
    """Internal async processing function"""
    
    try:
        await tracker.emit_progress("start", "Starting chat processing...", 5.0)
        await asyncio.sleep(0.1)  # Give time for the event to be processed
        
        # Check metadata availability with better error handling
        if not metadata:
            await tracker.emit_progress("error_check", "Checking system configuration...", 10.0)
            await asyncio.sleep(0.1)
            await tracker.emit_error("Metadata not available. Please check S3 access and ensure AWS credentials are configured properly.")
            return
        
        if use_fast_mode:
            # ChromaDB has been deprecated - fast mode is no longer supported
            await tracker.emit_error("Fast mode (ChromaDB) has been deprecated. Please use the /chat/stream endpoint instead.")
            return
        else:
            await _process_traditional_chat(request, s3_client, metadata, tracker)
            
    except Exception as e:
        logger.error(f"Error in streaming chat processing: {str(e)}")
        await tracker.emit_error(f"Error generating response: {str(e)}")

async def _process_fast_chat(
    request: StreamingChatRequest,
    metadata: Dict,
    tracker: ProgressTracker,
    collection: Any,
    meta_collection: Any
):
    """Process chat using vector database (fast mode)"""
    
    await tracker.emit_progress("query_prep", "Preparing search query...", 10.0)
    await asyncio.sleep(0.1)  # Give time for the event to be processed
    
    # Step 1: Build enhanced query
    retrieval_query = request.query
    if request.memory_enabled and request.conversation_history:
        recent_history = [
            msg["content"] for msg in request.conversation_history[-10:] if msg.get("role") == "user"
        ]
        retrieval_query = " ".join(recent_history + [request.query])
    
    await tracker.emit_progress("doc_selection", "Selecting relevant documents...", 25.0)
    await asyncio.sleep(0.1)  # Give time for the event to be processed
    
    # Step 2: Semantic pre-selection of documents
    auto_load_message: Optional[str] = None
    semantic_filenames: list[str] = []
    
    if request.auto_load_documents:
        selected_docs = semantic_document_selection(
            request.query,
            metadata,
            request.conversation_history,
            meta_collection,
        )
        
        if selected_docs:
            companies = selected_docs['companies_mentioned']
            auto_load_message = f"Found mentions of: {', '.join(companies)}"
            
            semantic_filenames = [
                os.path.basename(doc["filename"]) for doc in selected_docs["documents_to_load"]
            ]
            
            # Convert PDF extensions to TXT for vector DB
            semantic_filenames = list({
                fn[:-4] + ".txt" if fn.lower().endswith(".pdf") else fn
                for fn in semantic_filenames
            })
            
            await tracker.emit_progress(
                "doc_selection", 
                f"Selected {len(semantic_filenames)} relevant documents", 
                35.0,
                {"companies": companies, "documents": len(semantic_filenames)}
            )
            await asyncio.sleep(0.1)  # Give time for the event to be processed
    
    await tracker.emit_progress("vector_search", "Searching vector database...", 50.0)
    await asyncio.sleep(0.1)  # Give time for the event to be processed
    
    # Step 3: Query ChromaDB
    where_filter = {"filename": {"$in": semantic_filenames}} if semantic_filenames else None
    
    sorted_results, context = chroma_query_collection(
        collection,
        query=retrieval_query,
        n_results=3,
        where=where_filter,
    )
    
    retrieved_doc_names = [
        meta.get("filename") or meta.get("source") or meta.get("id")
        for meta, _ in sorted_results
    ] if sorted_results else []
    
    await tracker.emit_progress(
        "vector_search", 
        f"Retrieved {len(sorted_results)} documents from vector database", 
        65.0,
        {"documents_found": len(sorted_results)}
    )
    await asyncio.sleep(0.1)  # Give time for the event to be processed
    
    await tracker.emit_progress("ai_generation", "Generating AI response...", 80.0)
    await asyncio.sleep(0.1)  # Give time for the event to be processed
    
    # Step 4: Generate response
    response_text = qa_bot(
        request.query,
        context,
        conversation_history=request.conversation_history if request.memory_enabled else None,
    )
    
    await tracker.emit_progress("finalizing", "Finalizing response...", 95.0)
    await asyncio.sleep(0.1)  # Give time for the event to be processed
    
    # Step 5: Update conversation history
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
    
    # Combine messages
    doc_selection_message_parts = []
    if auto_load_message:
        doc_selection_message_parts.append(auto_load_message.strip())
    doc_selection_message_parts.append(f"{len(sorted_results)} documents retrieved from vector database.")
    doc_selection_message = " ".join(doc_selection_message_parts)
    
    # Final result
    result = {
        "response": response_text,
        "documents_loaded": retrieved_doc_names,
        "document_selection_message": doc_selection_message,
        "conversation_history": updated_conversation_history,
    }
    
    await tracker.emit_progress("complete", "Response generation complete!", 100.0)
    await asyncio.sleep(0.1)  # Give time for the event to be processed
    await tracker.emit_final_result(result)

async def _process_traditional_chat(
    request: StreamingChatRequest,
    s3_client: Any,
    metadata: Dict,
    tracker: ProgressTracker
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
        await tracker.emit_progress("doc_loading", "Metadata not available, skipping document loading", 30.0)
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
                    concurrent_downloads=3  # Limit concurrency for streaming
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
                document_texts, document_selection_message, loaded_docs = await auto_load_relevant_documents_async(
                    query=request.query,
                    metadata=metadata,
                    conversation_history=request.conversation_history,
                    current_document_texts={},  # Start with empty document_texts since this is stateless
                    config=download_config,
                    progress_callback=download_progress_callback
                )
                
                await tracker.emit_progress(
                    "doc_loading", 
                    f"Robustly loaded {len(loaded_docs)} documents using async downloads", 
                    75.0,
                    {"documents_loaded": len(loaded_docs), "download_method": "async_concurrent"}
                )
                await asyncio.sleep(0.1)  # Give time for the event to be processed
                
            except Exception as e:
                logger.error(f"Error in async document loading: {str(e)}")
                await tracker.emit_progress("doc_loading", f"Async download error: {str(e)}", 30.0)
                document_selection_message = f"Error in async document loading: {str(e)}"
        else:
            await tracker.emit_progress("doc_loading", "Document auto-loading disabled", 30.0)
            document_selection_message = "Document auto-loading was disabled"
    
    await tracker.emit_progress("ai_generation", "Generating AI response...", 80.0)
    await asyncio.sleep(0.1)  # Give time for the event to be processed
    
    # Generate response
    response_text = generate_chat_response(
        request.query,
        document_texts,
        request.conversation_history,
        document_selection_message
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
            {"role": "assistant", "content": response_text}
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