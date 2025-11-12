from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from app.models import (
    ChatRequest, ChatResponse, 
    ChromaAddRequest, ChromaAddResponse, 
    ChromaQueryRequest, ChromaQueryResponse,
    ChromaMetaUpdateRequest, ChromaMetaUpdateResponse,
    ChromaMetaQueryRequest, ChromaMetaQueryResponse,
    StreamingChatRequest,
    FinancialDataRequest, FinancialDataResponse, FinancialDataFilters
)
from app.utils import (
    init_s3_client, 
    init_vertex_ai, 
    load_metadata_from_s3, 
    auto_load_relevant_documents, 
    generate_chat_response,
    semantic_document_selection,
    refresh_metadata_cache,
    get_cache_status,
)
from app.chroma_utils import (
    init_chroma_client,
    get_or_create_collection,
    add_documents as chroma_add_documents,
    query_collection as chroma_query_collection,
    qa_bot,
)
from app.streaming_chat import process_streaming_chat
from app.financial_utils import FinancialDataManager

from dotenv import load_dotenv
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Initialize S3 client
        app.state.s3_client = init_s3_client()
        # Initialize Vertex AI
        init_vertex_ai()
        # Load metadata from S3
        app.state.metadata = load_metadata_from_s3(app.state.s3_client)
        if app.state.metadata:
            logger.info(f"Metadata loaded: {len(app.state.metadata)} companies found")
        else:
            logger.warning("Failed to load metadata from S3")
        # -----------------------
        # Initialise ChromaDB
        # -----------------------
        try:
            app.state.chroma_client = init_chroma_client()
            app.state.chroma_collection = get_or_create_collection(app.state.chroma_client)
            try:
                collection_size = app.state.chroma_collection.count()
            except Exception:
                collection_size = "unknown"
                logger.warning("Could not retrieve Chroma collection size during startup.")
            logger.info(
                "ChromaDB initialised and collection ready | collection_size=%s",
                collection_size,
            )
            # Initialize the metadata collection for semantic document selection
            app.state.meta_collection = get_or_create_collection(app.state.chroma_client, "doc_meta")
            try:
                meta_collection_size = app.state.meta_collection.count()
            except Exception:
                meta_collection_size = "unknown"
                logger.warning("Could not retrieve metadata collection size during startup.")
            logger.info(
                "Metadata collection initialised | collection_size=%s",
                meta_collection_size,
            )
        except Exception as chroma_err:
            logger.error(f"Failed to initialise ChromaDB: {chroma_err}")
        # -----------------------
        # Initialize Financial Data Manager (BigQuery)
        # -----------------------
        try:
            app.state.financial_manager = FinancialDataManager()
            if app.state.financial_manager.metadata:
                logger.info("Financial data manager (BigQuery) initialized successfully with metadata")
            else:
                logger.warning("Financial data manager initialized but metadata loading failed")
        except Exception as financial_err:
            logger.error(f"Failed to initialize financial data manager: {financial_err}")
            app.state.financial_manager = None
        yield
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        # We'll continue and let individual endpoints handle errors

# Initialize FastAPI app
app = FastAPI(
    title="Jacie",
    description="API for chatting with Jacie, the JSE DataSphere Chatbot.",
    version="1.2.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# ----------------------------------------------------------
# Middleware: Log every request & response with latency
# ----------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs incoming requests and their responses including latency and status."""
    request_id = uuid.uuid4().hex
    logger.info(f"[{request_id}] {request.method} {request.url.path} - Start")
    start_time = time.time()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception(f"[{request_id}] {request.method} {request.url.path} - Unhandled error: {exc}")
        raise
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} - Completed in {duration_ms:.2f}ms with status {response.status_code}"
    )
    return response

# Dependency to get S3 client
def get_s3_client():
    return app.state.s3_client

# Dependency to get metadata
def get_metadata():
    return app.state.metadata

# Dependency to get Chroma collection
def get_chroma_collection():
    return app.state.chroma_collection

# Dependency to get metadata collection
def get_meta_collection():
    return app.state.meta_collection

# Dependency to get financial data manager
def get_financial_manager():
    return getattr(app.state, 'financial_manager', None)

@app.get("/")
async def root():
    """Root endpoint to check if API is running"""
    return {"message": "Document Chat API is running"}

@app.get("/health")
async def health_check():
    try:
        s3_status = "available" if hasattr(app.state, "s3_client") else "unavailable"
        metadata_status = "available" if hasattr(app.state, "metadata") and app.state.metadata else "unavailable"
        
        financial_status = "unavailable"
        financial_records = 0
        financial_metadata = False
        if hasattr(app.state, "financial_manager") and app.state.financial_manager:
            financial_status = "available"
            if app.state.financial_manager.metadata:
                financial_metadata = True
                # Count total records from metadata if available
                if 'companies' in app.state.financial_manager.metadata:
                    financial_records = len(app.state.financial_manager.metadata['companies'])

        return {
            "status": "healthy",
            "s3_client": s3_status,
            "metadata": metadata_status,
            "financial_data": {
                "status": financial_status,
                "metadata_loaded": financial_metadata,
                "companies_count": financial_records
            }
        }
    except Exception as e:
        return {
            "status": "healthy",
            "error": f"Health check degraded: {str(e)}"
        }

@app.get("/health/stream")
async def health_stream():
    """Simple streaming health check to test SSE functionality"""
    async def generate_health_stream():
        yield "event: status\ndata: {\"message\": \"Streaming health check started\"}\n\n"
        await asyncio.sleep(0.1)
        yield "event: status\ndata: {\"message\": \"Streaming is working correctly\"}\n\n"
        await asyncio.sleep(0.1)
        yield "event: complete\ndata: {\"status\": \"healthy\", \"streaming\": \"enabled\"}\n\n"
    
    return StreamingResponse(
        generate_health_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    s3_client: Any = Depends(get_s3_client),
    metadata: Dict = Depends(get_metadata)
):
    """
    Chat with documents using Gemini AI
    
    This endpoint allows you to ask questions about documents stored in S3.
    It uses Gemini AI to generate responses based on the document content.
    
    If auto_load_documents is enabled, it will automatically load relevant documents
    based on the query and conversation history.
    
    If memory_enabled is enabled, it will use the conversation history to provide context
    for the response.
    """
    logger.info(
        f"/chat called. query='{request.query[:200]}', auto_load_documents={request.auto_load_documents}, memory_enabled={request.memory_enabled}"
    )
    
    # Log conversation history for debugging
    if request.conversation_history:
        logger.info(f"📜 Conversation history received: {len(request.conversation_history)} messages")
        for idx, msg in enumerate(request.conversation_history[-5:]):  # Log last 5 messages
            content_preview = msg.get('content', '')[:100]
            logger.info(f"  [{idx}] {msg.get('role', 'unknown')}: {content_preview}...")
    else:
        logger.info("📜 No conversation history received")
    
    try:
        # Check if metadata is available
        if not metadata:
            raise HTTPException(status_code=500, detail="Metadata not available. Please check S3 access.")
        
        # Initialize variables
        document_texts = {}
        document_selection_message = None
        loaded_docs = []
        
        # Auto-load relevant documents if enabled
        if request.auto_load_documents:
            document_texts, document_selection_message, loaded_docs = auto_load_relevant_documents(
                s3_client,
                request.query,
                metadata,
                {},  # Start with empty document_texts since this is stateless
                request.conversation_history
            )
        
        # Generate response
        response_text = generate_chat_response(
            request.query,
            document_texts,
            request.conversation_history,
            document_selection_message
        )
        
        # Update conversation history if memory is enabled
        updated_conversation_history = None
        if request.memory_enabled and request.conversation_history:
            updated_conversation_history = request.conversation_history.copy()
            # Add user query
            updated_conversation_history.append({"role": "user", "content": request.query})
            # Add assistant response
            updated_conversation_history.append({"role": "assistant", "content": response_text})
        elif request.memory_enabled:
            # If memory is enabled but no history provided, create new history
            updated_conversation_history = [
                {"role": "user", "content": request.query},
                {"role": "assistant", "content": response_text}
            ]
        
        # Add document selection message to response if available
        # if document_selection_message and "Semantically selected" in document_selection_message:
        #     response_text += f"\n\n{document_selection_message}"
        
        logger.info(
            f"/chat completed successfully. response_chars={len(response_text)}, documents_loaded={len(loaded_docs)}"
        )

        # Return response
        return ChatResponse(
            response=response_text,
            documents_loaded=loaded_docs if loaded_docs else None,
            document_selection_message=document_selection_message,
            conversation_history=updated_conversation_history
        )
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/chat/stream")
async def chat_stream(
    request: StreamingChatRequest,
    s3_client: Any = Depends(get_s3_client),
    metadata: Dict = Depends(get_metadata)
):
    """
    Stream chat responses with real-time progress updates using Server-Sent Events
    
    This endpoint provides the same functionality as /chat but streams progress updates
    to the client, allowing the frontend to show real-time status messages like:
    - "Loading documents..."
    - "Analyzing query..."
    - "Generating response..."
    
    The stream will emit 'progress' events with status updates and a final 'result' 
    event with the complete response.
    """
    logger.info(
        f"/chat/stream called. query='{request.query[:200]}', auto_load_documents={request.auto_load_documents}, memory_enabled={request.memory_enabled}"
    )
    
    # Log conversation history for debugging
    if request.conversation_history:
        logger.info(f"📜 Conversation history received: {len(request.conversation_history)} messages")
        for idx, msg in enumerate(request.conversation_history[-5:]):  # Log last 5 messages
            content_preview = msg.get('content', '')[:100]
            logger.info(f"  [{idx}] {msg.get('role', 'unknown')}: {content_preview}...")
    else:
        logger.info("📜 No conversation history received")
    
    try:
        # Start the streaming chat process
        tracker = await process_streaming_chat(
            request=request,
            s3_client=s3_client,
            metadata=metadata,
            use_fast_mode=False
        )
        
        # Return streaming response with proper headers for SSE
        return StreamingResponse(
            tracker.stream_updates(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            }
        )
    except Exception as e:
        logger.error(f"Error in chat stream endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting chat stream: {str(e)}")

@app.post("/chroma/update", response_model=ChromaAddResponse)
async def chroma_update(
    request: ChromaAddRequest,
    collection: Any = Depends(get_chroma_collection),
):
    """Add or upsert documents into the ChromaDB vector store."""
    logger.info(f"/chroma/update called. num_documents={len(request.documents)}")
    try:
        ids = chroma_add_documents(collection, request.documents, request.metadatas, request.ids)
        logger.info(f"/chroma/update completed. ids={ids}")
        return ChromaAddResponse(status="success", ids=ids)
    except Exception as e:
        logger.error(f"Error updating ChromaDB: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating ChromaDB: {str(e)}")

@app.post("/chroma/query", response_model=ChromaQueryResponse)
async def chroma_query(
    request: ChromaQueryRequest,
    collection: Any = Depends(get_chroma_collection),
):
    """Query the ChromaDB vector store and retrieve most similar documents."""
    logger.info(f"/chroma/query called. query='{request.query}', n_results={request.n_results}, where={request.where}")
    try:
        # Our helper returns (sorted_results, context)
        sorted_results, _ = chroma_query_collection(
            collection,
            query=request.query,
            n_results=request.n_results,
            where=request.where,
        )

        # Build separate lists for the response schema
        ids = [meta.get("id") for meta, _ in sorted_results if meta.get("id")]
        documents = [doc for _, doc in sorted_results]
        metadatas = [meta for meta, _ in sorted_results] if sorted_results else None

        logger.info(
            f"/chroma/query completed. documents_returned={len(documents)}"
        )

        return ChromaQueryResponse(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying ChromaDB: {str(e)}")

@app.post("/chroma/meta/update", response_model=ChromaMetaUpdateResponse)
async def chroma_meta_update(
    request: ChromaMetaUpdateRequest,
    meta_collection: Any = Depends(get_meta_collection),
):
    """Add or upsert document metadata into the metadata collection."""
    logger.info(f"/chroma/meta/update called. num_documents={len(request.documents)}")
    try:
        # Build the documents list for embedding
        documents = []
        metadatas = []
        ids = []
        
        for doc_info in request.documents:
            # Create description text for embedding: "company - doc_type - period"
            description = f"{doc_info.company} - {doc_info.type} - {doc_info.period}"
            documents.append(description)
            
            # Create metadata with all fields
            metadata = {
                "filename": doc_info.filename,
                "company": doc_info.company,
                "period": doc_info.period,
                "type": doc_info.type
            }
            metadatas.append(metadata)
            
            # Use filename as ID (could be made more unique if needed)
            ids.append(doc_info.filename)
        
        # Add to collection
        result_ids = chroma_add_documents(meta_collection, documents, metadatas, ids)
        logger.info(f"/chroma/meta/update completed. ids={result_ids}")
        return ChromaMetaUpdateResponse(status="success", ids=result_ids)
    except Exception as e:
        logger.error(f"Error updating metadata collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating metadata collection: {str(e)}")

@app.post("/chroma/meta/query", response_model=ChromaMetaQueryResponse)
async def chroma_meta_query(
    request: ChromaMetaQueryRequest,
    meta_collection: Any = Depends(get_meta_collection),
):
    """Query the metadata collection to find relevant documents."""
    logger.info(f"/chroma/meta/query called. query='{request.query}', n_results={request.n_results}")
    try:
        # Query the metadata collection
        sorted_results, _ = chroma_query_collection(
            meta_collection,
            query=request.query,
            n_results=request.n_results,
            where=request.where,
        )

        # Build separate lists for the response schema
        ids = [meta.get("filename") for meta, _ in sorted_results if meta.get("filename")]
        documents = [doc for _, doc in sorted_results]
        metadatas = [meta for meta, _ in sorted_results] if sorted_results else None

        logger.info(
            f"/chroma/meta/query completed. documents_returned={len(documents)}"
        )

        return ChromaMetaQueryResponse(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
    except Exception as e:
        logger.error(f"Error querying metadata collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying metadata collection: {str(e)}")

# ---------------------------------------------------------------------------
# Fast Chat Endpoint (Vector DB → Gemini QA)
# ---------------------------------------------------------------------------

@app.post("/fast_chat", response_model=ChatResponse)
async def fast_chat(
    request: ChatRequest,
    s3_client: Any = Depends(get_s3_client),
    metadata: Dict = Depends(get_metadata),
    collection: Any = Depends(get_chroma_collection),
    meta_collection: Any = Depends(get_meta_collection),
):
    """A retrieval-augmented chat endpoint that reuses the ChromaDB query logic
    from the /chroma/query endpoint and then lets Gemini answer based on that context.

    This endpoint is DRY - it reuses the same ChromaDB query logic as /chroma/query.
    """

    logger.info(
        f"/fast_chat called. query='{request.query[:200]}' | memory_enabled={request.memory_enabled} | auto_load_documents={request.auto_load_documents}"
    )
    
    try:
        # -----------------------------
        # Step 1: Build enhanced query (if conversation history is available)
        # -----------------------------
        retrieval_query = request.query
        if request.memory_enabled and request.conversation_history:
            # Combine recent user messages with the current query for better retrieval
            recent_history = [
                msg["content"] for msg in request.conversation_history[-10:] if msg.get("role") == "user"
            ]
            retrieval_query = " ".join(recent_history + [request.query])

        # -----------------------------
        # Step 2: (Optional) Semantic pre-selection of documents by filename
        # -----------------------------
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
                auto_load_message = f"The user has mentioned the following companies: {', '.join(selected_docs['companies_mentioned'])}"
                # Normalise filenames: some LLM responses include full S3 paths – we only
                # store the *basename* (e.g. "my_report.pdf") in Chroma metadata.
                # Strip any directory components before building the filter.
                semantic_filenames = [
                    os.path.basename(doc["filename"]) for doc in selected_docs["documents_to_load"]
                ]

                # ------------------------------------------------------------------
                # Chroma stores the *summary* files (usually `.txt`) while the
                # metadata coming from S3 often refers to the original PDF
                # filename.  To keep the vector-DB filter effective, translate any
                # “.pdf” extension to “.txt” and de-duplicate the list.
                # ------------------------------------------------------------------
                semantic_filenames = list({
                    fn[:-4] + ".txt" if fn.lower().endswith(".pdf") else fn
                    for fn in semantic_filenames
                })

        # -----------------------------
        # Step 3: Query ChromaDB using the same logic as /chroma/query endpoint
        # -----------------------------
        where_filter = {"filename": {"$in": semantic_filenames}} if semantic_filenames else None

        # Use the same query logic as the /chroma/query endpoint
        sorted_results, context = chroma_query_collection(
            collection,
            query=retrieval_query,
            n_results=3,
            where=where_filter,
        )

        # Prepare helpful metadata for the caller
        retrieved_doc_names = [
            meta.get("filename")
            or meta.get("source")
            or meta.get("id")
            for meta, _ in sorted_results
        ] if sorted_results else []

        retrieval_message = f"{len(sorted_results)} documents retrieved from vector database."

        # Combine messages
        doc_selection_message_parts = []
        if auto_load_message:
            doc_selection_message_parts.append(auto_load_message.strip())
        doc_selection_message_parts.append(retrieval_message)
        doc_selection_message = " ".join(doc_selection_message_parts)

        logger.info(
            f"/fast_chat retrieval complete. documents_retrieved={len(sorted_results)}, context_chars={len(context)}, semantic_filter_docs={semantic_filenames}"
        )

        # -----------------------------
        # Step 4: Let LLM answer (include conversation history if provided)
        # -----------------------------
        response_text = qa_bot(
            request.query,
            context,
            conversation_history=request.conversation_history if request.memory_enabled else None,
        )

        # Additional log after generating the response
        logger.info(
            f"/fast_chat LLM answer generated. response_chars={len(response_text)}"
        )

        # -----------------------------
        # Step 5: Update conversation history (if memory is enabled)
        # -----------------------------
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

        return ChatResponse(
            response=response_text,
            documents_loaded=retrieved_doc_names,
            document_selection_message=doc_selection_message,
            conversation_history=updated_conversation_history,
        )
    except Exception as e:
        logger.error(f"Error in fast_chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/fast_chat/stream")
async def fast_chat_stream(
    request: StreamingChatRequest,
    s3_client: Any = Depends(get_s3_client),
    metadata: Dict = Depends(get_metadata),
    collection: Any = Depends(get_chroma_collection),
    meta_collection: Any = Depends(get_meta_collection),
):
    """
    Stream fast chat responses with real-time progress updates using Server-Sent Events
    
    This endpoint provides the same functionality as /fast_chat but streams progress updates
    to the client. It uses vector database retrieval for faster responses and provides
    real-time updates on:
    - Document selection process
    - Vector database search
    - AI response generation
    
    The stream will emit 'progress' events with status updates and a final 'result' 
    event with the complete response.
    """
    logger.info(
        f"/fast_chat/stream called. query='{request.query[:200]}', auto_load_documents={request.auto_load_documents}, memory_enabled={request.memory_enabled}"
    )
    
    try:
        # Start the streaming fast chat process
        tracker = await process_streaming_chat(
            request=request,
            s3_client=s3_client,
            metadata=metadata,
            collection=collection,
            meta_collection=meta_collection,
            use_fast_mode=True
        )
        
        # Return streaming response with proper headers for SSE
        return StreamingResponse(
            tracker.stream_updates(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            }
        )
    except Exception as e:
        logger.error(f"Error in fast_chat stream endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting fast chat stream: {str(e)}")

# ---------------------------------------------------------------------------
# Financial Data Query Endpoint (V2)
# ---------------------------------------------------------------------------

@app.post("/fast_chat_v2", response_model=FinancialDataResponse)
async def fast_chat_v2(
    request: FinancialDataRequest,
    financial_manager: Any = Depends(get_financial_manager),
):
    """
    Natural language financial data query endpoint
    
    This endpoint allows users to query financial data using natural language.
    It processes queries like:
    - "Show me MDS revenue for 2024"
    - "Compare JBG and CPJ profit margins"
    - "What about 2022?" (follow-up questions)
    
    The endpoint uses AI to parse natural language queries into structured filters,
    queries the financial database, and returns formatted responses with data insights.
    """
    logger.info(f"/fast_chat_v2 called. query='{request.query[:200]}', memory_enabled={request.memory_enabled}")
    
    # Log conversation history for debugging
    if request.conversation_history:
        logger.info(f"📜 Conversation history received: {len(request.conversation_history)} messages")
        for idx, msg in enumerate(request.conversation_history[-5:]):  # Log last 5 messages
            content_preview = msg.get('content', '')[:100]
            logger.info(f"  [{idx}] {msg.get('role', 'unknown')}: {content_preview}...")
    else:
        logger.info("📜 No conversation history received")
    
    try:
        if not financial_manager:
            raise HTTPException(
                status_code=503,
                detail="Financial data service is not available. Please ensure BigQuery is configured."
            )
        last_query_data = getattr(request, '_last_query_data', None)
        filters = financial_manager.parse_user_query(
            request.query,
            request.conversation_history,
            last_query_data
        )
        logger.info(f"Parsed filters: {filters}")
        logger.info(f"filters type: {type(filters)}, filters: {filters}")
        availability = financial_manager.validate_data_availability(filters)
        # logger.info(f"availability type: {type(availability)}, availability: {availability}")
        warnings = availability.get('warnings', [])
        suggestions = availability.get('suggestions', [])
        results = financial_manager.query_data(filters)
        logger.info(f"results type: {type(results)}, results: {results}")
        ai_response = financial_manager.format_response(
            results,
            request.query,
            filters.interpretation,
            filters.is_follow_up,
            request.conversation_history
        )
        # logger.info(f"ai_response type: {type(ai_response)}, ai_response: {ai_response}")
        data_preview = results if results else None
        updated_conversation_history = None
        if request.memory_enabled:
            if request.conversation_history:
                updated_conversation_history = request.conversation_history.copy()
            else:
                updated_conversation_history = []
            updated_conversation_history.append({"role": "user", "content": request.query})
            updated_conversation_history.append({"role": "assistant", "content": ai_response})
            if len(updated_conversation_history) > 20:
                updated_conversation_history = updated_conversation_history[-20:]
        logger.info(f"/fast_chat_v2 completed. data_found={bool(results)}, record_count={len(results) if results else 0}")
        return FinancialDataResponse(
            response=ai_response,
            data_found=bool(results),
            record_count=len(results) if results else 0,
            filters_used=filters,
            data_preview=data_preview,
            conversation_history=updated_conversation_history,
            warnings=warnings if warnings else None,
            suggestions=suggestions if suggestions else None
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in fast_chat_v2 endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing financial data query: {str(e)}")

@app.post("/fast_chat_v2/stream")
async def fast_chat_v2_stream(
    request: StreamingChatRequest,
    financial_manager: Any = Depends(get_financial_manager),
):
    """
    Stream financial data query responses with real-time progress updates using Server-Sent Events
    
    This endpoint provides the same functionality as /fast_chat_v2 but streams progress updates
    to the client. It provides real-time updates on:
    - Natural language query parsing
    - Data availability validation
    - Financial database queries
    - AI response generation
    
    The stream will emit 'progress' events with status updates and a final 'result' 
    event with the complete response.
    """
    logger.info(
        f"/fast_chat_v2/stream called. query='{request.query[:200]}', memory_enabled={request.memory_enabled}"
    )
    
    # Log conversation history for debugging
    if request.conversation_history:
        logger.info(f"📜 Conversation history received: {len(request.conversation_history)} messages")
        for idx, msg in enumerate(request.conversation_history[-5:]):  # Log last 5 messages
            content_preview = msg.get('content', '')[:100]
            logger.info(f"  [{idx}] {msg.get('role', 'unknown')}: {content_preview}...")
    else:
        logger.info("📜 No conversation history received")
    
    try:
        # Import the streaming financial chat processor
        from app.streaming_financial_chat import process_streaming_financial_chat
        
        # Start the streaming financial chat process
        tracker = await process_streaming_financial_chat(
            request=request,
            financial_manager=financial_manager
        )
        
        # Return streaming response with proper headers for SSE
        return StreamingResponse(
            tracker.stream_updates(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            }
        )
    except Exception as e:
        logger.error(f"Error in fast_chat_v2 stream endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting financial data stream: {str(e)}")

@app.get("/financial/metadata")
async def get_financial_metadata(
    financial_manager: Any = Depends(get_financial_manager),
):
    """Get financial data metadata including available companies, symbols, years, and metrics"""
    try:
        if not financial_manager:
            raise HTTPException(
                status_code=503, 
                detail="Financial data service is not available."
            )
        
        if not financial_manager.metadata:
            raise HTTPException(
                status_code=404,
                detail="Financial metadata not available."
            )
        
        # Return the metadata but limit large lists for better API response size
        metadata = financial_manager.metadata.copy()
        
        # If there are too many items, provide sample + count
        for key in ['companies', 'symbols', 'standard_items']:
            if key in metadata and len(metadata[key]) > 50:
                sample = metadata[key][:50]
                metadata[key] = {
                    "sample": sample,
                    "total_count": len(metadata[key]),
                    "note": f"Showing first 50 of {len(metadata[key])} items"
                }
        
        return {
            "status": "success",
            "metadata": metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving financial metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving financial metadata: {str(e)}")

@app.get("/cache/status")
async def get_cache_status_endpoint():
    """Get the current status of the metadata cache"""
    try:
        status = get_cache_status()
        return {
            "success": True,
            "cache_status": status
        }
    except Exception as e:
        logger.error(f"Error getting cache status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting cache status: {str(e)}")

@app.post("/cache/refresh")
async def refresh_cache_endpoint():
    """Force refresh of the metadata cache using current S3 metadata"""
    try:
        # Load current metadata from S3
        metadata = load_metadata_from_s3()
        if not metadata:
            raise HTTPException(status_code=500, detail="Failed to load metadata from S3")
        
        # Refresh the cache
        success = refresh_metadata_cache(metadata)
        
        if success:
            return {
                "success": True,
                "message": "Cache refreshed successfully",
                "cache_status": get_cache_status()
            }
        else:
            return {
                "success": False,
                "message": "Failed to refresh cache",
                "cache_status": get_cache_status()
            }
    except Exception as e:
        logger.error(f"Error refreshing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error refreshing cache: {str(e)}")
