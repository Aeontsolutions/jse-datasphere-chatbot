from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from app.models import ChatRequest, ChatResponse, ChromaAddRequest, ChromaAddResponse, ChromaQueryRequest, ChromaQueryResponse
from app.utils import (
    init_s3_client, 
    init_vertex_ai, 
    load_metadata_from_s3, 
    auto_load_relevant_documents, 
    generate_chat_response,
    semantic_document_selection,
)
from app.chroma_utils import (
    init_chroma_client,
    get_or_create_collection,
    add_documents as chroma_add_documents,
    query_collection as chroma_query_collection,
    qa_bot,
)

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
        except Exception as chroma_err:
            logger.error(f"Failed to initialise ChromaDB: {chroma_err}")
        yield
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        # We'll continue and let individual endpoints handle errors

# Initialize FastAPI app
app = FastAPI(
    title="Jacie",
    description="API for chatting with Jacie, the JSE DataSphere Chatbot.",
    version="1.1.0",
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

@app.get("/")
async def root():
    """Root endpoint to check if API is running"""
    return {"message": "Document Chat API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check if S3 client and metadata are available
    s3_status = "available" if hasattr(app.state, "s3_client") else "unavailable"
    metadata_status = "available" if hasattr(app.state, "metadata") and app.state.metadata else "unavailable"
    
    return {
        "status": "healthy",
        "s3_client": s3_status,
        "metadata": metadata_status
    }

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

# ---------------------------------------------------------------------------
# Fast Chat Endpoint (Vector DB → Gemini QA)
# ---------------------------------------------------------------------------

@app.post("/fast_chat", response_model=ChatResponse)
async def fast_chat(
    request: ChatRequest,
    s3_client: Any = Depends(get_s3_client),
    metadata: Dict = Depends(get_metadata),
    collection: Any = Depends(get_chroma_collection),
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
            )
            
            if selected_docs:
                auto_load_message = f"The user has mentioned the following companies: {', '.join(selected_docs['companies_mentioned'])}"
                # Normalise filenames: some LLM responses include full S3 paths – we only
                # store the *basename* (e.g. "my_report.pdf") in Chroma metadata.
                # Strip any directory components before building the filter.
                semantic_filenames = [
                    os.path.basename(doc["filename"]) for doc in selected_docs["documents_to_load"]
                ]

        # -----------------------------
        # Step 3: Query ChromaDB using the same logic as /chroma/query endpoint
        # -----------------------------
        where_filter = {"filename": {"$in": semantic_filenames}} if semantic_filenames else None

        # Use the same query logic as the /chroma/query endpoint
        sorted_results, context = chroma_query_collection(
            collection,
            query=retrieval_query,
            n_results=5,
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
