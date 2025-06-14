from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import logging
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from app.models import ChatRequest, ChatResponse, ChromaAddRequest, ChromaAddResponse, ChromaQueryRequest, ChromaQueryResponse
from app.utils import (
    init_s3_client, 
    init_vertex_ai, 
    load_metadata_from_s3, 
    auto_load_relevant_documents, 
    generate_chat_response
)
from app.chroma_utils import (
    init_chroma_client,
    get_or_create_collection,
    add_documents as chroma_add_documents,
    query_collection as chroma_query_collection,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
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
            logger.info("ChromaDB initialised and collection ready")
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
    version="1.0.0",
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
    try:
        ids = chroma_add_documents(collection, request.documents, request.metadatas, request.ids)
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
    try:
        results = chroma_query_collection(
            collection,
            query=request.query,
            n_results=request.n_results,
            where=request.where,
        )
        return ChromaQueryResponse(
            ids=results.get("ids", [])[0] if results.get("ids") else [],
            documents=results.get("documents", [])[0] if results.get("documents") else [],
            metadatas=results.get("metadatas", [])[0] if results.get("metadatas") else None,
        )
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying ChromaDB: {str(e)}")
