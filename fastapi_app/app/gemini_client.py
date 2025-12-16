"""
Gemini AI client initialization and response generation.

This module provides functions for initializing Vertex AI and Google GenerativeAI,
managing metadata caching for improved performance, and generating chat responses
using Gemini models.
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta

from google.oauth2 import service_account
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel
import google.generativeai as genai
from fastapi import HTTPException

from app.config import get_config

logger = logging.getLogger(__name__)


# Global variables to store the cached context
_metadata_cache = None
_cache_expiry = None
_cache_hash = None


# =============================================================================
# INITIALIZATION FUNCTIONS
# =============================================================================


def init_vertex_ai():
    """Initialize Vertex AI using service account credentials"""
    try:
        config = get_config()

        # Get the service account info from config
        service_account_info = config.gcp.service_account_dict

        # Create credentials object from service account info
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        # Get project ID from config
        project_id = config.gcp.project_id

        # Initialize Vertex AI with project details and credentials
        aiplatform.init(
            project=project_id,
            location=config.gcp.location,
            credentials=credentials,
        )

        logger.info(
            "vertex_ai_initialized",
            extra={"project_id": project_id, "location": config.gcp.location},
        )
        return True
    except Exception as e:
        logger.error(
            "vertex_ai_init_failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "project_id": config.gcp.project_id if hasattr(config.gcp, "project_id") else None,
            },
        )
        raise HTTPException(status_code=503, detail="Failed to initialize AI service")


def init_genai():
    """Initialize Google GenerativeAI client"""
    try:
        config = get_config()
        api_key = config.gcp.api_key

        if not api_key:
            logger.error(
                "genai_api_key_missing",
                extra={"error": "GOOGLE_API_KEY not found in environment variables"},
            )
            raise HTTPException(status_code=503, detail="AI service not configured")

        genai.configure(api_key=api_key)
        logger.info("genai_initialized", extra={"status": "success"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error("genai_init_failed", extra={"error": str(e), "error_type": type(e).__name__})
        raise HTTPException(status_code=503, detail="Failed to initialize AI service")


# =============================================================================
# METADATA CACHING FUNCTIONS
# =============================================================================


def get_metadata_hash(metadata):
    """Generate a hash of the metadata to detect changes"""
    metadata_str = json.dumps(metadata, sort_keys=True)
    return hashlib.md5(metadata_str.encode()).hexdigest()


def create_metadata_cache(metadata):
    """Create or update the metadata cache using Google Gemini context caching"""
    global _metadata_cache, _cache_expiry, _cache_hash

    try:
        # Initialize genai if not already done
        init_genai()

        # Generate hash for metadata to detect changes
        current_hash = get_metadata_hash(metadata)

        # Check if cache exists and is still valid
        if (
            _metadata_cache
            and _cache_expiry
            and datetime.now() < _cache_expiry
            and _cache_hash == current_hash
        ):
            logger.info("Using existing valid metadata cache")
            return _metadata_cache

        logger.info("Creating new metadata cache with Google Gemini context caching")

        # Format metadata for caching
        metadata_str = json.dumps(metadata, indent=2)

        # Create cached content - this will be the system context that gets cached
        cached_content = f"""You are a document selection assistant. You have access to the following document metadata:

            {metadata_str}

            Your job is to analyze user queries and conversation history to select the most relevant documents from this metadata. When asked to select documents, you should:

            1. Consider all companies mentioned or implied in the query/conversation
            2. Consider aliases, abbreviations, or partial references to companies
            3. Select a maximum of 3 documents total, prioritizing the most relevant ones
            4. Return results as valid JSON with this structure:
            {{
            "companies_mentioned": ["company1", "company2"],
            "documents_to_load": [
                {{
                "company": "company name",
                "document_link": "url",
                "filename": "filename",
                "reason": "brief explanation why this document is relevant"
                }}
            ]
            }}"""

        # Build a dedicated client – required for the caches API
        client = genai.Client()

        # Create the cache using Google Gemini explicit context caching
        cache = client.caches.create(
            model="gemini-2.0-flash-001",
            config=genai.types.CreateCachedContentConfig(
                system_instruction=cached_content,
                display_name="document-metadata-cache",
                ttl_seconds=3600,  # 1-hour TTL
            ),
        )

        # Store cache info globally
        _metadata_cache = cache
        _cache_expiry = datetime.now() + timedelta(
            seconds=3500
        )  # Expire 5 minutes before actual TTL
        _cache_hash = current_hash

        logger.info(f"Created metadata cache with name: {cache.name}")
        return cache

    except Exception as e:
        logger.warning(
            "metadata_cache_creation_failed",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        return None


def get_cached_model(metadata):
    """Get a model instance that uses the cached metadata context"""
    try:
        cache = create_metadata_cache(metadata)
        if cache:
            # Create model that uses the cached context
            model = genai.GenerativeModel(model_name="gemini-2.0-flash-001", cached_content=cache)
            logger.info("Created model with cached metadata context")
            return model, True
        else:
            logger.info("Cache creation failed, falling back to regular model")
            return None, False
    except Exception as e:
        logger.warning(
            "cached_model_get_failed", extra={"error": str(e), "error_type": type(e).__name__}
        )
        return None, False


def refresh_metadata_cache(metadata):
    """Force refresh of the metadata cache (useful when metadata is updated)"""
    global _metadata_cache, _cache_expiry, _cache_hash

    try:
        # Clear existing cache
        _metadata_cache = None
        _cache_expiry = None
        _cache_hash = None

        # Create new cache
        cache = create_metadata_cache(metadata)
        if cache:
            logger.info("Successfully refreshed metadata cache")
            return True
        else:
            logger.warning("Failed to refresh metadata cache")
            return False
    except Exception as e:
        logger.error(
            "metadata_cache_refresh_failed", extra={"error": str(e), "error_type": type(e).__name__}
        )
        return False


def get_cache_status():
    """Get current cache status for monitoring/debugging"""
    global _metadata_cache, _cache_expiry, _cache_hash

    if not _metadata_cache:
        return {"status": "no_cache", "cache_name": None, "expires_at": None, "hash": None}

    is_expired = _cache_expiry and datetime.now() >= _cache_expiry

    return {
        "status": "expired" if is_expired else "active",
        "cache_name": getattr(_metadata_cache, "name", "unknown"),
        "expires_at": _cache_expiry.isoformat() if _cache_expiry else None,
        "hash": _cache_hash,
    }


# =============================================================================
# CHAT RESPONSE GENERATION
# =============================================================================


def generate_chat_response(
    query, document_texts, conversation_history=None, auto_load_message=None
):
    """Generate chat response using Gemini model"""
    try:
        model = GenerativeModel("gemini-2.5-pro")

        # Format conversation history if available
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            # Get the last few exchanges to provide context
            recent_history = conversation_history[-20:]  # Keep context length reasonable
            conversation_context = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in recent_history]
            )
        else:
            # If no history, just use the current input
            conversation_context = f"user: {query}"

        # Prepare document context if available
        document_context = ""
        if document_texts:
            # Combine all document texts with document names as headers
            for doc_name, doc_text in document_texts.items():
                document_context += f"Document: {doc_name}\n{doc_text}\n\n"

        # Prepare prompt for model
        if document_context:
            prompt = f"""
            The following are documents that have been uploaded:

            {document_context}

            Previous conversation:
            {conversation_context}

            Based on the above documents and our conversation history, please answer the following question:
            {query}

            If the question relates to our previous conversation, use that context in your answer.

            IMPORTANT: Use plain text formatting for all financial data. Do not use special formatting for dollar amounts or numbers.
            """

            # Add auto-load message if relevant
            if auto_load_message and "Semantically selected" in auto_load_message:
                prompt += f"\n\nNote: {auto_load_message}"

            response = model.generate_content(prompt)
        else:
            # No document context, just use conversation
            prompt = f"""
            Previous conversation:
            {conversation_context}

            Based on our conversation history, please answer the following question:
            {query}

            If the question relates to our previous conversation, use that context in your answer.

            IMPORTANT: Use plain text formatting for all financial data. Do not use special formatting for dollar amounts or numbers.
            """
            response = model.generate_content(prompt)

        logger.info(
            "chat_response_generated",
            extra={
                "response_length": len(response.text),
                "has_documents": bool(document_texts),
                "has_history": bool(conversation_history),
            },
        )

        return response.text

    except Exception as e:
        logger.error(
            "chat_response_generation_failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "has_documents": bool(document_texts),
                "has_history": bool(conversation_history),
            },
        )
        raise HTTPException(status_code=503, detail="Failed to generate AI response")
