"""
Document selection and auto-loading operations.

This module provides intelligent document selection using LLM-based semantic analysis
and supports both synchronous and asynchronous document loading operations.
"""

import json
import logging
import asyncio
from typing import Dict, List, Tuple, Optional

import google.generativeai as genai

from app.config import get_config, S3DownloadConfig
from app.s3_client import (
    download_and_extract_from_s3,
    download_and_extract_from_s3_async,
    DownloadResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SEMANTIC DOCUMENT SELECTION
# =============================================================================


def semantic_document_selection(query, metadata, conversation_history=None, meta_collection=None):
    """
    Use embedding-based search in metadata collection with LLM fallback.
    Now supports multi-company queries by extracting companies from the query
    and running separate searches for each company.

    Operators can set the environment variable ``FORCE_LLM_FALLBACK`` to
    ``true`` (case-insensitive) to bypass the embedding-based branch and jump
    straight to the LLM fallback.  This is handy for load-testing the cache
    mechanism.

    Args:
        query: User query
        metadata: S3 metadata (used for fallback)
        conversation_history: Optional conversation context
        meta_collection: ChromaDB metadata collection (if None, falls back to LLM)

    Returns:
        Dictionary with companies_mentioned and documents_to_load
    """
    # ------------------------------------------------------------------
    # ChromaDB has been deprecated - always use LLM-based selection
    # ------------------------------------------------------------------
    logger.info("Using LLM-based document selection (ChromaDB deprecated)")
    return semantic_document_selection_llm_fallback(query, metadata, conversation_history)


def semantic_document_selection_llm_fallback(query, metadata, conversation_history=None):
    """
    Use LLM to determine which documents to load based on query and conversation history (fallback method).
    Now optimized with Google Gemini context caching to reduce latency from ~20s to ~2-3s.
    """
    try:
        # Fallback to traditional approach if caching fails
        logger.info("Cache unavailable, using traditional LLM fallback (~20s expected)")
        model = genai.GenerativeModel("gemini-2.5-flash-lite")

        # Format metadata in a readable format for the LLM
        metadata_str = json.dumps(metadata, indent=2)

        # Format conversation history if available
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            # Get the last few exchanges to provide context (limit to last 10 exchanges to keep it focused)
            recent_history = conversation_history[-10:]
            conversation_context = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in recent_history]
            )
            conversation_context = f"""
            Previous conversation history:
            {conversation_context}
            """

        # Full prompt with metadata included (original behavior)
        prompt = f"""
        Based on the following user question, conversation history, and available document metadata, determine which documents are most relevant to answer the question.

        {conversation_context}

        Current user question: "{query}"

        Available document metadata:
        {metadata_str}

        For each company mentioned or implied in the question or previous conversation, select the most relevant documents.
        Consider aliases, abbreviations, or partial references to companies.
        Consider the full context of the conversation when determining relevance.

        IMPORTANT: Select a maximum of 3 documents total, prioritizing the most relevant ones.

        Return your response as a valid JSON object with this structure:
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
        }}

        ONLY return valid JSON. Do not include any explanations or text outside the JSON structure.
        """

        response = model.generate_content(prompt)
        response_text = response.text

        # Extract the JSON part from the response
        try:
            # Try to find JSON in the response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                response_text = response_text[json_start:json_end]

            recommendation = json.loads(response_text)

            logger.info("Successfully completed LLM fallback with traditional approach")

            return recommendation
        except json.JSONDecodeError as e:
            logger.error(
                "llm_response_parse_failed",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "response_preview": response_text[:200] if response_text else None,
                },
            )
            return None

    except Exception as e:
        logger.error(
            "semantic_selection_failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "has_metadata": bool(metadata),
                "has_history": bool(conversation_history),
            },
        )
        return None


# =============================================================================
# SYNCHRONOUS DOCUMENT AUTO-LOADING
# =============================================================================


def auto_load_relevant_documents(
    s3_client, query, metadata, current_document_texts, conversation_history=None
):
    """Load relevant documents based on query and conversation history"""
    document_texts = current_document_texts.copy() if current_document_texts else {}
    loaded_docs = []

    # Use LLM to determine which documents to load, including conversation history
    recommendation = semantic_document_selection(query, metadata, conversation_history)

    if recommendation and "documents_to_load" in recommendation:
        docs_to_load = recommendation["documents_to_load"]

        # Load the recommended documents (limited to 3)
        for doc_info in docs_to_load[:3]:
            doc_link = doc_info["document_link"]
            doc_name = doc_info["filename"]

            # Only load if not already loaded
            if doc_name not in document_texts:
                try:
                    text = download_and_extract_from_s3(s3_client, doc_link)
                    if text:
                        document_texts[doc_name] = text
                        loaded_docs.append(doc_name)
                except Exception as e:
                    logger.error(
                        "document_load_failed",
                        extra={
                            "document": doc_name,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )

        # Generate message about what was loaded
        if loaded_docs:
            message = f"Semantically selected {len(loaded_docs)} documents based on your query:\n"
            for doc_name in loaded_docs:
                matching_doc = next((d for d in docs_to_load if d["filename"] == doc_name), None)
                if matching_doc:
                    message += f"• {doc_name} - {matching_doc.get('reason', '')}\n"
            return document_texts, message, loaded_docs
        else:
            return (
                document_texts,
                "No documents were loaded. Please check S3 access permissions or document availability.",
                [],
            )

    return document_texts, "No relevant documents were identified for your query.", []


# =============================================================================
# ASYNC DOCUMENT AUTO-LOADING
# =============================================================================


async def auto_load_relevant_documents_async(
    query: str,
    metadata: Dict,
    conversation_history: Optional[List] = None,
    current_document_texts: Optional[Dict] = None,
    config: Optional[S3DownloadConfig] = None,
    progress_callback: Optional[callable] = None,
) -> Tuple[Dict[str, str], str, List[str]]:
    """
    Asynchronously load relevant documents based on query and conversation history with concurrent downloads.

    Args:
        query: User query
        metadata: Document metadata
        conversation_history: Optional conversation context
        current_document_texts: Existing loaded documents
        config: Download configuration
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (document_texts, message, loaded_docs)
    """
    if config is None:
        config = get_config().s3_download

    document_texts = current_document_texts.copy() if current_document_texts else {}
    loaded_docs = []

    try:
        if progress_callback:
            await progress_callback(
                "document_selection_start", "Analyzing query to select relevant documents"
            )

        # Use LLM to determine which documents to load, including conversation history
        recommendation = semantic_document_selection(query, metadata, conversation_history)

        if not recommendation or "documents_to_load" not in recommendation:
            message = "No relevant documents were identified for your query."
            if progress_callback:
                await progress_callback("document_selection_complete", message)
            return document_texts, message, []

        docs_to_load = recommendation["documents_to_load"]
        companies_mentioned = recommendation.get("companies_mentioned", [])

        if progress_callback:
            await progress_callback(
                "document_selection_complete",
                f"Selected {len(docs_to_load)} documents from {len(companies_mentioned)} companies",
            )

        # Filter documents that aren't already loaded
        docs_to_download = []
        for doc_info in docs_to_load[:3]:  # Limit to 3 documents
            doc_name = doc_info["filename"]
            if doc_name not in document_texts:
                docs_to_download.append(doc_info)

        if not docs_to_download:
            message = "All relevant documents are already loaded."
            if progress_callback:
                await progress_callback("documents_already_loaded", message)
            return document_texts, message, []

        if progress_callback:
            await progress_callback(
                "concurrent_download_start",
                f"Starting concurrent download of {len(docs_to_download)} documents",
            )

        # Create download tasks for concurrent execution
        download_tasks = []
        for doc_info in docs_to_download:
            task = _download_single_document_async(doc_info, config, progress_callback)
            download_tasks.append(task)

        # Execute downloads concurrently with semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(config.concurrent_downloads)
        download_results = await asyncio.gather(
            *[_download_with_semaphore(semaphore, task) for task in download_tasks],
            return_exceptions=True,
        )

        # Process results
        successful_downloads = 0
        failed_downloads = 0

        for i, result in enumerate(download_results):
            doc_info = docs_to_download[i]
            doc_name = doc_info["filename"]

            if isinstance(result, Exception):
                logger.error(f"Download task failed for {doc_name}: {str(result)}")
                failed_downloads += 1
            elif isinstance(result, DownloadResult):
                if result.success and result.content:
                    document_texts[doc_name] = result.content
                    loaded_docs.append(doc_name)
                    successful_downloads += 1
                    logger.info(
                        f"Successfully loaded {doc_name} ({result.download_time:.2f}s, {result.retry_count} retries)"
                    )
                else:
                    logger.error(f"Failed to load document {doc_name}: {result.error}")
                    failed_downloads += 1
            else:
                logger.error(f"Unexpected result type for {doc_name}: {type(result)}")
                failed_downloads += 1

        # Generate summary message
        if successful_downloads > 0:
            message = f"Successfully loaded {successful_downloads} documents concurrently"
            if failed_downloads > 0:
                message += f" ({failed_downloads} failed)"
            message += ":\n"

            for doc_name in loaded_docs:
                matching_doc = next((d for d in docs_to_load if d["filename"] == doc_name), None)
                if matching_doc:
                    message += f"• {doc_name} - {matching_doc.get('reason', '')}\n"
        else:
            message = "Failed to load any documents. Please check S3 access permissions."
            if failed_downloads > 0:
                message += f" ({failed_downloads} download failures)"

        if progress_callback:
            await progress_callback(
                "concurrent_download_complete",
                f"Completed: {successful_downloads} successful, {failed_downloads} failed",
            )

        return document_texts, message, loaded_docs

    except Exception as e:
        error_msg = f"Error in async document loading: {str(e)}"
        logger.error(
            "async_document_load_failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "documents_loaded": len(loaded_docs),
            },
        )
        if progress_callback:
            await progress_callback("document_load_error", error_msg)
        return document_texts, error_msg, []


async def _download_single_document_async(
    doc_info: Dict, config: S3DownloadConfig, progress_callback: Optional[callable] = None
) -> DownloadResult:
    """Download a single document asynchronously"""
    doc_link = doc_info["document_link"]
    doc_name = doc_info["filename"]

    try:
        if progress_callback:
            await progress_callback("single_download_start", f"Downloading {doc_name}")

        result = await download_and_extract_from_s3_async(doc_link, config, progress_callback)

        if progress_callback:
            status = "success" if result.success else "failed"
            await progress_callback("single_download_complete", f"{doc_name}: {status}")

        return result

    except Exception as e:
        error_msg = f"Error downloading {doc_name}: {str(e)}"
        logger.error(
            "single_document_download_failed",
            extra={"document": doc_name, "error": str(e), "error_type": type(e).__name__},
        )
        return DownloadResult(success=False, error=error_msg)


async def _download_with_semaphore(semaphore: asyncio.Semaphore, download_task) -> DownloadResult:
    """Execute download task with semaphore to limit concurrent connections"""
    async with semaphore:
        return await download_task
