"""
Document selection and auto-loading operations.

This module provides intelligent document selection using LLM-based semantic analysis
and supports both synchronous and asynchronous document loading operations.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple

from google.genai import types

from app.config import S3DownloadConfig, get_config
from app.gemini_client import get_genai_client
from app.logging_config import get_logger
from app.s3_client import (
    DownloadResult,
    download_and_extract_from_s3,
    download_and_extract_from_s3_async,
)
from app.utils.monitoring import record_document_load, record_document_selection

logger = get_logger(__name__)


# =============================================================================
# COMPANY EXTRACTION AND RESOLUTION (Two-Stage Approach)
# =============================================================================


def extract_companies_from_query(
    query: str,
    available_companies: List[str],
    available_symbols: Optional[List[str]] = None,
    conversation_history: Optional[List] = None,
) -> Dict:
    """
    Use fast LLM to extract company names/symbols from user query.

    This is Stage 1 of the two-stage document selection approach.
    Uses a smaller, faster model (gemini-2.5-flash) for simple entity extraction.

    Args:
        query: User query
        available_companies: List of valid company names from metadata
        available_symbols: Optional list of valid trading symbols
        conversation_history: Optional conversation context

    Returns:
        Dict with companies and symbols mentioned
    """
    start_time = time.time()
    try:
        client = get_genai_client()
        model_name = "gemini-2.5-flash"  # Fast model for simple extraction

        # Format conversation history if available
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-6:]  # Last 3 exchanges
            conversation_context = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in recent_history]
            )

        # Build a concise prompt with just company/symbol lists
        symbols_section = ""
        if available_symbols:
            symbols_section = f"Available trading symbols: {', '.join(available_symbols)}"

        prompt = f"""Extract company names and trading symbols mentioned in the user query.

Available companies:
{', '.join(available_companies)}

{symbols_section}

{"Previous conversation:" if conversation_context else ""}
{conversation_context}

Current query: "{query}"

Instructions:
- Match company names even with partial mentions, abbreviations, or aliases
- For follow-up questions like "what about their 2023 report?", identify companies from conversation context
- Trading symbols are typically 2-5 uppercase letters (e.g., NCB, GK, MDS)
- Return empty arrays if no companies/symbols are mentioned

Return ONLY valid JSON in this format:
{{"companies": ["company1", "company2"], "symbols": ["SYM1", "SYM2"]}}
"""

        # Build content for the SDK
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

        # Use low temperature for deterministic extraction
        generate_config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=512,
        )

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generate_config,
        )
        response_text = response.text.strip()

        # Parse JSON response
        clean_text = response_text
        if clean_text.startswith("```"):
            first_newline = clean_text.find("\n")
            if first_newline != -1:
                clean_text = clean_text[first_newline + 1 :]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3].strip()

        json_start = clean_text.find("{")
        json_end = clean_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            clean_text = clean_text[json_start:json_end]

        result = json.loads(clean_text)

        duration = time.time() - start_time
        logger.info(
            f"Company extraction completed in {duration:.2f}s: "
            f"companies={result.get('companies', [])}, symbols={result.get('symbols', [])}"
        )

        return result

    except Exception as e:
        logger.error(
            "company_extraction_failed",
            extra={"error": str(e), "error_type": type(e).__name__, "query": query[:100]},
        )
        return {"companies": [], "symbols": []}


def resolve_companies(
    extracted: Dict,
    available_companies: List[str],
    symbol_to_company: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """
    Resolve extracted companies/symbols to valid company names using Python.

    This is Stage 2 of the two-stage document selection approach.
    Uses case-insensitive matching, partial matching, and symbol resolution.
    Deduplicates by lowercase key to ensure no duplicate companies with different casing.

    Args:
        extracted: Dict with 'companies' and 'symbols' from extraction step
        available_companies: List of valid company names from metadata
        symbol_to_company: Optional mapping from symbols to company names

    Returns:
        List of resolved, valid company names (using original metadata casing)
    """
    # Create a lowercase -> original casing map for consistent deduplication
    available_map = {c.lower(): c for c in available_companies}
    resolved_lower = set()  # Track by lowercase for deduplication
    resolved_map = {}  # lowercase -> original casing from metadata

    # Resolve symbols to companies first
    if extracted.get("symbols") and symbol_to_company:
        for symbol in extracted["symbols"]:
            symbol_upper = symbol.upper()
            matched_companies = None

            # Try exact match
            if symbol_upper in symbol_to_company:
                matched_companies = symbol_to_company[symbol_upper]
            else:
                # Try case-insensitive search through keys
                for key, companies in symbol_to_company.items():
                    if key.upper() == symbol_upper:
                        matched_companies = companies
                        break

            if matched_companies:
                for company in matched_companies:
                    # Normalize to metadata casing
                    company_lower = company.lower()
                    if company_lower in available_map and company_lower not in resolved_lower:
                        resolved_lower.add(company_lower)
                        resolved_map[company_lower] = available_map[company_lower]
                    elif company_lower not in available_map:
                        logger.warning(f"Symbol {symbol} mapped to unknown company: '{company}'")

    # Match company names
    for company in extracted.get("companies", []):
        company_lower = company.lower()

        # Skip if already resolved
        if company_lower in resolved_lower:
            continue

        matched = False

        # Try exact match (case-insensitive)
        if company_lower in available_map:
            resolved_lower.add(company_lower)
            resolved_map[company_lower] = available_map[company_lower]
            matched = True
        else:
            # Try partial match if no exact match
            for valid_lower, valid_company in available_map.items():
                # Check if extracted name is contained in valid name or vice versa
                if company_lower in valid_lower or valid_lower in company_lower:
                    if valid_lower not in resolved_lower:
                        resolved_lower.add(valid_lower)
                        resolved_map[valid_lower] = valid_company
                        matched = True
                        break

        if not matched:
            logger.warning(f"Could not resolve company: '{company}'")

    result = list(resolved_map.values())
    logger.info(f"Resolved {len(result)} companies: {result}")
    return result


def filter_documents_by_companies(metadata: Dict, companies: List[str]) -> Dict:
    """
    Filter document metadata to only include specified companies.

    Args:
        metadata: Full S3 metadata (company_name -> list of documents)
        companies: List of company names to filter by

    Returns:
        Filtered metadata containing only the specified companies
    """
    companies_set = {c.lower() for c in companies}
    filtered = {}

    for company_key, documents in metadata.items():
        if company_key.lower() in companies_set:
            filtered[company_key] = documents

    logger.info(
        f"Filtered metadata: {len(filtered)} companies, "
        f"{sum(len(docs) for docs in filtered.values())} documents"
    )
    return filtered


def select_documents_from_filtered(
    query: str,
    filtered_metadata: Dict,
    conversation_history: Optional[List] = None,
    max_documents: int = 3,
) -> Dict:
    """
    Select specific documents from the filtered company list.

    This is the final stage: select the most relevant documents from
    the pre-filtered set. Uses LLM with much smaller context.

    Args:
        query: User query
        filtered_metadata: Metadata filtered to relevant companies only
        conversation_history: Optional conversation context
        max_documents: Maximum number of documents to select

    Returns:
        Dictionary with companies_mentioned and documents_to_load
    """
    if not filtered_metadata:
        return {"companies_mentioned": [], "documents_to_load": []}

    start_time = time.time()
    try:
        client = get_genai_client()
        model_name = "gemini-2.5-flash"  # Fast model for selection

        # Format the filtered metadata (much smaller than full metadata)
        metadata_str = json.dumps(filtered_metadata, indent=2)

        # Format conversation history
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-6:]
            conversation_context = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in recent_history]
            )

        prompt = f"""Select the most relevant documents to answer the user's question.

{"Previous conversation:" if conversation_context else ""}
{conversation_context}

Current question: "{query}"

Available documents:
{metadata_str}

Instructions:
- Select a maximum of {max_documents} documents
- Prefer annual reports for general company questions
- Prefer financial statements for specific financial questions
- Prefer the most recent documents unless a specific year is mentioned
- Consider document_type and period when selecting

Return ONLY valid JSON:
{{
  "companies_mentioned": ["company1", "company2"],
  "documents_to_load": [
    {{
      "company": "company name",
      "document_link": "full s3 link",
      "filename": "filename.pdf",
      "reason": "brief reason"
    }}
  ]
}}
"""

        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

        generate_config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=2048,  # Increased to prevent JSON truncation
        )

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generate_config,
        )
        response_text = response.text.strip()

        # Parse JSON response
        clean_text = response_text
        if clean_text.startswith("```"):
            first_newline = clean_text.find("\n")
            if first_newline != -1:
                clean_text = clean_text[first_newline + 1 :]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3].strip()

        json_start = clean_text.find("{")
        json_end = clean_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            clean_text = clean_text[json_start:json_end]

        result = json.loads(clean_text)

        duration = time.time() - start_time
        logger.info(
            f"Document selection completed in {duration:.2f}s: "
            f"{len(result.get('documents_to_load', []))} documents selected"
        )

        return result

    except Exception as e:
        logger.error(
            "document_selection_failed",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        return {"companies_mentioned": list(filtered_metadata.keys()), "documents_to_load": []}


# =============================================================================
# SEMANTIC DOCUMENT SELECTION (Main Entry Point)
# =============================================================================


def semantic_document_selection(
    query: str,
    metadata: Dict,
    conversation_history: Optional[List] = None,
    associations: Optional[Dict] = None,
):
    """
    Use two-stage LLM approach to determine which documents to load.

    Stage 1: Extract company names/symbols from query (fast model)
    Stage 2: Resolve to valid companies using Python (deterministic)
    Stage 3: Filter documents to matched companies (deterministic)
    Stage 4: Select specific documents from filtered list (fast model)

    This approach reduces hallucination risk and improves determinism by:
    - Using smaller context windows
    - Separating entity extraction from document selection
    - Using Python for validation and filtering

    Args:
        query: User query
        metadata: S3 metadata containing available documents (company -> documents)
        conversation_history: Optional conversation context
        associations: Optional dict with 'symbol_to_company' mapping from FinancialDataManager

    Returns:
        Dictionary with companies_mentioned and documents_to_load
    """
    start_time = time.time()
    try:
        logger.info("Using two-stage document selection")

        # Get list of available companies from metadata keys
        available_companies = list(metadata.keys())

        # Get symbol mapping if associations provided
        symbol_to_company = None
        available_symbols = None
        if associations:
            symbol_to_company = associations.get("symbol_to_company", {})
            available_symbols = list(symbol_to_company.keys()) if symbol_to_company else None

        # Stage 1: Extract companies/symbols from query using fast model
        extracted = extract_companies_from_query(
            query=query,
            available_companies=available_companies,
            available_symbols=available_symbols,
            conversation_history=conversation_history,
        )

        # Stage 2: Resolve extracted entities to valid company names
        resolved_companies = resolve_companies(
            extracted=extracted,
            available_companies=available_companies,
            symbol_to_company=symbol_to_company,
        )

        # If no companies resolved, return empty result
        if not resolved_companies:
            logger.warning("No companies could be resolved from query")
            duration = time.time() - start_time
            record_document_selection(duration=duration, num_documents=0)
            return {"companies_mentioned": [], "documents_to_load": []}

        # Stage 3: Filter metadata to only include resolved companies
        filtered_metadata = filter_documents_by_companies(metadata, resolved_companies)

        # Stage 4: Select specific documents from filtered list
        result = select_documents_from_filtered(
            query=query,
            filtered_metadata=filtered_metadata,
            conversation_history=conversation_history,
            max_documents=3,
        )

        # Record metrics
        duration = time.time() - start_time
        num_docs = len(result.get("documents_to_load", []))
        record_document_selection(duration=duration, num_documents=num_docs)

        logger.info(
            f"Two-stage document selection completed in {duration:.2f}s: "
            f"{len(resolved_companies)} companies, {num_docs} documents"
        )

        return result

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
    s3_client,
    query,
    metadata,
    current_document_texts,
    conversation_history=None,
    associations=None,
):
    """
    Load relevant documents based on query and conversation history.

    Args:
        s3_client: S3 client for downloading documents
        query: User query
        metadata: S3 metadata (company -> documents)
        current_document_texts: Already loaded documents
        conversation_history: Optional conversation context
        associations: Optional dict with 'symbol_to_company' mapping for better resolution
    """
    document_texts = current_document_texts.copy() if current_document_texts else {}
    loaded_docs = []

    # Use two-stage LLM approach to determine which documents to load
    recommendation = semantic_document_selection(
        query, metadata, conversation_history, associations
    )

    if recommendation and "documents_to_load" in recommendation:
        docs_to_load = recommendation["documents_to_load"]

        # Load the recommended documents (limited to 3)
        for doc_info in docs_to_load[:3]:
            doc_link = doc_info["document_link"]
            doc_name = doc_info["filename"]

            # Only load if not already loaded
            if doc_name not in document_texts:
                load_start = time.time()
                try:
                    text = download_and_extract_from_s3(s3_client, doc_link)
                    load_duration = time.time() - load_start
                    if text:
                        document_texts[doc_name] = text
                        loaded_docs.append(doc_name)
                        # Record successful document load from S3
                        record_document_load(source="s3", duration=load_duration, success=True)
                    else:
                        # Record failed document load
                        record_document_load(source="error", duration=load_duration, success=False)
                except Exception as e:
                    load_duration = time.time() - load_start
                    # Record failed document load
                    record_document_load(source="error", duration=load_duration, success=False)
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
    associations: Optional[Dict] = None,
) -> Tuple[Dict[str, str], str, List[str]]:
    """
    Asynchronously load relevant documents based on query and conversation history with concurrent downloads.

    Args:
        query: User query
        metadata: Document metadata (company -> documents)
        conversation_history: Optional conversation context
        current_document_texts: Existing loaded documents
        config: Download configuration
        progress_callback: Optional callback for progress updates
        associations: Optional dict with 'symbol_to_company' mapping for better resolution

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

        # Use two-stage LLM approach to determine which documents to load
        recommendation = semantic_document_selection(
            query, metadata, conversation_history, associations
        )

        if not recommendation or "documents_to_load" not in recommendation:
            message = "No relevant documents were identified for your query."
            if progress_callback:
                await progress_callback("document_selection_complete", message)
            return document_texts, message, []

        docs_to_load = recommendation["documents_to_load"]
        companies_mentioned = recommendation.get("companies_mentioned", [])

        # Note: Document selection metrics already recorded in semantic_document_selection

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
                # Record failed document load
                record_document_load(source="error", duration=0, success=False)
            elif isinstance(result, DownloadResult):
                if result.success and result.content:
                    document_texts[doc_name] = result.content
                    loaded_docs.append(doc_name)
                    successful_downloads += 1
                    # Record successful async document load from S3
                    record_document_load(source="s3", duration=result.download_time, success=True)
                    logger.info(
                        f"Successfully loaded {doc_name} ({result.download_time:.2f}s, {result.retry_count} retries)"
                    )
                else:
                    logger.error(f"Failed to load document {doc_name}: {result.error}")
                    failed_downloads += 1
                    # Record failed document load
                    record_document_load(
                        source="error", duration=result.download_time, success=False
                    )
            else:
                logger.error(f"Unexpected result type for {doc_name}: {type(result)}")
                failed_downloads += 1
                # Record failed document load
                record_document_load(source="error", duration=0, success=False)

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
