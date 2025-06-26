#!/usr/bin/env python3
"""
Script to populate the doc_meta collection via the /chroma/meta/update API endpoint.

This script reads metadata from S3 and uses the FastAPI application's 
/chroma/meta/update endpoint to populate the doc_meta collection with 
document metadata for embedding-based semantic document selection.

SETUP REQUIREMENTS:
1. Environment variables must be set:
   - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
   - DOCUMENT_METADATA_S3_BUCKET
   - SUMMARIZER_API_KEY (Google API key for embeddings)

2. FastAPI application must be running and accessible

USAGE:
    python scripts/populate_doc_meta.py
    
    # Or specify custom API URL and batch size:
    python scripts/populate_doc_meta.py --api-url http://localhost:8000 --batch-size 50
"""

import os
import sys
import json
import logging
import argparse
import requests
import time
from typing import List, Dict, Any

# Add the fastapi_app directory to the path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'fastapi_app'))

from app.utils import init_s3_client, load_metadata_from_s3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_metadata_for_api(metadata: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Extract and format metadata for the /chroma/meta/update API endpoint.
    
    Args:
        metadata: S3 metadata list
        
    Returns:
        List of MetaDocumentInfo dictionaries for the API
    """
    doc_entries: List[Dict[str, str]] = []

    # The metadata loaded from S3 can have two possible shapes:
    # 1. A list where each item is a dict with the keys "company_name", "documents", etc.
    # 2. A dict keyed by company name where the value is another dict that contains a
    #    "documents" list.  In the second case, iterating directly over the metadata
    #    gives us strings (the company_name keys), which caused the original
    #    AttributeError.  To support both shapes we normalise the data to a list of
    #    dictionaries that *always* contain the key "company_name".

    if isinstance(metadata, dict):
        # Convert to a uniform list of dicts that mimic the shape in option (1).
        normalised_metadata: List[Dict[str, Any]] = []
        for company_name, company_info in metadata.items():
            # Two possibilities here:
            # a) company_info is already a mapping with a key "documents" (old shape)
            # b) company_info is *itself* a list[dict] of documents (new shape shown by the user)

            if isinstance(company_info, dict):
                # Old shape → just tack on company_name and push through.
                company_copy = company_info.copy()
                company_copy.setdefault("company_name", company_name)
                normalised_metadata.append(company_copy)
            elif isinstance(company_info, list):
                # New shape → wrap it into a dict with key "documents" so the processing loop below can stay the same.
                normalised_metadata.append({
                    "company_name": company_name,
                    "documents": company_info,
                })
            else:
                logger.warning(
                    "Skipping company '%s' because the associated metadata type is unsupported (type=%s)",
                    company_name,
                    type(company_info).__name__,
                )
                continue
    else:
        # Assume the metadata is already an iterable of dicts.
        normalised_metadata = metadata  # type: ignore[assignment]

    # Now process the normalised list.
    for company_data in normalised_metadata:
        # Validate expected structure.
        if not isinstance(company_data, dict):
            logger.warning(
                "Encountered non-dict item in metadata list of type %s – skipping",
                type(company_data).__name__
            )
            continue

        company_name: str = company_data.get("company_name", "")

        # Process each document for this company
        for doc in company_data.get("documents", []):
            # Each doc should be a dict too – skip otherwise.
            if not isinstance(doc, dict):
                logger.warning(
                    "Encountered non-dict document entry for company '%s' (type=%s) – skipping",
                    company_name,
                    type(doc).__name__
                )
                continue

            filename = os.path.basename(doc.get("filename", ""))
            doc_type = doc.get("type", "unknown")
            period = doc.get("period", "unknown")

            # Skip if essential data is missing
            if not filename or not company_name:
                logger.warning("Skipping document with missing data: %s", doc)
                continue

            # Create description for embedding: "company - doc_type - period"
            description = f"{company_name} - {doc_type} - {period}"

            # Format according to MetaDocumentInfo model
            doc_entry: Dict[str, str] = {
                "filename": filename,
                "company": company_name,
                "period": period,
                "type": doc_type,
                "description": description,
            }

            doc_entries.append(doc_entry)

    logger.info("Extracted %s document entries from metadata", len(doc_entries))
    return doc_entries


def populate_via_api(doc_entries: List[Dict[str, str]], api_url: str, batch_size: int = 100) -> int:
    """
    Populate the metadata collection via the /chroma/meta/update API endpoint.
    
    Args:
        doc_entries: List of document metadata entries
        api_url: Base URL of the FastAPI application
        batch_size: Number of documents to send per request
        
    Returns:
        Total number of documents successfully added
    """
    if not doc_entries:
        logger.warning("No document entries to populate")
        return 0
    
    endpoint_url = f"{api_url.rstrip('/')}/chroma/meta/update"
    total_added = 0
    
    # Process in batches
    for i in range(0, len(doc_entries), batch_size):
        batch = doc_entries[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(doc_entries) + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
        
        # Prepare request payload
        request_data = {
            "documents": batch
        }
        
        # ------------------------------------------------------------
        # Send request with retry logic for transient 5xx errors
        # ------------------------------------------------------------
        max_retries = 3
        backoff_seconds = 5
        attempt = 0
        while attempt < max_retries:
            try:
                response = requests.post(endpoint_url, json=request_data, timeout=60)
                response.raise_for_status()

                # Parse response on success
                result = response.json()
                batch_added = len(result.get("ids", []))
                total_added += batch_added

                logger.info(
                    "Batch %s completed: %s documents added (attempt %s/%s)",
                    batch_num,
                    batch_added,
                    attempt + 1,
                    max_retries,
                )
                break  # Success – exit retry loop

            except requests.exceptions.RequestException as e:
                attempt += 1

                # ------------------------------------------------------
                # 1) Duplicate-ID error handling
                # ------------------------------------------------------
                if hasattr(e, "response") and e.response is not None:
                    status_code = e.response.status_code
                    body_text = e.response.text

                    # If duplicate-ID error, strip duplicates and retry once
                    if status_code == 500 and "Expected IDs to be unique" in body_text and "duplicates of:" in body_text:
                        dup_part = body_text.split("duplicates of:")[-1]
                        dup_part = dup_part.split("in add")[0]
                        dup_part = dup_part.strip().strip('"').strip()
                        duplicate_ids = [d.strip() for d in dup_part.replace("\n", ",").split(",") if d.strip()]

                        if duplicate_ids:
                            logger.warning(
                                "Batch %s attempt %s/%s – removing %s duplicate IDs and retrying immediately.",
                                batch_num,
                                attempt,
                                max_retries,
                                len(duplicate_ids),
                            )

                            # Remove duplicates from current batch and rebuild payload
                            batch = [doc for doc in batch if doc.get("filename") not in duplicate_ids]

                            if not batch:
                                logger.info("All documents in batch %s were duplicates – skipping batch.", batch_num)
                                break  # Exit retry loop successfully (nothing to add)

                            # Refresh request_data & reset attempt counter so we still get 3 network tries
                            request_data = {"documents": batch}
                            attempt = 0  # reset attempts for cleaned batch
                            continue  # Immediate retry with cleaned batch

                    # --------------------------------------------------
                    # 2) Transient 5xx errors (except duplicates) → retry
                    # --------------------------------------------------
                    if 500 <= status_code < 600:
                        if attempt < max_retries:
                            logger.warning(
                                "Transient %s error on batch %s (attempt %s/%s). Retrying in %s seconds...",
                                status_code,
                                batch_num,
                                attempt,
                                max_retries,
                                backoff_seconds,
                            )
                            time.sleep(backoff_seconds * attempt)
                            continue

                # Out of retries or not retryable → raise
                logger.error(
                    "Failed request for batch %s on attempt %s/%s: %s",
                    batch_num,
                    attempt,
                    max_retries,
                    e,
                )
                if attempt >= max_retries:
                    raise

        else:
            # Exceeded retries without success
            logger.error(
                "Failed to process batch %s after %s attempts – aborting.",
                batch_num,
                max_retries,
            )
            raise

    logger.info(f"Successfully added {total_added} documents total")
    return total_added


def test_api_connection(api_url: str) -> bool:
    """Test connection to the FastAPI application."""
    try:
        # Test basic connectivity to the API
        response = requests.get(f"{api_url.rstrip('/')}/", timeout=10)
        if response.status_code == 200:
            logger.info("Successfully connected to FastAPI application")
            return True
        else:
            logger.error(f"API responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to API at {api_url}: {e}")
        return False


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Populate doc_meta collection via /chroma/meta/update API endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default settings
    python scripts/populate_doc_meta.py
    
    # Specify custom API URL and batch size
    python scripts/populate_doc_meta.py --api-url http://localhost:8000 --batch-size 50
    
    # Test API connection only
    python scripts/populate_doc_meta.py --test-connection
        """
    )
    
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the FastAPI application (default: http://localhost:8000)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of documents to send per API request (default: 100)"
    )
    
    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Test API connection and exit"
    )
    
    return parser.parse_args()


def check_required_env_vars():
    """Check that all required environment variables are set."""
    required_vars = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY", 
        "AWS_DEFAULT_REGION",
        "DOCUMENT_METADATA_S3_BUCKET",
        "SUMMARIZER_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error("Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"  - {var}")
        logger.error("\nPlease set these environment variables and try again.")
        return False
    
    return True


def main():
    """Main function to populate the doc_meta collection via API."""
    # Parse command-line arguments
    args = parse_arguments()
    
    logger.info("Starting doc_meta collection population via API")
    
    # Test API connection if requested
    if args.test_connection:
        logger.info(f"Testing API connection to {args.api_url}...")
        if test_api_connection(args.api_url):
            logger.info("✅ API connection test successful!")
            return 0
        else:
            logger.error("❌ API connection test failed!")
            return 1
    
    # Check required environment variables
    if not check_required_env_vars():
        return 1
    
    try:
        # Test API connection first
        logger.info(f"Testing connection to FastAPI application at {args.api_url}...")
        if not test_api_connection(args.api_url):
            logger.error("Cannot proceed - API is not accessible")
            logger.error("Make sure the FastAPI application is running and accessible")
            return 1
        
        # Initialize S3 client and load metadata
        logger.info("Initializing S3 client...")
        s3_client = init_s3_client()
        
        logger.info("Loading metadata from S3...")
        metadata = load_metadata_from_s3(s3_client)
        
        if not metadata:
            logger.error("Failed to load metadata from S3")
            return 1
        
        logger.info(f"Loaded metadata for {len(metadata)} companies")
        
        # Extract document entries
        logger.info("Extracting document entries...")
        doc_entries = extract_metadata_for_api(metadata)
        
        if not doc_entries:
            logger.error("No document entries found in metadata")
            return 1
        
        # Populate via API
        logger.info(f"Populating collection via API (batch size: {args.batch_size})...")
        total_added = populate_via_api(doc_entries, args.api_url, args.batch_size)
        
        # Log summary
        logger.info(f"✅ Successfully populated doc_meta collection with {total_added} documents")
        return 0
        
    except Exception as e:
        logger.error(f"Error during population: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)