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
    doc_entries = []
    
    for company_data in metadata:
        company_name = company_data.get("company_name", "")
        
        # Process each document for this company
        for doc in company_data.get("documents", []):
            filename = os.path.basename(doc.get("filename", ""))
            doc_type = doc.get("type", "unknown")
            period = doc.get("period", "unknown")
            
            # Skip if essential data is missing
            if not filename or not company_name:
                logger.warning(f"Skipping document with missing data: {doc}")
                continue
            
            # Create description for embedding: "company - doc_type - period"
            description = f"{company_name} - {doc_type} - {period}"
            
            # Format according to MetaDocumentInfo model
            doc_entry = {
                "filename": filename,
                "company": company_name,
                "period": period,
                "type": doc_type,
                "description": description
            }
            
            doc_entries.append(doc_entry)
    
    logger.info(f"Extracted {len(doc_entries)} document entries from metadata")
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
        
        try:
            # Make API request
            response = requests.post(endpoint_url, json=request_data, timeout=60)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            batch_added = len(result.get("ids", []))
            total_added += batch_added
            
            logger.info(f"Batch {batch_num} completed: {batch_added} documents added")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending batch {batch_num} to API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing batch {batch_num}: {e}")
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