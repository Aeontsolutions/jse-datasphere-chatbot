#!/usr/bin/env python3
"""
Script to populate the doc_meta ChromaDB collection from S3 metadata.

This script reads the metadata from S3 and populates the doc_meta collection
with document metadata for embedding-based semantic document selection.

SETUP REQUIREMENTS:
1. Environment variables must be set:
   - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
   - DOCUMENT_METADATA_S3_BUCKET
   - SUMMARIZER_API_KEY (Google API key for embeddings)

2. ChromaDB setup - choose one:
   Option A (Local storage - recommended for development):
     unset CHROMA_HOST
     export CHROMA_PERSIST_DIRECTORY="./chroma_db"
   
   Option B (Remote server):
     pip install chromadb[server]
     chroma run --host localhost --port 8000  # in separate terminal
     export CHROMA_HOST="localhost"
     export CHROMA_PORT="8000"

USAGE:
    python scripts/populate_doc_meta.py
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any

# Add the fastapi_app directory to the path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'fastapi_app'))

from app.utils import init_s3_client, load_metadata_from_s3
from app.chroma_utils import init_chroma_client, get_or_create_collection, add_documents
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_metadata_for_embedding(metadata: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Extract and format metadata for the doc_meta collection.
    
    Args:
        metadata: S3 metadata list
        
    Returns:
        List of formatted metadata entries
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


def populate_meta_collection(doc_entries: List[Dict[str, str]], meta_collection):
    """
    Populate the metadata collection with document entries.
    
    Args:
        doc_entries: List of document metadata entries
        meta_collection: ChromaDB collection for metadata
    """
    if not doc_entries:
        logger.warning("No document entries to populate")
        return
    
    # Prepare data for ChromaDB
    documents = []
    metadatas = []
    ids = []
    
    for entry in doc_entries:
        documents.append(entry["description"])  # This gets embedded
        
        # Store all metadata
        metadata = {
            "filename": entry["filename"],
            "company": entry["company"],
            "period": entry["period"],
            "type": entry["type"]
        }
        metadatas.append(metadata)
        
        # Use filename as ID (assuming filenames are unique)
        ids.append(entry["filename"])
    
    try:
        # Add documents to collection
        result_ids = add_documents(meta_collection, documents, metadatas, ids)
        logger.info(f"Successfully added {len(result_ids)} documents to meta collection")
        return result_ids
    except Exception as e:
        logger.error(f"Error adding documents to meta collection: {e}")
        raise


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
    """Main function to populate the doc_meta collection."""
    logger.info("Starting doc_meta collection population")
    
    # Check environment variables first
    if not check_required_env_vars():
        return 1
    
    try:
        # Initialize clients
        logger.info("Initializing S3 client...")
        s3_client = init_s3_client()
        
        logger.info("Initializing ChromaDB client...")
        try:
            chroma_client = init_chroma_client()
        except Exception as chroma_error:
            logger.error(f"Failed to initialize ChromaDB client: {chroma_error}")
            logger.error("\n" + "="*60)
            logger.error("CHROMADB CONNECTION ERROR - SETUP REQUIRED")
            logger.error("="*60)
            chroma_host = os.getenv("CHROMA_HOST")
            if chroma_host:
                logger.error(f"Trying to connect to ChromaDB server at: {chroma_host}")
                logger.error("SOLUTION 1 - Start ChromaDB server:")
                logger.error("  pip install chromadb[server]")
                logger.error("  chroma run --host localhost --port 8000")
                logger.error("")
                logger.error("SOLUTION 2 - Use local storage instead:")
                logger.error("  unset CHROMA_HOST")
                logger.error("  unset CHROMA_PORT")
                logger.error("  export CHROMA_PERSIST_DIRECTORY='./chroma_db'")
            else:
                logger.error("Using local ChromaDB storage but connection failed")
                logger.error("Make sure you have write permissions to the chroma directory")
                persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/chroma_db")
                logger.error(f"ChromaDB persist directory: {persist_dir}")
            logger.error("="*60)
            return 1
        
        # Get or create the metadata collection
        logger.info("Getting or creating doc_meta collection...")
        meta_collection = get_or_create_collection(chroma_client, "doc_meta")
        
        # Load metadata from S3
        logger.info("Loading metadata from S3...")
        metadata = load_metadata_from_s3(s3_client)
        
        if not metadata:
            logger.error("Failed to load metadata from S3")
            return 1
        
        logger.info(f"Loaded metadata for {len(metadata)} companies")
        
        # Extract document entries
        logger.info("Extracting document entries...")
        doc_entries = extract_metadata_for_embedding(metadata)
        
        if not doc_entries:
            logger.error("No document entries found in metadata")
            return 1
        
        # Populate the collection
        logger.info("Populating meta collection...")
        result_ids = populate_meta_collection(doc_entries, meta_collection)
        
        # Log summary
        logger.info(f"Successfully populated doc_meta collection with {len(result_ids)} documents")
        
        # Get final collection size
        try:
            collection_size = meta_collection.count()
            logger.info(f"Final collection size: {collection_size}")
        except Exception as e:
            logger.warning(f"Could not get collection size: {e}")
        
        logger.info("doc_meta collection population completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error during population: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)