import os
import json
import logging
import tempfile
import hashlib
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

import boto3
import PyPDF2
from dotenv import load_dotenv
from google.oauth2 import service_account
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel
import google.generativeai as genai
from app.chroma_utils import query_meta_collection, get_companies_from_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# Initialize AWS S3 client
def init_s3_client():
    """Initialize and return an S3 client using environment variables"""
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION")
    
    if not all([aws_access_key_id, aws_secret_access_key, aws_region]):
        logger.error("AWS credentials not found in environment variables")
        raise ValueError("AWS credentials not found in environment variables")
    
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        logger.info("Successfully connected to AWS S3")
        return s3_client
    except Exception as e:
        logger.error(f"Error connecting to AWS S3: {str(e)}")
        raise

# Initialize Vertex AI
def init_vertex_ai():
    """Initialize Vertex AI using service account credentials"""
    try:
        # Get the service account info from environment
        service_account_info_str = os.getenv("GCP_SERVICE_ACCOUNT_INFO")
        if not service_account_info_str:
            logger.error("GCP_SERVICE_ACCOUNT_INFO not found in environment variables")
            raise ValueError("GCP_SERVICE_ACCOUNT_INFO not found in environment variables")
        
        service_account_info = json.loads(service_account_info_str)
        
        # Create credentials object from service account info
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        
        # Extract project ID from service account info
        project_id = service_account_info.get("project_id")
        if not project_id:
            logger.error("Could not find project_id in service account info")
            raise ValueError("Could not find project_id in service account info")
        
        # Initialize Vertex AI with project details and credentials
        aiplatform.init(
            project=project_id,
            location="us-central1",  # You may need to change this to your preferred region
            credentials=credentials
        )
        
        logger.info(f"Successfully initialized Vertex AI with project: {project_id}")
        return True
    except Exception as e:
        logger.error(f"Error setting up Vertex AI: {str(e)}")
        raise

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            text += page_text + "\n\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return None

# Function to download PDF from S3 and extract text
def download_and_extract_from_s3(s3_client, s3_path):
    """Download a PDF from S3 and extract its text"""
    try:
        # Parse S3 path to get bucket and key
        # s3://jse-renamed-docs/organized/... format
        if not s3_path.startswith("s3://"):
            logger.error(f"Invalid S3 path format: {s3_path}")
            return None
        
        path_without_prefix = s3_path[5:]  # Remove "s3://"
        bucket_name = path_without_prefix.split('/')[0]
        key = '/'.join(path_without_prefix.split('/')[1:])
        
        # Log the attempt
        logger.info(f"Attempting to download S3 object: Bucket='{bucket_name}', Key='{key}' from Path='{s3_path}'")
        
        # Create a temporary file to store the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Download the file from S3
            s3_client.download_fileobj(bucket_name, key, tmp_file)
            tmp_file_path = tmp_file.name
        
        # Extract text from the downloaded PDF
        with open(tmp_file_path, 'rb') as pdf_file:
            text = extract_text_from_pdf(pdf_file)
        
        # Clean up the temporary file
        os.unlink(tmp_file_path)
        
        return text
    except Exception as e:
        # Log the error with details
        logger.error(f"Error downloading/processing PDF from S3 Path='{s3_path}'. Bucket='{bucket_name}', Key='{key}'. Error: {str(e)}")
        return None

# Function to download metadata from S3
def download_metadata_from_s3(s3_client, bucket_name, key="metadata.json"):
    """Download metadata JSON file from S3"""
    try:
        # Download the metadata file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        metadata_content = response['Body'].read().decode('utf-8')
        return metadata_content
    except Exception as e:
        logger.error(f"Error downloading metadata from S3: {str(e)}")
        return None

# Function to parse metadata file
def parse_metadata_file(metadata_content):
    """Parse metadata JSON content"""
    try:
        # Parse the metadata JSON
        metadata = json.loads(metadata_content)
        return metadata
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing metadata file: {str(e)}")
        return None

# Function to load metadata from S3 bucket specified in .env
def load_metadata_from_s3(s3_client):
    """Load metadata from S3 bucket specified in environment variables"""
    try:
        bucket_name = os.getenv("DOCUMENT_METADATA_S3_BUCKET")
        if not bucket_name:
            logger.error("DOCUMENT_METADATA_S3_BUCKET not found in environment variables")
            return None
        
        # Default key for the metadata file
        metadata_key = "metadata.json"
        
        # Download metadata from S3
        metadata_content = download_metadata_from_s3(s3_client, bucket_name, metadata_key)
        if not metadata_content:
            return None
            
        # Parse the downloaded metadata
        return parse_metadata_file(metadata_content)
    except Exception as e:
        logger.error(f"Error loading metadata from S3: {str(e)}")
        return None

# Global variable to store the cached context
_metadata_cache = None
_cache_expiry = None
_cache_hash = None

def init_genai():
    """Initialize Google GenerativeAI client"""
    api_key = os.getenv("SUMMARIZER_API_KEY")
    if not api_key:
        logger.error("SUMMARIZER_API_KEY not found in environment variables")
        raise ValueError("SUMMARIZER_API_KEY not found for Google GenerativeAI")
    
    genai.configure(api_key=api_key)
    logger.info("Google GenerativeAI client initialized")

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
        if (_metadata_cache and _cache_expiry and 
            datetime.now() < _cache_expiry and 
            _cache_hash == current_hash):
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
            model='gemini-2.0-flash-001',
            config=genai.types.CreateCachedContentConfig(
                system_instruction=cached_content,
                display_name='document-metadata-cache',
                ttl_seconds=3600,  # 1-hour TTL
            ),
        )
        
        # Store cache info globally
        _metadata_cache = cache
        _cache_expiry = datetime.now() + timedelta(seconds=3500)  # Expire 5 minutes before actual TTL
        _cache_hash = current_hash
        
        logger.info(f"Created metadata cache with name: {cache.name}")
        return cache
        
    except Exception as e:
        logger.warning(f"Failed to create metadata cache: {str(e)}")
        return None

def get_cached_model(metadata):
    """Get a model instance that uses the cached metadata context"""
    try:
        cache = create_metadata_cache(metadata)
        if cache:
            # Create model that uses the cached context
            model = genai.GenerativeModel(
                model_name='gemini-2.0-flash-001',
                cached_content=cache
            )
            logger.info("Created model with cached metadata context")
            return model, True
        else:
            logger.info("Cache creation failed, falling back to regular model")
            return None, False
    except Exception as e:
        logger.warning(f"Failed to get cached model: {str(e)}")
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
        logger.error(f"Error refreshing metadata cache: {str(e)}")
        return False

def get_cache_status():
    """Get current cache status for monitoring/debugging"""
    global _metadata_cache, _cache_expiry, _cache_hash
    
    if not _metadata_cache:
        return {"status": "no_cache", "cache_name": None, "expires_at": None, "hash": None}
    
    is_expired = _cache_expiry and datetime.now() >= _cache_expiry
    
    return {
        "status": "expired" if is_expired else "active",
        "cache_name": getattr(_metadata_cache, 'name', 'unknown'),
        "expires_at": _cache_expiry.isoformat() if _cache_expiry else None,
        "hash": _cache_hash
    }

# Function to use embedding-based document selection with LLM fallback
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
    # Optional override: skip embedding search entirely (test LLM fallback)
    # ------------------------------------------------------------------
    if os.getenv("FORCE_LLM_FALLBACK", "false").lower() == "true":
        logger.info("FORCE_LLM_FALLBACK is set – bypassing embedding-based selection")
        return semantic_document_selection_llm_fallback(query, metadata, conversation_history)

    # Try embedding-based approach first if meta_collection is available
    if meta_collection is not None:
        try:
            logger.info("Attempting embedding-based document selection")
            
            # Extract companies from the query
            companies_from_query = get_companies_from_query(query)
            logger.info(f"Extracted companies from query: {companies_from_query}")
            
            all_documents_to_load = []
            all_companies_mentioned = set()
            
            if companies_from_query:
                # Run individual searches for each detected company
                for company in companies_from_query:
                    logger.info(f"Searching for documents from company: {company}")
                    company_result = query_meta_collection(
                        meta_collection=meta_collection,
                        query=query,
                        n_results=3,  # Get top 3 documents per company
                        where={"company": {"$eq": company}},
                        conversation_history=conversation_history
                    )
                    
                    if company_result and company_result.get("documents_to_load"):
                        all_documents_to_load.extend(company_result["documents_to_load"])
                        all_companies_mentioned.update(company_result.get("companies_mentioned", []))
                        logger.info(f"Found {len(company_result['documents_to_load'])} documents for {company}")
                
                # Deduplicate documents by filename
                seen_filenames = set()
                deduplicated_documents = []
                for doc in all_documents_to_load:
                    if doc["filename"] not in seen_filenames:
                        deduplicated_documents.append(doc)
                        seen_filenames.add(doc["filename"])
                
                if deduplicated_documents:
                    result = {
                        "companies_mentioned": list(all_companies_mentioned),
                        "documents_to_load": deduplicated_documents
                    }
                    logger.info(f"Multi-company embedding-based selection found {len(deduplicated_documents)} documents from {len(all_companies_mentioned)} companies")
                    return result
                else:
                    logger.info("Multi-company search returned no results, trying broader search")
            
            # Fallback to broader search if no companies detected or no results
            logger.info("Running broader embedding search (no company-specific filtering)")
            result = query_meta_collection(
                meta_collection=meta_collection,
                query=query,
                n_results=15,  # Increased for broader search
                conversation_history=conversation_history
            )
            
            if result and result.get("documents_to_load"):
                logger.info(f"Broader embedding-based selection found {len(result['documents_to_load'])} documents")
                return result
            else:
                logger.info("Embedding-based selection returned no results, falling back to LLM")
        except Exception as e:
            logger.warning(f"Embedding-based selection failed: {str(e)}, falling back to LLM")
    else:
        logger.info("No meta_collection provided, using LLM approach")
    
    # Fallback to LLM-based approach
    logger.info("Using LLM fallback for document selection")
    return semantic_document_selection_llm_fallback(query, metadata, conversation_history)

# Function to use the LLM to determine which documents to load based on the query and conversation history (FALLBACK)
def semantic_document_selection_llm_fallback(query, metadata, conversation_history=None):
    """
    Use LLM to determine which documents to load based on query and conversation history (fallback method).
    Now optimized with Google Gemini context caching to reduce latency from ~20s to ~2-3s.
    """
    try:
        # Try to get cached model first for faster performance
        cached_model, using_cache = get_cached_model(metadata)
        
        if using_cache and cached_model:
            logger.info("Using cached metadata context for LLM fallback (~2-3s expected)")
            model = cached_model
            
            # Format conversation history if available
            conversation_context = ""
            if conversation_history and len(conversation_history) > 0:
                # Get the last few exchanges to provide context (limit to last 10 exchanges to keep it focused)
                recent_history = conversation_history[-10:]
                conversation_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
                conversation_context = f"""
                Previous conversation history:
                {conversation_context}
                """
            
            # Simplified prompt since metadata is already cached in the model context
            prompt = f"""
            {conversation_context}
            
            Current user question: "{query}"
            
            Based on this question and conversation history, select the most relevant documents from the available metadata.
            
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
        else:
            # Fallback to traditional approach if caching fails
            logger.info("Cache unavailable, using traditional LLM fallback (~20s expected)")
            model = GenerativeModel("gemini-2.0-flash-001")
            
            # Format metadata in a readable format for the LLM
            metadata_str = json.dumps(metadata, indent=2)
            
            # Format conversation history if available
            conversation_context = ""
            if conversation_history and len(conversation_history) > 0:
                # Get the last few exchanges to provide context (limit to last 10 exchanges to keep it focused)
                recent_history = conversation_history[-10:]
                conversation_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
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
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                response_text = response_text[json_start:json_end]
            
            recommendation = json.loads(response_text)
            
            if using_cache:
                logger.info("Successfully completed LLM fallback with cached context")
            else:
                logger.info("Successfully completed LLM fallback with traditional approach")
                
            return recommendation
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response as JSON: {e}")
            logger.debug(response_text)  # Log the response for debugging
            return None
            
    except Exception as e:
        logger.error(f"Error in semantic document selection: {str(e)}")
        return None

# Function to load relevant documents based on the query and conversation history
def auto_load_relevant_documents(s3_client, query, metadata, current_document_texts, conversation_history=None):
    """Load relevant documents based on query and conversation history"""
    document_texts = current_document_texts.copy() if current_document_texts else {}
    loaded_docs = []
    
    # Use LLM to determine which documents to load, including conversation history
    recommendation = semantic_document_selection(query, metadata, conversation_history)
    
    if recommendation and "documents_to_load" in recommendation:
        docs_to_load = recommendation["documents_to_load"]
        companies_mentioned = recommendation.get("companies_mentioned", [])
        
        # Load the recommended documents (limited to 3)
        for doc_info in docs_to_load[:3]:
            doc_link = doc_info["document_link"]
            doc_name = doc_info["filename"]
            reason = doc_info.get("reason", "")
            
            # Only load if not already loaded
            if doc_name not in document_texts:
                text = download_and_extract_from_s3(s3_client, doc_link)
                if text:
                    document_texts[doc_name] = text
                    loaded_docs.append(doc_name)
                else:
                    logger.error(f"Failed to load document: {doc_name}")
        
        # Generate message about what was loaded
        if loaded_docs:
            message = f"Semantically selected {len(loaded_docs)} documents based on your query:\n"
            for doc_name in loaded_docs:
                matching_doc = next((d for d in docs_to_load if d["filename"] == doc_name), None)
                if matching_doc:
                    message += f"• {doc_name} - {matching_doc.get('reason', '')}\n"
            return document_texts, message, loaded_docs
        else:
            return document_texts, "No documents were loaded. Please check S3 access permissions or document availability.", []
    
    return document_texts, "No relevant documents were identified for your query.", []

# Function to generate chat response using Gemini
def generate_chat_response(query, document_texts, conversation_history=None, auto_load_message=None):
    """Generate chat response using Gemini model"""
    try:
        model = GenerativeModel("gemini-2.5-pro")
        
        # Format conversation history if available
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            # Get the last few exchanges to provide context
            recent_history = conversation_history[-20:] # Keep context length reasonable
            conversation_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
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
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise
