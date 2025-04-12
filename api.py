from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import logging
from utils.query_cache import qa_workflow, answer_found
from google.oauth2 import service_account
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel
import json
import boto3
import tempfile
import PyPDF2
from io import BytesIO
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="JSE Document Chat API")

# Pydantic models for request/response
class ChatRequest(BaseModel):
    query: str
    conversation_history: Optional[List[Dict[str, str]]] = None
    memory_enabled: bool = True

class DocumentInfo(BaseModel):
    filename: str
    content: str
    reason: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    context_used: str
    answer_type: str  # "structured" or "fallback"
    documents_used: List[DocumentInfo]
    document_recommendation: Optional[str] = None

# Initialize AWS S3 client
try:
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION")
    metadata_s3_bucket = os.getenv("DOCUMENT_METADATA_S3_BUCKET")

    if not all([aws_access_key_id, aws_secret_access_key, aws_region, metadata_s3_bucket]):
        raise ValueError("Missing required AWS credentials or bucket name in environment variables")

    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    logger.info("Successfully connected to AWS S3")
except Exception as e:
    logger.error(f"Error connecting to AWS S3: {str(e)}")
    raise

# Initialize Vertex AI
try:
    service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not service_account_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not found in environment variables")

    with open(service_account_path, 'r') as f:
        service_account_info = json.load(f)
    
    project_id = service_account_info.get("project_id")
    if not project_id:
        raise ValueError("Could not find project_id in service account file")
    
    credentials = service_account.Credentials.from_service_account_file(
        service_account_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
    
    aiplatform.init(
        project=project_id,
        location="us-central1",
        credentials=credentials
    )
    
    logger.info(f"Successfully initialized Vertex AI with project: {project_id}")
    
except Exception as e:
    logger.error(f"Error setting up Vertex AI: {str(e)}")
    raise

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
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

def download_and_extract_from_s3(s3_path):
    """Download PDF from S3 and extract text."""
    try:
        if not s3_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path format: {s3_path}")
        
        path_without_prefix = s3_path[5:]  # Remove "s3://"
        bucket_name = path_without_prefix.split('/')[0]
        key = '/'.join(path_without_prefix.split('/')[1:])
        
        logger.info(f"Attempting to download S3 object: Bucket='{bucket_name}', Key='{key}'")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            s3_client.download_fileobj(bucket_name, key, tmp_file)
            tmp_file_path = tmp_file.name
        
        with open(tmp_file_path, 'rb') as pdf_file:
            text = extract_text_from_pdf(pdf_file)
        
        os.unlink(tmp_file_path)
        return text
    except Exception as e:
        logger.error(f"Error downloading/processing PDF from S3: {str(e)}")
        return None

def download_metadata_from_s3():
    """Download metadata from S3 bucket."""
    try:
        response = s3_client.get_object(Bucket=metadata_s3_bucket, Key="metadata.json")
        metadata_content = response['Body'].read().decode('utf-8')
        return json.loads(metadata_content)
    except Exception as e:
        logger.error(f"Error downloading metadata from S3: {str(e)}")
        return None

def semantic_document_selection(query, metadata, conversation_history=None):
    """Use LLM to determine which documents to load based on the query."""
    try:
        model = GenerativeModel("gemini-2.0-flash-001")
        
        metadata_str = json.dumps(metadata, indent=2)
        
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-10:]
            conversation_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
            conversation_context = f"""
            Previous conversation history:
            {conversation_context}
            """
        
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
        
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                response_text = response_text[json_start:json_end]
            
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response as JSON: {e}")
            return None
            
    except Exception as e:
        logger.error(f"Error in semantic document selection: {str(e)}")
        return None

def auto_load_relevant_documents(query, metadata, conversation_history=None):
    """Load relevant documents based on the query and conversation history."""
    document_texts = {}
    loaded_docs = []
    
    recommendation = semantic_document_selection(query, metadata, conversation_history)
    
    if recommendation and "documents_to_load" in recommendation:
        docs_to_load = recommendation["documents_to_load"]
        
        for doc_info in docs_to_load[:3]:
            doc_link = doc_info["document_link"]
            doc_name = doc_info["filename"]
            
            text = download_and_extract_from_s3(doc_link)
            if text:
                document_texts[doc_name] = text
                loaded_docs.append(doc_name)
        
        if loaded_docs:
            message = f"Semantically selected {len(loaded_docs)} documents based on your query:\n"
            for doc_name in loaded_docs:
                matching_doc = next((d for d in docs_to_load if d["filename"] == doc_name), None)
                if matching_doc:
                    message += f"• {doc_name} - {matching_doc.get('reason', '')}\n"
            return document_texts, message
        else:
            return document_texts, "No documents were loaded. Please check S3 access permissions or document availability."
    
    return document_texts, "No relevant documents were identified for your query."

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint that handles both structured and fallback QA approaches.
    """
    try:
        # Load metadata and relevant documents
        metadata = download_metadata_from_s3()
        if not metadata:
            raise HTTPException(status_code=500, detail="Failed to load document metadata")
        
        document_texts, document_recommendation = auto_load_relevant_documents(
            request.query,
            metadata,
            request.conversation_history
        )
        
        # Convert document_texts to DocumentInfo objects
        documents_used = []
        for doc_name, doc_text in document_texts.items():
            # Find the reason for this document from the recommendation
            reason = None
            if document_recommendation and "Semantically selected" in document_recommendation:
                for line in document_recommendation.split('\n'):
                    if doc_name in line and '-' in line:
                        reason = line.split('-', 1)[1].strip()
                        break
            
            documents_used.append(DocumentInfo(
                filename=doc_name,
                content=doc_text,
                reason=reason
            ))
        
        # First try the structured QA implementation
        logger.info("Attempting structured QA implementation...")
        structured_answer = None
        _answer_found = False
        
        try:
            if request.conversation_history:
                structured_answer = qa_workflow(request.query, request.conversation_history[-10:])
            else:
                structured_answer = qa_workflow(request.query)
                
            if structured_answer:
                _answer_found = answer_found(structured_answer, request.query)
                logger.info(f"Answer found check result: {_answer_found}")
            else:
                logger.info("Structured QA returned None, skipping answer_found check.")

        except Exception as e:
            logger.error(f"Error during structured QA or answer check: {str(e)}")
            _answer_found = False
            
        # Decide based on whether the answer was found
        if _answer_found:
            logger.info("Structured answer considered valid.")
            return ChatResponse(
                answer=structured_answer,
                context_used="Structured QA with document summaries",
                answer_type="structured",
                documents_used=documents_used,
                document_recommendation=document_recommendation
            )
        else:
            logger.info("Structured answer not found or invalid, falling back to dynamic QA implementation.")
            model = GenerativeModel("gemini-2.0-flash-001")
            
            # Construct conversation context
            conversation_context = ""
            if request.memory_enabled and request.conversation_history:
                recent_history = request.conversation_history[-20:]
                conversation_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
            else:
                conversation_context = f"user: {request.query}"
            
            # Prepare prompt for fallback model
            if documents_used:
                document_context = "\n\n".join([
                    f"Document: {doc.filename}\n{doc.content}"
                    for doc in documents_used
                ])
                
                prompt = f"""
                The following are documents that have been uploaded:
                
                {document_context}
                
                Previous conversation:
                {conversation_context}
                
                Based on the above documents and our conversation history, please answer the following question:
                {request.query}
                
                If the question relates to our previous conversation, use that context in your answer.
                
                IMPORTANT: Use plain text formatting for all financial data. Do not use special formatting for dollar amounts or numbers.
                """
            else:
                prompt = f"""
                Previous conversation:
                {conversation_context}
                
                Based on our conversation history, please answer the following question:
                {request.query}
                
                If the question relates to our previous conversation, use that context in your answer.
                
                IMPORTANT: Use plain text formatting for all financial data. Do not use special formatting for dollar amounts or numbers.
                """
            
            response = model.generate_content(prompt)
            ai_message = response.text
            
            return ChatResponse(
                answer=ai_message,
                context_used=document_context if documents_used else "Conversation history only",
                answer_type="fallback",
                documents_used=documents_used,
                document_recommendation=document_recommendation
            )
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint that provides basic API information."""
    return JSONResponse(
        content={
            "name": "JSE Document Chat API",
            "version": "1.0.0",
            "description": "API for querying and analyzing JSE documents using AI",
            "endpoints": {
                "/chat": {
                    "method": "POST",
                    "description": "Main chat endpoint for document queries",
                    "request_body": {
                        "query": "The question to answer",
                        "conversation_history": "Optional list of previous messages",
                        "memory_enabled": "Whether to use conversation history (default: true)"
                    }
                }
            },
            "documentation": "/docs"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 