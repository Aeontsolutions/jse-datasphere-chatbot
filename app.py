import streamlit as st
import os
import requests
import boto3
from dotenv import load_dotenv
import json
from google.oauth2 import service_account
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel, Part
import tempfile
import PyPDF2
from io import BytesIO
import time
import logging
from utils.query_cache import qa_workflow, answer_found

st.set_page_config(page_title="JSE Document Chat", page_icon=":material/chat:", layout="wide")

# Initialize session state variables (moved to top)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "document_texts" not in st.session_state:
    st.session_state.document_texts = {}
if "document_context" not in st.session_state:
    st.session_state.document_context = ""
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
# Add state for feedback comments
if "pending_feedback_comment" not in st.session_state:
    st.session_state.pending_feedback_comment = None # Stores index of message needing comment
if "current_feedback_value" not in st.session_state:
    st.session_state.current_feedback_value = None # Stores the 0 or 1 from the initial click

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get AWS credentials from environment variables
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION")
metadata_s3_bucket = os.getenv("DOCUMENT_METADATA_S3_BUCKET")

# Validate AWS credentials
if not all([aws_access_key_id, aws_secret_access_key, aws_region]):
    st.error("AWS credentials not found in environment variables. Please add them to your .env file.")
    st.info("Required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION")
    st.stop()

if not metadata_s3_bucket:
    st.error("DOCUMENT_METADATA_S3_BUCKET not found in environment variables. Please add it to your .env file.")
    st.info("Required: DOCUMENT_METADATA_S3_BUCKET for metadata file storage.")
    st.stop()

# Initialize S3 client
try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    st.success("Successfully connected to AWS S3")
except Exception as e:
    st.error(f"Error connecting to AWS S3: {str(e)}")
    st.stop()

# Get service account file path from environment variables
service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not service_account_path:
    st.error("GOOGLE_APPLICATION_CREDENTIALS not found in environment variables. Please add it to your .env file.")
    st.stop()

# Check if the service account file exists
if not os.path.exists(service_account_path):
    st.error(f"Service account file not found at: {service_account_path}")
    st.stop()

try:
    # Load service account info to get project details
    with open(service_account_path, 'r') as f:
        service_account_info = json.load(f)
    
    # Extract project ID from service account
    project_id = service_account_info.get("project_id")
    if not project_id:
        st.error("Could not find project_id in service account file")
        st.stop()
    
    # Create credentials object from service account file
    credentials = service_account.Credentials.from_service_account_file(
        service_account_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    
    # Set the environment variable for other Google libraries
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
    
    # Initialize Vertex AI with project details and credentials
    aiplatform.init(
        project=project_id,
        location="us-central1",  # You may need to change this to your preferred region
        credentials=credentials
    )
    
    st.success(f"Successfully initialized Vertex AI with project: {project_id}")
    
except Exception as e:
    st.error(f"Error setting up Vertex AI: {str(e)}")
    st.stop()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            text += page_text + "\n\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

# Function to download PDF from S3 and extract text
def download_and_extract_from_s3(s3_path):
    try:
        # Parse S3 path to get bucket and key
        # s3://jse-renamed-docs/organized/... format
        if not s3_path.startswith("s3://"):
            st.error(f"Invalid S3 path format: {s3_path}")
            return None
        
        path_without_prefix = s3_path[5:]  # Remove "s3://"
        bucket_name = path_without_prefix.split('/')[0]
        key = '/'.join(path_without_prefix.split('/')[1:])
        
        # Log the attempt
        logger.info(f"Attempting to download S3 object: Bucket='{bucket_name}', Key='{key}' from Path='{s3_path}'")
        
        with st.spinner(f"Downloading from S3: {key}..."):
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
        st.error(f"Error downloading/processing PDF from S3: {str(e)}")
        return None

# Function to download metadata from S3
def download_metadata_from_s3(bucket_name, key="metadata.json"):
    try:
        with st.spinner(f"Downloading metadata from S3: {bucket_name}/{key}..."):
            # Download the metadata file from S3
            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            metadata_content = response['Body'].read().decode('utf-8')
            return metadata_content
    except Exception as e:
        st.error(f"Error downloading metadata from S3: {str(e)}")
        return None

# Function to parse metadata file
def parse_metadata_file(metadata_content):
    try:
        # Parse the metadata JSON
        metadata = json.loads(metadata_content)
        return metadata
    except json.JSONDecodeError as e:
        st.error(f"Error parsing metadata file: {str(e)}")
        st.info("Please check the format of your metadata file. It should be valid JSON.")
        return None

# Function to use the LLM to determine which documents to load based on the query and conversation history
def semantic_document_selection(query, metadata, conversation_history=None):
    try:
        # Create a prompt for the LLM to analyze the query and metadata
        model = GenerativeModel("gemini-2.0-flash-001")
        
        # Format metadata in a readable format for the LLM
        metadata_str = json.dumps(metadata, indent=2)
        
        # Format conversation history if available
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            # Get the last few exchanges to provide context (limit to last 5 exchanges to keep it focused)
            recent_history = conversation_history[-10:]
            conversation_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
            conversation_context = f"""
            Previous conversation history:
            {conversation_context}
            """
        
        # Prompt the LLM to determine relevant documents (LIMIT TO 3)
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
            return recommendation
        except json.JSONDecodeError as e:
            st.error(f"Error parsing LLM response as JSON: {e}")
            st.code(response_text)  # Show the response for debugging
            return None
            
    except Exception as e:
        st.error(f"Error in semantic document selection: {str(e)}")
        return None

# Function to load relevant documents based on the query and conversation history
def auto_load_relevant_documents(query, metadata, current_document_texts):
    document_texts = current_document_texts.copy()
    loaded_docs = []
    
    # Get conversation history if available
    conversation_history = []
    if "conversation_history" in st.session_state and st.session_state.conversation_history:
        conversation_history = st.session_state.conversation_history
    
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
                with st.spinner(f"Loading document: {doc_name}..."):
                    text = download_and_extract_from_s3(doc_link)
                    if text:
                        document_texts[doc_name] = text
                        loaded_docs.append(doc_name)
                    else:
                        st.error(f"Failed to load document: {doc_name}")
        
        # Generate message about what was loaded
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

# Function to load metadata from S3 bucket specified in .env
def load_metadata_from_s3():
    try:
        bucket_name = os.getenv("DOCUMENT_METADATA_S3_BUCKET")
        if not bucket_name:
            st.error("DOCUMENT_METADATA_S3_BUCKET not found in environment variables. Please add it to your .env file.")
            return None
        
        # Default key for the metadata file
        metadata_key = "metadata.json"
        
        # Download metadata from S3
        metadata_content = download_metadata_from_s3(bucket_name, metadata_key)
        if not metadata_content:
            return None
            
        # Parse the downloaded metadata
        return parse_metadata_file(metadata_content)
    except Exception as e:
        st.error(f"Error loading metadata from S3: {str(e)}")
        return None

# Streamlit UI
st.title("PDF Document Chat with Gemini")

# Add document reference toggle in the main area
auto_load_documents = st.checkbox("Use semantic document selection", value=True,
                            help="When enabled, the AI will analyze your question and intelligently select up to 3 most relevant documents")

# Metadata configuration section
st.header("Metadata Configuration")
metadata_option = st.radio("Select metadata source:", ["Use metadata from S3", "Upload metadata file"])

# Use session state for metadata
# metadata = None # Remove local variable initialization
if metadata_option == "Use metadata from S3":
    # Load metadata from the S3 bucket specified in .env only if not already loaded
    if st.session_state.metadata is None:
        st.session_state.metadata = load_metadata_from_s3()

    # Display status based on session state metadata
    if st.session_state.metadata:
        st.success(f"Metadata loaded from S3 bucket 'jse-metadata-bucket': {len(st.session_state.metadata)} companies found.")
    else:
        st.warning(f"Failed to load metadata from S3 bucket 'jse-metadata-bucket'. Please check your AWS credentials and bucket name.")
else:
    # Allow user to upload a metadata file
    uploaded_metadata = st.file_uploader("Upload metadata JSON file", type="json")
    if uploaded_metadata:
        # Check if this is a new upload to avoid reprocessing
        # (Simple check based on file uploader state, might need more robust logic if needed)
        if st.session_state.get("last_uploaded_metadata_name") != uploaded_metadata.name:
            metadata_content = uploaded_metadata.read().decode('utf-8')
            # Store parsed metadata in session state
            st.session_state.metadata = parse_metadata_file(metadata_content)
            st.session_state.last_uploaded_metadata_name = uploaded_metadata.name # Track uploaded file

        if st.session_state.metadata:
            st.success(f"Metadata loaded: {len(st.session_state.metadata)} companies found.")
        else:
            # Clear tracker if parsing failed
             st.session_state.pop("last_uploaded_metadata_name", None)

# Sidebar for PDF upload and settings
with st.sidebar:
    st.header("Document Sources")
    
    # Option to choose document source
    doc_source = st.radio(
        "Select document source:",
        ["Upload PDFs", "Manual Document Selection", "Automatic Only"]
    )
    
    # Use session state for document_texts
    # document_texts = {} # Remove local variable initialization
    
    # Upload PDFs option
    if doc_source in ["Upload PDFs", "Manual Document Selection"]:
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
        
        # Process uploaded PDFs
        if uploaded_files:
            for file in uploaded_files:
                # Check if file is already processed and stored in session state
                if file.name not in st.session_state.document_texts:
                    with st.spinner(f"Processing {file.name}..."):
                        text = extract_text_from_pdf(BytesIO(file.getvalue()))
                        if text:
                            # Store extracted text in session state
                            st.session_state.document_texts[file.name] = text
                            st.success(f"Processed: {file.name}")
                        else:
                            st.error(f"Failed to process: {file.name}")
                else:
                     st.info(f"Already processed: {file.name}") # Inform user if already loaded
    
    # Manual document selection option
    if doc_source == "Manual Document Selection" and st.session_state.metadata:
        st.subheader("Manual Document Selection")
        
        # Get list of companies from session state metadata
        companies = list(st.session_state.metadata.keys())
        
        # Company selection dropdown
        selected_company = st.selectbox("Select Company:", companies)
        
        if selected_company:
            # Get documents for selected company from session state metadata
            company_docs = st.session_state.metadata[selected_company]
            
            # Create document selection checkboxes
            st.write("Select documents to load (max 3):")
            
            selected_docs = []
            for i, doc in enumerate(company_docs):
                doc_label = f"{doc['filename']} ({doc['document_type']}, {doc['period']})"
                if st.checkbox(doc_label, key=f"doc_{selected_company}_{i}"):
                    selected_docs.append(doc)
            
            # Load selected documents button
            if selected_docs and st.button("Load Selected Documents"):
                # Limit to 3 documents
                selected_docs = selected_docs[:3]
                for doc in selected_docs:
                    doc_url = doc["document_link"]
                    doc_name = doc["filename"]
                    
                    # Check if document is already loaded in session state
                    if doc_name not in st.session_state.document_texts:
                        with st.spinner(f"Downloading {doc_name}..."):
                            text = download_and_extract_from_s3(doc_url)
                            if text:
                                # Store downloaded text in session state
                                st.session_state.document_texts[doc_name] = text
                                st.success(f"Loaded: {doc_name}")
                            else:
                                st.error(f"Failed to load: {doc_name}")
                    else:
                        st.info(f"Already loaded: {doc_name}") # Inform user
    elif doc_source == "Manual Document Selection" and not st.session_state.metadata:
        st.error("Metadata file not found or contains invalid JSON")
        st.info("Please upload a metadata file or use the example metadata")
    
    # Show currently loaded documents from session state
    if st.session_state.document_texts:
        st.subheader("Currently Loaded Documents")
        for doc_name in st.session_state.document_texts.keys():
            st.info(f"✅ {doc_name}")
            
        # Add button to clear loaded documents from session state
        if st.button("Clear All Loaded Documents"):
            st.session_state.document_texts = {} # Clear session state dict
            st.session_state.document_context = ""
            st.rerun()
    else:
        if auto_load_documents:
            st.info("No documents loaded yet. Documents will be loaded automatically when you ask questions relevant to them.")
        else:
            st.warning("No documents loaded. Please upload or select documents, or enable auto-loading.")
        
    # Conversation memory settings
    st.header("Conversation Settings")
    memory_enabled = st.checkbox("Enable conversation memory", value=True, 
                              help="When enabled, the AI will remember previous questions and answers")
    
    if st.button("Clear Conversation Memory"):
        st.session_state.conversation_history = []
        st.success("Conversation memory cleared!")
        
    # Display conversation memory status
    if "conversation_history" in st.session_state and st.session_state.conversation_history:
        memory_turns = len(st.session_state.conversation_history) // 2
        st.info(f"Conversation memory: {memory_turns} turns")

# Update document context when new documents are loaded (using session state)
if st.session_state.document_texts and "document_context" in st.session_state:
    # Combine all document texts from session state with document names as headers
    combined_text = ""
    for doc_name, doc_text in st.session_state.document_texts.items():
        combined_text += f"Document: {doc_name}\\n{doc_text}\\n\\n"
    
    st.session_state.document_context = combined_text

# Display chat history
# Use enumerate to get index for feedback keys
for i, message in enumerate(st.session_state.messages):
    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        # Use code block for assistant to avoid markdown issues
        if role == "assistant":
            st.code(content)
            # Add feedback for historical assistant messages
            feedback_key = f"feedback_{i}"
            feedback = st.feedback(options="thumbs", key=feedback_key)
            
            # --- New Comment Form Logic (History) ---
            # Check if this message needs a comment
            if st.session_state.pending_feedback_comment == i:
                comment_key = f"comment_{i}"
                comment = st.text_area("Why this rating?", key=comment_key)
                submit_key = f"submit_{i}"
                if st.button("Submit Comment", key=submit_key):
                    original_feedback = st.session_state.current_feedback_value
                    print(f"Feedback for message {i}: Value={original_feedback}, Comment='{comment}'")
                    # Reset state to hide form
                    st.session_state.pending_feedback_comment = None
                    st.session_state.current_feedback_value = None
                    st.rerun()

            # Handle initial feedback click
            if feedback:
                # Store feedback value and index, then rerun to show comment box
                st.session_state.current_feedback_value = feedback
                st.session_state.pending_feedback_comment = i
                # print(f"Feedback for message {i}: {feedback}") # Original print removed
                st.rerun()
            # --- End Comment Form Logic --- 
        else:
            st.write(content)

# User input
user_input = st.chat_input("Ask me about the uploaded documents...")
if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Auto-load relevant documents if enabled (limited to 3)
    auto_load_message = ""
    if auto_load_documents and st.session_state.metadata:
        # Pass session state document_texts to the function
        st.session_state.document_texts, auto_load_message = auto_load_relevant_documents(
            user_input,
            st.session_state.metadata,
            st.session_state.document_texts # Pass session state dict
        )
        
        # If memory is disabled, we should mention this could affect document selection
        if not memory_enabled and "conversation_history" in st.session_state and len(st.session_state.conversation_history) > 0:
            auto_load_message += "\n\nNote: Conversation memory is disabled. Enabling it could improve document selection for follow-up questions."
        
        # Update document context with any newly loaded documents (reading from session state)
        if auto_load_message and "Semantically selected" in auto_load_message:
            combined_text = ""
            for doc_name, doc_text in st.session_state.document_texts.items():
                combined_text += f"Document: {doc_name}\\n{doc_text}\\n\\n"
            st.session_state.document_context = combined_text
            
    # Store message for later use
    document_recommendation = auto_load_message

    try:
        # First try the structured QA implementation
        logger.info("Attempting structured QA implementation...")
        structured_answer = None
        _answer_found = False  # Initialize to False
        try:
            if st.session_state.conversation_history:
                structured_answer = qa_workflow(user_input, st.session_state.conversation_history[-10:])
            else:
                structured_answer = qa_workflow(user_input)
                
            # Check if the answer was found only if structured_answer is not None
            if structured_answer:
                _answer_found = answer_found(structured_answer, user_input)
                logger.info(f"Answer found check result: {_answer_found}")
            else:
                 logger.info("Structured QA returned None, skipping answer_found check.")

        except Exception as e:
            logger.error(f"Error during structured QA or answer check: {str(e)}")
            # Ensure _answer_found remains False if an error occurs
            _answer_found = False 
            
        # Decide based on whether the answer was found
        if _answer_found:
            logger.info("Structured answer considered valid.")
            ai_message = structured_answer
        else:
            logger.info("Structured answer not found or invalid, falling back to dynamic QA implementation.")
            # Fallback logic starts here...
            model = GenerativeModel("gemini-2.0-flash-001")
        
            # Append the new user message to conversation history if memory is enabled
            if memory_enabled:
                st.session_state.conversation_history.append({"role": "user", "content": user_input})
                
                # Construct the conversation context from history (limited to maintain token limits)
                recent_history = st.session_state.conversation_history[-20:] # Keep context length reasonable
                conversation_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
            else:
                # If memory is off, just use the current input
                conversation_context = f"user: {user_input}"
            
            # Prepare prompt for fallback model
            if st.session_state.document_context:
                prompt = f"""
                The following are documents that have been uploaded:
                
                {st.session_state.document_context}
                
                Previous conversation:
                {conversation_context}
                
                Based on the above documents and our conversation history, please answer the following question:
                {user_input}
                
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
                {user_input}
                
                If the question relates to our previous conversation, use that context in your answer.
                
                IMPORTANT: Use plain text formatting for all financial data. Do not use special formatting for dollar amounts or numbers.
                """
                response = model.generate_content(prompt)
            
            ai_message = response.text

        # Add document recommendation to the AI response if relevant (outside the if/else block)
        if document_recommendation and "Semantically selected" in document_recommendation:
            # Avoid duplicating the message if it's already part of the fallback prompt
             if not (not _answer_found and auto_load_message and "Semantically selected" in auto_load_message):
                 ai_message += f"\n\n{document_recommendation}"
        
        # Add AI response to conversation history if memory is enabled
        if memory_enabled:
            # Ensure history is not excessively long (optional limit)
            # if len(st.session_state.conversation_history) > 50: 
            #     st.session_state.conversation_history = st.session_state.conversation_history[-50:]
            st.session_state.conversation_history.append({"role": "assistant", "content": ai_message})
        
        # Display AI message using code block to avoid formatting issues
        # Store the index before appending for the feedback key
        new_message_index = len(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": ai_message})
        # Display the new message and add feedback
        with st.chat_message("assistant"):
            st.code(ai_message)
            feedback_key = f"feedback_{new_message_index}"
            feedback = st.feedback(options="thumbs", key=feedback_key)

            # --- New Comment Form Logic (New Message) ---
            # Check if this new message needs a comment
            if st.session_state.pending_feedback_comment == new_message_index:
                comment_key = f"comment_{new_message_index}"
                comment = st.text_area("Why this rating?", key=comment_key)
                submit_key = f"submit_{new_message_index}"
                if st.button("Submit Comment", key=submit_key):
                    original_feedback = st.session_state.current_feedback_value
                    print(f"Feedback for new message {new_message_index}: Value={original_feedback}, Comment='{comment}'")
                    # Reset state to hide form
                    st.session_state.pending_feedback_comment = None
                    st.session_state.current_feedback_value = None
                    st.rerun()
            
            # Handle initial feedback click for the new message
            if feedback:
                 # Store feedback value and index, then rerun to show comment box
                 st.session_state.current_feedback_value = feedback
                 st.session_state.pending_feedback_comment = new_message_index
                 # print(f"Feedback for new message {new_message_index}: {feedback}") # Original print removed
                 st.rerun()
            # --- End Comment Form Logic ---
            
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        st.info("If you're seeing authentication errors, ensure your service account has the necessary permissions for Vertex AI.")

# Add a clear conversation button
if st.button("Clear Conversation"):
    st.session_state.messages = []
    st.session_state.conversation_history = []
    st.rerun()