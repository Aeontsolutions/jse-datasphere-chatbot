import streamlit as st
from typing import Dict, Any, List, Optional
import logging
from ..config.settings import settings

logger = logging.getLogger(__name__)

class StreamlitUI:
    def __init__(self):
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
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
        if "pending_feedback_comment" not in st.session_state:
            st.session_state.pending_feedback_comment = None
        if "current_feedback_value" not in st.session_state:
            st.session_state.current_feedback_value = None
            
    def setup_page(self):
        """Set up the Streamlit page configuration."""
        st.set_page_config(
            page_title="JSE Document Chat",
            page_icon=":material/chat:",
            layout="wide"
        )
        
    def display_chat_history(self):
        """Display the chat history with feedback options."""
        for i, message in enumerate(st.session_state.messages):
            role = message["role"]
            content = message["content"]

            with st.chat_message(role):
                if role == "assistant":
                    st.code(content)
                    self._handle_feedback(i)
                else:
                    st.write(content)
                    
    def _handle_feedback(self, message_index: int):
        """Handle feedback for assistant messages."""
        feedback_key = f"feedback_{message_index}"
        feedback = st.feedback(options="thumbs", key=feedback_key)
        
        # Handle feedback comment form
        if st.session_state.pending_feedback_comment == message_index:
            self._display_feedback_comment_form(message_index)
            
        # Handle initial feedback click
        if feedback:
            st.session_state.current_feedback_value = feedback
            st.session_state.pending_feedback_comment = message_index
            
    def _display_feedback_comment_form(self, message_index: int):
        """Display the feedback comment form."""
        comment_key = f"comment_{message_index}"
        comment = st.text_area("Why this rating?", key=comment_key)
        submit_key = f"submit_{message_index}"
        
        if st.button("Submit Comment", key=submit_key):
            original_feedback = st.session_state.current_feedback_value
            logger.info(f"Feedback for message {message_index}: Value={original_feedback}, Comment='{comment}'")
            st.session_state.pending_feedback_comment = None
            st.session_state.current_feedback_value = None
            st.rerun()
            
    def display_sidebar(self, document_processor: Any, ai_model: Any):
        """Display the sidebar with document and conversation controls."""
        with st.sidebar:
            st.header("Document Sources")
            
            # Document source selection
            doc_source = st.radio(
                "Select document source:",
                ["Upload PDFs", "Manual Document Selection", "Automatic Only"]
            )
            
            # Document upload and selection
            if doc_source in ["Upload PDFs", "Manual Document Selection"]:
                self._handle_document_upload(document_processor)
                
            if doc_source == "Manual Document Selection" and st.session_state.metadata:
                self._handle_manual_document_selection(document_processor)
                
            # Display loaded documents
            self._display_loaded_documents()
            
            # Conversation settings
            self._display_conversation_settings()
            
            # Export conversation
            self._display_export_options()
            
    def _handle_document_upload(self, document_processor: Any):
        """Handle document upload functionality."""
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
        
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in st.session_state.document_texts:
                    with st.spinner(f"Processing {file.name}..."):
                        text = document_processor.extract_text_from_pdf(file)
                        if text:
                            st.session_state.document_texts[file.name] = text
                            st.success(f"Processed: {file.name}")
                        else:
                            st.error(f"Failed to process: {file.name}")
                else:
                    st.info(f"Already processed: {file.name}")
                    
    def _handle_manual_document_selection(self, document_processor: Any):
        """Handle manual document selection functionality."""
        st.subheader("Manual Document Selection")
        
        companies = list(st.session_state.metadata.keys())
        selected_company = st.selectbox("Select Company:", companies)
        
        if selected_company:
            company_docs = st.session_state.metadata[selected_company]
            st.write("Select documents to load (max 3):")
            
            selected_docs = []
            for i, doc in enumerate(company_docs):
                doc_label = f"{doc['filename']} ({doc['document_type']}, {doc['period']})"
                if st.checkbox(doc_label, key=f"doc_{selected_company}_{i}"):
                    selected_docs.append(doc)
                    
            if selected_docs and st.button("Load Selected Documents"):
                for doc in selected_docs[:settings.max_documents]:
                    if doc["filename"] not in st.session_state.document_texts:
                        with st.spinner(f"Downloading {doc['filename']}..."):
                            text = document_processor.download_and_extract_from_s3(doc["document_link"])
                            if text:
                                st.session_state.document_texts[doc["filename"]] = text
                                st.success(f"Loaded: {doc['filename']}")
                            else:
                                st.error(f"Failed to load: {doc['filename']}")
                    else:
                        st.info(f"Already loaded: {doc['filename']}")
                        
    def _display_loaded_documents(self):
        """Display currently loaded documents."""
        if st.session_state.document_texts:
            st.subheader("Currently Loaded Documents")
            for doc_name in st.session_state.document_texts.keys():
                st.info(f"✅ {doc_name}")
                
            if st.button("Clear All Loaded Documents"):
                st.session_state.document_texts = {}
                st.session_state.document_context = ""
                st.rerun()
        else:
            st.warning("No documents loaded. Please upload or select documents.")
            
    def _display_conversation_settings(self):
        """Display conversation settings."""
        st.header("Conversation Settings")
        memory_enabled = st.checkbox(
            "Enable conversation memory",
            value=True,
            help="When enabled, the AI will remember previous questions and answers"
        )
        
        if st.button("Clear Conversation Memory"):
            st.session_state.conversation_history = []
            st.success("Conversation memory cleared!")
            
        if st.session_state.conversation_history:
            memory_turns = len(st.session_state.conversation_history) // 2
            st.info(f"Conversation memory: {memory_turns} turns")
            
    def _display_export_options(self):
        """Display conversation export options."""
        st.header("Export")
        if st.session_state.messages:
            export_data = ""
            for message in st.session_state.messages:
                role = message["role"].capitalize()
                content = message["content"]
                export_data += f"{role}: {content}\n{'-'*20}\n"
                
            st.download_button(
                label="Export Conversation",
                data=export_data.encode('utf-8'),
                file_name="conversation_history.txt",
                mime="text/plain"
            )
        else:
            st.info("No conversation history to export yet.") 