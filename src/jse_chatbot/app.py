import streamlit as st
import logging
from typing import Dict, Any, Optional
from .config.settings import settings
from .core.document_processor import DocumentProcessor
from .models.ai_model import AIModel
from .ui.streamlit_ui import StreamlitUI
from utils.query_cache import qa_workflow, answer_found

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JSEChatbot:
    def __init__(self):
        self.ui = StreamlitUI()
        self.document_processor = DocumentProcessor()
        self.ai_model = AIModel()
        
    def initialize(self):
        """Initialize the application."""
        self.ui.setup_page()
        
        # Validate settings
        errors = settings.validate()
        if errors:
            for error in errors.values():
                st.error(error)
            st.stop()
            
        # Load metadata if not already loaded
        if st.session_state.metadata is None:
            self._load_metadata()
            
    def _load_metadata(self):
        """Load metadata from S3."""
        metadata_content = self.document_processor.download_metadata_from_s3(
            settings.metadata_s3_bucket
        )
        if metadata_content:
            st.session_state.metadata = self.document_processor.parse_metadata_file(
                metadata_content
            )
            if st.session_state.metadata:
                st.success(f"Metadata loaded: {len(st.session_state.metadata)} companies found.")
            else:
                st.error("Failed to parse metadata file.")
        else:
            st.error("Failed to load metadata from S3.")
            
    def _auto_load_relevant_documents(self, query: str) -> Optional[str]:
        """Automatically load relevant documents based on the query."""
        if not st.session_state.metadata:
            return None
            
        recommendation = self.ai_model.semantic_document_selection(
            query,
            st.session_state.metadata,
            st.session_state.conversation_history
        )
        
        if not recommendation or "documents_to_load" not in recommendation:
            return None
            
        loaded_docs = []
        for doc_info in recommendation["documents_to_load"][:settings.max_documents]:
            doc_name = doc_info["filename"]
            if doc_name not in st.session_state.document_texts:
                text = self.document_processor.download_and_extract_from_s3(
                    doc_info["document_link"]
                )
                if text:
                    st.session_state.document_texts[doc_name] = text
                    loaded_docs.append(doc_name)
                    
        if loaded_docs:
            message = f"Semantically selected {len(loaded_docs)} documents based on your query:\n"
            for doc_name in loaded_docs:
                matching_doc = next(
                    (d for d in recommendation["documents_to_load"] 
                     if d["filename"] == doc_name),
                    None
                )
                if matching_doc:
                    message += f"• {doc_name} - {matching_doc.get('reason', '')}\n"
            return message
            
        return None
        
    def _update_document_context(self):
        """Update the document context with all loaded documents."""
        if st.session_state.document_texts:
            combined_text = ""
            for doc_name, doc_text in st.session_state.document_texts.items():
                combined_text += f"Document: {doc_name}\n{doc_text}\n\n"
            st.session_state.document_context = combined_text
            
    def run(self):
        """Run the main application loop."""
        self.initialize()
        
        # Display the main UI
        st.title("PDF Document Chat with Gemini")
        
        # Add document reference toggle
        auto_load_documents = st.checkbox(
            "Use semantic document selection",
            value=True,
            help="When enabled, the AI will analyze your question and intelligently select up to 3 most relevant documents"
        )
        
        # Display sidebar
        self.ui.display_sidebar(self.document_processor, self.ai_model)
        
        # Update document context
        self._update_document_context()
        
        # Display chat history
        self.ui.display_chat_history()
        
        # Handle user input
        user_input = st.chat_input("Ask me about the uploaded documents...")
        if user_input:
            # Display user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)
            
            # Auto-load relevant documents if enabled
            document_recommendation = None
            if auto_load_documents:
                document_recommendation = self._auto_load_relevant_documents(user_input)
                if document_recommendation:
                    self._update_document_context()
                    
            try:
                # Try structured QA first
                structured_answer = None
                _answer_found = False
                
                if st.session_state.conversation_history:
                    structured_answer = qa_workflow(
                        user_input,
                        st.session_state.conversation_history[-10:]
                    )
                else:
                    structured_answer = qa_workflow(user_input)
                    
                if structured_answer:
                    _answer_found = answer_found(structured_answer, user_input)
                    
                # Generate response
                if _answer_found:
                    ai_message = structured_answer
                else:
                    ai_message = self.ai_model.generate_response(
                        user_input,
                        st.session_state.document_context,
                        st.session_state.conversation_history
                    )
                    
                # Add document recommendation if relevant
                if document_recommendation and not _answer_found:
                    ai_message += f"\n\n{document_recommendation}"
                    
                # Update conversation history
                st.session_state.conversation_history.append(
                    {"role": "user", "content": user_input}
                )
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": ai_message}
                )
                
                # Display AI response
                st.session_state.messages.append(
                    {"role": "assistant", "content": ai_message}
                )
                with st.chat_message("assistant"):
                    st.code(ai_message)
                    self.ui._handle_feedback(len(st.session_state.messages) - 1)
                    
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                st.error("An error occurred while generating the response.")
                
        # Add clear conversation button
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()
            
def main():
    app = JSEChatbot()
    app.run()
    
if __name__ == "__main__":
    main() 