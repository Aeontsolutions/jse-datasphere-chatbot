import logging
from typing import List, Dict, Any, Optional
from google.oauth2 import service_account
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel
from ..config.settings import settings

logger = logging.getLogger(__name__)

class AIModel:
    def __init__(self):
        self.model = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the Gemini model with proper credentials."""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                settings.google_credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            
            aiplatform.init(
                project=settings.google_project_id,
                location="us-central1",
                credentials=credentials
            )
            
            self.model = GenerativeModel("gemini-2.0-flash-001")
            logger.info(f"Successfully initialized Gemini model with project: {settings.google_project_id}")
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
            raise
            
    def semantic_document_selection(self, query: str, metadata: Dict[str, Any], 
                                  conversation_history: Optional[List[Dict[str, str]]] = None) -> Optional[Dict[str, Any]]:
        """Use the model to determine which documents are relevant to the query."""
        try:
            # Format metadata for the model
            metadata_str = self._format_metadata(metadata)
            
            # Format conversation history if available
            conversation_context = self._format_conversation_history(conversation_history)
            
            prompt = self._create_document_selection_prompt(query, metadata_str, conversation_context)
            
            response = self.model.generate_content(prompt)
            return self._parse_model_response(response.text)
        except Exception as e:
            logger.error(f"Error in semantic document selection: {str(e)}")
            return None
            
    def generate_response(self, query: str, document_context: str, 
                         conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate a response to the user's query using the model."""
        try:
            conversation_context = self._format_conversation_history(conversation_history)
            prompt = self._create_response_prompt(query, document_context, conversation_context)
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response."
            
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for the model's consumption."""
        import json
        return json.dumps(metadata, indent=2)
        
    def _format_conversation_history(self, history: Optional[List[Dict[str, str]]]) -> str:
        """Format conversation history for the model's consumption."""
        if not history:
            return ""
            
        recent_history = history[-settings.max_conversation_history:]
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
        
    def _create_document_selection_prompt(self, query: str, metadata_str: str, 
                                        conversation_context: str) -> str:
        """Create a prompt for document selection."""
        return f"""
        Based on the following user question, conversation history, and available document metadata, 
        determine which documents are most relevant to answer the question.
        
        {conversation_context}
        
        Current user question: "{query}"
        
        Available document metadata:
        {metadata_str}
        
        For each company mentioned or implied in the question or previous conversation, 
        select the most relevant documents. Consider aliases, abbreviations, or partial references.
        
        IMPORTANT: Select a maximum of {settings.max_documents} documents total.
        
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
        """
        
    def _create_response_prompt(self, query: str, document_context: str, 
                              conversation_context: str) -> str:
        """Create a prompt for generating a response."""
        return f"""
        The following are documents that have been uploaded:
        
        {document_context}
        
        Previous conversation:
        {conversation_context}
        
        Based on the above documents and our conversation history, please answer the following question:
        {query}
        
        If the question relates to our previous conversation, use that context in your answer.
        
        IMPORTANT: Use plain text formatting for all financial data. 
        Do not use special formatting for dollar amounts or numbers.
        """
        
    def _parse_model_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse the model's response into a structured format."""
        try:
            import json
            # Try to find JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                response_text = response_text[json_start:json_end]
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing model response as JSON: {e}")
            return None 