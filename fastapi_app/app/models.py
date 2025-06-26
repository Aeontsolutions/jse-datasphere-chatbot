from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class ChatRequest(BaseModel):
    """
    Request model for chat endpoint
    """
    query: str = Field(..., description="User query/question")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None, 
        description="Previous conversation history as a list of role-content pairs"
    )
    auto_load_documents: bool = Field(
        default=True, 
        description="Whether to automatically load relevant documents"
    )
    memory_enabled: bool = Field(
        default=True, 
        description="Whether to use conversation memory"
    )

class DocumentInfo(BaseModel):
    """
    Information about a document selected by the LLM
    """
    company: str = Field(..., description="Company name")
    document_link: str = Field(..., description="S3 URL of the document")
    filename: str = Field(..., description="Filename of the document")
    reason: Optional[str] = Field(None, description="Reason why this document was selected")

class DocumentSelectionResponse(BaseModel):
    """
    Response from the semantic document selection
    """
    companies_mentioned: List[str] = Field(..., description="Companies mentioned in the query")
    documents_to_load: List[DocumentInfo] = Field(..., description="Documents to load")

class ChatResponse(BaseModel):
    """
    Response model for chat endpoint
    """
    response: str = Field(..., description="AI response to the query")
    documents_loaded: Optional[List[str]] = Field(
        default=None, 
        description="List of documents that were loaded to answer the query"
    )
    document_selection_message: Optional[str] = Field(
        default=None, 
        description="Message about document selection process"
    )
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None, 
        description="Updated conversation history"
    )

class ChromaAddRequest(BaseModel):
    """Request model for adding documents to ChromaDB"""
    documents: List[str] = Field(..., description="Raw document texts to embed and store")
    metadatas: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional metadata dictionaries for each document (must align 1-to-1)",
    )
    ids: Optional[List[str]] = Field(
        default=None,
        description="Optional custom IDs to assign to the documents; if omitted they are auto-generated",
    )

class ChromaAddResponse(BaseModel):
    """Response after adding documents to ChromaDB"""
    status: str = Field(..., description="Operation status message")
    ids: List[str] = Field(..., description="IDs of the documents that were added")

class ChromaQueryRequest(BaseModel):
    """Request model for querying ChromaDB"""
    query: str = Field(..., description="Search query or user question")
    n_results: int = Field(5, description="Number of most similar documents to return")
    where: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filter for the query (Chroma `where` syntax)",
    )

class ChromaQueryResponse(BaseModel):
    """Response model for ChromaDB query results"""
    ids: List[str] = Field(..., description="IDs of the retrieved documents")
    documents: List[str] = Field(..., description="Content of the retrieved documents")
    metadatas: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Metadata associated with the retrieved documents",
    )

class MetaDocumentInfo(BaseModel):
    """Information about a document in the metadata collection"""
    filename: str = Field(..., description="Document filename")
    company: str = Field(..., description="Company name")
    period: str = Field(..., description="Reporting period")
    type: str = Field(..., description="Document type (e.g., financial, non-financial)")
    description: str = Field(..., description="Document description for embedding")

class ChromaMetaUpdateRequest(BaseModel):
    """Request model for updating the metadata collection"""
    documents: List[MetaDocumentInfo] = Field(..., description="Metadata documents to add/update")

class ChromaMetaUpdateResponse(BaseModel):
    """Response after updating the metadata collection"""
    status: str = Field(..., description="Operation status message")
    ids: List[str] = Field(..., description="IDs of the metadata documents that were added")

class ChromaMetaQueryRequest(BaseModel):
    """Request model for querying the metadata collection"""
    query: str = Field(..., description="Search query for document selection")
    n_results: int = Field(3, description="Number of most similar documents to return")
    where: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filter for the query",
    )

class ChromaMetaQueryResponse(BaseModel):
    """Response model for metadata collection query results"""
    ids: List[str] = Field(..., description="IDs of the retrieved document metadata")
    documents: List[str] = Field(..., description="Description text of the retrieved documents")
    metadatas: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Metadata associated with the retrieved documents",
    )
