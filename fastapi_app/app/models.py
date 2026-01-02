from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Request model for chat endpoint
    """

    query: str = Field(..., description="User query/question")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None, description="Previous conversation history as a list of role-content pairs"
    )
    auto_load_documents: bool = Field(
        default=True, description="Whether to automatically load relevant documents"
    )
    memory_enabled: bool = Field(default=True, description="Whether to use conversation memory")


class ProgressUpdate(BaseModel):
    """
    Model for streaming progress updates
    """

    step: str = Field(..., description="Current processing step")
    message: str = Field(..., description="Human-readable status message")
    progress: float = Field(..., description="Progress percentage (0-100)")
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp of the update")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional details about the step"
    )


class StreamingChatRequest(BaseModel):
    """
    Request model for streaming chat endpoint
    """

    query: str = Field(..., description="User query/question")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None, description="Previous conversation history as a list of role-content pairs"
    )
    auto_load_documents: bool = Field(
        default=True, description="Whether to automatically load relevant documents"
    )
    memory_enabled: bool = Field(default=True, description="Whether to use conversation memory")


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
        default=None, description="List of documents that were loaded to answer the query"
    )
    document_selection_message: Optional[str] = Field(
        default=None, description="Message about document selection process"
    )
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None, description="Updated conversation history"
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


class FinancialDataRequest(BaseModel):
    """
    Request model for financial data query endpoint
    """

    query: str = Field(..., description="Natural language query about financial data")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None, description="Previous conversation history as a list of role-content pairs"
    )
    memory_enabled: bool = Field(default=True, description="Whether to use conversation memory")


class FinancialDataFilters(BaseModel):
    """
    Parsed filters from natural language query
    """

    companies: List[str] = Field(default=[], description="Company names to filter by")
    symbols: List[str] = Field(default=[], description="Stock symbols to filter by")
    years: List[str] = Field(default=[], description="Years to filter by")
    standard_items: List[str] = Field(default=[], description="Financial metrics to filter by")
    interpretation: str = Field(
        default="", description="Human-readable interpretation of the query"
    )
    data_availability_note: str = Field(default="", description="Notes about data availability")
    is_follow_up: bool = Field(default=False, description="Whether this is a follow-up query")
    context_used: str = Field(default="", description="Context used from previous queries")


class FinancialDataRecord(BaseModel):
    """
    Single financial data record
    """

    company: str = Field(..., description="Company name")
    symbol: str = Field(..., description="Stock symbol")
    year: str = Field(..., description="Year")
    standard_item: str = Field(..., description="Financial metric name")
    item: Optional[float] = Field(None, description="Financial metric value, or null if missing")
    unit_multiplier: Optional[int] = Field(
        1, description="Unit multiplier (1, 1M, 1B), or null if missing"
    )
    formatted_value: str = Field(
        ..., description="Human-readable formatted value, 'N/A' if missing"
    )


class ChartSpec(BaseModel):
    """
    Vega-Lite chart specification for visualizing financial data
    """

    chart_type: str = Field(
        ..., description="Type of chart: line, bar, grouped_bar, horizontal_bar"
    )
    title: str = Field(..., description="Chart title")
    description: str = Field(..., description="Human-readable description of the chart")
    vega_lite: Dict[str, Any] = Field(..., description="Full Vega-Lite specification")


class FinancialDataResponse(BaseModel):
    """
    Response model for financial data query endpoint
    """

    response: str = Field(..., description="AI-generated natural language response")
    data_found: bool = Field(..., description="Whether any data was found")
    record_count: int = Field(..., description="Number of records found")
    filters_used: FinancialDataFilters = Field(..., description="Filters that were applied")
    data_preview: Optional[List[FinancialDataRecord]] = Field(
        default=None, description="Preview of the first few data records"
    )
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None, description="Updated conversation history"
    )
    warnings: Optional[List[str]] = Field(
        default=None, description="Any warnings about data availability"
    )
    suggestions: Optional[List[str]] = Field(
        default=None, description="Suggestions for alternative queries"
    )
    chart: Optional[ChartSpec] = Field(
        default=None, description="Vega-Lite chart specification if data is chartable"
    )


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    expired = "expired"


class JobCreateResponse(BaseModel):
    job_id: str = Field(..., description="Identifier for the newly created job")
    status: JobStatus = Field(..., description="Initial job status")
    job_type: str = Field(..., description="Logical job category")
    polling_url: str = Field(..., description="Relative URL to poll for job status")


class JobStatusResponse(BaseModel):
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    job_type: str = Field(..., description="Logical job category")
    created_at: str = Field(..., description="Creation timestamp in ISO format")
    updated_at: str = Field(..., description="Last update timestamp in ISO format")
    progress: List[ProgressUpdate] = Field(
        default_factory=list, description="Historical progress updates"
    )
    latest_progress: Optional[ProgressUpdate] = Field(
        default=None, description="Most recent progress update"
    )
    result: Optional[Dict[str, Any]] = Field(
        default=None, description="Final job payload if completed"
    )
    error: Optional[str] = Field(default=None, description="Error message if job failed")


# ==============================================================================
# AGENT CHAT MODELS
# ==============================================================================


class AgentChatRequest(BaseModel):
    """
    Request model for agent chat endpoint.
    Similar to FinancialDataRequest but with tool toggles.

    Backward compatible with StreamingChatRequest - accepts auto_load_documents
    which maps to enable_financial_data for seamless endpoint migration.
    """

    query: str = Field(..., description="Natural language query about financial data or companies")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None, description="Previous conversation history as a list of role-content pairs"
    )
    memory_enabled: bool = Field(default=True, description="Whether to use conversation memory")
    enable_web_search: bool = Field(
        default=True, description="Whether to enable Google Search for web grounding"
    )
    enable_financial_data: bool = Field(
        default=True, description="Whether to enable SQL financial data queries"
    )
    # Backward compatibility: StreamingChatRequest used auto_load_documents
    auto_load_documents: Optional[bool] = Field(
        default=None,
        description="Deprecated: Use enable_financial_data. Maps to enable_financial_data for backward compatibility.",
    )


class AgentChatResponse(BaseModel):
    """
    Response model for agent chat endpoint.
    BACKWARD COMPATIBLE with FinancialDataResponse - includes all existing fields
    plus new agent-specific fields.
    """

    # === EXISTING FIELDS (from FinancialDataResponse for backward compatibility) ===
    response: str = Field(..., description="AI-generated response with source citations")
    data_found: bool = Field(..., description="Whether any financial data was found")
    record_count: int = Field(..., description="Number of financial records found")
    filters_used: Optional[FinancialDataFilters] = Field(
        default=None, description="Filters that were applied to SQL query"
    )
    data_preview: Optional[List[FinancialDataRecord]] = Field(
        default=None, description="Preview of financial data records"
    )
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None, description="Updated conversation history"
    )
    warnings: Optional[List[str]] = Field(
        default=None, description="Any warnings about data availability"
    )
    suggestions: Optional[List[str]] = Field(
        default=None, description="Suggested follow-up questions"
    )
    chart: Optional[ChartSpec] = Field(
        default=None, description="Vega-Lite chart specification if data is chartable"
    )

    # === NEW FIELDS (agent-specific, additive only) ===
    sources: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of sources cited in the response (values may be strings or lists)",
    )
    web_search_results: Optional[Dict[str, Any]] = Field(
        default=None, description="Google Search grounding metadata"
    )
    tools_executed: Optional[List[str]] = Field(
        default=None, description="List of tools that were executed"
    )

    # === CLARIFICATION FIELDS ===
    needs_clarification: bool = Field(
        default=False,
        description="Whether the system needs user clarification before proceeding",
    )
    clarification_question: Optional[str] = Field(
        default=None, description="Question to ask user if clarification is needed"
    )

    # === INTERNAL/DEBUG FIELDS ===
    cost_summary: Optional["CostSummary"] = Field(
        default=None,
        description="Internal: Cost tracking summary for this request (developer/ops monitoring)",
    )


# ==============================================================================
# COST TRACKING MODELS (Internal/Debug)
# ==============================================================================


class PhaseCost(BaseModel):
    """Cost breakdown for a single agent phase."""

    phase: str = Field(..., description="Agent phase name (e.g., 'classification', 'synthesis')")
    model: str = Field(..., description="Model used for this phase")
    input_tokens: int = Field(default=0, description="Number of input tokens")
    output_tokens: int = Field(default=0, description="Number of output tokens")
    cached_tokens: int = Field(default=0, description="Number of cached tokens")
    input_cost_usd: float = Field(default=0.0, description="Cost for input tokens in USD")
    output_cost_usd: float = Field(default=0.0, description="Cost for output tokens in USD")
    total_cost_usd: float = Field(default=0.0, description="Total cost for this phase in USD")


class CostSummary(BaseModel):
    """
    Cost summary for an agent request.

    Internal/debug field - not part of the public API contract.
    Used for developer monitoring and UAT testing.
    """

    total_input_tokens: int = Field(default=0, description="Total input tokens across all phases")
    total_output_tokens: int = Field(default=0, description="Total output tokens across all phases")
    total_cached_tokens: int = Field(default=0, description="Total cached tokens across all phases")
    total_cost_usd: float = Field(default=0.0, description="Total cost in USD across all phases")
    phases: List[PhaseCost] = Field(default_factory=list, description="Cost breakdown by phase")


# ==============================================================================
# PROMPT OPTIMIZATION MODELS
# ==============================================================================


class ClarificationReason(str, Enum):
    """Reasons why clarification might be needed."""

    NO_ENTITY = "no_entity"  # No company/symbol identified
    AMBIGUOUS_COMPARISON = "ambiguous_comparison"  # "compare banks" without specifics
    UNRESOLVED_PRONOUN = "unresolved_pronoun"  # "they" with no context


class PromptOptimizationResult(BaseModel):
    """
    Result of prompt optimization with context resolution and clarification detection.
    """

    optimized_query: str = Field(..., description="The query with resolved references")
    needs_clarification: bool = Field(
        default=False, description="Whether clarification is needed before proceeding"
    )
    clarification_question: Optional[str] = Field(
        default=None, description="Question to ask user if clarification is needed"
    )
    clarification_reason: Optional[ClarificationReason] = Field(
        default=None, description="Why clarification is needed"
    )
    resolved_context: Dict[str, Any] = Field(
        default_factory=dict, description="Context resolved from conversation history"
    )
    defaults_applied: List[str] = Field(
        default_factory=list, description="List of defaults that were applied"
    )
    confidence: str = Field(default="high", description="Confidence level: high, medium, low")
    llm_routing: Optional[Dict[str, Any]] = Field(
        default=None,
        description="LLM-determined tool routing (use_financial, use_web_search, etc.)",
    )
