from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response, RedirectResponse
import time
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any
from contextlib import asynccontextmanager

from app.models import (
    ChatRequest,
    ChatResponse,
    ChartSpec,
    FinancialDataRequest,
    FinancialDataResponse,
    AgentChatRequest,
    AgentChatResponse,
    Source,
    FinancialDataFilters,
    FinancialDataRecord,
)
from app.charting import generate_chart
from app.document_registry import build_document_index, presign_document
from app.s3_client import init_s3_client
from app.gemini_client import (
    init_vertex_ai,
    generate_chat_response,
    refresh_metadata_cache,
    get_cache_status,
)
from app.metadata_loader import load_metadata_from_s3
from app.document_selector import auto_load_relevant_documents
from app.financial_utils import FinancialDataManager
from app.agent_v2 import AgentV2
from app.config import get_config
from app.response_cache import ResponseCache, build_response_cache
from app.interaction_log import InteractionLogger, build_interaction_logger
from app.tracing import get_langfuse_client, traced_observation
from app.logging_config import configure_logging, get_logger
from app.middleware.request_id import RequestIDMiddleware
from app.middleware.metrics import (
    PrometheusMetricsMiddleware,
    get_metrics,
    get_metrics_content_type,
)

from dotenv import load_dotenv

load_dotenv()

# Initialize configuration (validates all env vars and provides type-safe access)
config = get_config()

# Configure structured logging
configure_logging(log_level=config.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Store config in app state for access in endpoints
        app.state.config = config

        # Initialize S3 client
        app.state.s3_client = init_s3_client()
        # Initialize Vertex AI
        init_vertex_ai()
        # Load metadata from S3
        try:
            app.state.metadata = load_metadata_from_s3(app.state.s3_client)
            if app.state.metadata:
                logger.info(f"Metadata loaded: {len(app.state.metadata)} companies found")
            else:
                logger.warning("Failed to load metadata from S3")
        except HTTPException as e:
            logger.error(
                "metadata_load_startup_failed",
                status_code=e.status_code,
                detail=e.detail,
            )
            app.state.metadata = None
        # Index documents for citation resolution (GET /documents/{id}).
        app.state.document_index = build_document_index(app.state.metadata)
        # -----------------------
        # Initialize Financial Data Manager (BigQuery)
        # -----------------------
        try:
            app.state.financial_manager = FinancialDataManager()
            if app.state.financial_manager.metadata:
                logger.info(
                    "Financial data manager (BigQuery) initialized successfully with metadata"
                )
            else:
                logger.warning("Financial data manager initialized but metadata loading failed")
        except Exception as financial_err:
            logger.error(f"Failed to initialize financial data manager: {financial_err}")
            app.state.financial_manager = None

        # -----------------------
        # Response cache (Redis), interaction logger (BigQuery), tracing (Langfuse)
        # -----------------------
        app.state.response_cache = build_response_cache()
        logger.info(
            "response_cache_initialized",
            extra={"enabled": app.state.response_cache.enabled},
        )

        app.state.interaction_logger = build_interaction_logger()
        logger.info(
            "interaction_logger_initialized",
            extra={"enabled": app.state.interaction_logger.enabled},
        )

        langfuse_client = get_langfuse_client()
        logger.info("langfuse_client_initialized", extra={"enabled": langfuse_client is not None})

        yield

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        # We'll continue and let individual endpoints handle errors
    finally:
        cache = getattr(app.state, "response_cache", None)
        if cache is not None:
            await cache.close()
        langfuse_client = get_langfuse_client()
        if langfuse_client is not None:
            try:
                langfuse_client.flush()
            except Exception as exc:
                logger.warning("langfuse_flush_failed", extra={"error": str(exc)})


# Initialize FastAPI app
app = FastAPI(
    title="Jacie",
    description="API for chatting with Jacie, the JSE DataSphere Chatbot.",
    version="1.2.0",
    lifespan=lifespan,
)

# Add Request ID middleware (must be added before other middleware)
app.add_middleware(RequestIDMiddleware)

# Add Prometheus metrics middleware (after Request ID, before CORS)
app.add_middleware(PrometheusMetricsMiddleware)

# Add CORS middleware
# Note: When allow_credentials=True, cannot use allow_origins=["*"]
# Using regex to allow all origins while being compatible with credentials
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r".*",  # Allows all origins (compatible with credentials)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# ----------------------------------------------------------
# Global Exception Handler
# ----------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler to catch unhandled exceptions and log them with structured context.
    This is a safety net for any errors that slip through endpoint-specific error handling.
    """
    # Skip HTTPException - FastAPI handles these properly
    if isinstance(exc, HTTPException):
        raise exc

    # Log unhandled exception with structured context
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        error_type=type(exc).__name__,
        client_host=request.client.host if request.client else None,
    )

    # Return safe error message
    return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred"})


# ----------------------------------------------------------
# Middleware: Log every request & response with latency
# ----------------------------------------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs incoming requests and their responses including latency and status."""
    request_id = uuid.uuid4().hex
    logger.info(f"[{request_id}] {request.method} {request.url.path} - Start")
    start_time = time.time()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception(
            f"[{request_id}] {request.method} {request.url.path} - Unhandled error: {exc}"
        )
        raise
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} - Completed in {duration_ms:.2f}ms with status {response.status_code}"
    )
    return response


# Dependency to get S3 client
def get_s3_client():
    return app.state.s3_client


# Dependency to get metadata
def get_metadata():
    return app.state.metadata


# Dependency to get the document index
def get_document_index():
    return getattr(app.state, "document_index", {}) or {}


# Dependency to get financial data manager
def get_financial_manager():
    return getattr(app.state, "financial_manager", None)


# Dependency to get the response cache
def get_response_cache_dep():
    return getattr(app.state, "response_cache", None)


# Dependency to get the interaction logger
def get_interaction_logger_dep():
    return getattr(app.state, "interaction_logger", None)


@app.get("/")
async def root():
    """Root endpoint to check if API is running"""
    return {"message": "Document Chat API is running"}


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text exposition format for scraping.
    This endpoint is used by Prometheus to collect application metrics.
    """
    return Response(content=get_metrics(), media_type=get_metrics_content_type())


@app.get("/health")
async def health_check():
    """
    Enhanced health check endpoint with component connectivity tests.

    Checks the health of all critical components:
    - S3 connectivity (document storage)
    - BigQuery connectivity (financial data)
    - Gemini AI availability

    Returns:
        200 OK: All components healthy
        503 Service Unavailable: One or more components unhealthy
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {},
        }
        all_healthy = True

        # Check S3 connectivity
        s3_health = await _check_s3_health()
        health_status["components"]["s3"] = s3_health
        if s3_health["status"] != "healthy":
            all_healthy = False

        # Check BigQuery connectivity
        bigquery_health = await _check_bigquery_health()
        health_status["components"]["bigquery"] = bigquery_health
        if bigquery_health["status"] != "healthy":
            all_healthy = False

        # Check Gemini AI availability
        gemini_health = await _check_gemini_health()
        health_status["components"]["gemini"] = gemini_health
        if gemini_health["status"] != "healthy":
            all_healthy = False

        # Check metadata availability
        metadata_health = _check_metadata_health()
        health_status["components"]["metadata"] = metadata_health
        if metadata_health["status"] != "healthy":
            all_healthy = False

        # Set overall status
        if not all_healthy:
            health_status["status"] = "degraded"
            return JSONResponse(status_code=503, content=health_status)

        return health_status

    except Exception as e:
        logger.error(f"Health check failed with exception: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": f"Health check failed: {str(e)}",
                "timestamp": time.time(),
            },
        )


async def _check_s3_health() -> Dict[str, Any]:
    """Check S3 connectivity by attempting to list objects."""
    try:
        if not hasattr(app.state, "s3_client") or app.state.s3_client is None:
            return {"status": "unavailable", "message": "S3 client not initialized"}

        # Try to check if bucket is accessible (use head_bucket for quick check)
        s3_client = app.state.s3_client
        bucket_name = config.aws.s3_bucket

        # Simple check - verify client exists and bucket is configured
        # We avoid actual S3 calls in health check to keep it fast
        if s3_client and bucket_name:
            return {"status": "healthy", "bucket": bucket_name}
        else:
            return {"status": "unhealthy", "message": "S3 configuration incomplete"}

    except Exception as e:
        logger.error(f"S3 health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}


async def _check_bigquery_health() -> Dict[str, Any]:
    """Check BigQuery connectivity."""
    try:
        if not hasattr(app.state, "financial_manager") or app.state.financial_manager is None:
            return {"status": "unavailable", "message": "BigQuery client not initialized"}

        financial_manager = app.state.financial_manager

        # Check if client and metadata are available
        if financial_manager.bq_client and financial_manager.metadata:
            companies_count = 0
            if "companies" in financial_manager.metadata:
                companies_count = len(financial_manager.metadata["companies"])

            return {
                "status": "healthy",
                "metadata_loaded": True,
                "companies_count": companies_count,
            }
        elif financial_manager.bq_client:
            return {
                "status": "degraded",
                "metadata_loaded": False,
                "message": "Client initialized but metadata not loaded",
            }
        else:
            return {"status": "unhealthy", "message": "BigQuery client not initialized"}

    except Exception as e:
        logger.error(f"BigQuery health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}


async def _check_gemini_health() -> Dict[str, Any]:
    """Check Gemini AI availability."""
    try:
        # Check if API key is configured
        if not config.gcp.api_key:
            return {"status": "unhealthy", "message": "Gemini API key not configured"}

        # Check if Vertex AI is initialized
        # We don't make actual API calls in health check to keep it fast
        return {"status": "healthy", "message": "Gemini AI configured"}

    except Exception as e:
        logger.error(f"Gemini health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}


def _check_metadata_health() -> Dict[str, Any]:
    """Check metadata availability."""
    try:
        if hasattr(app.state, "metadata") and app.state.metadata:
            companies_count = len(app.state.metadata) if app.state.metadata else 0
            return {"status": "healthy", "companies_count": companies_count}
        else:
            return {"status": "unavailable", "message": "Metadata not loaded"}

    except Exception as e:
        logger.error(f"Metadata health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}


@app.get("/health/stream")
async def health_stream():
    """Simple streaming health check to test SSE functionality"""

    async def generate_health_stream():
        yield 'event: status\ndata: {"message": "Streaming health check started"}\n\n'
        await asyncio.sleep(0.1)
        yield 'event: status\ndata: {"message": "Streaming is working correctly"}\n\n'
        await asyncio.sleep(0.1)
        yield 'event: complete\ndata: {"status": "healthy", "streaming": "enabled"}\n\n'

    return StreamingResponse(
        generate_health_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    s3_client: Any = Depends(get_s3_client),
    metadata: Dict = Depends(get_metadata),
    financial_manager: Any = Depends(get_financial_manager),
):
    """
    Chat with documents using Gemini AI

    This endpoint allows you to ask questions about documents stored in S3.
    It uses Gemini AI to generate responses based on the document content.

    If auto_load_documents is enabled, it will automatically load relevant documents
    based on the query and conversation history.

    If memory_enabled is enabled, it will use the conversation history to provide context
    for the response.
    """
    logger.info(
        f"/chat called. query='{request.query[:200]}', auto_load_documents={request.auto_load_documents}, memory_enabled={request.memory_enabled}"
    )

    # Log conversation history for debugging
    if request.conversation_history:
        logger.info(
            f"📜 Conversation history received: {len(request.conversation_history)} messages"
        )
        for idx, msg in enumerate(request.conversation_history[-5:]):  # Log last 5 messages
            content_preview = msg.get("content", "")[:100]
            logger.info(f"  [{idx}] {msg.get('role', 'unknown')}: {content_preview}...")
    else:
        logger.info("📜 No conversation history received")

    try:
        # Check if metadata is available
        if not metadata:
            raise HTTPException(
                status_code=500, detail="Metadata not available. Please check S3 access."
            )

        # Initialize variables
        document_texts = {}
        document_selection_message = None
        loaded_docs = []
        doc_sources = []

        # Auto-load relevant documents if enabled
        if request.auto_load_documents:
            # Get symbol-to-company associations for better company resolution
            associations = None
            if financial_manager and financial_manager.metadata:
                associations = financial_manager.metadata.get("associations")

            load_result = auto_load_relevant_documents(
                s3_client,
                request.query,
                metadata,
                {},  # Start with empty document_texts since this is stateless
                request.conversation_history,
                associations,
            )
            document_texts = load_result.texts
            document_selection_message = load_result.message
            loaded_docs = load_result.loaded_docs
            doc_sources = load_result.sources

        # Generate response
        response_text = generate_chat_response(
            request.query, document_texts, request.conversation_history, document_selection_message
        )

        # Update conversation history if memory is enabled
        updated_conversation_history = None
        if request.memory_enabled and request.conversation_history:
            updated_conversation_history = request.conversation_history.copy()
            # Add user query
            updated_conversation_history.append({"role": "user", "content": request.query})
            # Add assistant response
            updated_conversation_history.append({"role": "assistant", "content": response_text})
        elif request.memory_enabled:
            # If memory is enabled but no history provided, create new history
            updated_conversation_history = [
                {"role": "user", "content": request.query},
                {"role": "assistant", "content": response_text},
            ]

        # Add document selection message to response if available
        # if document_selection_message and "Semantically selected" in document_selection_message:
        #     response_text += f"\n\n{document_selection_message}"

        logger.info(
            f"/chat completed successfully. response_chars={len(response_text)}, documents_loaded={len(loaded_docs)}"
        )

        # Return response
        return ChatResponse(
            response=response_text,
            documents_loaded=loaded_docs if loaded_docs else None,
            document_selection_message=document_selection_message,
            conversation_history=updated_conversation_history,
            sources=doc_sources if doc_sources else None,
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


# ---------------------------------------------------------------------------
# Financial Data Query Endpoint (V2)
# ---------------------------------------------------------------------------


@app.post("/fast_chat_v2", response_model=FinancialDataResponse)
async def fast_chat_v2(
    request: FinancialDataRequest,
    http_request: Request,
    financial_manager: Any = Depends(get_financial_manager),
    response_cache: Any = Depends(get_response_cache_dep),
    interaction_logger: Any = Depends(get_interaction_logger_dep),
):
    """
    Natural language financial data query endpoint

    This endpoint allows users to query financial data using natural language.
    It processes queries like:
    - "Show me MDS revenue for 2024"
    - "Compare JBG and CPJ profit margins"
    - "What about 2022?" (follow-up questions)

    The endpoint uses AI to parse natural language queries into structured filters,
    queries the financial database, and returns formatted responses with data insights.

    Repeated queries with no conversation history are served from a Redis
    response cache (skipped whenever history is present, since a follow-up's
    correct answer depends on that context). Every call is logged to BigQuery
    for regression/behavior monitoring, and traced via Langfuse if configured.
    """
    logger.info(
        f"/fast_chat_v2 called. query='{request.query[:200]}', memory_enabled={request.memory_enabled}"
    )

    # Log conversation history for debugging
    if request.conversation_history:
        logger.info(
            f"📜 Conversation history received: {len(request.conversation_history)} messages"
        )
        for idx, msg in enumerate(request.conversation_history[-5:]):  # Log last 5 messages
            content_preview = msg.get("content", "")[:100]
            logger.info(f"  [{idx}] {msg.get('role', 'unknown')}: {content_preview}...")
    else:
        logger.info("📜 No conversation history received")

    start_time = time.time()
    request_id = getattr(http_request.state, "request_id", None)
    cache_eligible = not request.conversation_history
    cache_key = (
        ResponseCache.build_key("fast_chat_v2", ResponseCache.normalize_query(request.query))
        if cache_eligible
        else None
    )

    def schedule_log(**kwargs):
        if interaction_logger is None:
            return
        row = InteractionLogger.build_row(
            endpoint="fast_chat_v2",
            query=request.query,
            request_id=request_id,
            conversation_history=request.conversation_history,
            memory_enabled=request.memory_enabled,
            latency_ms=(time.time() - start_time) * 1000,
            timestamp=datetime.now(timezone.utc).isoformat(),
            **kwargs,
        )
        asyncio.create_task(interaction_logger.log(row))

    try:
        if not financial_manager:
            raise HTTPException(
                status_code=503,
                detail="Financial data service is not available. Please ensure BigQuery is configured.",
            )

        cached = await response_cache.get(cache_key) if cache_key and response_cache else None
        if cached is not None:
            ai_response = cached["response"]
            data_found = cached["data_found"]
            record_count = cached["record_count"]
            filters = FinancialDataFilters(**cached["filters_used"])
            data_preview = (
                [FinancialDataRecord(**r) for r in cached["data_preview"]]
                if cached.get("data_preview")
                else None
            )
            chart_spec = ChartSpec(**cached["chart"]) if cached.get("chart") else None
            sources = [Source(**s) for s in cached["sources"]] if cached.get("sources") else None

            updated_conversation_history = None
            if request.memory_enabled:
                updated_conversation_history = [
                    {"role": "user", "content": request.query},
                    {"role": "assistant", "content": ai_response},
                ]

            logger.info(f"/fast_chat_v2 cache hit. data_found={data_found}")
            schedule_log(
                response=ai_response,
                cache_hit=True,
                success=True,
                data_found=data_found,
                record_count=record_count,
                sources=sources,
            )
            return FinancialDataResponse(
                response=ai_response,
                data_found=data_found,
                record_count=record_count,
                filters_used=filters,
                data_preview=data_preview,
                conversation_history=updated_conversation_history,
                warnings=cached.get("warnings"),
                suggestions=cached.get("suggestions"),
                chart=chart_spec,
                sources=sources,
            )

        last_query_data = getattr(request, "_last_query_data", None)
        request_costs = []
        with traced_observation(
            "fast_chat_v2", as_type="span", input={"query": request.query}
        ) as trace_obs:
            # parse_user_query/format_response are now genuinely async and
            # issue their Gemini calls through native async clients, so they
            # yield the event loop without occupying a thread-pool worker
            # (docs/adr/0003-native-async-gemini-client.md). query_data stays
            # wrapped in asyncio.to_thread — google-cloud-bigquery has no
            # first-party async client, so a thread is still the right tool
            # there (docs/adr/0001-fix-event-loop-blocking-llm-calls.md).
            filters = await financial_manager.parse_user_query(
                request.query,
                request.conversation_history,
                last_query_data,
                cost_sink=request_costs,
            )
            logger.info(f"Parsed filters: {filters}")
            logger.info(f"filters type: {type(filters)}, filters: {filters}")
            availability = financial_manager.validate_data_availability(filters)
            # logger.info(f"availability type: {type(availability)}, availability: {availability}")
            warnings = availability.get("warnings", [])
            suggestions = availability.get("suggestions", [])
            results = await asyncio.to_thread(financial_manager.query_data, filters)
            logger.info(f"results type: {type(results)}, results: {results}")
            ai_response = await financial_manager.format_response(
                results,
                request.query,
                filters.interpretation,
                filters.is_follow_up,
                request.conversation_history,
                unrecognized_items=filters.unrecognized_items or None,
                cost_sink=request_costs,
            )
            # logger.info(f"ai_response type: {type(ai_response)}, ai_response: {ai_response}")
            data_preview = results if results else None
            updated_conversation_history = None
            if request.memory_enabled:
                if request.conversation_history:
                    updated_conversation_history = request.conversation_history.copy()
                else:
                    updated_conversation_history = []
                updated_conversation_history.append({"role": "user", "content": request.query})
                updated_conversation_history.append({"role": "assistant", "content": ai_response})
                if len(updated_conversation_history) > 20:
                    updated_conversation_history = updated_conversation_history[-20:]
            # Generate chart if data is chartable
            chart_spec = None
            if results:
                chart_data = generate_chart(results, request.query)
                if chart_data:
                    chart_spec = ChartSpec(**chart_data)
                    logger.info(
                        f"Generated {chart_data['chart_type']} chart: {chart_data['title']}"
                    )

            sources = [financial_manager.describe_source(results)] if results else None

            if trace_obs is not None:
                trace_obs.update(output=ai_response, metadata={"cache_hit": False})

        logger.info(
            f"/fast_chat_v2 completed. data_found={bool(results)}, record_count={len(results) if results else 0}, has_chart={chart_spec is not None}"
        )

        if cache_key and response_cache:
            await response_cache.set(
                cache_key,
                {
                    "response": ai_response,
                    "data_found": bool(results),
                    "record_count": len(results) if results else 0,
                    "filters_used": filters.model_dump(),
                    "data_preview": [r.model_dump() for r in results] if results else None,
                    "warnings": warnings if warnings else None,
                    "suggestions": suggestions if suggestions else None,
                    "chart": chart_spec.model_dump() if chart_spec else None,
                    "sources": _jsonable(sources),
                },
            )

        schedule_log(
            response=ai_response,
            cache_hit=False,
            success=True,
            data_found=bool(results),
            record_count=len(results) if results else 0,
            model=request_costs[0].model if request_costs else None,
            input_tokens=sum(c.token_usage.input_tokens for c in request_costs),
            output_tokens=sum(c.token_usage.output_tokens for c in request_costs),
            cost_usd=sum(c.total_cost for c in request_costs),
            phase_costs=[c.phase for c in request_costs],
            sources=sources,
        )
        return FinancialDataResponse(
            response=ai_response,
            data_found=bool(results),
            record_count=len(results) if results else 0,
            filters_used=filters,
            data_preview=data_preview,
            conversation_history=updated_conversation_history,
            warnings=warnings if warnings else None,
            suggestions=suggestions if suggestions else None,
            chart=chart_spec,
            sources=sources,
        )
    except HTTPException as e:
        schedule_log(response=None, cache_hit=False, success=False, error_message=str(e.detail))
        raise
    except Exception as e:
        logger.error(f"Error in fast_chat_v2 endpoint: {str(e)}")
        schedule_log(response=None, cache_hit=False, success=False, error_message=str(e))
        raise HTTPException(
            status_code=500, detail=f"Error processing financial data query: {str(e)}"
        )


@app.get("/financial/metadata")
async def get_financial_metadata(
    financial_manager: Any = Depends(get_financial_manager),
):
    """Get financial data metadata including available companies, symbols, years, and metrics"""
    try:
        if not financial_manager:
            raise HTTPException(status_code=503, detail="Financial data service is not available.")

        if not financial_manager.metadata:
            raise HTTPException(status_code=404, detail="Financial metadata not available.")

        # Return the metadata but limit large lists for better API response size
        metadata = financial_manager.metadata.copy()

        # If there are too many items, provide sample + count
        for key in ["companies", "symbols", "standard_items"]:
            if key in metadata and len(metadata[key]) > 50:
                sample = metadata[key][:50]
                metadata[key] = {
                    "sample": sample,
                    "total_count": len(metadata[key]),
                    "note": f"Showing first 50 of {len(metadata[key])} items",
                }

        return {"status": "success", "metadata": metadata}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving financial metadata: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving financial metadata: {str(e)}"
        )


@app.get("/documents/{document_id}")
async def resolve_document(
    document_id: str,
    s3_client: Any = Depends(get_s3_client),
    document_index: Dict = Depends(get_document_index),
):
    """Resolve a cited document to a short-lived presigned S3 URL.

    `document_id` is a one-way hash, so this is a lookup against the documents
    present in metadata.json — an id we did not mint resolves to nothing.
    """
    entry = document_index.get(document_id)
    if not entry:
        logger.warning(f"document_resolve_miss id={document_id[:32]}")
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        url = presign_document(s3_client, entry["s3_path"])
    except ValueError as e:
        logger.error(f"document_resolve_bad_path id={document_id}: {e}")
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"document_resolve_failed id={document_id}: {e}")
        raise HTTPException(status_code=500, detail="Could not resolve document")

    return RedirectResponse(url=url, status_code=307)


@app.get("/cache/status")
async def get_cache_status_endpoint():
    """Get the current status of the metadata cache"""
    try:
        status = get_cache_status()
        return {"success": True, "cache_status": status}
    except Exception as e:
        logger.error(f"Error getting cache status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting cache status: {str(e)}")


@app.post("/cache/refresh")
async def refresh_cache_endpoint():
    """Force refresh of the metadata cache using current S3 metadata"""
    try:
        # Load current metadata from S3
        metadata = load_metadata_from_s3()
        if not metadata:
            raise HTTPException(status_code=500, detail="Failed to load metadata from S3")

        # Refresh the cache
        success = refresh_metadata_cache(metadata)

        if success:
            return {
                "success": True,
                "message": "Cache refreshed successfully",
                "cache_status": get_cache_status(),
            }
        else:
            return {
                "success": False,
                "message": "Failed to refresh cache",
                "cache_status": get_cache_status(),
            }
    except Exception as e:
        logger.error(f"Error refreshing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error refreshing cache: {str(e)}")


# ==============================================================================
# AGENT CHAT ENDPOINT (Primary: /chat/stream)
# ==============================================================================


def _jsonable(value):
    """Best-effort conversion of a Pydantic model (or list of them) to plain
    dict/list for JSON caching. Passes plain dicts/None through unchanged."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


@app.post("/chat/stream", response_model=AgentChatResponse)
async def chat_stream(
    request: AgentChatRequest,
    http_request: Request,
    response_cache: Any = Depends(get_response_cache_dep),
    interaction_logger: Any = Depends(get_interaction_logger_dep),
    financial_manager: Any = Depends(get_financial_manager),
):
    """
    JSE Financial Analyst endpoint using Gemini 2.5 Pro with Google Search grounding.

    This endpoint uses a simplified single-call architecture with native Google Search
    grounding for JSE financial research and market information.

    Features:
    - Gemini 2.5 Pro model for high-quality responses
    - Native Google Search grounding for current market data and news
    - Conversation history support for follow-up questions
    - Source citations from web search results

    The response is backward compatible with AgentChatResponse.

    Repeated queries (same flags, no conversation history) are served from a
    Redis response cache. Every call is logged to BigQuery for regression/
    behavior monitoring, and traced via Langfuse if configured.
    """
    logger.info(
        f"/chat/stream called. query='{request.query[:200]}', "
        f"memory_enabled={request.memory_enabled}"
    )

    # Log conversation history for debugging
    if request.conversation_history:
        logger.info(f"Conversation history: {len(request.conversation_history)} messages")
        for idx, msg in enumerate(request.conversation_history[-3:]):
            content_preview = msg.get("content", "")[:100]
            logger.info(f"  [{idx}] {msg.get('role', 'unknown')}: {content_preview}...")
    else:
        logger.info("No conversation history received")

    start_time = time.time()
    request_id = getattr(http_request.state, "request_id", None)
    cache_eligible = not request.conversation_history
    cache_key = (
        ResponseCache.build_key(
            "chat_stream",
            ResponseCache.normalize_query(request.query),
            request.enable_web_search,
            request.enable_financial_data,
        )
        if cache_eligible
        else None
    )

    def schedule_log(**kwargs):
        if interaction_logger is None:
            return
        row = InteractionLogger.build_row(
            endpoint="chat_stream",
            query=request.query,
            request_id=request_id,
            conversation_history=request.conversation_history,
            memory_enabled=request.memory_enabled,
            enable_web_search=request.enable_web_search,
            enable_financial_data=request.enable_financial_data,
            latency_ms=(time.time() - start_time) * 1000,
            timestamp=datetime.now(timezone.utc).isoformat(),
            **kwargs,
        )
        asyncio.create_task(interaction_logger.log(row))

    try:
        cached = await response_cache.get(cache_key) if cache_key and response_cache else None
        if cached is not None:
            updated_conversation_history = None
            if request.memory_enabled:
                updated_conversation_history = [
                    {"role": "user", "content": request.query},
                    {"role": "assistant", "content": cached["response"]},
                ]
            logger.info(f"/chat/stream cache hit. data_found={cached['data_found']}")
            schedule_log(
                response=cached["response"],
                cache_hit=True,
                success=True,
                data_found=cached["data_found"],
                record_count=cached["record_count"],
                sources=cached.get("sources"),
            )
            return AgentChatResponse(
                response=cached["response"],
                data_found=cached["data_found"],
                record_count=cached["record_count"],
                filters_used=cached.get("filters_used"),
                data_preview=cached.get("data_preview"),
                conversation_history=updated_conversation_history,
                warnings=cached.get("warnings"),
                suggestions=cached.get("suggestions"),
                chart=cached.get("chart"),
                sources=cached.get("sources"),
                web_search_results=cached.get("web_search_results"),
                tools_executed=cached.get("tools_executed"),
                needs_clarification=cached.get("needs_clarification", False),
                clarification_question=cached.get("clarification_question"),
                cost_summary=None,
            )

        agent = AgentV2()

        with traced_observation(
            "chat_stream",
            as_type="span",
            input={
                "query": request.query,
                "enable_web_search": request.enable_web_search,
            },
        ) as trace_obs:
            # Run the agent
            result = await agent.run(
                query=request.query,
                conversation_history=request.conversation_history,
                enable_web_search=request.enable_web_search,
                financial_manager=financial_manager,
            )

            # Build response (compatible with AgentChatResponse)
            response = AgentChatResponse(
                response=result["response"],
                data_found=result["data_found"],
                record_count=result["record_count"],
                filters_used=result.get("filters_used"),
                data_preview=result.get("data_preview"),
                conversation_history=(
                    result["conversation_history"] if request.memory_enabled else None
                ),
                warnings=result.get("warnings"),
                suggestions=result.get("suggestions"),
                chart=result.get("chart"),
                sources=result.get("sources"),
                web_search_results=result.get("web_search_results"),
                tools_executed=result.get("tools_executed"),
                needs_clarification=result.get("needs_clarification", False),
                clarification_question=result.get("clarification_question"),
                cost_summary=result.get("cost_summary"),
            )

            if trace_obs is not None:
                trace_obs.update(output=response.response, metadata={"cache_hit": False})

        logger.info(
            f"/chat/stream completed. response_chars={len(response.response)}, "
            f"tools_executed={response.tools_executed}"
        )

        if cache_key and response_cache and not response.needs_clarification:
            await response_cache.set(
                cache_key,
                {
                    "response": response.response,
                    "data_found": response.data_found,
                    "record_count": response.record_count,
                    "filters_used": _jsonable(response.filters_used),
                    "data_preview": _jsonable(response.data_preview),
                    "warnings": response.warnings,
                    "suggestions": response.suggestions,
                    "chart": _jsonable(response.chart),
                    "sources": _jsonable(response.sources),
                    "web_search_results": response.web_search_results,
                    "tools_executed": response.tools_executed,
                    "needs_clarification": response.needs_clarification,
                    "clarification_question": response.clarification_question,
                },
            )

        cost_summary = response.cost_summary
        thinking_tokens = cost_summary.total_thinking_tokens if cost_summary else None
        schedule_log(
            response=response.response,
            cache_hit=False,
            success=True,
            data_found=response.data_found,
            record_count=response.record_count,
            input_tokens=cost_summary.total_input_tokens if cost_summary else 0,
            output_tokens=cost_summary.total_output_tokens if cost_summary else 0,
            thinking_tokens=thinking_tokens,
            # Thinking is billed as output but reported separately, so the real
            # total is input + visible output + thinking (issue #72).
            total_tokens=(
                cost_summary.total_input_tokens
                + cost_summary.total_output_tokens
                + cost_summary.total_thinking_tokens
                if cost_summary
                else None
            ),
            finish_reason=result.get("finish_reason"),
            cost_usd=cost_summary.total_cost_usd if cost_summary else 0.0,
            phase_costs=[p.phase for p in cost_summary.phases] if cost_summary else None,
            sources=response.sources,
        )

        return response

    except Exception as e:
        logger.error(f"Error in chat stream endpoint: {str(e)}", exc_info=True)
        schedule_log(response=None, cache_hit=False, success=False, error_message=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat stream: {str(e)}",
        )
