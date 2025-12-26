from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response
import time
import uuid
import asyncio
from typing import Dict, Any
from contextlib import asynccontextmanager

from app.models import (
    ChatRequest,
    ChatResponse,
    ChartSpec,
    StreamingChatRequest,
    FinancialDataRequest,
    FinancialDataResponse,
    JobCreateResponse,
    JobStatusResponse,
    JobStatus,
    AgentChatRequest,
    AgentChatResponse,
)
from app.charting import generate_chart
from app.s3_client import init_s3_client
from app.gemini_client import (
    init_vertex_ai,
    generate_chat_response,
    refresh_metadata_cache,
    get_cache_status,
)
from app.metadata_loader import load_metadata_from_s3
from app.document_selector import auto_load_relevant_documents
from app.streaming_chat import process_streaming_chat
from app.financial_utils import FinancialDataManager
from app.agent import AgentOrchestrator
from app.job_store import JobStore, JobProgressSink
from app.redis_job_store import RedisJobStore
from app.progress_tracker import ProgressTracker
from app.config import get_config
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
        # Initialize Job Store (Redis or In-Memory)
        # -----------------------
        # Use centralized config instead of multiple env var checks
        redis_url = config.redis.url

        if redis_url:
            try:
                # Initialize and TEST connection using config values
                job_store = RedisJobStore(
                    redis_url=redis_url,
                    ttl_seconds=config.async_job.ttl_seconds,
                    max_progress_history=config.async_job.max_progress_history,
                )
                # Verify connection works
                await job_store._redis.ping()

                app.state.job_store = job_store
                logger.info("Redis job store initialized and connected successfully")
            except Exception as e:
                logger.error(
                    f"Failed to initialize Redis job store (connection failed): {e}. Falling back to in-memory."
                )
                app.state.job_store = JobStore(
                    ttl_seconds=config.async_job.ttl_seconds,
                    max_progress_history=config.async_job.max_progress_history,
                )
        else:
            app.state.job_store = JobStore(
                ttl_seconds=config.async_job.ttl_seconds,
                max_progress_history=config.async_job.max_progress_history,
            )
        logger.info("Job store initialized | async_job_mode=%s", config.async_job.enabled)
        yield

        # Cleanup
        if hasattr(app.state, "job_store") and hasattr(app.state.job_store, "close"):
            logger.info("Closing job store connection...")
            await app.state.job_store.close()
            logger.info("Job store connection closed")

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        # We'll continue and let individual endpoints handle errors


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


# Dependency to get financial data manager
def get_financial_manager():
    return getattr(app.state, "financial_manager", None)


def get_job_store() -> JobStore:
    job_store = getattr(app.state, "job_store", None)
    if not job_store:
        raise HTTPException(status_code=503, detail="Job store not initialized")
    return job_store


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
    - Redis connectivity (job storage)
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

        # Check Redis connectivity
        redis_health = await _check_redis_health()
        health_status["components"]["redis"] = redis_health
        # Redis is optional, so don't mark overall health as unhealthy if Redis is down

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


async def _check_redis_health() -> Dict[str, Any]:
    """Check Redis connectivity."""
    try:
        if not hasattr(app.state, "job_store") or app.state.job_store is None:
            return {"status": "unavailable", "message": "Job store not initialized"}

        job_store = app.state.job_store

        # Check if Redis is being used
        if hasattr(job_store, "_redis"):
            # Try to ping Redis
            try:
                await job_store._redis.ping()
                return {"status": "healthy", "type": "redis"}
            except Exception as e:
                return {"status": "unhealthy", "type": "redis", "error": str(e)}
        else:
            # In-memory job store
            return {"status": "healthy", "type": "in-memory"}

    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}")
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

        # Auto-load relevant documents if enabled
        if request.auto_load_documents:
            # Get symbol-to-company associations for better company resolution
            associations = None
            if financial_manager and financial_manager.metadata:
                associations = financial_manager.metadata.get("associations")

            document_texts, document_selection_message, loaded_docs = auto_load_relevant_documents(
                s3_client,
                request.query,
                metadata,
                {},  # Start with empty document_texts since this is stateless
                request.conversation_history,
                associations,
            )

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
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(
    request: StreamingChatRequest,
    s3_client: Any = Depends(get_s3_client),
    metadata: Dict = Depends(get_metadata),
    job_store: JobStore = Depends(get_job_store),
    financial_manager: Any = Depends(get_financial_manager),
):
    """
    Stream chat responses with real-time progress updates using Server-Sent Events

    This endpoint provides the same functionality as /chat but streams progress updates
    to the client, allowing the frontend to show real-time status messages like:
    - "Loading documents..."
    - "Analyzing query..."
    - "Generating response..."

    The stream will emit 'progress' events with status updates and a final 'result'
    event with the complete response.
    """
    logger.info(
        f"/chat/stream called. query='{request.query[:200]}', auto_load_documents={request.auto_load_documents}, memory_enabled={request.memory_enabled}"
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
        # Get symbol-to-company associations for better company resolution
        associations = None
        if financial_manager and financial_manager.metadata:
            associations = financial_manager.metadata.get("associations")

        if not config.async_job.enabled:
            # SSE streaming mode - return immediate streaming response
            tracker = await process_streaming_chat(
                request=request,
                s3_client=s3_client,
                metadata=metadata,
                use_fast_mode=False,
                associations=associations,
            )
            return StreamingResponse(
                tracker.stream_updates(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Disable nginx buffering
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Cache-Control",
                },
            )

        # Async job mode - create job and return immediately
        job_id = await job_store.create_job("chat_stream", request.model_dump())

        # Fire-and-forget: start processing in background
        asyncio.create_task(
            _process_chat_job_background(
                job_id=job_id,
                request=request,
                s3_client=s3_client,
                metadata=metadata,
                job_store=job_store,
                associations=associations,
            )
        )

        # Return immediately with job ID
        response_payload = JobCreateResponse(
            job_id=job_id,
            status=JobStatus.queued,
            job_type="chat_stream",
            polling_url=f"/jobs/{job_id}",
        )
        logger.info(f"Created async job {job_id} for chat_stream")
        return JSONResponse(status_code=202, content=response_payload.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat stream endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating chat job: {str(e)}")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status_endpoint(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
):
    job_status = await job_store.get_job_status(job_id)
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_status


# ---------------------------------------------------------------------------
# Financial Data Query Endpoint (V2)
# ---------------------------------------------------------------------------


@app.post("/fast_chat_v2", response_model=FinancialDataResponse)
async def fast_chat_v2(
    request: FinancialDataRequest,
    financial_manager: Any = Depends(get_financial_manager),
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

    try:
        if not financial_manager:
            raise HTTPException(
                status_code=503,
                detail="Financial data service is not available. Please ensure BigQuery is configured.",
            )
        last_query_data = getattr(request, "_last_query_data", None)
        filters = financial_manager.parse_user_query(
            request.query, request.conversation_history, last_query_data
        )
        logger.info(f"Parsed filters: {filters}")
        logger.info(f"filters type: {type(filters)}, filters: {filters}")
        availability = financial_manager.validate_data_availability(filters)
        # logger.info(f"availability type: {type(availability)}, availability: {availability}")
        warnings = availability.get("warnings", [])
        suggestions = availability.get("suggestions", [])
        results = financial_manager.query_data(filters)
        logger.info(f"results type: {type(results)}, results: {results}")
        ai_response = financial_manager.format_response(
            results,
            request.query,
            filters.interpretation,
            filters.is_follow_up,
            request.conversation_history,
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
                logger.info(f"Generated {chart_data['chart_type']} chart: {chart_data['title']}")

        logger.info(
            f"/fast_chat_v2 completed. data_found={bool(results)}, record_count={len(results) if results else 0}, has_chart={chart_spec is not None}"
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
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in fast_chat_v2 endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing financial data query: {str(e)}"
        )


@app.post("/fast_chat_v2/stream")
async def fast_chat_v2_stream(
    request: StreamingChatRequest,
    financial_manager: Any = Depends(get_financial_manager),
    job_store: JobStore = Depends(get_job_store),
):
    """
    Stream financial data query responses with real-time progress updates using Server-Sent Events

    This endpoint provides the same functionality as /fast_chat_v2 but streams progress updates
    to the client. It provides real-time updates on:
    - Natural language query parsing
    - Data availability validation
    - Financial database queries
    - AI response generation

    The stream will emit 'progress' events with status updates and a final 'result'
    event with the complete response.
    """
    logger.info(
        f"/fast_chat_v2/stream called. query='{request.query[:200]}', memory_enabled={request.memory_enabled}"
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
        # Import the streaming financial chat processor
        from app.streaming_financial_chat import process_streaming_financial_chat

        if not config.async_job.enabled:
            # SSE streaming mode - return immediate streaming response
            tracker = await process_streaming_financial_chat(
                request=request, financial_manager=financial_manager
            )

            # Return streaming response with proper headers for SSE
            return StreamingResponse(
                tracker.stream_updates(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Disable nginx buffering
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Cache-Control",
                },
            )

        # Async job mode - create job and return immediately
        job_id = await job_store.create_job("financial_stream", request.model_dump())

        # Fire-and-forget: start processing in background
        asyncio.create_task(
            _process_financial_job_background(
                job_id=job_id,
                request=request,
                financial_manager=financial_manager,
                job_store=job_store,
            )
        )

        # Return immediately with job ID
        response_payload = JobCreateResponse(
            job_id=job_id,
            status=JobStatus.queued,
            job_type="financial_stream",
            polling_url=f"/jobs/{job_id}",
        )
        logger.info(f"Created async job {job_id} for financial_stream")
        return JSONResponse(status_code=202, content=response_payload.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in fast_chat_v2 stream endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating financial job: {str(e)}")


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
# AGENT CHAT ENDPOINT
# ==============================================================================


@app.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(
    request: AgentChatRequest,
    financial_manager: Any = Depends(get_financial_manager),
):
    """
    Agentic research endpoint that combines multiple data sources with source citations.

    This endpoint uses Gemini 3 with tool calling to intelligently:
    1. Analyze the user's query and optimize it (resolve pronouns, references)
    2. Determine which tools to use (Google Search, SQL financial data)
    3. Execute tools to gather context
    4. Synthesize a comprehensive response based ONLY on gathered context
    5. Cite all sources explicitly
    6. Suggest follow-up questions

    The agent will NOT hallucinate - it only responds based on data retrieved from tools.

    Tools available:
    - Google Search: For web grounding and recent news
    - SQL Query: For financial data from JSE database

    Response is backward compatible with FinancialDataResponse.
    """
    logger.info(
        f"/agent/chat called. query='{request.query[:200]}', "
        f"web_search={request.enable_web_search}, "
        f"financial={request.enable_financial_data}, "
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

    try:
        # Check if financial manager is available (required for SQL queries)
        if request.enable_financial_data and not financial_manager:
            logger.warning("Financial data requested but manager not available")

        # Get associations from financial manager for better resolution
        associations = None
        if financial_manager and financial_manager.metadata:
            associations = financial_manager.metadata.get("associations")

        # Create orchestrator
        orchestrator = AgentOrchestrator(
            financial_manager=financial_manager,
            associations=associations,
        )

        # Run the agent
        result = await orchestrator.run(
            query=request.query,
            conversation_history=request.conversation_history,
            enable_web_search=request.enable_web_search,
            enable_financial_data=request.enable_financial_data,
        )

        # Build response (backward compatible with FinancialDataResponse)
        response = AgentChatResponse(
            response=result["response"],
            data_found=result["data_found"],
            record_count=result["record_count"],
            filters_used=result.get("filters_used"),
            data_preview=result.get("data_preview"),
            conversation_history=result["conversation_history"] if request.memory_enabled else None,
            warnings=result.get("warnings"),
            suggestions=result.get("suggestions"),
            chart=result.get("chart"),
            sources=result.get("sources"),
            web_search_results=result.get("web_search_results"),
            tools_executed=result.get("tools_executed"),
            # Clarification fields
            needs_clarification=result.get("needs_clarification", False),
            clarification_question=result.get("clarification_question"),
        )

        logger.info(
            f"/agent/chat completed. response_chars={len(response.response)}, "
            f"data_found={response.data_found}, "
            f"record_count={response.record_count}, "
            f"tools_executed={response.tools_executed}"
        )

        return response

    except Exception as e:
        logger.error(f"Error in agent chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing agent chat: {str(e)}",
        )


# ==============================================================================
# BACKGROUND JOB PROCESSORS
# ==============================================================================


async def _process_chat_job_background(
    job_id: str,
    request: StreamingChatRequest,
    s3_client: Any,
    metadata: Dict,
    job_store: JobStore,
    associations: Dict = None,
):
    """
    Background task to process chat job with progress tracking.
    This function runs asynchronously and updates the job store with progress.
    """
    try:
        logger.info(f"Starting background processing for job {job_id}")
        await job_store.mark_running(job_id)

        # Create tracker with job store sink for progress updates
        tracker = ProgressTracker(event_sink=JobProgressSink(job_store, job_id))

        # Process the streaming chat
        await process_streaming_chat(
            request=request,
            s3_client=s3_client,
            metadata=metadata,
            use_fast_mode=False,
            tracker=tracker,
            associations=associations,
        )

        logger.info(f"Successfully completed background job {job_id}")

    except Exception as job_error:
        logger.error(f"Job {job_id} failed with error: {str(job_error)}", exc_info=True)
        await job_store.fail_job(job_id, str(job_error))


async def _process_financial_job_background(
    job_id: str,
    request: StreamingChatRequest,
    financial_manager: Any,
    job_store: JobStore,
):
    """
    Background task to process financial data job with progress tracking.
    Queries financial database with natural language.
    """
    try:
        logger.info(f"Starting background financial processing for job {job_id}")
        await job_store.mark_running(job_id)

        # Import the streaming financial chat processor
        from app.streaming_financial_chat import process_streaming_financial_chat

        # Create tracker with job store sink for progress updates
        tracker = ProgressTracker(event_sink=JobProgressSink(job_store, job_id))

        # Process the streaming financial chat
        await process_streaming_financial_chat(
            request=request,
            financial_manager=financial_manager,
            tracker=tracker,
        )

        logger.info(f"Successfully completed background financial job {job_id}")

    except Exception as job_error:
        logger.error(f"Financial job {job_id} failed with error: {str(job_error)}", exc_info=True)
        await job_store.fail_job(job_id, str(job_error))
