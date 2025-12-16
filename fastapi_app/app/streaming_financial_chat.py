import asyncio
import logging
from typing import Optional, Any
from app.progress_tracker import ProgressTracker
from app.models import StreamingChatRequest

logger = logging.getLogger(__name__)


async def process_streaming_financial_chat(
    request: StreamingChatRequest, financial_manager: Any, tracker: Optional[ProgressTracker] = None
) -> ProgressTracker:
    """
    Process a financial data query with streaming progress updates

    Args:
        request: The streaming chat request
        financial_manager: Financial data manager instance
        tracker: Optional progress tracker (creates new one if not provided)

    Returns:
        ProgressTracker instance that can be used to stream updates
    """

    if tracker is None:
        tracker = ProgressTracker()

    # Start the processing in the background
    asyncio.create_task(_process_financial_chat_async(request, financial_manager, tracker))

    return tracker


async def _process_financial_chat_async(
    request: StreamingChatRequest, financial_manager: Any, tracker: ProgressTracker
):
    """Internal async processing function for financial data queries"""

    try:
        await tracker.emit_progress("start", "Starting financial data query...", 5.0)

        if not financial_manager:
            await tracker.emit_error(
                "Financial data service is not available. Please ensure BigQuery is configured."
            )
            return

        await tracker.emit_progress("query_parsing", "Parsing natural language query...", 15.0)

        # Step 1: Parse the natural language query
        last_query_data = getattr(request, "_last_query_data", None)
        filters = financial_manager.parse_user_query(
            request.query, request.conversation_history, last_query_data
        )

        await tracker.emit_progress(
            "query_parsing",
            "Query parsed successfully",
            25.0,
            {
                "companies": filters.companies,
                "years": filters.years,
                "metrics": filters.standard_items,
            },
        )

        await tracker.emit_progress("data_validation", "Validating data availability...", 35.0)

        # Step 2: Validate data availability
        availability = financial_manager.validate_data_availability(filters)
        warnings = availability.get("warnings", [])
        suggestions = availability.get("suggestions", [])

        await tracker.emit_progress(
            "data_validation",
            "Data availability validated",
            45.0,
            {"warnings": len(warnings), "suggestions": len(suggestions)},
        )

        await tracker.emit_progress("data_query", "Querying financial database...", 55.0)

        # Step 3: Query the financial data
        results = financial_manager.query_data(filters)

        await tracker.emit_progress(
            "data_query",
            f"Retrieved {len(results) if results else 0} financial records",
            75.0,
            {"records_found": len(results) if results else 0},
        )

        await tracker.emit_progress("ai_generation", "Generating AI response...", 85.0)

        # Step 4: Generate AI response
        ai_response = financial_manager.format_response(
            results,
            request.query,
            filters.interpretation,
            filters.is_follow_up,
            request.conversation_history,
        )

        await tracker.emit_progress("finalizing", "Finalizing response...", 95.0)

        # Step 5: Update conversation history
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

        # Prepare final result
        data_preview = results if results else None

        result = {
            "response": ai_response,
            "data_found": bool(results),
            "record_count": len(results) if results else 0,
            "filters_used": filters,
            "data_preview": data_preview,
            "conversation_history": updated_conversation_history,
            "warnings": warnings if warnings else None,
            "suggestions": suggestions if suggestions else None,
        }

        await tracker.emit_progress("complete", "Financial data query complete!", 100.0)
        await tracker.emit_final_result(result)

    except Exception as e:
        logger.error(f"Error in streaming financial chat processing: {str(e)}")
        await tracker.emit_error(f"Error processing financial data query: {str(e)}")
