import json
import asyncio
import logging
from datetime import datetime
from typing import AsyncGenerator, Optional, Dict, Any, Protocol
from app.models import ProgressUpdate

logger = logging.getLogger(__name__)


class ProgressEventSink(Protocol):
    async def on_progress(self, update: ProgressUpdate) -> None: ...

    async def on_result(self, result: Dict[str, Any]) -> None: ...

    async def on_error(self, error: str) -> None: ...


class ProgressTracker:
    """
    Utility class for tracking and streaming progress updates via Server-Sent Events
    """

    def __init__(self, event_sink: Optional[ProgressEventSink] = None):
        self.current_step = ""
        self.current_progress = 0.0
        self.updates_queue = asyncio.Queue()
        self.event_sink = event_sink

    async def emit_progress(
        self, step: str, message: str, progress: float, details: Optional[Dict[str, Any]] = None
    ):
        """Emit a progress update"""
        self.current_step = step
        self.current_progress = progress

        update = ProgressUpdate(
            step=step,
            message=message,
            progress=progress,
            timestamp=datetime.utcnow().isoformat() + "Z",
            details=details,
        )

        await self.updates_queue.put(update)
        if self.event_sink:
            await self.event_sink.on_progress(update)
        logger.info(f"Emitted progress: {step} - {progress}% - {message}")  # Changed to info level

    async def emit_final_result(self, result: Dict[str, Any]):
        """Emit the final result"""
        await self.updates_queue.put({"type": "result", "data": result})
        if self.event_sink:
            await self.event_sink.on_result(result)
        logger.info("Emitted final result")

    async def emit_error(self, error: str):
        """Emit an error"""
        await self.updates_queue.put({"type": "error", "error": error})
        if self.event_sink:
            await self.event_sink.on_error(error)
        logger.error(f"Emitted error: {error}")

    async def stream_updates(self) -> AsyncGenerator[str, None]:
        """Generate SSE-formatted updates"""
        try:
            while True:
                # Wait for the next update with a timeout
                try:
                    update = await asyncio.wait_for(self.updates_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Send a heartbeat to keep connection alive
                    yield "event: heartbeat\ndata: {}\n\n"
                    continue

                if isinstance(update, ProgressUpdate):
                    # Progress update
                    try:
                        event_data = update.model_dump_json()
                        logger.info(f"Streaming progress event: {update.step} - {update.progress}%")
                        yield f"event: progress\ndata: {event_data}\n\n"
                    except Exception as e:
                        logger.error(f"Error serializing progress update: {e}")
                        yield f"event: error\ndata: {json.dumps({'error': 'Failed to serialize progress update'})}\n\n"
                        break
                elif isinstance(update, dict):
                    if update.get("type") == "result":
                        # Final result
                        try:
                            yield f"event: result\ndata: {json.dumps(update['data'])}\n\n"
                        except Exception as e:
                            logger.error(f"Error serializing final result: {e}")
                            yield f"event: error\ndata: {json.dumps({'error': 'Failed to serialize final result'})}\n\n"
                        break
                    elif update.get("type") == "error":
                        # Error
                        try:
                            yield f"event: error\ndata: {json.dumps({'error': update['error']})}\n\n"
                        except Exception as e:
                            logger.error(f"Error serializing error message: {e}")
                            yield f"event: error\ndata: {json.dumps({'error': 'Failed to serialize error message'})}\n\n"
                        break

        except Exception as e:
            logger.error(f"Unexpected error in stream_updates: {e}")
            # Send error and close
            try:
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            except Exception:
                # Last resort - send a simple error message
                yield 'event: error\ndata: {"error": "Internal streaming error"}\n\n'


def format_sse_message(event: str, data: str) -> str:
    """Format a message for Server-Sent Events"""
    return f"event: {event}\ndata: {data}\n\n"
