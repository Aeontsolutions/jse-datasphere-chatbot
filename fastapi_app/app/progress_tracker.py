import json
import asyncio
from datetime import datetime
from typing import AsyncGenerator, Optional, Dict, Any
from app.models import ProgressUpdate

class ProgressTracker:
    """
    Utility class for tracking and streaming progress updates via Server-Sent Events
    """
    
    def __init__(self):
        self.current_step = ""
        self.current_progress = 0.0
        self.updates_queue = asyncio.Queue()
        
    async def emit_progress(
        self, 
        step: str, 
        message: str, 
        progress: float, 
        details: Optional[Dict[str, Any]] = None
    ):
        """Emit a progress update"""
        self.current_step = step
        self.current_progress = progress
        
        update = ProgressUpdate(
            step=step,
            message=message,
            progress=progress,
            timestamp=datetime.utcnow().isoformat() + "Z",
            details=details
        )
        
        await self.updates_queue.put(update)
    
    async def emit_final_result(self, result: Dict[str, Any]):
        """Emit the final result"""
        await self.updates_queue.put({"type": "result", "data": result})
    
    async def emit_error(self, error: str):
        """Emit an error"""
        await self.updates_queue.put({"type": "error", "error": error})
    
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
                    event_data = update.model_dump_json()
                    yield f"event: progress\ndata: {event_data}\n\n"
                elif isinstance(update, dict):
                    if update.get("type") == "result":
                        # Final result
                        yield f"event: result\ndata: {json.dumps(update['data'])}\n\n"
                        break
                    elif update.get("type") == "error":
                        # Error
                        yield f"event: error\ndata: {json.dumps({'error': update['error']})}\n\n"
                        break
                
        except Exception as e:
            # Send error and close
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

def format_sse_message(event: str, data: str) -> str:
    """Format a message for Server-Sent Events"""
    return f"event: {event}\ndata: {data}\n\n" 