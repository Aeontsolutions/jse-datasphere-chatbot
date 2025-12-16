import asyncio
import json
from datetime import datetime
from typing import Any, Dict, Optional

import redis.asyncio as redis

from app.job_store import JobRecord, JobStore
from app.logging_config import get_logger
from app.models import JobStatus, JobStatusResponse, ProgressUpdate

logger = get_logger(__name__)


class RedisJobStore(JobStore):
    """
    Redis-backed job registry with TTL-based cleanup.
    Replaces the in-memory JobStore for multi-instance support.
    """

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 900,
        max_progress_history: int = 50,
    ) -> None:
        # Use decode_responses=True to get strings instead of bytes
        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._ttl = ttl_seconds
        self._max_progress_history = max_progress_history
        # We don't need a lock for Redis as it's atomic, but we keep the interface consistent
        self._lock = asyncio.Lock()

    async def close(self):
        """Close the Redis connection."""
        await self._redis.close()

    def _serialize_record(self, record: JobRecord) -> str:
        """Convert JobRecord to JSON string."""
        data = {
            "job_id": record.job_id,
            "job_type": record.job_type,
            "status": record.status.value,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
            "expires_at": record.expires_at.isoformat(),
            "request_payload": record.request_payload,
            "progress": [p.model_dump() for p in record.progress],
            "result": record.result,
            "error": record.error,
        }
        return json.dumps(data)

    def _deserialize_record(self, data: str) -> JobRecord:
        """Convert JSON string back to JobRecord."""
        d = json.loads(data)
        return JobRecord(
            job_id=d["job_id"],
            job_type=d["job_type"],
            status=JobStatus(d["status"]),
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
            expires_at=datetime.fromisoformat(d["expires_at"]),
            request_payload=d.get("request_payload"),
            progress=[ProgressUpdate(**p) for p in d.get("progress", [])],
            result=d.get("result"),
            error=d.get("error"),
        )

    async def create_job(
        self,
        job_type: str,
        request_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        # Re-implementation to avoid using parent's memory store
        # We generate ID and save directly to Redis
        import uuid
        from datetime import timedelta

        job_id = uuid.uuid4().hex
        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=self._ttl)

        record = JobRecord(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.queued,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
            request_payload=request_payload,
        )

        await self._save_record(record)
        return job_id

    async def _save_record(self, record: JobRecord) -> None:
        """Save record to Redis with TTL."""
        key = f"job:{record.job_id}"
        data = self._serialize_record(record)
        # Set the key with the configured TTL
        await self._redis.setex(key, self._ttl, data)

    async def _get_record(self, job_id: str) -> Optional[JobRecord]:
        """Retrieve record from Redis."""
        key = f"job:{job_id}"
        data = await self._redis.get(key)
        if not data:
            return None
        return self._deserialize_record(data)

    async def mark_running(self, job_id: str) -> None:
        record = await self._get_record(job_id)
        if not record:
            return
        record.status = JobStatus.running
        record.updated_at = datetime.utcnow()
        await self._save_record(record)

    async def record_progress(self, job_id: str, update: ProgressUpdate) -> None:
        record = await self._get_record(job_id)
        if not record:
            return
        record.progress.append(update)
        if len(record.progress) > self._max_progress_history:
            record.progress = record.progress[-self._max_progress_history :]
        record.updated_at = datetime.utcnow()
        await self._save_record(record)

    async def complete_job(self, job_id: str, result: Dict[str, Any]) -> None:
        record = await self._get_record(job_id)
        if not record:
            return
        record.status = JobStatus.succeeded
        record.result = result
        record.error = None
        record.updated_at = datetime.utcnow()
        await self._save_record(record)

    async def fail_job(self, job_id: str, error: str) -> None:
        record = await self._get_record(job_id)
        if not record:
            return
        record.status = JobStatus.failed
        record.error = error
        record.updated_at = datetime.utcnow()
        await self._save_record(record)

    async def get_job_status(self, job_id: str) -> Optional[JobStatusResponse]:
        record = await self._get_record(job_id)
        if not record:
            return None
        return self._record_to_response(record)

    async def prune_job(self, job_id: str) -> None:
        key = f"job:{job_id}"
        await self._redis.delete(key)
