from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from app.models import ProgressUpdate, JobStatus, JobStatusResponse


@dataclass
class JobRecord:
    job_id: str
    job_type: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    request_payload: Optional[Dict[str, Any]] = None
    progress: List[ProgressUpdate] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class JobProgressSink:
    """Bridges progress tracker events into the in-memory job store."""

    def __init__(self, job_store: "JobStore", job_id: str):
        self._job_store = job_store
        self._job_id = job_id

    async def on_progress(self, update: ProgressUpdate) -> None:
        await self._job_store.record_progress(self._job_id, update)

    async def on_result(self, result: Dict[str, Any]) -> None:
        await self._job_store.complete_job(self._job_id, result)

    async def on_error(self, error: str) -> None:
        await self._job_store.fail_job(self._job_id, error)


class JobStore:
    """Simple in-memory job registry with TTL-based cleanup."""

    def __init__(
        self,
        ttl_seconds: int = 900,
        max_progress_history: int = 50,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_progress_history = max_progress_history
        self._lock = asyncio.Lock()
        self._jobs: Dict[str, JobRecord] = {}

    async def create_job(
        self,
        job_type: str,
        request_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
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
        async with self._lock:
            self._cleanup_locked(now)
            self._jobs[job_id] = record
        return job_id

    async def mark_running(self, job_id: str) -> None:
        async with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return
            record.status = JobStatus.running
            record.updated_at = datetime.utcnow()

    async def record_progress(self, job_id: str, update: ProgressUpdate) -> None:
        async with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return
            record.progress.append(update)
            if len(record.progress) > self._max_progress_history:
                record.progress = record.progress[-self._max_progress_history :]
            record.updated_at = datetime.utcnow()

    async def complete_job(self, job_id: str, result: Dict[str, Any]) -> None:
        async with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return
            record.status = JobStatus.succeeded
            record.result = result
            record.error = None
            record.updated_at = datetime.utcnow()

    async def fail_job(self, job_id: str, error: str) -> None:
        async with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return
            record.status = JobStatus.failed
            record.error = error
            record.updated_at = datetime.utcnow()

    async def get_job_status(self, job_id: str) -> Optional[JobStatusResponse]:
        async with self._lock:
            self._cleanup_locked()
            record = self._jobs.get(job_id)
            if not record:
                return None
            return self._record_to_response(record)

    async def prune_job(self, job_id: str) -> None:
        async with self._lock:
            self._jobs.pop(job_id, None)

    def _cleanup_locked(self, now: Optional[datetime] = None) -> None:
        now = now or datetime.utcnow()
        expired_keys = [job_id for job_id, rec in self._jobs.items() if rec.expires_at < now]
        for job_id in expired_keys:
            self._jobs.pop(job_id, None)

    def _record_to_response(self, record: JobRecord) -> JobStatusResponse:
        latest_progress = record.progress[-1] if record.progress else None
        return JobStatusResponse(
            job_id=record.job_id,
            status=record.status,
            job_type=record.job_type,
            created_at=record.created_at.isoformat() + "Z",
            updated_at=record.updated_at.isoformat() + "Z",
            progress=record.progress,
            latest_progress=latest_progress,
            result=record.result,
            error=record.error,
        )
