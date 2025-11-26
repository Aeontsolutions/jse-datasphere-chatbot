# Chat Stream Polling Implementation Guide

## Overview

The chat stream endpoints support two modes of operation:
1. **SSE Streaming Mode** (when `ASYNC_JOB_MODE=false`) - Server-Sent Events with real-time streaming
2. **Polling Mode** (when `ASYNC_JOB_MODE=true`) - Asynchronous job-based polling (Recommended for Production)

This guide focuses on the **Polling Mode** implementation.

---

## Architecture

### Backend Behavior

When `ASYNC_JOB_MODE=true` is set in the environment:

1. **Request**: The server receives a request (e.g., `/chat/stream`).
2. **Job Creation**: A job record is created in **Redis**.
3. **Response**: The server returns HTTP 202 (Accepted) immediately with a `job_id`.
4. **Processing**: The job is processed in the background by a worker.
5. **State Sharing**: Since Redis is used, **any server instance** can retrieve the job status.

### Advantages of Redis Architecture
- **Stateless API**: No need for sticky sessions or server affinity.
- **Scalability**: Works correctly with multiple backend instances (horizontal scaling).
- **Resilience**: Job state survives application restarts (until TTL expires).

### Affected Endpoints

All streaming endpoints support this mode:
- `POST /chat/stream` - Deep research chat with document loading
- `POST /fast_chat/stream` - Vector database chat
- `POST /fast_chat_v2/stream` - Financial data queries

---

## Request/Response Flow

### Step 1: Initial Request

**Client sends POST request** to a streaming endpoint:

```typescript
const response = await fetch(`${baseUrl}/chat/stream`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  // credentials: 'include' is optional with Redis, but good for auth cookies
  credentials: 'include', 
  body: JSON.stringify({
    query: "What are the main financial highlights for MTN Group?",
    auto_load_documents: true,
    memory_enabled: true
  })
});
```

### Step 2: Detect Response Type

**Check the Content-Type** to determine if server is in polling mode:

```typescript
const contentType = response.headers.get('content-type') || '';

if (contentType.includes('text/event-stream')) {
  // SSE Streaming mode
  handleSSEStream(response);
} else if (contentType.includes('application/json')) {
  // Polling mode
  handleJobPolling(response);
}
```

### Step 3: Parse Job Response

**Server returns** HTTP 202 with job payload:

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "queued",
  "job_type": "chat_stream",
  "polling_url": "/jobs/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

**Client extracts** job details:

```typescript
const jobPayload = await response.json();
const pollingUrl = jobPayload.polling_url.startsWith('http') 
  ? jobPayload.polling_url 
  : `${baseUrl}${jobPayload.polling_url}`;
```

### Step 4: Start Polling

**Client polls** the status endpoint at regular intervals:

```typescript
const JOB_POLL_INTERVAL_MS = 1000; // Poll every 1 second

const pollInterval = setInterval(async () => {
  try {
    const statusResponse = await fetch(pollingUrl);
    
    if (statusResponse.status === 404) {
        console.error('Job not found (expired or invalid ID)');
        stopPolling();
        return;
    }
    
    const jobStatus = await statusResponse.json();
    handleJobStatus(jobStatus);
  } catch (error) {
    console.error('Polling error:', error);
  }
}, JOB_POLL_INTERVAL_MS);
```

### Step 5: Handle Job Status Updates

**Server returns** job status with progress:

```json
{
  "job_id": "...",
  "status": "running",
  "progress": [ ... ],
  "latest_progress": {
    "step": "generating_response",
    "message": "Generating AI response...",
    "progress": 85,
    "timestamp": "2025-11-25T10:30:05.000Z"
  },
  "result": null
}
```

### Step 6: Display Final Result

**When job succeeds**, the result contains the chat response:

```json
{
  "job_id": "...",
  "status": "succeeded",
  "result": {
    "response": "Based on the documents...",
    "documents_loaded": ["report.pdf"],
    ...
  }
}
```

---

## Job Status States

| Status | Description | Action |
|--------|-------------|--------|
| `queued` | Job created, waiting to start | Continue polling |
| `running` | Job is being processed | Continue polling |
| `succeeded` | Job completed successfully | Stop polling, show result |
| `failed` | Job encountered an error | Stop polling, show error |

---

## Configuration

### Environment Variables (Backend)

```bash
# .env file
ASYNC_JOB_MODE=true          # Enable polling mode
REDIS_URL=redis://...        # Redis connection string (Required for production)
ASYNC_JOB_TTL_SECONDS=900    # Job expiration (15 minutes)
```

### Frontend Configuration

No special configuration is needed for the frontend other than supporting the JSON polling flow.

---

## Troubleshooting

### Issue: Job Not Found (404)
**Cause**: The job ID does not exist in Redis.
- **Expired**: The job creation time exceeded `ASYNC_JOB_TTL_SECONDS` (default 15 mins).
- **Redis Flushed**: The Redis cache was cleared.
- **Wrong Environment**: Client pointing to a different environment than where job was created (rare with Redis).

### Issue: Polling never stops
**Cause**: Client logic failing to handle terminal states.
**Solution**: Ensure your client code checks for `status === 'succeeded'` and `status === 'failed'`.

---

## Summary

With the **Redis-backed Job Store** now implemented:
1.  **Sticky Sessions are NOT required** for job polling to work (though may be used for other reasons).
2.  The system is **robust** against container restarts and scaling events.
3.  The frontend implementation remains standard JSON polling.
