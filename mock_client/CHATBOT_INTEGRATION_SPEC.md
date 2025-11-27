# Chatbot Integration Specification

## Overview

This document specifies the integration contract between the JSE Analytics Platform and the external Deep Research Chatbot service. The integration supports two response patterns:
1. **SSE Streaming** - Real-time Server-Sent Events
2. **Async Job Polling** - Long-running task pattern with polling

## Platform Architecture

### Request Flow
```
Frontend (Svelte)
    ↓ Fetch API + SSE
POST /analytics/chat/stream (Django View)
    ↓ Threading + Queue
streaming_chat_service.py
    ↓ requests.Session (maintains cookies)
External Chatbot Service
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Frontend Service | `frontend/src/services/streamingService.ts` | Initiates SSE connection, parses events |
| Django View | `analytics/views.py:938` (deep_chat_stream_view) | Proxies requests, saves responses |
| Streaming Service | `analytics/services/streaming_chat_service.py` | Handles both SSE and job polling |
| State Management | `frontend/src/stores/streamingStore.ts` | Reactive UI state |

---

## API Contract

### 1. Chatbot Endpoint Configuration

#### Environment Variables
```bash
# Primary endpoint (must end with /stream for SSE)
CHATBOT_DEEP_STREAM_URL=https://your-chatbot.com/api/v1/stream

# Alternative (will append /stream if missing)
CHATBOT_DEEP_URL=https://your-chatbot.com/api/v1

# Polling configuration
CHATBOT_POLL_INTERVAL=2.0  # seconds between polls (default: 2.0)
MAX_POLL_TIMEOUT_SECONDS=300  # 5 minutes max
```

### 2. Request Format

#### Endpoint
```
POST /api/v1/stream
Content-Type: application/json
```

#### Request Payload
```json
{
  "query": "What are the key financial metrics for company XYZ?",
  "conversation_history": [
    {
      "role": "user",
      "content": "Previous question"
    },
    {
      "role": "assistant",
      "content": "Previous answer"
    }
  ],
  "auto_load_documents": true,
  "memory_enabled": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | User's question |
| `conversation_history` | array | No | Previous Q&A pairs for context |
| `auto_load_documents` | boolean | No | Enable automatic document retrieval (default: true) |
| `memory_enabled` | boolean | No | Enable conversation memory (default: true) |

---

## Response Patterns

### Pattern 1: SSE Streaming (Real-time)

#### Response Headers
```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
Set-Cookie: session_id=abc123; Path=/; HttpOnly  # CRITICAL for sticky sessions
```

#### SSE Event Format
```
event: progress
data: {"step": "loading_documents", "message": "Loading documents...", "progress": 25, "details": {...}}

event: progress
data: {"step": "analyzing", "message": "Analyzing data...", "progress": 50, "details": {...}}

event: result
data: {"response": "The key financial metrics are...", "sources": [...], "metadata": {...}}

event: complete
data: {"progress": 100, "message": "Analysis complete", "step": "complete"}
```

#### Event Types

##### `progress` Event
Indicates ongoing processing step.

```json
{
  "step": "loading_documents",
  "message": "Loading 5 documents...",
  "progress": 25,
  "details": {
    "documents_loaded": 3,
    "total_documents": 5
  }
}
```

**Fields:**
- `step` (string): Machine-readable step identifier (e.g., `loading_documents`, `analyzing`, `generating_response`)
- `message` (string): Human-readable progress message
- `progress` (number): 0-100 percentage
- `details` (object, optional): Step-specific metadata

##### `result` Event
Contains the final response. **CRITICAL: This event MUST contain the response text.**

```json
{
  "response": "Based on the analysis of financial documents...",
  "sources": [
    {
      "title": "Annual Report 2023",
      "url": "https://...",
      "relevance": 0.95
    }
  ],
  "metadata": {
    "processing_time": 12.5,
    "documents_analyzed": 5
  }
}
```

**Fields:**
- `response` (string, **REQUIRED**): The chatbot's answer
- `sources` (array, optional): Source documents used
- `metadata` (object, optional): Additional response metadata

##### `complete` Event
Status update indicating completion (does NOT contain response).

```json
{
  "progress": 100,
  "message": "Analysis complete",
  "step": "complete"
}
```

##### `error` Event
Indicates failure.

```json
{
  "error": "Failed to load documents: Timeout after 30s"
}
```

##### `heartbeat` Event
Keep-alive event (no data required).

```json
{}
```

**Purpose:** Prevents proxy/browser timeout. Sent every 120 seconds by platform if no events received.

---

### Pattern 2: Async Job Polling

For long-running tasks, the chatbot can return a job ID immediately and be polled for status.

#### Initial Response Headers
```
Content-Type: application/json
Set-Cookie: session_id=abc123; Path=/; HttpOnly  # CRITICAL - same session for polling
```

#### Initial Response Body
```json
{
  "job_id": "job_abc123xyz",
  "status": "pending",
  "polling_url": "/jobs/job_abc123xyz"
}
```

**Fields:**
- `job_id` (string, **REQUIRED**): Unique job identifier
- `status` (string): Initial status (e.g., `pending`, `running`)
- `polling_url` (string, optional): Relative or absolute polling URL

#### Polling Request

The platform constructs the polling URL:
```
GET /jobs/job_abc123xyz
Cookie: session_id=abc123  # Uses same session from initial request
```

**CRITICAL:** The platform uses `requests.Session()` to maintain cookies across all requests.

#### Polling Response Format

```json
{
  "job_id": "job_abc123xyz",
  "status": "running",
  "latest_progress": {
    "step": "analyzing",
    "message": "Analyzing financial metrics...",
    "progress": 60,
    "details": {
      "metrics_analyzed": 12,
      "total_metrics": 20
    }
  },
  "result": null
}
```

##### Terminal State: Success
```json
{
  "job_id": "job_abc123xyz",
  "status": "succeeded",
  "latest_progress": {
    "step": "complete",
    "message": "Analysis complete",
    "progress": 100
  },
  "result": {
    "response": "The key financial metrics are...",
    "sources": [...],
    "metadata": {...}
  }
}
```

##### Terminal State: Failure
```json
{
  "job_id": "job_abc123xyz",
  "status": "failed",
  "error": "Document retrieval failed: Connection timeout"
}
```

#### Job Status Values

| Status | Description | Terminal | Next Action |
|--------|-------------|----------|-------------|
| `pending` | Queued, not started | No | Continue polling |
| `running` | In progress | No | Continue polling |
| `succeeded` | Completed successfully | Yes | Extract `result` |
| `completed` | Alternative success status | Yes | Extract `result` |
| `failed` | Error occurred | Yes | Show error message |

---

## Session Management (CRITICAL)

### Why Sessions Matter
The chatbot service uses **sticky sessions** to route requests to the same backend worker. Without proper session management, polling requests may go to different workers that don't have the job state.

### Platform Implementation

```python
# streaming_chat_service.py:39
session = requests.Session()

# Initial request sets cookies
with session.post(chatbot_stream_url, ...) as resp:
    if resp.cookies:
        logger.info(f"✅ Cookies received: {dict(resp.cookies)}")
    # Session automatically stores cookies

# Polling requests reuse same session
resp = session.get(status_url, timeout=30)
logger.info(f"🔑 Sending cookies: {dict(session.cookies)}")
```

### Chatbot Requirements

1. **Set session cookie in initial response:**
   ```
   Set-Cookie: session_id=unique_value; Path=/; HttpOnly; Secure
   ```

2. **Validate session cookie in polling requests:**
   ```python
   session_id = request.cookies.get('session_id')
   if not session_id:
       return {"error": "Session expired"}, 401
   ```

3. **Use session to route to same worker:**
   - Sticky sessions via load balancer
   - Session-based job storage (Redis, in-memory cache)

### Debugging Sessions

Platform logs all cookie operations:
```
✅ Cookies received from chatbot: {'session_id': 'abc123'}
✅ Session cookies now contain: {'session_id': 'abc123'}
🔑 Sending cookies with polling request: {'session_id': 'abc123'}
📥 Polling response status: 200
```

If you see:
```
⚠️ NO COOKIES received from chatbot - sticky sessions won't work!
❌ Error polling job status: 404 Not Found
```

**Problem:** Chatbot didn't set session cookie, or polling request went to different worker.

---

## Event Transformation (Job Polling → SSE)

The platform transforms job polling responses to SSE events for unified frontend handling.

### Transformation Logic

```python
# _transform_job_status_to_events() in streaming_chat_service.py:220

# Progress event from latest_progress
if latest_progress:
    yield 'progress', json.dumps({
        'step': latest_progress.get('step'),
        'message': latest_progress.get('message'),
        'progress': latest_progress.get('progress'),
        'details': latest_progress.get('details')
    })

# Result event when job succeeds
if status == 'succeeded':
    yield 'result', json.dumps(job_status.get('result'))

# Error event when job fails
if status == 'failed':
    yield 'error', json.dumps({"error": job_status.get('error')})
```

This allows the frontend to consume both patterns identically.

---

## Frontend Integration

### Request Initiation

```typescript
// streamingService.ts:16
async *startDeepResearchStream(requestData) {
  const url = `${this.baseUrl}/analytics/chat/stream`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': this.getCSRFToken(),
    },
    body: JSON.stringify({
      query: requestData.query,
      conversation_history: requestData.conversation_history || [],
      auto_load_documents: requestData.auto_load_documents ?? true,
      memory_enabled: requestData.memory_enabled ?? true,
      chat_session_id: requestData.chat_session_id
    })
  });

  yield* this.processStream(response);
}
```

### SSE Processing

```typescript
// streamingService.ts:46
private async *processStream(response: Response) {
  const reader = response.body?.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let currentEvent = 'message';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.substring(7);
      } else if (line.startsWith('data: ')) {
        const data = line.substring(6);
        const parsed = JSON.parse(data);
        yield { event: currentEvent, data: parsed };
      }
    }
  }
}
```

### Event Handling

```typescript
// Assistant.svelte:134
for await (const {event, data} of streamingService.startDeepResearchStream(...)) {
  if (event === 'progress') {
    chatContext.streamingActions.updateProgress(
      data.progress,
      data.message,
      data.step,
      data.details
    );
  } else if (event === 'result') {
    const responseText = data.response;
    chatContext.streamingActions.setResponse(responseText, data);
    selectedChatMessages.push({
      text: responseText,
      message_type: "Answer",
    });
    break;
  } else if (event === 'error') {
    chatContext.streamingActions.setError(data.error);
    break;
  }
}
```

---

## Error Handling

### Platform Error Handling

| Scenario | Platform Response |
|----------|-------------------|
| Cannot connect to chatbot | `event: error` with "Failed to connect to external chatbot service." |
| Unexpected content-type | `event: error` with "Unexpected response type: {type}" |
| Job polling timeout | `event: error` with "Job polling timeout exceeded" (after 5 minutes) |
| SSE stream interrupted | `event: error` with "External chatbot connection ended prematurely." |
| Missing job_id | `event: error` with "Invalid job response: missing job_id" |

### Expected Chatbot Error Responses

#### SSE Stream Error
```
event: error
data: {"error": "Document retrieval failed: S3 connection timeout"}
```

#### Job Polling Error
```json
{
  "job_id": "job_abc123",
  "status": "failed",
  "error": "Query processing failed: OpenAI API rate limit exceeded"
}
```

### Frontend Error Display

```typescript
// Assistant.svelte:174
if (event === 'error') {
  chatContext.streamingActions.setError(data.error);
  // Stops streaming, displays error to user
}
```

---

## Performance Characteristics

### Timeouts

| Component | Timeout | Configurable |
|-----------|---------|--------------|
| Initial connection | 120s | No (hardcoded) |
| Polling interval | 2s | Yes (`CHATBOT_POLL_INTERVAL`) |
| Max polling duration | 300s | Yes (`MAX_POLL_TIMEOUT_SECONDS`) |
| Polling request timeout | 30s | No |
| Heartbeat interval | 120s | No |

### Resource Usage

- **Memory:** Queue size capped at 100 events (`queue.Queue(maxsize=100)`)
- **Threads:** 1 background thread per active chat session
- **Connections:** Persistent `requests.Session` per request (closed after completion)

### Backpressure Handling

```python
# views.py:972
try:
    q.put((event, data), timeout=1)
except queue.Full:
    logger.warning("SSE queue is full; dropping event to avoid backpressure")
    continue
```

If chatbot sends events faster than Django can stream, events are dropped with warning.

---

## Testing Checklist

### SSE Streaming Tests

- [ ] Happy path: Query → Progress events → Result event → Complete
- [ ] Heartbeat handling (no events for 120s)
- [ ] Progress with detailed metadata
- [ ] Multiple progress updates
- [ ] Error during processing
- [ ] Stream interruption (connection drop)
- [ ] Empty event name (should default to 'message')
- [ ] Non-JSON data payload
- [ ] Cookie setting and validation

### Job Polling Tests

- [ ] Happy path: Job created → Polling → Success with result
- [ ] Job failure with error message
- [ ] Polling timeout (5+ minutes)
- [ ] Session cookie persistence across polls
- [ ] Missing job_id in initial response
- [ ] Invalid polling URL
- [ ] Job status: pending → running → succeeded
- [ ] Progress updates during polling
- [ ] 404 on polling (job not found / wrong worker)

### Integration Tests

- [ ] Frontend parses all event types correctly
- [ ] Chat message saved to database on result
- [ ] Conversation history passed correctly
- [ ] CSRF token validation
- [ ] Concurrent chat sessions
- [ ] Session persistence across page refresh

---

## Common Issues & Solutions

### Issue: Polling returns 404 "Job not found"

**Cause:** Request routed to different worker without job state.

**Solution:**
1. Ensure chatbot sets `Set-Cookie` header in initial response
2. Verify session cookie included in polling requests
3. Configure load balancer for sticky sessions
4. Use shared job storage (Redis, database)

### Issue: No progress updates shown

**Cause:** Missing or malformed `latest_progress` in job status.

**Solution:** Include valid progress object in every polling response:
```json
{
  "latest_progress": {
    "step": "current_step",
    "message": "What's happening",
    "progress": 0-100
  }
}
```

### Issue: Response not displayed in UI

**Cause:** `result` event missing `response` field.

**Solution:** Ensure result event contains:
```json
{
  "response": "The actual answer text here"
}
```

### Issue: Stream timeout after 2 minutes

**Cause:** No events sent, no heartbeat.

**Solution:** Send heartbeat every 60-90 seconds:
```
event: heartbeat
data: {}

```

---

## Reference Implementation

See complete working implementation:

- **Backend Service:** [analytics/services/streaming_chat_service.py](../analytics/services/streaming_chat_service.py)
- **Django View:** [analytics/views.py:938](../analytics/views.py#L938) (`deep_chat_stream_view`)
- **Frontend Service:** [frontend/src/services/streamingService.ts](../frontend/src/services/streamingService.ts)
- **UI Component:** [frontend/src/pages/Assistant/Assistant.svelte](../frontend/src/pages/Assistant/Assistant.svelte)
- **State Management:** [frontend/src/stores/streamingStore.ts](../frontend/src/stores/streamingStore.ts)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-26 | Initial specification based on current implementation |
