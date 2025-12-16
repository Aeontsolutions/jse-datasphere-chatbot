# Chatbot Integration Test Scenarios

This document provides comprehensive test scenarios for validating the integration between the JSE Analytics Platform and the external Deep Research Chatbot service.

## Prerequisites

1. **Install dependencies:**
   ```bash
   pip install requests flask
   ```

2. **Start mock server:**
   ```bash
   python docs/mock_server.py
   ```

3. **Verify server is running:**
   ```bash
   curl http://localhost:5000/health
   ```

---

## Test Scenario Categories

- [Basic Connectivity](#basic-connectivity)
- [SSE Streaming Mode](#sse-streaming-mode)
- [Job Polling Mode](#job-polling-mode)
- [Session Management](#session-management)
- [Error Handling](#error-handling)
- [Performance & Timeouts](#performance--timeouts)
- [Integration Tests](#integration-tests)

---

## Basic Connectivity

### TC-001: Health Check
**Purpose:** Verify server is accessible

```bash
curl http://localhost:5000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-26T12:00:00.000000",
  "active_jobs": 0
}
```

**Pass Criteria:** 200 OK status, `status: "healthy"`

---

### TC-002: Endpoint Availability
**Purpose:** Verify streaming endpoint exists

```bash
curl -X POST http://localhost:5000/api/v1/stream \
     -H "Content-Type: application/json" \
     -d '{"query": "test"}'
```

**Expected Response:** SSE stream starts (not 404)

**Pass Criteria:** Response begins with `event:` or `data:`

---

## SSE Streaming Mode

### TC-101: Basic SSE Stream
**Purpose:** Verify complete SSE streaming flow

```bash
python docs/mock_client.py \
  --url http://localhost:5000/api/v1/stream \
  --query "What are the key financial metrics?"
```

**Expected Events (in order):**
1. `progress` - initializing (10%)
2. `progress` - loading_documents (25%)
3. `progress` - analyzing (50%)
4. `progress` - generating_response (75%)
5. `result` - final response with sources
6. `complete` - 100%

**Pass Criteria:**
- All 6 events received
- `result` event contains `response` field
- Final progress is 100%
- No errors

**Sample Output:**
```
[PROGRESS]
{
  "step": "analyzing",
  "message": "Analyzing financial data...",
  "progress": 50,
  "details": {
    "metrics_analyzed": 15,
    "total_metrics": 30
  }
}

[RESULT]
{
  "response": "Based on my analysis...",
  "sources": [...],
  "metadata": {...}
}
```

---

### TC-102: SSE with Conversation History
**Purpose:** Verify conversation context handling

**Step 1:** Create history file `history.json`:
```json
[
  {
    "role": "user",
    "content": "What is the stock market?"
  },
  {
    "role": "assistant",
    "content": "The stock market is..."
  }
]
```

**Step 2:** Send query with history:
```bash
python docs/mock_client.py \
  --url http://localhost:5000/api/v1/stream \
  --query "What about the JSE specifically?" \
  --history history.json
```

**Pass Criteria:**
- Server logs show: "Conversation history: 2 messages"
- Response acknowledges context (mock server mentions history length)

---

### TC-103: SSE Progress Details
**Purpose:** Verify detailed progress metadata

```bash
python docs/mock_client.py \
  --url http://localhost:5000/api/v1/stream \
  --query "Analyze company XYZ" \
  --verbose
```

**Expected:** Each progress event includes `details` field:
```json
{
  "step": "loading_documents",
  "message": "Loading relevant documents...",
  "progress": 25,
  "details": {
    "documents_found": 12,
    "sources": ["Annual Report 2023", "Q3 Financial Statement"]
  }
}
```

**Pass Criteria:**
- At least one progress event has non-null `details`
- Details contain meaningful metadata

---

### TC-104: SSE Event Ordering
**Purpose:** Verify events arrive in correct sequence

```bash
python docs/mock_client.py \
  --url http://localhost:5000/api/v1/stream \
  --query "Test ordering" \
  --save-output output.json
```

**Step 2:** Validate order:
```python
import json
with open('output.json') as f:
    events = json.load(f)

# Progress values should be monotonically increasing
progress_events = [e for e in events if e['event'] == 'progress']
progress_values = [json.loads(e['data'])['progress'] for e in progress_events]
assert progress_values == sorted(progress_values), "Progress not monotonic!"

# Result should come before complete
result_idx = next(i for i, e in enumerate(events) if e['event'] == 'result')
complete_idx = next(i for i, e in enumerate(events) if e['event'] == 'complete')
assert result_idx < complete_idx, "Result must come before complete!"
```

**Pass Criteria:** All assertions pass

---

## Job Polling Mode

### TC-201: Basic Job Polling
**Purpose:** Verify async job creation and polling

```bash
python docs/mock_client.py \
  --url "http://localhost:5000/api/v1/stream?mode=job" \
  --query "Deep analysis request" \
  --expect-job
```

**Expected Flow:**
1. Initial response: `{"job_id": "job_...", "status": "pending", "polling_url": "/jobs/..."}`
2. Poll 1: `status: "running"`, progress: 25%
3. Poll 2: `status: "running"`, progress: 50%
4. Poll 3: `status: "running"`, progress: 75%
5. Poll 4: `status: "succeeded"`, progress: 100%, result present

**Pass Criteria:**
- Job ID returned
- Status progresses: pending → running → succeeded
- Final result contains response text
- Client logs show cookie management

---

### TC-202: Job Status Progression
**Purpose:** Verify job state transitions

**Step 1:** Create job:
```bash
curl -X POST "http://localhost:5000/api/v1/stream?mode=job" \
     -H "Content-Type: application/json" \
     -d '{"query": "test"}' \
     -c cookies.txt
```

**Step 2:** Poll manually (3 times):
```bash
# Extract job_id from response, then:
JOB_ID="job_abc123..."

curl -b cookies.txt http://localhost:5000/jobs/$JOB_ID
sleep 2
curl -b cookies.txt http://localhost:5000/jobs/$JOB_ID
sleep 2
curl -b cookies.txt http://localhost:5000/jobs/$JOB_ID
```

**Expected:**
- Poll 1: `status: "running"`, `progress: 25`
- Poll 2: `status: "running"`, `progress: 50`
- Poll 3: `status: "succeeded"`, `result` field present

**Pass Criteria:** Status and progress advance with each poll

---

### TC-203: Job Polling URL Construction
**Purpose:** Verify polling URL handling

**Test Case A - Relative URL:**
```json
{
  "job_id": "job_123",
  "polling_url": "/jobs/job_123"
}
```
**Expected:** Client constructs `http://localhost:5000/jobs/job_123`

**Test Case B - Absolute URL:**
```json
{
  "job_id": "job_123",
  "polling_url": "http://localhost:5000/jobs/job_123"
}
```
**Expected:** Client uses URL as-is

**Test Case C - No polling_url:**
```json
{
  "job_id": "job_123"
}
```
**Expected:** Client constructs `/jobs/job_123`

**Pass Criteria:** All URL formats work correctly

---

### TC-204: Job Completion Detection
**Purpose:** Verify client stops polling when job completes

```bash
python docs/mock_client.py \
  --url "http://localhost:5000/api/v1/stream?mode=job" \
  --query "Quick job" \
  --poll-interval 1 \
  --verbose
```

**Expected Logs:**
```
Poll attempt 1/150
Poll attempt 2/150
Poll attempt 3/150
Poll attempt 4/150
✅ Job finished with status: succeeded
```

**Pass Criteria:**
- Client stops polling after status becomes terminal
- No polls after "succeeded" status
- Fewer than 10 polls needed (mock server completes in ~4 polls)

---

## Session Management

### TC-301: Cookie Setting (SSE Mode)
**Purpose:** Verify session cookie is set in SSE response

```bash
curl -X POST http://localhost:5000/api/v1/stream \
     -H "Content-Type: application/json" \
     -d '{"query": "test"}' \
     -v 2>&1 | grep -i "set-cookie"
```

**Expected:**
```
< Set-Cookie: session_id=...; HttpOnly; Path=/; SameSite=Lax
```

**Pass Criteria:** `Set-Cookie` header present with session_id

---

### TC-302: Cookie Setting (Job Mode)
**Purpose:** Verify session cookie is set in job response

```bash
curl -X POST "http://localhost:5000/api/v1/stream?mode=job" \
     -H "Content-Type: application/json" \
     -d '{"query": "test"}' \
     -v 2>&1 | grep -i "set-cookie"
```

**Expected:**
```
< Set-Cookie: session_id=...; HttpOnly; Path=/; SameSite=Lax
```

**Pass Criteria:** `Set-Cookie` header present

---

### TC-303: Cookie Persistence Across Polls
**Purpose:** Verify client sends same cookie in all polling requests

```bash
python docs/mock_client.py \
  --url "http://localhost:5000/api/v1/stream?mode=job" \
  --query "Session test" \
  --verbose
```

**Expected Logs:**
```
✅ Cookies received: {'session_id': 'abc-123-xyz'}
✅ Session cookies: {'session_id': 'abc-123-xyz'}
...
🔑 Sending cookies: {'session_id': 'abc-123-xyz'}
📥 Poll response status: 200
```

**Pass Criteria:**
- Cookie received in initial response
- Same cookie sent in all polls
- Server logs show matching session IDs

---

### TC-304: Session Mismatch Handling
**Purpose:** Verify behavior when session cookie is missing/wrong

**Step 1:** Create job with cookie:
```bash
curl -X POST "http://localhost:5000/api/v1/stream?mode=job" \
     -H "Content-Type: application/json" \
     -d '{"query": "test"}' \
     -c cookies.txt
```

**Step 2:** Poll WITHOUT cookie:
```bash
JOB_ID="..." # from step 1
curl http://localhost:5000/jobs/$JOB_ID
```

**Expected:** Server logs show session mismatch warning

**Pass Criteria:**
- Request succeeds (mock server is lenient)
- Server logs: "⚠️ Session mismatch!"
- (In production chatbot, this might return 401/404)

---

## Error Handling

### TC-401: Simulated Processing Error
**Purpose:** Verify error event handling

```bash
python docs/mock_client.py \
  --url http://localhost:5000/api/v1/stream/error \
  --query "This will fail"
```

**Expected Output:**
```
[PROGRESS]
{
  "step": "initializing",
  "message": "Starting...",
  "progress": 10
}

[ERROR]
{
  "error": "Simulated processing error: Document retrieval failed"
}

SUMMARY
Total events received: 2
Result found: False
Error occurred: True
```

**Pass Criteria:**
- Error event received
- Client exits with error code (1)
- Error message displayed to user

---

### TC-402: Job Failure
**Purpose:** Verify failed job status handling

**Step 1:** Create job:
```bash
curl -X POST "http://localhost:5000/api/v1/stream?mode=job" \
     -H "Content-Type: application/json" \
     -d '{"query": "test"}' \
     -c cookies.txt \
     | jq -r '.job_id'
```

**Step 2:** Manually fail the job:
```bash
JOB_ID="..." # from step 1
curl -X POST -b cookies.txt http://localhost:5000/jobs/$JOB_ID/fail
```

**Step 3:** Poll the job:
```bash
curl -b cookies.txt http://localhost:5000/jobs/$JOB_ID
```

**Expected Response:**
```json
{
  "job_id": "job_...",
  "status": "failed",
  "error": "Manually failed for testing purposes",
  "latest_progress": {...}
}
```

**Pass Criteria:**
- Status is "failed"
- Error message present
- Client would emit error event

---

### TC-403: Invalid Request Payload
**Purpose:** Verify validation errors

```bash
curl -X POST http://localhost:5000/api/v1/stream \
     -H "Content-Type: application/json" \
     -d '{"invalid": "payload"}'
```

**Expected:** Request still succeeds (query defaults to empty string)

**Alternative - Invalid JSON:**
```bash
curl -X POST http://localhost:5000/api/v1/stream \
     -H "Content-Type: application/json" \
     -d 'not valid json'
```

**Expected:** 400 Bad Request

**Pass Criteria:** Server handles invalid input gracefully

---

### TC-404: Job Not Found
**Purpose:** Verify 404 handling for invalid job ID

```bash
curl http://localhost:5000/jobs/invalid_job_id
```

**Expected Response:**
```json
{
  "error": "Job not found"
}
```
**Status:** 404

**Pass Criteria:** 404 status, clear error message

---

## Performance & Timeouts

### TC-501: Polling Timeout
**Purpose:** Verify client respects max polling timeout

```bash
# Mock server won't complete in 5 minutes, so reduce timeout for testing
python docs/mock_client.py \
  --url "http://localhost:5000/api/v1/stream?mode=job" \
  --query "Long running job" \
  --poll-interval 10
```

**Modify mock_client.py temporarily:**
```python
MAX_POLL_TIMEOUT_SECONDS = 30  # 30 seconds instead of 300
```

**Expected Output:**
```
Poll attempt 1/3
Poll attempt 2/3
Poll attempt 3/3
❌ Job polling timeout
[ERROR] {"error": "Job polling timeout exceeded"}
```

**Pass Criteria:**
- Client stops after max polls
- Timeout error emitted

---

### TC-502: Slow Response Handling
**Purpose:** Verify handling of very slow responses

```bash
# This will take 150 seconds - test in background
python docs/mock_client.py \
  --url http://localhost:5000/api/v1/stream/timeout \
  --query "Slow response" &
```

**Expected Behavior:**
- Client waits patiently for SSE events
- May timeout after 120s (initial connection timeout)
- Platform logs "Chunked encoding error" or similar

**Pass Criteria:** Client handles gracefully (no crash)

---

### TC-503: Concurrent Requests
**Purpose:** Verify server handles multiple simultaneous requests

```bash
# Terminal 1
python docs/mock_client.py \
  --url http://localhost:5000/api/v1/stream \
  --query "Request 1" &

# Terminal 2
python docs/mock_client.py \
  --url http://localhost:5000/api/v1/stream \
  --query "Request 2" &

# Terminal 3
python docs/mock_client.py \
  --url http://localhost:5000/api/v1/stream \
  --query "Request 3" &

wait
```

**Expected:** All 3 requests complete successfully

**Pass Criteria:**
- All clients receive complete responses
- No server errors
- Responses are independent (not mixed)

---

### TC-504: Heartbeat Mechanism
**Purpose:** Verify heartbeat events prevent timeout

**Note:** Platform sends heartbeat every 120s if no events received.

**Test Setup:** Create a slow mock endpoint:
```python
@app.route('/api/v1/stream/slow', methods=['POST'])
def slow_stream():
    def generate():
        yield "event: progress\ndata: {\"progress\": 10}\n\n"
        time.sleep(130)  # Longer than heartbeat interval
        yield "event: result\ndata: {\"response\": \"Done\"}\n\n"
    return Response(generate(), mimetype='text/event-stream')
```

**Expected:** Client receives heartbeat after 120s, keeps connection alive

**Pass Criteria:** No premature timeout

---

## Integration Tests

### TC-601: Full Platform Flow (SSE)
**Purpose:** End-to-end test with platform (if available)

1. Start platform (Django + Frontend)
2. Navigate to `/analytics/assistant`
3. Select "Deep Research" mode
4. Enter query: "What are the key metrics for company XYZ?"
5. Submit

**Expected:**
- Progress indicators appear
- Progress advances through steps
- Final response displays in chat
- Response saved to database
- No console errors

**Verify in Django Logs:**
```
Deep chat stream view called
Deep chat request - Query: What are the key metrics...
Starting event stream for deep chat
✅ Cookies received from chatbot: {...}
Yielding to frontend - event: progress
Yielding to frontend - event: result
Saving Chat Message
```

**Verify in Browser Console:**
```
Frontend received chunk: event: progress...
Frontend yielding event: progress with data: {...}
Streaming store updating progress: {...}
Setting final response - event: result
```

**Pass Criteria:** Complete flow with no errors

---

### TC-602: Full Platform Flow (Job Polling)
**Purpose:** End-to-end job polling test

**Prerequisite:** Configure chatbot to return job response

**Steps:**
1. Start platform
2. Navigate to assistant
3. Enter query and submit
4. Observe polling behavior

**Expected:**
- Initial request receives job ID
- Platform polls every 2 seconds
- Progress updates in real-time
- Final result displays
- Chat message saved

**Verify Session Logs:**
```
✅ Cookies received from chatbot: {'session_id': 'abc123'}
🔑 Sending cookies with polling request: {'session_id': 'abc123'}
Polling attempt 1/150
Job status: running (50%)
```

**Pass Criteria:** Successful completion via polling

---

### TC-603: Database Persistence
**Purpose:** Verify chat messages are saved

**Steps:**
1. Send query via platform
2. Wait for response
3. Query database:
   ```sql
   SELECT * FROM analytics_chatmessages
   ORDER BY date_created DESC LIMIT 1;
   ```

**Expected:**
- Row created with query_str and response_str
- response_str matches final chatbot response
- Linked to correct chat_session

**Pass Criteria:** Database record created correctly

---

### TC-604: Frontend State Management
**Purpose:** Verify Svelte store updates correctly

**Steps:**
1. Open browser DevTools
2. Add breakpoint in `Assistant.svelte:146` (updateProgress call)
3. Submit query
4. Inspect state at each progress event

**Expected State Evolution:**
```javascript
// Initial
{isStreaming: true, progress: 0, message: "Starting...", step: "start"}

// Progress updates
{isStreaming: true, progress: 25, message: "Loading documents...", step: "loading_documents"}
{isStreaming: true, progress: 50, message: "Analyzing...", step: "analyzing"}

// Final
{isStreaming: false, progress: 100, response: "...", step: "complete"}
```

**Pass Criteria:** State transitions match event flow

---

## Test Execution Checklist

### Quick Smoke Test (5 minutes)
- [ ] TC-001: Health check
- [ ] TC-101: Basic SSE stream
- [ ] TC-201: Basic job polling
- [ ] TC-301: Cookie setting

### Core Functionality (15 minutes)
- [ ] TC-101 - TC-104: All SSE tests
- [ ] TC-201 - TC-204: All job polling tests
- [ ] TC-301 - TC-303: Session management
- [ ] TC-401: Error handling

### Comprehensive Test Suite (45 minutes)
- [ ] All Basic Connectivity tests
- [ ] All SSE Streaming tests
- [ ] All Job Polling tests
- [ ] All Session Management tests
- [ ] All Error Handling tests
- [ ] Key Performance tests (TC-501, TC-503)

### Full Integration Test (60+ minutes)
- [ ] All test categories
- [ ] All Integration tests (TC-601 - TC-604)
- [ ] Performance under load

---

## Troubleshooting Guide

### Issue: Client hangs, no output

**Diagnosis:**
```bash
curl -v http://localhost:5000/api/v1/stream ...
```
Check if response starts immediately.

**Common Causes:**
- Server not running
- Firewall blocking port 5000
- Wrong URL

---

### Issue: "Job not found" errors

**Diagnosis:** Check session cookies
```bash
python docs/mock_client.py --url ... --query "test" --verbose
```

**Look for:**
```
⚠️ NO COOKIES received
```

**Solution:** Ensure server sets cookies correctly

---

### Issue: Progress never reaches 100%

**Diagnosis:** Save output and inspect
```bash
python docs/mock_client.py ... --save-output out.json
cat out.json | jq '.[] | select(.event == "progress") | .data | fromjson | .progress'
```

**Common Causes:**
- Server terminates stream early
- Missing `complete` or `result` event
- Network interruption

---

### Issue: Concurrent requests interfere

**Diagnosis:** Check session isolation
```bash
curl http://localhost:5000/jobs
```

**Expected:** Each job has unique session_id

**Common Causes:**
- Shared session storage without proper isolation
- Missing session_id in job data

---

## Success Criteria Summary

A complete, production-ready integration should pass:

| Category | Tests | Critical | Pass Rate |
|----------|-------|----------|-----------|
| Basic Connectivity | 2 | Yes | 100% |
| SSE Streaming | 4 | Yes | 100% |
| Job Polling | 4 | Yes | 100% |
| Session Management | 4 | Yes | 100% |
| Error Handling | 4 | Yes | 100% |
| Performance | 4 | No | ≥75% |
| Integration | 4 | Yes | 100% |

**Overall:** Minimum 95% pass rate across all tests.

---

## Automated Test Script

Create `run_tests.sh`:
```bash
#!/bin/bash
set -e

echo "Starting mock server..."
python docs/mock_server.py &
SERVER_PID=$!
sleep 2

echo "Running tests..."

# Quick smoke test
echo "Test: Health check"
curl -s http://localhost:5000/health | grep "healthy" || exit 1

echo "Test: SSE streaming"
python docs/mock_client.py --url http://localhost:5000/api/v1/stream --query "Test SSE" || exit 1

echo "Test: Job polling"
python docs/mock_client.py --url "http://localhost:5000/api/v1/stream?mode=job" --query "Test job" --expect-job || exit 1

echo "Test: Error handling"
python docs/mock_client.py --url http://localhost:5000/api/v1/stream/error --query "Test error" && exit 1 || echo "Error test passed"

echo "All tests passed!"

# Cleanup
kill $SERVER_PID
```

Run with:
```bash
chmod +x run_tests.sh
./run_tests.sh
```

---

## Next Steps

After passing all tests:

1. **Deploy mock server** to staging environment
2. **Point platform to mock server** via `CHATBOT_DEEP_STREAM_URL`
3. **Run integration tests** (TC-601 - TC-604)
4. **Swap to production chatbot** and repeat tests
5. **Monitor production** logs for session/cookie issues

---

## Additional Resources

- Integration Specification: [CHATBOT_INTEGRATION_SPEC.md](./CHATBOT_INTEGRATION_SPEC.md)
- Mock Client Source: [mock_client.py](./mock_client.py)
- Mock Server Source: [mock_server.py](./mock_server.py)
- Platform Implementation: [streaming_chat_service.py](../analytics/services/streaming_chat_service.py)
