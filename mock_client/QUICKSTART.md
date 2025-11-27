# Chatbot Integration Testing - Quick Start Guide

Get up and running with integration testing in under 5 minutes.

## What You'll Build

A complete testing environment that lets you:
1. **Mock the JSE Analytics Platform** - Test your chatbot without the full platform
2. **Simulate both integration patterns** - SSE streaming and async job polling
3. **Validate the contract** - Ensure your chatbot meets all requirements

---

## Prerequisites

- Python 3.10+
- `pip install requests flask`

---

## 3-Step Setup

### Step 1: Start Mock Server

```bash
# From the project root
python docs/mock_server.py
```

You should see:
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         MOCK CHATBOT SERVER                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Server running at: http://127.0.0.1:5000
```

**What this does:** Simulates your external chatbot service with both SSE and job polling support.

---

### Step 2: Test SSE Streaming Mode

**Open a new terminal:**

```bash
python docs/mock_client.py \
  --url http://localhost:5000/api/v1/stream \
  --query "What are the key financial metrics for company XYZ?"
```

**Expected output:**
```
================================================================================
STREAMING EVENTS
================================================================================

[PROGRESS]
{
  "step": "loading_documents",
  "message": "Loading relevant documents...",
  "progress": 25,
  "details": {
    "documents_found": 12
  }
}

[RESULT]
{
  "response": "Based on my analysis of your query...",
  "sources": [...],
  "metadata": {...}
}

================================================================================
FINAL RESPONSE:
================================================================================
Based on my analysis of your query "What are the key financial metrics..."
1. The market shows strong growth indicators
2. Financial metrics demonstrate stability
...
```

✅ **Success Indicators:**
- Progress events from 10% → 100%
- Final response displayed
- Exit code 0

---

### Step 3: Test Job Polling Mode

```bash
python docs/mock_client.py \
  --url "http://localhost:5000/api/v1/stream?mode=job" \
  --query "Perform deep analysis on market trends" \
  --expect-job
```

**Expected output:**
```
📋 Detected JSON job response
Job payload: {'job_id': 'job_abc123...', 'status': 'pending', ...}
🔄 Polling job at: http://localhost:5000/jobs/job_abc123...
✅ Cookies received: {'session_id': 'xyz789'}

🔄 Poll attempt 1/150
📊 Job status: {"status": "running", "latest_progress": {"progress": 25, ...}}

🔄 Poll attempt 2/150
📊 Job status: {"status": "running", "latest_progress": {"progress": 50, ...}}

...

✅ Job finished with status: succeeded

[RESULT]
{
  "response": "Based on my analysis...",
  ...
}
```

✅ **Success Indicators:**
- Job ID received
- Session cookies logged
- Status progresses: pending → running → succeeded
- Final result contains response

---

## Testing Your Own Chatbot

### Option 1: Replace Mock Server with Your Chatbot

```bash
# Point mock client to your chatbot
python docs/mock_client.py \
  --url https://your-chatbot.com/api/v1/stream \
  --query "Test query"
```

**Look for:**
- ✅ Cookies received (critical for sticky sessions)
- Progress events in correct format
- Result event with `response` field

**Common issues:**
- ⚠️ NO COOKIES received → Add `Set-Cookie` header
- Missing `response` field → Check result event format
- Timeout → Ensure heartbeat or completion within 120s

---

### Option 2: Test Platform → Your Chatbot

**1. Configure platform environment:**
```bash
# In platform .env file
CHATBOT_DEEP_STREAM_URL=https://your-chatbot.com/api/v1/stream
CHATBOT_POLL_INTERVAL=2.0
```

**2. Start platform:**
```bash
make dev
```

**3. Navigate to:** `http://localhost:5173/analytics/assistant`

**4. Submit a query in "Deep Research" mode**

**5. Watch Django logs:**
```
Deep chat stream view called
Starting event stream for deep chat
✅ Cookies received from chatbot: {'session_id': '...'}
Yielding to frontend - event: progress
...
Yielding to frontend - event: result
Saving Chat Message
```

**6. Check browser console:**
```
Frontend received chunk: event: progress...
Streaming store updating progress: {progress: 25, ...}
Setting final response
```

---

## Understanding the Mock Server

### Available Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/stream` | POST | SSE streaming (default) |
| `/api/v1/stream?mode=job` | POST | Job polling mode |
| `/jobs/<job_id>` | GET | Poll job status |
| `/api/v1/stream/error` | POST | Simulate error |
| `/api/v1/stream/timeout` | POST | Simulate timeout |
| `/health` | GET | Health check |
| `/jobs` | GET | List all jobs |

### Simulating Different Scenarios

**Success with progress:**
```bash
python docs/mock_client.py --url http://localhost:5000/api/v1/stream --query "Test"
```

**Error during processing:**
```bash
python docs/mock_client.py --url http://localhost:5000/api/v1/stream/error --query "Test"
```

**Slow response (timeout test):**
```bash
python docs/mock_client.py --url http://localhost:5000/api/v1/stream/timeout --query "Test"
```

---

## Customizing the Mock Server

### Add Custom Responses

Edit `docs/mock_server.py`:

```python
# Modify steps in generate_sse_stream() function
steps = [
    {
        'event': 'progress',
        'data': {
            'step': 'your_custom_step',
            'message': 'Your custom message',
            'progress': 30,
            'details': {'your': 'data'}
        }
    },
    # ... more steps
]
```

### Change Timing

```python
# In generate_sse_stream(), adjust sleep duration
time.sleep(0.5)  # Faster response (default: 1.5)
```

### Simulate Your Data Format

```python
# In result event
'data': {
    'response': 'Your actual response format',
    'sources': [...],  # Your sources structure
    'metadata': {...}  # Your metadata
}
```

---

## Conversation History Testing

**Create `history.json`:**
```json
[
  {
    "role": "user",
    "content": "What is the stock market?"
  },
  {
    "role": "assistant",
    "content": "The stock market is a collection of markets where stocks are traded..."
  },
  {
    "role": "user",
    "content": "How does it work in Jamaica?"
  },
  {
    "role": "assistant",
    "content": "In Jamaica, the Jamaica Stock Exchange (JSE) operates..."
  }
]
```

**Test with history:**
```bash
python docs/mock_client.py \
  --url http://localhost:5000/api/v1/stream \
  --query "What about recent trends?" \
  --history history.json
```

**Server logs:**
```
Conversation history: 4 messages
```

---

## Debugging Tips

### Enable Verbose Logging

```bash
python docs/mock_client.py \
  --url http://localhost:5000/api/v1/stream \
  --query "Debug test" \
  --verbose
```

Shows:
- Raw SSE chunks
- Each line processed
- Cookie operations
- Detailed event data

---

### Save Output for Analysis

```bash
python docs/mock_client.py \
  --url http://localhost:5000/api/v1/stream \
  --query "Test" \
  --save-output results.json
```

**Analyze:**
```python
import json

with open('results.json') as f:
    events = json.load(f)

# Check event types
event_types = [e['event'] for e in events]
print(f"Event sequence: {event_types}")

# Verify progress values
progress_events = [e for e in events if e['event'] == 'progress']
for pe in progress_events:
    data = json.loads(pe['data'])
    print(f"{data['step']}: {data['progress']}%")

# Check final result
result_events = [e for e in events if e['event'] == 'result']
if result_events:
    result = json.loads(result_events[0]['data'])
    print(f"Response length: {len(result['response'])} chars")
```

---

### Monitor Server Logs

Mock server logs everything:
```
INFO - Received request - Query: Test query
INFO - ✅ Created new session: abc-123-xyz
INFO - 📡 Using SSE streaming mode
INFO - SSE -> progress: {"step": "loading_documents", ...}
INFO - SSE -> result: {"response": "Based on my analysis...", ...}
```

Look for:
- ✅ Session created/reused
- 🔑 Cookies sent/received
- 📊 Job status updates
- ❌ Errors

---

## Validation Checklist

Before deploying your chatbot to production:

### SSE Streaming
- [ ] Sets `Content-Type: text/event-stream`
- [ ] Sets `Set-Cookie` header with session ID
- [ ] Sends progress events with increasing percentages
- [ ] Sends result event with `response` field
- [ ] Sends heartbeat every 60-120s (if processing takes long)
- [ ] Completes within 5 minutes or sends error

### Job Polling
- [ ] Returns `job_id` and `polling_url` in initial response
- [ ] Sets `Set-Cookie` header with session ID
- [ ] Polling endpoint returns valid job status
- [ ] `latest_progress` field present in all polling responses
- [ ] Status transitions: pending → running → succeeded/failed
- [ ] Final response includes `result` with `response` field
- [ ] Session cookie validates across polls

### Error Handling
- [ ] Returns error event/status on failure
- [ ] Error message is descriptive
- [ ] No uncaught exceptions

---

## Next Steps

1. **Read the full spec:** [CHATBOT_INTEGRATION_SPEC.md](./CHATBOT_INTEGRATION_SPEC.md)
2. **Run comprehensive tests:** [TEST_SCENARIOS.md](./TEST_SCENARIOS.md)
3. **Test with real platform:** Point platform to your chatbot
4. **Monitor production:** Check logs for session issues

---

## Common Issues & Solutions

### Issue: "Failed to connect to chatbot"

**Check:**
- Is server running? `curl http://localhost:5000/health`
- Correct URL? Should end with `/stream`
- Firewall blocking?

---

### Issue: "No progress updates"

**For SSE:** Ensure events are formatted correctly:
```
event: progress
data: {"step": "...", "message": "...", "progress": 25}

```
(Note the blank line after data)

**For Job Polling:** Include `latest_progress` in every response:
```json
{
  "status": "running",
  "latest_progress": {
    "step": "analyzing",
    "message": "Analyzing...",
    "progress": 50
  }
}
```

---

### Issue: "Result not displayed"

**Check result event format:**
```json
{
  "response": "The actual answer text"  // ← This field is REQUIRED
}
```

Not:
```json
{
  "result": "..."  // ← Wrong field name
}
```

---

### Issue: "Job not found" (404)

**Cause:** Session cookie not being sent/validated

**Solution:**
1. Ensure initial response sets `Set-Cookie`
2. Verify client sends cookie in polls
3. Check server validates session
4. Use session storage (Redis) for multi-worker setups

---

## Support & Resources

- **Documentation:** All docs in `docs/` folder
- **Source Code:** Reference implementation in `analytics/services/streaming_chat_service.py`
- **Issues:** Check server/client logs for detailed error messages

---

## Summary

You now have:
✅ Mock server simulating your chatbot
✅ Mock client simulating the platform
✅ Test scenarios for validation
✅ Tools to debug integration issues

**Ready to integrate?** Start by pointing the mock client at your chatbot URL and verify all events are correctly formatted.
