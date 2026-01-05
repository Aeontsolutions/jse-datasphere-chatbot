# Production Polling Setup Guide

## Problem Solved

The 404 polling error was caused by **sticky sessions not being properly configured** in the production environment override.

## Root Cause

When you override the `http` configuration in environment-specific settings, you must **re-declare all HTTP settings**, including `stickiness`:

```yaml
# ❌ WRONG - This loses the stickiness setting
environments:
  prod:
    http:
      deregistration_delay: 120s  # Only this is set, stickiness is lost!

# ✅ CORRECT - Re-declare stickiness
environments:
  prod:
    http:
      deregistration_delay: 120s
      stickiness: true  # Must explicitly set again
```

## Fix Applied

Updated `fastapi_app/copilot/api/manifest.yml` to properly configure sticky sessions:

```yaml
environments:
  prod:
    count: 2
    cpu: 1024
    memory: 2048
    deployment:
      rolling: 'recreate'
    http:
      deregistration_delay: 120s
      stickiness: true     # ✅ Re-declared for production
    variables:
      ASYNC_JOB_MODE: "true"  # ✅ Re-enabled
```

## Deployment Steps

### 1. Deploy to Production

```bash
cd fastapi_app

# Deploy the updated configuration
copilot svc deploy --name api --env prod
```

This will:
- Update the load balancer to enable sticky sessions
- Configure target group with session affinity
- Apply the ASYNC_JOB_MODE=true setting
- Rolling restart of both instances

**Expected deployment time**: 5-10 minutes

### 2. Verify Sticky Sessions in AWS Console

After deployment, verify the configuration:

1. **Go to AWS Console** → EC2 → Target Groups
2. **Find your target group** (should include "jse-datasphere-api-prod")
3. **Check Attributes tab** → Look for:
   - ✅ **Stickiness**: Enabled
   - ✅ **Stickiness type**: Application-based cookie
   - ✅ **Stickiness duration**: Should be set (typically 86400 seconds = 1 day)

### 3. Test with Your HTML Client

Open `streaming_test_client.html` and:

1. **Enable Debug Mode** (checkbox at bottom)
2. **Select "Prod" environment**
3. **Choose "Deep Research"** (uses `/chat/stream`)
4. **Enter a test query**: "how did MDS perform in 2023?"
5. **Click Send Message**

### 4. Expected Debug Output (SUCCESS)

If sticky sessions are working, you'll see:

```
[Time] 🚀 Starting streaming chat request
[Time] 📝 Query: "how did MDS perform in 2023?"
[Time] 🌐 Environment: prod
[Time] 🔗 URL: http://jse-da-Publi-ehai7dwBXRyV-969154490.us-east-1.elb.amazonaws.com/chat/stream
[Time] ⚙️ Auto-load: true, Memory: true
[Time] ℹ️ Server responded with job payload instead of SSE stream
[Time] 🆔 Job created: [job_id]
[Time] 📬 Job status ([job_id]): queued
[Time] 📬 Job status ([job_id]): running
[Time] 📊 Progress: loading_documents - 30% - Loading relevant documents...
[Time] 📬 Job status ([job_id]): running
[Time] 📊 Progress: generating_response - 80% - Generating AI response...
[Time] 📬 Job status ([job_id]): succeeded
[Time] ✅ Final result received
```

**No 404 errors!** ✅

### 5. What to Look For

**✅ Success indicators:**
- Job created message
- Job status updates (queued → running → succeeded)
- Progress updates every ~1 second
- Final result displayed

**❌ Failure indicators:**
- `⚠️ Job polling error: Job polling failed (404)` - Sticky sessions not working
- Immediate 404 after job creation - Load balancer routing to wrong instance

## How Sticky Sessions Work

```
┌─────────────┐
│   Client    │
│  (Browser)  │
└──────┬──────┘
       │
       │ 1. POST /chat/stream
       │
       ▼
┌─────────────────────┐
│  Load Balancer      │
│  (ELB)              │
│  Sticky: ENABLED ✅ │
└──────┬──────────────┘
       │
       │ Sets cookie: AWSALB=[instance_id]
       ▼
┌──────────────┐           ┌──────────────┐
│  Instance A  │           │  Instance B  │
│  (ECS Task)  │           │  (ECS Task)  │
└──────────────┘           └──────────────┘
       │
       │ 2. Creates job in memory
       │    Job ID: abc123
       │
       ▼
     Stored
       │
       │ 3. GET /jobs/abc123
       │    Cookie: AWSALB=[instance_id]
       │
       ▼
┌─────────────────────┐
│  Load Balancer      │
│  Reads cookie       │
│  Routes to same     │
│  instance           │
└──────┬──────────────┘
       │
       │ Routes back to Instance A
       ▼
┌──────────────┐
│  Instance A  │ ✅ Job found!
│  Returns job │
│  status      │
└──────────────┘
```

## Verify Sticky Session Cookies

### Using Browser DevTools

1. Open browser DevTools (F12)
2. Go to **Network** tab
3. Send a chat request
4. Look for the initial POST request to `/chat/stream`
5. Check **Response Headers** for:
   ```
   Set-Cookie: AWSALB=...; Path=/; ...
   ```
6. Subsequent polling requests should include:
   ```
   Cookie: AWSALB=...
   ```

### Using curl

```bash
# Step 1: Create job and capture cookies
curl -v -c cookies.txt -X POST \
  'http://jse-da-Publi-ehai7dwBXRyV-969154490.us-east-1.elb.amazonaws.com/chat/stream' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "test",
    "auto_load_documents": true,
    "memory_enabled": false
  }'

# Look for job_id in response
# Example: {"job_id":"abc123","status":"queued",...}

# Step 2: Poll job status using the saved cookies
curl -v -b cookies.txt \
  'http://jse-da-Publi-ehai7dwBXRyV-969154490.us-east-1.elb.amazonaws.com/jobs/abc123'

# Should return 200 OK with job status
```

## Troubleshooting

### Issue: Still getting 404 after deployment

**Check 1: Verify deployment completed**
```bash
copilot svc status --name api --env prod
```

Look for:
- ✅ Service Status: ACTIVE
- ✅ Running count: 2/2
- ✅ Latest deployment: SUCCESS

**Check 2: Check environment variables**
```bash
copilot svc exec --name api --env prod
# Inside container:
env | grep ASYNC_JOB_MODE
# Should show: ASYNC_JOB_MODE=true
```

**Check 3: Check load balancer attributes**

Go to AWS Console → EC2 → Target Groups → Attributes

Verify:
- Stickiness: Enabled
- Stickiness type: lb_cookie (Load balancer generated cookie)

**Check 4: Browser cache**

Clear browser cache and cookies, then test again.

### Issue: Cookies not being sent

**Cause**: Missing `credentials: 'include'` in fetch requests

**🚨 CRITICAL FIX**: Every fetch request must include credentials:

```javascript
// ❌ WRONG - Browser won't send cookies
const response = await fetch(url, {
  method: 'POST',
  body: JSON.stringify(data)
});

// ✅ CORRECT - Browser will send cookies
const response = await fetch(url, {
  method: 'POST',
  credentials: 'include',  // Required for sticky sessions!
  body: JSON.stringify(data)
});
```

**This applies to:**
- Initial POST request to `/chat/stream` (to receive the cookie)
- All GET requests to `/jobs/{job_id}` (to send the cookie back)

**Backend**: Your backend already has CORS configured properly in `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,  # ✅ This enables cookies
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue: Deployment fails

**Check logs:**
```bash
copilot svc logs --name api --env prod --follow
```

Look for:
- Startup errors
- Configuration issues
- Health check failures

## Alternative: Single Instance (Not Recommended)

If sticky sessions still don't work, you can temporarily scale down to 1 instance:

```yaml
environments:
  prod:
    count: 1  # Only one instance, no sticky sessions needed
```

**Disadvantages:**
- ❌ No high availability
- ❌ Zero downtime deployments harder
- ❌ Can't handle high load
- ❌ Single point of failure

## Future: Redis-Based Job Store

For a more robust solution, implement a Redis-based job store:

**Advantages:**
- ✅ Works with any number of instances
- ✅ No sticky sessions required
- ✅ Jobs survive instance restarts
- ✅ Can scale horizontally

**Implementation sketch:**
```python
# app/redis_job_store.py
import redis
import json

class RedisJobStore:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)

    async def create_job(self, job_type: str, request_data: dict) -> str:
        job_id = str(uuid.uuid4())
        job_data = {
            "job_id": job_id,
            "status": "queued",
            "job_type": job_type,
            "created_at": datetime.utcnow().isoformat(),
        }
        # Store in Redis with TTL
        self.redis.setex(
            f"job:{job_id}",
            900,  # 15 minutes TTL
            json.dumps(job_data)
        )
        return job_id
```

## Summary

✅ **Fix applied**: Sticky sessions properly configured in production
🚀 **Next step**: Deploy to production
⏱️ **Deployment time**: 5-10 minutes
🎯 **Expected result**: Polling works with 2 instances
📊 **Monitoring**: Use debug mode in test client to verify

## Deploy Command

```bash
cd fastapi_app
copilot svc deploy --name api --env prod
```

Then test with your HTML client in production mode!
