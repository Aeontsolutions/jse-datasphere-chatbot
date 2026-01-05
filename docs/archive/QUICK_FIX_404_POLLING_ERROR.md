# Quick Fix: 404 Polling Error in Production

## Problem

Getting this error in production:
```
⚠️ Job polling error: Job polling failed (404)
```

## Root Cause

Your production environment has **2 app instances** behind a load balancer. The job polling uses an **in-memory store**, which requires **sticky sessions** to work properly.

The issue was that **sticky sessions were not properly configured** in the production environment override. When you override the `http` section in environment-specific config, you must re-declare ALL http settings, including `stickiness: true`.

## Solution Applied

I've updated your `copilot/api/manifest.yml` to **properly configure sticky sessions in production**. This enables job polling to work correctly with multiple instances.

### Changes Made

**File**: `fastapi_app/copilot/api/manifest.yml`

Fixed the production environment configuration:

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
      stickiness: true     # ✅ FIXED: Re-declared stickiness in prod override
    variables:
      ASYNC_JOB_MODE: "true"  # Polling enabled with sticky sessions
```

**The key fix**: When you override `http:` in environment-specific config, you must re-declare `stickiness: true`, otherwise it gets lost!

## Next Steps

### 1. Deploy the Changes

```bash
cd fastapi_app
copilot svc deploy --name api --env prod
```

**Deployment time**: 5-10 minutes

### 2. Test with HTML Client

After deployment:

1. Open `streaming_test_client.html`
2. **Enable Debug Mode** (checkbox at bottom)
3. Select **"Prod"** environment
4. Choose **"Deep Research"** endpoint
5. Enter a test query: "how did MDS perform in 2023?"
6. Click **Send Message**

### 3. Verify Success

You should now see in the debug log:

```
✅ SUCCESS - Job polling working!
ℹ️ Server responded with job payload instead of SSE stream
🆔 Job created: [job_id]
📬 Job status ([job_id]): queued
📬 Job status ([job_id]): running
📊 Progress: loading_documents - 30% - Loading relevant documents...
📬 Job status ([job_id]): succeeded
```

**No 404 errors!** The load balancer will now route all requests from the same client to the same instance using sticky session cookies (`AWSALB` cookie).

## How Sticky Sessions Fix This

**With Sticky Sessions Enabled:**

```
Client → Load Balancer → Instance A (creates job)
         ↓ Sets cookie: AWSALB=[instance_id]

Client → Load Balancer → Instance A (polls job) ✅
         ↑ Reads cookie, routes to same instance
```

**What happens now:**

1. **First request** (`POST /chat/stream`): Load balancer routes to Instance A, sets cookie
2. **All polling requests** (`GET /jobs/{job_id}`): Load balancer reads cookie, routes back to Instance A
3. **Job found** in Instance A's memory → No 404 error! ✅

**Why sticky sessions are needed:**
- Job data is stored **in-memory** on each instance
- Without sticky sessions: Load balancer round-robins between Instance A and B
- With sticky sessions: All requests from same client go to same instance

## Configuration Summary

| Environment | ASYNC_JOB_MODE | Sticky Sessions | Instances | Behavior |
|-------------|----------------|-----------------|-----------|----------|
| Local | `true` | N/A | 1 | Job Polling |
| Dev | `true` | Yes | 1 | Job Polling |
| Prod | `true` | **Yes** ✅ | 2 | Job Polling (with sticky sessions) |

## How to Verify Sticky Sessions Work

### Check in AWS Console

1. Go to **AWS Console** → **EC2** → **Target Groups**
2. Find your target group (includes "jse-datasphere-api-prod")
3. Click **Attributes** tab
4. Verify:
   - ✅ Stickiness: **Enabled**
   - ✅ Stickiness type: Load balancer generated cookie
   - ✅ Stickiness duration: 86400 seconds (1 day)

### Check in Browser

1. Open DevTools (F12) → Network tab
2. Send a request to production
3. Look for **Response Headers** containing:
   ```
   Set-Cookie: AWSALB=...; Path=/; Expires=...
   ```
4. Subsequent requests should include:
   ```
   Cookie: AWSALB=...
   ```

## Alternative: Redis Job Store (Future Enhancement)

For even better reliability, you could implement a Redis-based job store:

**Advantages:**
- ✅ No sticky sessions needed
- ✅ Works with any number of instances
- ✅ Jobs survive instance restarts
- ✅ Can scale horizontally without limits

**Trade-offs:**
- Requires Redis deployment
- Additional infrastructure cost
- More complex setup

For now, **sticky sessions are the simplest solution** that works well for your use case.

## Troubleshooting

### Still getting 404 after deployment?

**1. 🚨 MOST COMMON ISSUE: Missing `credentials: 'include'` in fetch**

Your browser needs to send cookies for sticky sessions to work:

```javascript
// ❌ WRONG - No credentials
fetch(url, {
  method: 'POST',
  body: JSON.stringify(data)
})

// ✅ CORRECT - Include credentials
fetch(url, {
  method: 'POST',
  credentials: 'include',  // Required!
  body: JSON.stringify(data)
})
```

**This is required for BOTH:**
- Initial request to create the job
- All subsequent polling requests

**Without `credentials: 'include'`:**
- Browser ignores `Set-Cookie` headers
- Cookies not sent on polling requests
- Load balancer routes to different instances
- Job not found → 404 error

**2. Verify deployment completed successfully:**
```bash
copilot svc status --name api --env prod
```

Look for: Service Status: ACTIVE, Running count: 2/2

**3. Check if sticky sessions are actually enabled:**

AWS Console → EC2 → Target Groups → Your prod target group → Attributes

Verify "Stickiness" shows as "Enabled"

**4. Verify cookies are being set:**

Open browser DevTools → Network tab → Look for `Set-Cookie: AWSALB=...` in response headers

**5. Clear browser cache:**

Cookies might be cached. Clear cache and try again.

**6. Check the logs:**
```bash
copilot svc logs --name api --env prod --follow
```

Look for any configuration errors or health check failures.

## Summary

✅ **Root cause**: Sticky sessions not declared in production environment override
✅ **Changes made**: Added `stickiness: true` to production http config
🚀 **Action required**: Deploy to production
⏱️ **Estimated time**: 5-10 minutes
🎯 **Expected result**: Job polling works with 2 instances using sticky sessions

**The fix enables your production environment to use job polling with multiple instances while maintaining high availability!**
