# Redis Job Store Implementation

## Overview

We have transitioned from an in-memory job store to a **Redis-backed job store** to resolve persistent 404 errors in the multi-instance production environment.

## Why Redis?

The in-memory store had a fundamental flaw in load-balanced environments:
- Request A creates job on Instance 1 (stored in Instance 1's RAM)
- Request B polls job on Instance 2 (Instance 2 has no knowledge of the job) -> **404 Error**

While "sticky sessions" can mitigate this, they are fragile (browser settings, CORS issues, network changes).

**Redis provides a shared, persistent storage layer** accessible by all instances.

## Changes Made

### 1. Dependencies
Added `redis` to `requirements.txt`.

### 2. Backend Code
- Created `app/redis_job_store.py`: Implements `JobStore` interface using Redis.
- Updated `app/main.py`: Automatically initializes `RedisJobStore` if `REDIS_URL` environment variable is present.

### 3. Infrastructure (AWS Copilot)
Created `copilot/api/addons/redis.yml` to provision an **AWS ElastiCache Redis** cluster.
- Instance Type: `cache.t3.micro` (Cost-effective)
- Networking: Automatically configured in private subnets
- Security: Only accessible from the ECS tasks
- **Auto-Injection**: The Redis URL is automatically injected as `REDIS_URL` into the application container.

## Deployment

To deploy these changes and provision the Redis cluster:

```bash
cd fastapi_app
copilot svc deploy --name api --env prod
```

**Note**: The first deployment will take longer (~10-15 minutes) as it needs to provision the ElastiCache cluster.

## Verification

After deployment, the application logs will show:
```
Redis job store initialized | url=redis://...
```

You can verify the fix by disabling sticky sessions (optional) or simply testing with the chat client. The 404 errors should disappear completely, regardless of which instance serves the request.

