# Response Caching, Interaction Logging & Tracing

Applies to the two production endpoints, `POST /fast_chat_v2` and `POST
/chat/stream` (`fastapi_app/app/main.py`). `/chat` (S3 document RAG) is not
covered.

## What each piece does

- **Response cache** (`fastapi_app/app/response_cache.py`, Redis-backed): skips
  the Gemini call entirely for a repeated/identical query, cutting cost and
  latency. Phase 1 only — normalized exact-string match, no semantic
  similarity. Cache is skipped whenever `conversation_history` is present
  (a follow-up's correct answer depends on that context), and for
  `chat_stream` the cache key also includes `enable_web_search`/
  `enable_financial_data` since those change which tools run. No-ops safely
  if Redis isn't configured.
- **Interaction logging** (`fastapi_app/app/interaction_log.py`, BigQuery):
  the permanent record of every call (success and failure) — query,
  response, tokens, cost, latency, cache hit/miss, and which deploy
  `environment` produced it — used for regression/behavior monitoring.
  Chosen over relying on Langfuse's own (time-limited) trace retention.
  Fires via `asyncio.create_task(...)` so a logging failure never affects
  the user-facing request or its latency.
- **Langfuse tracing** (`fastapi_app/app/tracing.py`): trace-level
  observability and cost dashboards only — not used for caching (Langfuse
  doesn't do response caching) and not the permanent log (BigQuery is). No-ops
  safely if `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` aren't set.

## New environment variables

See `fastapi_app/.env.example` for the full list with defaults
(`RESPONSE_CACHE_*`, `INTERACTION_LOGGING_ENABLED`, `BIGQUERY_INTERACTIONS_*`,
`LANGFUSE_*`). `REDIS_URL` is already handled by the existing ElastiCache
Copilot addon (`fastapi_app/copilot/examples/api/addons/redis.yml.example`) —
no new Redis-specific env var was added.

## Distinguishing environments in BigQuery

`GCP_PROJECT_ID`/`BIGQUERY_DATASET`/`BIGQUERY_INTERACTIONS_TABLE` are declared
in the Copilot manifest's top-level `variables:` block, not per-environment
overrides — so unless a project/dataset is deliberately split out per AWS
environment, dev/staging/prod all write to the same BigQuery table. Every row
now carries an `environment` column, sourced from `COPILOT_ENVIRONMENT_NAME`
(injected automatically by AWS Copilot into every container — no new secret
or config needed). Locally, it falls back to a plain `ENVIRONMENT` env var if
set, otherwise `NULL`.

**If you already have an `interactions` table without this column** (e.g. the
dev table from before this change), add it with a lightweight ALTER — BigQuery
supports adding a NULLABLE column without rewriting existing rows:
```bash
bq query --use_legacy_sql=false \
  'ALTER TABLE `<PROJECT>.<DATASET>.interactions` ADD COLUMN environment STRING'
```
Existing rows will show `NULL` for `environment` — safe to treat as "dev"
if that's the only environment that's been live so far.

## One-time setup a human needs to do (cannot be run from a sandboxed session)

1. **Langfuse**: sign up for Langfuse Cloud (free Hobby tier is fine to
   start — traces retained 30 days; see the tradeoff note below), create a
   project, copy the Public/Secret keys.
2. **Create the BigQuery `interactions` table** (idempotent, safe to re-run):
   ```bash
   python scripts/create_interactions_table.py \
     --project <GCP_PROJECT_ID> --dataset <BIGQUERY_DATASET> \
     --table interactions --location <BIGQUERY_LOCATION>
   ```
3. **Push the two new secrets to SSM**, per environment, after adding
   `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` to your local `fastapi_app/.env`
   (the script already includes them in its `secret_keys` list):
   ```bash
   python scripts/push_copilot_secrets.py --env staging --profile ats-jse-elroy --env-file fastapi_app/.env
   ```
4. **Populate the real (gitignored) Copilot manifest**: copy
   `fastapi_app/copilot/examples/api/manifest.yml.example` into
   `fastapi_app/copilot/api/manifest.yml` if you haven't already, uncomment
   the two new `LANGFUSE_*` lines in `secrets:`, and fill in the new plain
   `variables:` (cache TTL, table names, etc.) per environment.
5. **Confirm the Redis ElastiCache addon is actually deployed** — a template
   existing at `redis.yml.example` doesn't mean it's live in a given
   environment. Check `copilot svc show --name api --env staging` or the
   environment's CloudFormation stack Outputs for `RedisUrl`. If missing,
   copy `redis.yml.example` to the real `copilot/api/addons/redis.yml` before
   deploying.
6. **Deploy**: `copilot svc deploy --name api --env staging --profile ats-jse-elroy` (repeat per environment).
7. **Smoke test**: send the same `/fast_chat_v2` or `/chat/stream` query
   twice with no conversation history — the second call should be
   noticeably faster and should not appear as a fresh Gemini call in
   Langfuse. Check the Langfuse Cloud dashboard for traces and the BigQuery
   console for new rows in `interactions`.

## Known limitation

Langfuse's free Hobby tier only retains trace data for 30 days, with no
built-in export to a data warehouse (that requires a paid add-on). This is
why BigQuery, not Langfuse, is the permanent interaction-log store — Langfuse
is purely for the trace UI and cost dashboards, which don't need to be
permanent.
