# _archive

This directory holds code that is **not active** but is preserved for reference.

Files here are intentionally excluded from the running application. They are kept
because they document non-obvious capabilities (tool declarations, 3-phase pipeline
patterns, financial DB query logic) that may be revived in a future ticket (see R10).

## Contents

| File | Original path | What it is |
|---|---|---|
| `agent_orchestrator.py` | `app/agent.py` | `AgentOrchestrator` — a 3-phase agent (clarification → routing → synthesis) with both Google Search grounding *and* JSE financial-DB tool calling via Gemini function calling. Was imported in `main.py` but never instantiated; all live endpoints use the simpler `AgentV2` instead. |
| `streaming_chat.py` | `app/streaming_chat.py` | Backed the `POST /chat/stream/v0` endpoint (SSE/async-job variant of the legacy `/chat` document-RAG pipeline). Docstring explicitly said `[DEPRECATED] ... use /chat/stream instead`. Archived 2026-08-14 during the prod-cleanup pass that scoped the live surface down to `/chat/stream` + `/fast_chat_v2` (`/chat` itself was kept — see project memory on the S3 doc-metadata rebuild). |
| `streaming_financial_chat.py` | `app/streaming_financial_chat.py` | Backed `POST /fast_chat_v2/stream`, the SSE sibling of the live `/fast_chat_v2`. Confirmed unused by any frontend and archived alongside the async-job infra it depended on. |
| `job_store.py` | `app/job_store.py` | In-memory `JobStore`/`JobProgressSink` used only by the async-job mode of `/chat/stream/v0` and `/fast_chat_v2/stream`, plus the now-removed `GET /jobs/{job_id}` polling endpoint. |
| `redis_job_store.py` | `app/redis_job_store.py` | `RedisJobStore`, the Redis-backed alternative to `job_store.py`. Also removed the `/health` endpoint's Redis connectivity check, which only ever tested this. |
| `progress_tracker.py` | `app/progress_tracker.py` | `ProgressTracker`/SSE progress-event formatting, used only by `streaming_chat.py` and `streaming_financial_chat.py` above. |

Note: the three job/streaming-support files above import from each other via
`app._archive.<module>` (rewritten from `app.<module>` at archive time) rather than
relative imports, matching the style of the rest of the app package.

## Restoring

> **ATS-334 update (2026-05-31):** The financial function-calling primitives
> (`get_financial_data_tool_declaration`, `execute_financial_query`,
> `build_financial_context`) now live in `app/financial_tool.py` and are used by
> `AgentV2` on `/chat/stream` (Option 2). The 3-phase `AgentOrchestrator` below
> was **not** restored (Option 1, PR #32, was closed in favor of the smaller
> AgentV2 path). This file remains archived for reference only.

To wire `AgentOrchestrator` back up (ticket R10):
1. Move `agent_orchestrator.py` back to `app/agent.py` and remove the `ARCHIVED` header from the module docstring
2. Re-add `from app.agent import AgentOrchestrator` in `main.py`
3. In the `lifespan` context manager in `main.py`, instantiate the orchestrator after `financial_manager` is ready and store it on app state:
   ```python
   app.state.agent_orchestrator = AgentOrchestrator(financial_manager=app.state.financial_manager)
   ```
4. In the `/chat/stream` endpoint, swap out `AgentV2` for `app.state.agent_orchestrator` — note that `AgentOrchestrator.run()` returns a full dict (not a streaming generator), so the endpoint response shape will need to be reconciled with the current streaming contract
