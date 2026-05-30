# ATS-334: Restore Financial Tool Calling on /chat/stream

**Date:** 2026-05-30
**Ticket:** [ATS-334](https://linear.app/galbraith-family/issue/ATS-334/r10-restore-financial-tool-calling-on-chatstream)
**Status:** Approved — ready for implementation

---

## Problem

`/chat/stream` runs `AgentV2` — a single Gemini 2.5 Pro call with Google Search grounding only. It has no path to the BigQuery financial-statement data. Queries that the DB could answer in ~1s instead trigger exhaustive web grounding, inflating median latency to 87s per conversation.

The `enable_financial_data` field already exists on `AgentChatRequest` but is currently ignored by `AgentV2`.

---

## Decision

Restore `AgentOrchestrator` (archived in ATS-330) as the handler for `/chat/stream`.

**Why not Option B (add financial tool to AgentV2)?**
Gemini cannot run function calling and GoogleSearch in the same call. A routing step is required regardless, which means reimplementing the orchestrator piecemeal. The archived version is complete and battle-tested.

---

## Architecture

`AgentOrchestrator` is instantiated once at startup (stored on `app.state`) and receives the already-initialized `financial_manager`. The `/chat/stream` endpoint delegates to it instead of creating a fresh `AgentV2` per request.

### Split-model strategy

| Phase | Model | Reason |
|---|---|---|
| Phase 1: Routing / clarification / pronoun resolution | `gemini-2.5-flash` | Small prompt, low latency |
| Phase 2: Financial extraction (function calling) | `gemini-2.5-flash` | Structured output, no quality gap |
| Phase 3: Synthesis (final response) | `gemini-2.5-pro` | User-facing; higher quality justified |

`AgentV2` stays in place serving `/chat/stream/v2` unchanged.

---

## Code Changes

### 1. `fastapi_app/app/_archive/agent_orchestrator.py` → `fastapi_app/app/agent.py`

- Remove the `ARCHIVED` header from the module docstring
- Split `model_name: str = "gemini-2.5-flash"` into two class attributes:
  - `flash_model = "gemini-2.5-flash"` — used in phases 1 and 2
  - `pro_model = "gemini-2.5-pro"` — used in phase 3 (synthesis)
- Update `_track_cost()` calls to pass the correct model name per phase
- **Fix conversation-history bug:** the success return path returns the original `conversation_history` unchanged. Append the new user/assistant turn before returning, matching `AgentV2` behaviour.

### 2. `fastapi_app/app/main.py` — lifespan

```python
from app.agent import AgentOrchestrator

# After financial_manager is initialized:
app.state.agent_orchestrator = AgentOrchestrator(
    financial_manager=app.state.financial_manager
)
```

If `financial_manager` is `None`, still instantiate — the orchestrator's internal guard (`routing["use_financial"] and self.financial_manager`) skips DB queries safely.

### 3. `fastapi_app/app/main.py` — `/chat/stream` endpoint

Replace:
```python
agent = AgentV2()
result = await agent.run(
    query=request.query,
    conversation_history=request.conversation_history,
    enable_web_search=request.enable_web_search,
)
```

With:
```python
agent = app.state.agent_orchestrator
result = await agent.run(
    query=request.query,
    conversation_history=request.conversation_history,
    enable_web_search=request.enable_web_search,
    enable_financial_data=request.enable_financial_data,
)
```

Also pass `cost_summary` through to `AgentChatResponse` (the field already exists on the model).

### 4. `fastapi_app/app/_archive/README.md`

Update the restoration steps table to mark R10 as complete.

---

## Error Handling

No new error handling required. `AgentOrchestrator` already has `try/except` in every phase with `_build_error_response()` as fallback. Financial queries are skipped gracefully when `financial_manager is None` or `enable_financial_data=False`.

---

## Testing

- Existing `/chat/stream` tests pass unchanged (response shape is identical to `AgentV2`)
- Add a focused integration test: POST `/chat/stream` with a financial query and `enable_financial_data=True` → assert `record_count > 0` and `filters_used` is not null
- `uat/test_uat_agent_chat.py` exercises the endpoint end-to-end and will surface regressions

---

## Expected Impact

- Financial queries answered from the DB (~1s) instead of exhaustive web grounding (~87s median)
- `record_count`, `filters_used`, `data_preview`, and `chart` fields populated for DB-answerable queries
- Web-only queries unaffected (routing phase correctly skips DB for news/current-events queries)
