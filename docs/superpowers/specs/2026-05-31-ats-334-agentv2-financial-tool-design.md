# ATS-334 (Option 2): Add `query_financial_data` to AgentV2

**Date:** 2026-05-31
**Ticket:** [ATS-334 — R10: Restore financial tool calling on `/chat/stream`](https://linear.app/galbraith-family/issue/ATS-334)
**Status:** Design — approved approach, pending spec review

## Context

`/chat/stream` runs `AgentV2` — a single Gemini 2.5 Pro call with Google Search
grounding only. It has no path to the BigQuery financial-statement data, so
financial questions fall back to slow, often-stale web grounding (baseline
median ~87s/conversation).

ATS-334 offered two fixes. **Option 1** (restore the archived 3-phase
`AgentOrchestrator`) was attempted in PR #32 and **closed** — the 3-phase
pipeline added too much latency. We are proceeding with **Option 2: add the
financial tool directly to AgentV2**, which is a smaller, lower-latency surface.

### Corrected facts about the current code

- `streaming_financial_chat.py` is a 5-step SSE pipeline
  (`parse_user_query → validate_data_availability → query_data → format_response`).
  It does **not** use Gemini function calling. `/fast_chat_v2` uses this NL-parse
  path. It is **not touched** by this work.
- The **only** Gemini function-calling financial implementation that exists is in
  archived dead code: [`_archive/agent_orchestrator.py`](../../../fastapi_app/app/_archive/agent_orchestrator.py):
  - `get_financial_data_tool_declaration()` — the `query_financial_data` tool
    (params: `symbols`, `years`, `standard_items`).
  - `execute_financial_query(manager, args)` — builds `FinancialDataFilters`,
    post-processes via metadata associations, calls `manager.query_data()`,
    generates a chart, returns `(records, filters, chart, sources)`.
  - `_build_financial_context(records)` — formats records into a compact text
    block for synthesis.
- `FinancialDataManager.query_data(filters)` is **synchronous** (blocking
  BigQuery). The manager is already live on `app.state.financial_manager` and
  exposed via `get_financial_manager()`.
- `AgentChatRequest` already carries `enable_financial_data: bool = True` and
  `enable_web_search`. `AgentChatResponse` already has every field we need
  (`record_count`, `filters_used`, `data_preview`, `chart`, `sources`,
  `tools_executed`, …). No model changes required.
- Gemini (this team's setup) does not reliably support combining `google_search`
  grounding with a function-declaration tool in one request — the archived
  orchestrator deliberately ran financial and web calls **separately**. Our
  design honors that constraint.

## Goal

When `/chat/stream` receives a question answerable from the JSE financial
database, AgentV2 calls `query_financial_data`, retrieves records from BigQuery
(~1s), and synthesizes a grounded answer — instead of doing exhaustive web
grounding. Web-only questions are unaffected.

## Non-goals

- No changes to `/fast_chat_v2`, `streaming_financial_chat.py`, or the NL-parse
  pipeline.
- No restoration of `AgentOrchestrator` / the 3-phase pipeline.
- `/chat/stream/v2` stays web-only by design (its docstring states "no SQL
  financial data queries").
- No changes to `AgentChatRequest` / `AgentChatResponse` models.

## Architecture

### 1. New module: `app/financial_tool.py`

A small, unit-tested live home for the function-calling primitives, lifted from
the archive (the archive stays as-is; a one-line pointer is added to
`_archive/README.md`). Contents:

- `get_financial_data_tool_declaration() -> types.FunctionDeclaration` — lifted
  verbatim (`query_financial_data`; params `symbols`, `years`, `standard_items`).
- `get_financial_tool() -> types.Tool` — wraps the declaration in
  `types.Tool(function_declarations=[...])`.
- `async execute_financial_query(manager, args) -> (records, filters, chart, sources)`
  — lifted from the archive, with one improvement: the blocking
  `manager.query_data(filters)` is wrapped in `asyncio.to_thread(...)` so it does
  not block the event loop.
- `build_financial_context(records) -> str` — lifted verbatim (compact, token-
  efficient summary used for synthesis).
- `extract_query_financial_data_call(response) -> Optional[function_call]` —
  returns the first `query_financial_data` function-call part, or `None`.

Dependencies are all live modules: `app.charting.generate_chart`,
`app.models.{FinancialDataFilters, FinancialDataRecord}`.

### 2. AgentV2 changes (`app/agent_v2.py`)

- `__init__(self, model_name="gemini-2.5-pro", financial_manager=None)` — store
  `self.financial_manager` (dependency-injected for testability; `None` disables
  the financial path).
- `_track_cost(self, response, phase, model=None)` — accept an optional `model`
  so per-call costs are attributed correctly (the decision call uses flash).
- `run(..., enable_financial_data: bool = True)` — new flow (sequential):

  ```
  if enable_financial_data and self.financial_manager:
      # Phase A — decision/extraction (cheap, fast)
      resp1 = generate_content(
          model="gemini-2.5-flash",
          contents=self._build_contents(history, query),   # full history for context
          config=GenerateContentConfig(
              system_instruction=FINANCIAL_DECISION_PROMPT,
              tools=[get_financial_tool()],
              tool_config=ToolConfig(FunctionCallingConfig(mode="AUTO")),
              temperature=0.3, max_output_tokens=512))
      track_cost(resp1, "financial_extraction", model="gemini-2.5-flash")
      fc = extract_query_financial_data_call(resp1)
      if fc:
          records, filters, chart, sources = await execute_financial_query(mgr, dict(fc.args))
          # Phase B — synthesis (quality)
          context = build_financial_context(records)
          contents = self._build_contents(history, query) + [Content(role="user",
              parts=[Part.from_text("Financial data:\n" + context +
                     "\n\nAnswer the question using ONLY this data.")])]
          resp2 = generate_content(model=self.model_name,   # pro
              config=GenerateContentConfig(system_instruction=SYSTEM_PROMPT_NO_SEARCH,
                                            temperature=0.3, max_output_tokens=8192))  # no tools
          track_cost(resp2, "synthesis")
          return { response: resp2.text, data_found: len(records)>0,
                   record_count: len(records), filters_used: filters,
                   data_preview: records[:10], chart: chart, sources: sources,
                   tools_executed: ["query_financial_data"], web_search_results: None,
                   needs_clarification: False, conversation_history: updated,
                   cost_summary: ... }
      # else: model declined → fall through to web/plain path

  # existing behavior, unchanged
  return <current single-call google_search (or no-search) path>
  ```

- **`FINANCIAL_DECISION_PROMPT`** instructs flash to call `query_financial_data`
  only for JSE financial-metric questions (decline otherwise) **and embeds
  available symbols / company names / years from `manager.metadata`** so the
  extracted `symbols` are valid (mirrors the archive's metadata-context
  injection — the tool exposes only `symbols`, not company names).
- **Cost/latency for DB-answerable queries:** flash-decide (~1–2s) + query (~1s) +
  pro-synth (~10s) ≈ ~13s vs ~87s baseline. **Web-only queries with financial
  enabled** add only the cheap flash-decide call before the existing web path.
  **`enable_financial_data=False`** is byte-for-byte the current behavior.
- Uses only verified SDK surface (`Part.from_text`, `generate_content`,
  `ToolConfig`/`FunctionCallingConfig` — all present in the archive). No
  dependence on `Part.from_function_response`.

### 3. Endpoint changes (`app/main.py`)

`/chat/stream` (and its `/agent/chat` alias) only:

```python
async def chat_stream(request: AgentChatRequest,
                      financial_manager: Any = Depends(get_financial_manager)):
    agent = AgentV2(financial_manager=financial_manager)
    result = await agent.run(
        query=request.query,
        conversation_history=request.conversation_history,
        enable_web_search=request.enable_web_search,
        enable_financial_data=request.enable_financial_data,
    )
```

No 503 guard: if `financial_manager` is `None` (BigQuery down), AgentV2 skips the
financial path and degrades gracefully to web search.

## Testing (TDD)

- `tests/test_financial_tool.py` (new, mocked): tool declaration shape;
  `execute_financial_query` builds correct filters (symbols upper, years str,
  items lower/underscored) and calls `query_data`; `build_financial_context`
  formatting; `extract_query_financial_data_call` present/absent.
- `tests/test_agent_v2_financial.py` (new, mocked Gemini client + mock manager):
  function-call → execute → synthesize returns populated
  `record_count`/`filters_used`/`data_preview`/`tools_executed`; model declines →
  web fallback; `enable_financial_data=False` skips financial entirely
  (`query_data` never called); `financial_manager=None` degrades gracefully.
- `tests/test_chat_stream_financial.py` (new): `/chat/stream` injects
  `financial_manager` and forwards `enable_financial_data` to `AgentV2.run`
  (TestClient with `AgentV2.run` patched).
- **Untouched / regression guards:** `test_streaming.py`, `test_streaming_units.py`
  (cover `/fast_chat_v2`) require **no changes** — we don't touch that path.

## Evaluation plan

The existing `chat_stream` personas are qualitative and barely exercise the tool
(metric-heavy personas are tagged `fast_chat_v2`). So:

1. **Add financial `chat_stream` personas** (port the 3 from closed PR #32 via
   `git show claude/ecstatic-lovelace-a785e5:evals/personas/...`, adapt to current
   schema): NCB revenue lookup, GK-vs-NCB profit compare, mixed financial+news.
2. **Persona eval:** `python scripts/run_eval.py --endpoint chat_stream` for
   groundedness + latency (using the main-dir `.env` workaround; confirm a
   baseline run dir exists before any before/after — do not assume baseline
   numbers).
3. **Direct verification:** POST metric queries to live `/chat/stream` with
   `enable_financial_data:true` (e.g. "What was NCB revenue in 2023?"); confirm
   `tools_executed` contains `query_financial_data`, `record_count > 0`, latency
   in seconds (not ~87s).

**Prerequisites / cost:** needs uvicorn + the real `.env` (BigQuery + Gemini
creds) from the **main** project dir, not the worktree. Live runs are billable
(Gemini + BigQuery).

## Risks & mitigations

- **flash extraction misses entities on follow-ups** → we pass full conversation
  history to the decision call (better than the archive's query-only call).
- **flash answers a web-only question instead of declining** → mitigated by the
  `FINANCIAL_DECISION_PROMPT` instructing it to call the tool only for JSE metric
  questions; on no function-call we route to web search and ignore phase-A text.
- **Tokens from large result sets** → `build_financial_context` already caps at
  50 records.

## Files changed

| File | Change |
|---|---|
| `fastapi_app/app/financial_tool.py` | **new** — tool declaration, `get_financial_tool`, async `execute_financial_query`, `build_financial_context`, `extract_query_financial_data_call` |
| `fastapi_app/app/agent_v2.py` | financial-data path in `run()`; `financial_manager` ctor arg; `_track_cost` model arg; `FINANCIAL_DECISION_PROMPT` |
| `fastapi_app/app/main.py` | `/chat/stream` injects `financial_manager`, forwards `enable_financial_data` |
| `fastapi_app/app/_archive/README.md` | one-line pointer to `app/financial_tool.py` |
| `fastapi_app/tests/test_financial_tool.py` | **new** unit tests |
| `fastapi_app/tests/test_agent_v2_financial.py` | **new** unit tests |
| `fastapi_app/tests/test_chat_stream_financial.py` | **new** endpoint test |
| `evals/personas/chatstream_*.yaml` | **new** — 3 financial chat_stream personas |
