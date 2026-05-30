# ATS-334: Restore Financial Tool Calling on /chat/stream — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `AgentOrchestrator` back into `/chat/stream` so that financial queries hit BigQuery instead of falling back to exhaustive web grounding.

**Architecture:** `AgentOrchestrator` (restored from `_archive/agent_orchestrator.py` as `app/agent.py`) replaces the per-request `AgentV2()` instantiation in `/chat/stream`. It is instantiated once in `lifespan` on `app.state`, uses `gemini-2.5-flash` for phases 1–2 (routing + extraction) and `gemini-2.5-pro` for phase 3 (synthesis), and correctly updates `conversation_history` in its return value.

**Tech Stack:** FastAPI, Gemini Python SDK (`google.genai`), BigQuery (via `FinancialDataManager`), pytest, `unittest.mock`

---

## File Map

| File | Action | What changes |
|---|---|---|
| `fastapi_app/app/agent.py` | **Create** (from archive) | Restored orchestrator with split models + history fix |
| `fastapi_app/app/main.py` | **Modify** | Import, lifespan instantiation, endpoint swap |
| `fastapi_app/app/_archive/README.md` | **Modify** | Mark R10 as complete |
| `fastapi_app/tests/unit/test_agent.py` | **Create** | Unit tests for `AgentOrchestrator` |
| `fastapi_app/tests/test_api.py` | **Modify** | Add endpoint test for financial flag passthrough |

---

## Task 1: Write failing unit tests for AgentOrchestrator

**Files:**
- Create: `fastapi_app/tests/unit/test_agent.py`

- [ ] **Step 1: Create the test file with two failing tests**

`fastapi_app/tests/unit/test_agent.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def financial_manager():
    fm = MagicMock()
    fm.metadata = {
        "symbols": ["NCB", "GK"],
        "years": ["2022", "2023"],
        "associations": {"symbol_to_company": {"NCB": ["NCB Financial Group"]}},
    }
    return fm


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_appends_new_turn_to_conversation_history(financial_manager):
    """AgentOrchestrator.run() must append the new user+assistant turn to history."""
    from app.agent import AgentOrchestrator

    orchestrator = AgentOrchestrator(financial_manager=financial_manager)
    prior_history = [
        {"role": "user", "content": "What is NCB's revenue?"},
        {"role": "assistant", "content": "NCB revenue was $50B."},
    ]

    with (
        patch.object(
            orchestrator,
            "_smart_optimize",
            new_callable=AsyncMock,
            return_value={
                "needs_clarification": False,
                "clarification_reason": None,
                "clarification_question": None,
                "optimized_query": "NCB profit 2023",
                "routing": {"use_financial": False, "use_web_search": True},
                "defaults_applied": [],
                "resolved_years": [],
            },
        ),
        patch.object(
            orchestrator,
            "_execute_web_search",
            new_callable=AsyncMock,
            return_value={
                "search_results": {},
                "sources": [],
                "context": "NCB profit context from web",
            },
        ),
        patch.object(
            orchestrator,
            "_synthesize",
            new_callable=AsyncMock,
            return_value="NCB's 2023 profit was $10B.",
        ),
        patch.object(orchestrator, "_track_cost"),
    ):
        result = await orchestrator.run(
            query="What about profit?",
            conversation_history=prior_history,
            enable_web_search=True,
            enable_financial_data=False,
        )

    history = result["conversation_history"]
    assert history[-2] == {"role": "user", "content": "What about profit?"}
    assert history[-1] == {"role": "assistant", "content": "NCB's 2023 profit was $10B."}
    assert len(history) == 4  # 2 prior + 2 new


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_with_empty_history_creates_new_history(financial_manager):
    """AgentOrchestrator.run() with no prior history returns a two-entry history."""
    from app.agent import AgentOrchestrator

    orchestrator = AgentOrchestrator(financial_manager=financial_manager)

    with (
        patch.object(
            orchestrator,
            "_smart_optimize",
            new_callable=AsyncMock,
            return_value={
                "needs_clarification": False,
                "clarification_reason": None,
                "clarification_question": None,
                "optimized_query": "NCB revenue",
                "routing": {"use_financial": False, "use_web_search": True},
                "defaults_applied": [],
                "resolved_years": [],
            },
        ),
        patch.object(
            orchestrator,
            "_execute_web_search",
            new_callable=AsyncMock,
            return_value={"search_results": {}, "sources": [], "context": "some context"},
        ),
        patch.object(
            orchestrator,
            "_synthesize",
            new_callable=AsyncMock,
            return_value="NCB revenue is $50B.",
        ),
        patch.object(orchestrator, "_track_cost"),
    ):
        result = await orchestrator.run(
            query="NCB revenue",
            conversation_history=None,
            enable_web_search=True,
            enable_financial_data=False,
        )

    history = result["conversation_history"]
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "NCB revenue"}
    assert history[1] == {"role": "assistant", "content": "NCB revenue is $50B."}
```

- [ ] **Step 2: Run the tests and confirm they fail**

```
cd fastapi_app && python -m pytest tests/unit/test_agent.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.agent'` (the file doesn't exist yet).

---

## Task 2: Create `app/agent.py` — restore, split models, fix history

**Files:**
- Create: `fastapi_app/app/agent.py`

- [ ] **Step 1: Copy the archived file to its new location**

```
copy fastapi_app\app\_archive\agent_orchestrator.py fastapi_app\app\agent.py
```

(On Linux/Mac: `cp fastapi_app/app/_archive/agent_orchestrator.py fastapi_app/app/agent.py`)

- [ ] **Step 2: Replace the module docstring — remove the ARCHIVED header**

In `fastapi_app/app/agent.py`, replace the entire block at the top of the file (lines 1–15) with:

```python
"""
AgentOrchestrator — 3-phase agent combining Google Search grounding with JSE
financial-DB tool calling via Gemini function calling.

Architecture: 3-phase pipeline
1. Smart Optimization - Single LLM call for clarification, routing, pronoun resolution
2. Tool Execution - Financial query and/or web search
3. Response Synthesis - Combine results into natural response
"""
```

- [ ] **Step 3: Replace `__init__` — split into flash and pro model attributes**

Find and replace `__init__` (inside `AgentOrchestrator`):

```python
# BEFORE:
def __init__(self, financial_manager: Any):
    self.financial_manager = financial_manager
    self.client = get_genai_client()
    self.model_name = "gemini-2.5-flash"
    self._phase_costs: List[PhaseCost] = []

# AFTER:
def __init__(self, financial_manager: Any):
    self.financial_manager = financial_manager
    self.client = get_genai_client()
    self.flash_model = "gemini-2.5-flash"
    self.pro_model = "gemini-2.5-pro"
    self._phase_costs: List[PhaseCost] = []
```

- [ ] **Step 4: Update `_track_cost` — require explicit model argument**

```python
# BEFORE:
def _track_cost(self, response: Any, phase: str) -> None:
    """Track cost from a Gemini response."""
    cost = calculate_cost_from_response(self.model_name, response, phase)
    record_ai_cost(
        model=cost.model,
        phase=cost.phase,
        input_tokens=cost.token_usage.input_tokens,
        output_tokens=cost.token_usage.output_tokens,
        input_cost=cost.input_cost,
        output_cost=cost.output_cost,
        total_cost=cost.total_cost,
        cached_tokens=cost.token_usage.cached_tokens,
    )
    self._add_phase_cost(
        phase=phase,
        model=self.model_name,
        input_tokens=cost.token_usage.input_tokens,
        output_tokens=cost.token_usage.output_tokens,
        cached_tokens=cost.token_usage.cached_tokens,
        input_cost=cost.input_cost,
        output_cost=cost.output_cost,
        total_cost=cost.total_cost,
    )

# AFTER:
def _track_cost(self, response: Any, phase: str, model: str) -> None:
    """Track cost from a Gemini response."""
    cost = calculate_cost_from_response(model, response, phase)
    record_ai_cost(
        model=cost.model,
        phase=cost.phase,
        input_tokens=cost.token_usage.input_tokens,
        output_tokens=cost.token_usage.output_tokens,
        input_cost=cost.input_cost,
        output_cost=cost.output_cost,
        total_cost=cost.total_cost,
        cached_tokens=cost.token_usage.cached_tokens,
    )
    self._add_phase_cost(
        phase=phase,
        model=model,
        input_tokens=cost.token_usage.input_tokens,
        output_tokens=cost.token_usage.output_tokens,
        cached_tokens=cost.token_usage.cached_tokens,
        input_cost=cost.input_cost,
        output_cost=cost.output_cost,
        total_cost=cost.total_cost,
    )
```

- [ ] **Step 5: Update `_smart_optimize` — use flash model**

In `_smart_optimize`, change the `generate_content` call and `_track_cost` call:

```python
# BEFORE:
response = self.client.models.generate_content(
    model=self.model_name,
    contents=prompt,
    config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=512),
)
self._track_cost(response, "smart_optimization")

# AFTER:
response = self.client.models.generate_content(
    model=self.flash_model,
    contents=prompt,
    config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=512),
)
self._track_cost(response, "smart_optimization", self.flash_model)
```

- [ ] **Step 6: Update `_execute_financial` — use flash model**

In `_execute_financial`, change the `generate_content` call and `_track_cost` call:

```python
# BEFORE:
response = self.client.models.generate_content(
    model=self.model_name,
    contents=[types.Content(role="user", parts=[types.Part.from_text(text=query)])],
    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[tool],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY", allowed_function_names=["query_financial_data"]
            )
        ),
        temperature=0.3,
        max_output_tokens=512,
    ),
)
self._track_cost(response, "financial_extraction")

# AFTER:
response = self.client.models.generate_content(
    model=self.flash_model,
    contents=[types.Content(role="user", parts=[types.Part.from_text(text=query)])],
    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[tool],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY", allowed_function_names=["query_financial_data"]
            )
        ),
        temperature=0.3,
        max_output_tokens=512,
    ),
)
self._track_cost(response, "financial_extraction", self.flash_model)
```

- [ ] **Step 7: Update `_execute_web_search` — use flash model**

In `_execute_web_search`, change the `generate_content` call and `_track_cost` call:

```python
# BEFORE:
response = self.client.models.generate_content(
    model=self.model_name,
    contents=[types.Content(role="user", parts=[types.Part.from_text(text=query)])],
    config=types.GenerateContentConfig(
        system_instruction="Search for Jamaica Stock Exchange information. Cite sources.",
        tools=[tool],
        temperature=0.7,
        max_output_tokens=2048,
    ),
)
self._track_cost(response, "web_search")

# AFTER:
response = self.client.models.generate_content(
    model=self.flash_model,
    contents=[types.Content(role="user", parts=[types.Part.from_text(text=query)])],
    config=types.GenerateContentConfig(
        system_instruction="Search for Jamaica Stock Exchange information. Cite sources.",
        tools=[tool],
        temperature=0.7,
        max_output_tokens=2048,
    ),
)
self._track_cost(response, "web_search", self.flash_model)
```

- [ ] **Step 8: Update `_synthesize` — use pro model**

In `_synthesize`, change the `generate_content` call and `_track_cost` call:

```python
# BEFORE:
response = self.client.models.generate_content(
    model=self.model_name,
    contents=user_prompt,
    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.7,
        max_output_tokens=4096,
    ),
)
self._track_cost(response, "synthesis")

# AFTER:
response = self.client.models.generate_content(
    model=self.pro_model,
    contents=user_prompt,
    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.7,
        max_output_tokens=4096,
    ),
)
self._track_cost(response, "synthesis", self.pro_model)
```

- [ ] **Step 9: Fix the conversation-history bug in `run()`**

In `run()`, after these two lines (near the end of the try block):

```python
follow_up = self._extract_follow_up_questions(response_text)
clean_response = self._clean_response_text(response_text)
```

Add:

```python
updated_history = list(conversation_history) if conversation_history else []
updated_history.append({"role": "user", "content": query})
updated_history.append({"role": "assistant", "content": clean_response})
if len(updated_history) > 20:
    updated_history = updated_history[-20:]
```

Then in the return dict, change:

```python
# BEFORE:
"conversation_history": conversation_history,

# AFTER:
"conversation_history": updated_history,
```

- [ ] **Step 10: Run the unit tests — confirm they pass**

```
cd fastapi_app && python -m pytest tests/unit/test_agent.py -v
```

Expected output:
```
tests/unit/test_agent.py::test_run_appends_new_turn_to_conversation_history PASSED
tests/unit/test_agent.py::test_run_with_empty_history_creates_new_history PASSED
2 passed
```

- [ ] **Step 11: Commit**

```
git add fastapi_app/app/agent.py fastapi_app/tests/unit/test_agent.py
git commit -m "feat(agent): restore AgentOrchestrator with split models and history fix"
```

---

## Task 3: Write failing endpoint integration test

**Files:**
- Modify: `fastapi_app/tests/test_api.py`

- [ ] **Step 1: Add the test at the bottom of `tests/test_api.py`**

```python
@pytest.mark.unit
def test_chat_stream_passes_enable_financial_data_to_orchestrator(test_client):
    """
    /chat/stream must delegate to app.state.agent_orchestrator and pass
    enable_financial_data from the request body.
    """
    from unittest.mock import AsyncMock, MagicMock
    from app.main import app

    mock_result = {
        "response": "NCB 2023 revenue was $50B.",
        "data_found": True,
        "record_count": 3,
        "filters_used": None,
        "data_preview": None,
        "conversation_history": [
            {"role": "user", "content": "NCB revenue 2023"},
            {"role": "assistant", "content": "NCB 2023 revenue was $50B."},
        ],
        "warnings": None,
        "suggestions": None,
        "chart": None,
        "sources": None,
        "web_search_results": None,
        "tools_executed": ["query_financial_data"],
        "needs_clarification": False,
        "clarification_question": None,
        "cost_summary": None,
    }

    mock_orchestrator = MagicMock()
    mock_orchestrator.run = AsyncMock(return_value=mock_result)
    app.state.agent_orchestrator = mock_orchestrator

    response = test_client.post(
        "/chat/stream",
        json={
            "query": "NCB revenue 2023",
            "enable_financial_data": True,
            "enable_web_search": False,
            "memory_enabled": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["record_count"] == 3
    assert data["tools_executed"] == ["query_financial_data"]

    mock_orchestrator.run.assert_called_once_with(
        query="NCB revenue 2023",
        conversation_history=None,
        enable_web_search=False,
        enable_financial_data=True,
    )
```

- [ ] **Step 2: Run the test and confirm it fails**

```
cd fastapi_app && python -m pytest tests/test_api.py::test_chat_stream_passes_enable_financial_data_to_orchestrator -v
```

Expected: Test fails — either `AttributeError: 'State' object has no attribute 'agent_orchestrator'` or the endpoint still calls `AgentV2` instead of the orchestrator.

---

## Task 4: Update `main.py` — wire AgentOrchestrator into lifespan and endpoint

**Files:**
- Modify: `fastapi_app/app/main.py`

- [ ] **Step 1: Add the import for AgentOrchestrator**

Near the top of `fastapi_app/app/main.py`, alongside the existing agent import:

```python
# BEFORE (existing line ~line 35):
from app.agent_v2 import AgentV2

# AFTER — add the line below it:
from app.agent_v2 import AgentV2
from app.agent import AgentOrchestrator
```

- [ ] **Step 2: Instantiate AgentOrchestrator in lifespan**

In the `lifespan` context manager, after the `financial_manager` initialization block (which ends around `app.state.financial_manager = None`), add:

```python
        # -----------------------
        # Initialize AgentOrchestrator (financial DB + web search)
        # -----------------------
        app.state.agent_orchestrator = AgentOrchestrator(
            financial_manager=app.state.financial_manager
        )
        logger.info("AgentOrchestrator initialized")
```

The `financial_manager` block ends with the `except` block that sets `app.state.financial_manager = None`. Place the new block immediately after it, before the job store block.

- [ ] **Step 3: Update the `/chat/stream` endpoint**

Find the `chat_stream` function (around line 951). Replace the try block body:

```python
# BEFORE:
    try:
        # Create simplified agent (uses Gemini 2.5 Pro with Google Search grounding)
        agent = AgentV2()

        # Run the agent
        result = await agent.run(
            query=request.query,
            conversation_history=request.conversation_history,
            enable_web_search=request.enable_web_search,
        )

        # Build response (compatible with AgentChatResponse)
        response = AgentChatResponse(
            response=result["response"],
            data_found=result["data_found"],
            record_count=result["record_count"],
            filters_used=result.get("filters_used"),
            data_preview=result.get("data_preview"),
            conversation_history=result["conversation_history"] if request.memory_enabled else None,
            warnings=result.get("warnings"),
            suggestions=result.get("suggestions"),
            chart=result.get("chart"),
            sources=result.get("sources"),
            web_search_results=result.get("web_search_results"),
            tools_executed=result.get("tools_executed"),
            needs_clarification=result.get("needs_clarification", False),
            clarification_question=result.get("clarification_question"),
        )

# AFTER:
    try:
        agent = app.state.agent_orchestrator

        result = await agent.run(
            query=request.query,
            conversation_history=request.conversation_history,
            enable_web_search=request.enable_web_search,
            enable_financial_data=request.enable_financial_data,
        )

        response = AgentChatResponse(
            response=result["response"],
            data_found=result["data_found"],
            record_count=result["record_count"],
            filters_used=result.get("filters_used"),
            data_preview=result.get("data_preview"),
            conversation_history=result["conversation_history"] if request.memory_enabled else None,
            warnings=result.get("warnings"),
            suggestions=result.get("suggestions"),
            chart=result.get("chart"),
            sources=result.get("sources"),
            web_search_results=result.get("web_search_results"),
            tools_executed=result.get("tools_executed"),
            needs_clarification=result.get("needs_clarification", False),
            clarification_question=result.get("clarification_question"),
            cost_summary=result.get("cost_summary"),
        )
```

- [ ] **Step 4: Run the integration test — confirm it passes**

```
cd fastapi_app && python -m pytest tests/test_api.py::test_chat_stream_passes_enable_financial_data_to_orchestrator -v
```

Expected:
```
tests/test_api.py::test_chat_stream_passes_enable_financial_data_to_orchestrator PASSED
```

- [ ] **Step 5: Run the full test suite to check for regressions**

```
cd fastapi_app && python -m pytest tests/ -v --ignore=tests/integration --ignore=tests/uat -x
```

Expected: all previously passing tests still pass; no new failures.

- [ ] **Step 6: Commit**

```
git add fastapi_app/app/main.py fastapi_app/tests/test_api.py
git commit -m "feat(main): wire AgentOrchestrator into /chat/stream with financial tool calling"
```

---

## Task 5: Update `_archive/README.md`

**Files:**
- Modify: `fastapi_app/app/_archive/README.md`

- [ ] **Step 1: Update the contents table and restoration steps**

Replace the existing content with:

```markdown
# _archive

This directory holds code that is **not active** but is preserved for reference.

Files here are intentionally excluded from the running application.

## Contents

| File | Original path | What it is | Status |
|---|---|---|---|
| `agent_orchestrator.py` | `app/agent.py` | `AgentOrchestrator` — a 3-phase agent (clarification → routing → synthesis) with both Google Search grounding *and* JSE financial-DB tool calling via Gemini function calling. | Restored in R10 (ATS-334). `app/agent.py` is the live version. This copy is kept for historical reference. |

## Restoration (R10 — ATS-334)

Completed. `AgentOrchestrator` is live at `app/agent.py` and wired into `/chat/stream`.
Changes made:
- Removed `ARCHIVED` header from module docstring
- Split `model_name` into `flash_model` ("gemini-2.5-flash", phases 1–2) and `pro_model` ("gemini-2.5-pro", phase 3)
- Fixed conversation-history bug (now appends new turn before returning)
- Instantiated in `lifespan` on `app.state.agent_orchestrator`
```

- [ ] **Step 2: Commit**

```
git add fastapi_app/app/_archive/README.md
git commit -m "docs(_archive): mark R10 complete — AgentOrchestrator restored in ATS-334"
```

---

## Definition of Done

- [ ] `python -m pytest tests/unit/test_agent.py -v` — 2 tests pass
- [ ] `python -m pytest tests/test_api.py::test_chat_stream_passes_enable_financial_data_to_orchestrator -v` — passes
- [ ] Full unit suite `python -m pytest tests/ -v --ignore=tests/integration --ignore=tests/uat -x` — no regressions
- [ ] `app/agent.py` exists, has no `ARCHIVED` header, no `self.model_name`, has `self.flash_model` and `self.pro_model`
- [ ] `/chat/stream` endpoint uses `app.state.agent_orchestrator` (not a new `AgentV2()`)
- [ ] `_archive/README.md` marks R10 complete
