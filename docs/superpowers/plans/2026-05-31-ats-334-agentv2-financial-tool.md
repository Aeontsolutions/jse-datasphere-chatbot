# ATS-334 (Option 2): Add `query_financial_data` to AgentV2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `/chat/stream`'s `AgentV2` a path to the JSE BigQuery financial data via a `query_financial_data` Gemini function-calling tool, so DB-answerable questions are answered from the database (~1s) instead of slow web grounding (~87s baseline).

**Architecture:** Lift the proven function-calling primitives out of `_archive/agent_orchestrator.py` into a new live `app/financial_tool.py`. AgentV2 gains a sequential financial path: a cheap `gemini-2.5-flash` decide/extract call (AUTO function-calling) → execute the BigQuery query off-thread → synthesize the answer with `gemini-2.5-pro`. If the model declines the tool, fall through to the existing Google-Search path unchanged. `enable_financial_data=False` reproduces today's behavior byte-for-byte.

**Tech Stack:** Python 3, FastAPI, `google-genai` SDK (`google.genai.types`), Pydantic, pytest. Spec: [docs/superpowers/specs/2026-05-31-ats-334-agentv2-financial-tool-design.md](../specs/2026-05-31-ats-334-agentv2-financial-tool-design.md).

---

## Conventions

- **All `pytest` / `python` commands run from the `fastapi_app/` directory** (imports are `from app...`, conftest lives in `fastapi_app/tests/`). On Windows PowerShell: `Set-Location fastapi_app` first, or prefix each command.
- Tests live flat in `fastapi_app/tests/` (e.g. `tests/test_financial_tool.py`) — match the existing convention used by `test_streaming.py`, `test_financial_utils.py`. (A `tests/unit/` subdir also exists for util tests; these new agent tests go flat.)
- **AgentV2 has no prior unit tests** — these are the first. The Gemini-client mock pattern: patch `app.agent_v2.get_genai_client` and drive `client.models.generate_content` via `return_value` / `side_effect`.
- **CRITICAL — mock responses must set `usage_metadata = None`.** `AgentV2._track_cost` runs for real in these tests; `TokenUsage.from_response` (in `app/utils/cost_tracking.py`) treats *any* truthy `usage_metadata` as real token counts. A bare `MagicMock()` response auto-creates a truthy `usage_metadata`, so its attributes become `MagicMock`s, and building `PhaseCost(input_tokens=<MagicMock>, ...)` raises a Pydantic `ValidationError`. Setting `usage_metadata = None` takes the zero-cost branch (`return cls()`), exercising `_track_cost` safely. Every mock response builder below does this.

## File Structure

| File | Responsibility |
|---|---|
| `fastapi_app/app/financial_tool.py` | **new** — the reusable financial function-calling primitives: tool declaration, `Tool` wrapper, async query executor, context formatter, function-call extractor. One clear job: "the `query_financial_data` tool." |
| `fastapi_app/app/agent_v2.py` | **modify** — financial path (`_try_financial`), `run()` integration, `financial_manager` ctor arg, `_track_cost` model arg, decision prompt + model constant. |
| `fastapi_app/app/main.py` | **modify** — `/chat/stream` injects `financial_manager`, forwards `enable_financial_data`. |
| `fastapi_app/app/_archive/README.md` | **modify** — one-line pointer noting the primitives now live in `app/financial_tool.py`. |
| `fastapi_app/tests/test_financial_tool.py` | **new** — unit tests for the new module. |
| `fastapi_app/tests/test_agent_v2_financial.py` | **new** — unit tests for the AgentV2 financial path. |
| `fastapi_app/tests/test_chat_stream_financial.py` | **new** — endpoint wiring test. |
| `evals/personas/chatstream_ncb_revenue_lookup.yaml` | **new** — financial chat_stream persona. |
| `evals/personas/chatstream_compare_gk_ncb_profit.yaml` | **new** — financial chat_stream persona. |
| `evals/personas/chatstream_mixed_financial_and_news.yaml` | **new** — mixed persona (measured at minimum bar). |

---

## Task 1: `financial_tool.py` — declaration, Tool wrapper, extractor

**Files:**
- Create: `fastapi_app/app/financial_tool.py`
- Test: `fastapi_app/tests/test_financial_tool.py`

- [ ] **Step 1: Write the failing test**

Create `fastapi_app/tests/test_financial_tool.py`:

```python
"""Unit tests for app.financial_tool — the query_financial_data primitives."""

from unittest.mock import MagicMock

import pytest

from app.financial_tool import (
    extract_query_financial_data_call,
    get_financial_data_tool_declaration,
    get_financial_tool,
)


def _fc_response(name="query_financial_data", args=None):
    """Build a Gemini-style response carrying a single function_call part."""
    fc = MagicMock()
    fc.name = name
    fc.args = args if args is not None else {"symbols": ["NCB"]}
    part = MagicMock()
    part.function_call = fc
    content = MagicMock()
    content.parts = [part]
    candidate = MagicMock()
    candidate.content = content
    response = MagicMock()
    response.candidates = [candidate]
    return response


class TestToolDeclaration:
    def test_declaration_name_and_params(self):
        decl = get_financial_data_tool_declaration()
        assert decl.name == "query_financial_data"
        props = decl.parameters.properties
        assert set(props.keys()) == {"symbols", "years", "standard_items"}

    def test_get_financial_tool_wraps_declaration(self):
        tool = get_financial_tool()
        assert tool.function_declarations[0].name == "query_financial_data"


class TestExtractCall:
    def test_extracts_matching_call(self):
        fc = extract_query_financial_data_call(
            _fc_response(args={"symbols": ["NCB"], "years": ["2023"]})
        )
        assert fc is not None
        assert fc.name == "query_financial_data"

    def test_returns_none_when_no_candidates(self):
        response = MagicMock()
        response.candidates = []
        assert extract_query_financial_data_call(response) is None

    def test_returns_none_for_other_function(self):
        assert extract_query_financial_data_call(_fc_response(name="something_else")) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `fastapi_app/`): `python -m pytest tests/test_financial_tool.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.financial_tool'`.

- [ ] **Step 3: Write minimal implementation**

Create `fastapi_app/app/financial_tool.py`:

```python
"""
query_financial_data tool primitives for Gemini function calling.

Lifted from the archived AgentOrchestrator (see _archive/agent_orchestrator.py)
into a live, reusable module so AgentV2 can call the JSE BigQuery financial
data on /chat/stream (ATS-334, Option 2).

Contents:
- get_financial_data_tool_declaration() / get_financial_tool(): the tool schema
- execute_financial_query(): build filters, query BigQuery (off-thread), chart, sources
- build_financial_context(): compact text summary of records for synthesis
- extract_query_financial_data_call(): pull the function call from a response
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from google.genai import types

from app.charting import generate_chart
from app.logging_config import get_logger
from app.models import FinancialDataFilters, FinancialDataRecord

logger = get_logger(__name__)


def get_financial_data_tool_declaration() -> types.FunctionDeclaration:
    """Define the financial query tool for Gemini function calling."""
    return types.FunctionDeclaration(
        name="query_financial_data",
        description="""Query financial data from the Jamaica Stock Exchange (JSE) database.
Use this tool when the user asks about:
- Company financial metrics (revenue, profit, EPS, margins, assets, liabilities)
- Financial comparisons between companies
- Historical financial data for specific years

Available metrics: revenue, net_profit, gross_profit, operating_profit, eps,
total_assets, total_liabilities, shareholders_equity, gross_profit_margin,
net_profit_margin, operating_profit_margin, roe, roa, current_ratio, debt_to_equity.""",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "symbols": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                    description="Stock trading symbols (e.g., ['NCB', 'JBG', 'GK']). Use uppercase.",
                ),
                "years": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                    description="Years to filter by (e.g., ['2022', '2023', '2024']).",
                ),
                "standard_items": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                    description="Financial metrics to retrieve (e.g., ['revenue', 'net_profit']).",
                ),
            },
        ),
    )


def get_financial_tool() -> types.Tool:
    """Wrap the financial data declaration in a Gemini Tool."""
    return types.Tool(function_declarations=[get_financial_data_tool_declaration()])


def extract_query_financial_data_call(response: Any) -> Optional[Any]:
    """Return the first query_financial_data function_call part, or None."""
    if not getattr(response, "candidates", None):
        return None
    candidate = response.candidates[0]
    content = getattr(candidate, "content", None)
    if not content:
        return None
    for part in getattr(content, "parts", None) or []:
        fc = getattr(part, "function_call", None)
        if fc and getattr(fc, "name", None) == "query_financial_data":
            return fc
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_financial_tool.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add fastapi_app/app/financial_tool.py fastapi_app/tests/test_financial_tool.py
git commit -m "feat(financial_tool): add query_financial_data declaration + extractor"
```

---

## Task 2: `financial_tool.py` — async executor + context formatter

**Files:**
- Modify: `fastapi_app/app/financial_tool.py` (append two functions)
- Test: `fastapi_app/tests/test_financial_tool.py` (append two test classes)

- [ ] **Step 1: Write the failing test**

Append to `fastapi_app/tests/test_financial_tool.py`:

```python
import asyncio

from app.financial_tool import build_financial_context, execute_financial_query
from app.models import FinancialDataRecord


def _record(company="NCB Financial Group", symbol="NCB", year="2023",
            item_name="revenue", item=123456789.0):
    return FinancialDataRecord(
        company=company, symbol=symbol, year=year, standard_item=item_name,
        item=item, unit_multiplier=1, formatted_value=f"{item:,.0f}",
    )


class TestExecuteFinancialQuery:
    def test_builds_filters_and_calls_query_data(self):
        manager = MagicMock()
        manager.metadata = {}  # no associations -> skip post-processing
        manager.query_data.return_value = [_record()]

        records, filters, chart, sources = asyncio.run(
            execute_financial_query(
                manager,
                {"symbols": ["ncb"], "years": [2023], "standard_items": ["Net Profit"]},
            )
        )

        # query_data called once with a FinancialDataFilters
        manager.query_data.assert_called_once()
        passed = manager.query_data.call_args.args[0]
        assert passed.symbols == ["NCB"]              # uppercased
        assert passed.years == ["2023"]               # stringified
        assert passed.standard_items == ["net_profit"]  # lower + underscored
        assert len(records) == 1
        assert sources[0]["type"] == "database"

    def test_empty_results_no_chart(self):
        manager = MagicMock()
        manager.metadata = {}
        manager.query_data.return_value = []
        records, filters, chart, sources = asyncio.run(
            execute_financial_query(manager, {"symbols": ["NCB"]})
        )
        assert records == []
        assert chart is None


class TestBuildFinancialContext:
    def test_empty(self):
        assert build_financial_context([]) == "No financial data found."

    def test_groups_by_company_and_formats(self):
        ctx = build_financial_context([_record(item=5_000_000.0)])
        assert "NCB Financial Group" in ctx
        assert "revenue" in ctx
        assert "$5.00M" in ctx
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_financial_tool.py -k "ExecuteFinancialQuery or BuildFinancialContext" -v`
Expected: FAIL — `ImportError: cannot import name 'build_financial_context'`.

- [ ] **Step 3: Write minimal implementation**

Append to `fastapi_app/app/financial_tool.py`:

```python
async def execute_financial_query(
    financial_manager: Any,
    args: Dict[str, Any],
) -> Tuple[List[FinancialDataRecord], FinancialDataFilters, Optional[Dict], List[Dict]]:
    """Execute financial data query and return results with source metadata.

    The BigQuery call (financial_manager.query_data) is synchronous/blocking, so
    it is run via asyncio.to_thread to avoid blocking the event loop.
    """
    start_time = time.time()

    try:
        raw_symbols = args.get("symbols") or []
        raw_years = args.get("years") or []
        raw_items = args.get("standard_items") or []

        symbols = [s.upper() for s in raw_symbols if s]
        years = [str(y) for y in raw_years if y]
        standard_items = [item.lower().replace(" ", "_") for item in raw_items if item]

        filters = FinancialDataFilters(
            companies=[],
            symbols=symbols,
            years=years,
            standard_items=standard_items,
            interpretation=f"Agent query: symbols={symbols}, years={years}, items={standard_items}",
            data_availability_note="",
            is_follow_up=False,
            context_used="",
        )

        # Post-process filters using associations from metadata (e.g. symbol->company)
        if financial_manager.metadata and "associations" in financial_manager.metadata:
            filters_dict = filters.model_dump()
            filters_dict = financial_manager._post_process_filters(filters_dict)
            filters = FinancialDataFilters(**filters_dict)

        # Query the data (blocking BigQuery call -> off-thread)
        records = await asyncio.to_thread(financial_manager.query_data, filters)

        # Generate chart if applicable
        chart_spec = None
        if records:
            chart_data = generate_chart(records, "")
            if chart_data:
                chart_spec = chart_data

        # Build source citations
        symbols_str = ", ".join(filters.symbols) if filters.symbols else "all"
        years_str = ", ".join(filters.years) if filters.years else "all years"
        source_entry = {
            "type": "database",
            "description": f"JSE Financial Database: {symbols_str} ({years_str})",
            "table": "financial_data",
        }
        if filters.symbols:
            source_entry["symbols"] = filters.symbols
        if filters.years:
            source_entry["years"] = filters.years
        if filters.standard_items:
            source_entry["metrics"] = filters.standard_items
        sources = [source_entry]

        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Financial query: {len(records)} records in {duration_ms:.2f}ms")

        return records, filters, chart_spec, sources

    except Exception as e:
        logger.error(f"Financial query failed: {e}", exc_info=True)
        raise


def build_financial_context(records: List[FinancialDataRecord]) -> str:
    """Build a compact context string from financial records (caps at 50)."""
    if not records:
        return "No financial data found."

    lines = [f"Financial Data ({len(records)} records):"]
    by_company: Dict[str, List[FinancialDataRecord]] = {}

    for record in records[:50]:
        company = record.company or record.symbol
        if company not in by_company:
            by_company[company] = []
        by_company[company].append(record)

    for company, company_records in by_company.items():
        lines.append(f"\n{company}:")
        for r in company_records:
            year = r.year or "N/A"
            metric = r.standard_item or "metric"
            if r.item is not None:
                if abs(r.item) >= 1_000_000:
                    formatted = f"${r.item/1_000_000:,.2f}M"
                elif abs(r.item) >= 1_000:
                    formatted = f"${r.item/1_000:,.2f}K"
                else:
                    formatted = f"{r.item:,.2f}"
                lines.append(f"  - {metric} ({year}): {formatted}")
            else:
                lines.append(f"  - {metric} ({year}): {r.formatted_value}")

    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_financial_tool.py -v`
Expected: PASS (all tests, ~9).

- [ ] **Step 5: Commit**

```bash
git add fastapi_app/app/financial_tool.py fastapi_app/tests/test_financial_tool.py
git commit -m "feat(financial_tool): add async execute_financial_query + build_financial_context"
```

---

## Task 3: AgentV2 plumbing — ctor, cost model arg, metadata context, constants

**Files:**
- Modify: `fastapi_app/app/agent_v2.py`
- Test: `fastapi_app/tests/test_agent_v2_financial.py`

- [ ] **Step 1: Write the failing test**

Create `fastapi_app/tests/test_agent_v2_financial.py`:

```python
"""Tests for the AgentV2 financial-data path (ATS-334)."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.agent_v2 import FINANCIAL_DECISION_MODEL, AgentV2


@pytest.fixture
def mock_genai_client():
    with patch("app.agent_v2.get_genai_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        yield mock_client


def test_constructor_stores_financial_manager(mock_genai_client):
    mgr = MagicMock()
    agent = AgentV2(financial_manager=mgr)
    assert agent.financial_manager is mgr


def test_constructor_defaults_financial_manager_none(mock_genai_client):
    agent = AgentV2()
    assert agent.financial_manager is None


def test_metadata_context_includes_symbols_and_years(mock_genai_client):
    mgr = MagicMock()
    mgr.metadata = {"symbols": ["NCB", "GK"], "years": ["2022", "2023"]}
    agent = AgentV2(financial_manager=mgr)
    ctx = agent._get_metadata_context()
    assert "NCB" in ctx and "2023" in ctx


def test_metadata_context_empty_when_no_manager(mock_genai_client):
    agent = AgentV2()
    assert agent._get_metadata_context() == ""


def test_decision_model_is_flash():
    assert FINANCIAL_DECISION_MODEL == "gemini-2.5-flash"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_v2_financial.py -v`
Expected: FAIL — `ImportError: cannot import name 'FINANCIAL_DECISION_MODEL'`.

- [ ] **Step 3: Write minimal implementation**

In `fastapi_app/app/agent_v2.py`, add to the imports near the top (after the existing `from app.utils.monitoring import record_ai_cost`):

```python
from app.financial_tool import (
    build_financial_context,
    execute_financial_query,
    extract_query_financial_data_call,
    get_financial_tool,
)
```

Add module constants just below `SYSTEM_PROMPT_NO_SEARCH = ...` (around line 75):

```python
# Model used for the cheap financial decide/extract call (Phase A).
FINANCIAL_DECISION_MODEL = "gemini-2.5-flash"

# Phase-A system prompt: decide whether the question is a JSE financial-metric
# lookup and, if so, call query_financial_data. {metadata_context} is filled with
# available symbols/years so extracted symbols are valid.
FINANCIAL_DECISION_PROMPT = """You decide whether a user's question can be answered from the Jamaica Stock Exchange (JSE) financial-statement database.

Call the query_financial_data tool ONLY when the user asks for specific company financial metrics (revenue, profit, EPS, margins, assets, liabilities, ratios) for JSE-listed companies, optionally for specific years.

Do NOT call the tool for: news, announcements, qualitative or opinion questions, market commentary, current stock prices, IPO timelines, or anything outside the metric list. If the question is not a financial-metric lookup, do not call the tool at all.

When you call the tool, use uppercase trading symbols.
{metadata_context}"""
```

Update `__init__` (currently around line 90):

```python
    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        financial_manager: Any = None,
    ):
        """
        Initialize the agent.

        Args:
            model_name: Gemini model for synthesis/web. Defaults to gemini-2.5-pro.
            financial_manager: FinancialDataManager for the query_financial_data
                tool. When None, the financial path is skipped (web/plain only).
        """
        self.client = get_genai_client()
        self.model_name = model_name
        self.financial_manager = financial_manager
        self._phase_costs: List[PhaseCost] = []
```

Update `_track_cost` (currently around line 144) to accept an optional model:

```python
    def _track_cost(self, response: Any, phase: str, model: Optional[str] = None) -> None:
        """Track cost from a Gemini response (model defaults to self.model_name)."""
        model = model or self.model_name
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

Add a helper method (place it just after `_build_contents`, before `_extract_grounding_metadata`):

```python
    def _get_metadata_context(self) -> str:
        """Available symbols/years from the financial manager metadata.

        Injected into the Phase-A decision prompt so the model emits valid
        trading symbols (the tool exposes only `symbols`, not company names).
        """
        manager = self.financial_manager
        if not manager or not getattr(manager, "metadata", None):
            return ""
        metadata = manager.metadata
        parts = []
        if metadata.get("symbols"):
            parts.append(f"Available symbols: {', '.join(metadata['symbols'][:50])}")
        if metadata.get("years"):
            parts.append(f"Available years: {', '.join(metadata['years'])}")
        return "\n".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_v2_financial.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Regression guard — imports still resolve**

Run (from `fastapi_app/`): `python -c "import app.agent_v2; import app.main; print('ok')"`
Expected: `ok` (the new `from app.financial_tool import ...` and ctor/`_track_cost` signature changes don't break imports; back-compat preserved — `AgentV2()` with no args, `_track_cost(resp, phase)` with no model still valid). There are no prior AgentV2 unit tests to run; the new file is the coverage.

- [ ] **Step 6: Commit**

```bash
git add fastapi_app/app/agent_v2.py fastapi_app/tests/test_agent_v2_financial.py
git commit -m "feat(agent_v2): financial_manager ctor arg, cost model arg, decision prompt scaffolding"
```

---

## Task 4: AgentV2 `_try_financial` + `run()` integration

**Files:**
- Modify: `fastapi_app/app/agent_v2.py`
- Test: `fastapi_app/tests/test_agent_v2_financial.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `fastapi_app/tests/test_agent_v2_financial.py`:

```python
from app.models import FinancialDataRecord


def _record():
    return FinancialDataRecord(
        company="NCB Financial Group", symbol="NCB", year="2023",
        standard_item="revenue", item=123456789.0, unit_multiplier=1,
        formatted_value="123,456,789",
    )


# NOTE: every builder sets usage_metadata = None so the real _track_cost takes
# the zero-cost branch instead of choking on MagicMock token counts (see the
# CRITICAL note in Conventions).

def _fc_response(args=None):
    """Phase-A response that calls query_financial_data."""
    fc = MagicMock()
    fc.name = "query_financial_data"
    fc.args = args if args is not None else {"symbols": ["NCB"], "years": ["2023"],
                                             "standard_items": ["revenue"]}
    part = MagicMock()
    part.function_call = fc
    content = MagicMock()
    content.parts = [part]
    candidate = MagicMock()
    candidate.content = content
    resp = MagicMock()
    resp.candidates = [candidate]
    resp.text = ""
    resp.usage_metadata = None
    return resp


def _decline_response():
    """Phase-A response with no function call (model declined)."""
    resp = MagicMock()
    resp.candidates = []
    resp.text = "I can help with that."
    resp.usage_metadata = None
    return resp


def _text_response(text):
    resp = MagicMock()
    resp.text = text
    resp.candidates = []
    resp.usage_metadata = None
    return resp


def _mock_manager(records):
    mgr = MagicMock()
    mgr.metadata = {"symbols": ["NCB"], "years": ["2023"]}
    mgr.query_data.return_value = records
    return mgr


def test_financial_path_calls_tool_and_synthesizes(mock_genai_client):
    # Phase A returns a function call; Phase B returns synthesized text.
    mock_genai_client.models.generate_content.side_effect = [
        _fc_response(),
        _text_response("NCB's 2023 revenue was J$123.5M."),
    ]
    agent = AgentV2(financial_manager=_mock_manager([_record()]))
    result = asyncio.run(agent.run(query="What was NCB revenue in 2023?"))

    assert result["tools_executed"] == ["query_financial_data"]
    assert result["record_count"] == 1
    assert result["filters_used"].symbols == ["NCB"]
    assert result["data_preview"] and len(result["data_preview"]) == 1
    assert "123.5M" in result["response"]
    # Two LLM calls: decide (flash) + synthesize (pro)
    assert mock_genai_client.models.generate_content.call_count == 2


def test_decline_falls_through_to_web(mock_genai_client):
    # Phase A declines; web path runs (1 more call) -> google_search.
    mock_genai_client.models.generate_content.side_effect = [
        _decline_response(),
        _text_response("Here is some recent JSE news."),
    ]
    agent = AgentV2(financial_manager=_mock_manager([]))
    result = asyncio.run(agent.run(query="Any recent JSE news?"))

    assert result["response"] == "Here is some recent JSE news."
    # query_financial_data NOT in tools_executed; query_data never called
    assert "query_financial_data" not in (result.get("tools_executed") or [])
    agent.financial_manager.query_data.assert_not_called()


def test_enable_financial_false_skips_financial(mock_genai_client):
    mock_genai_client.models.generate_content.return_value = _text_response("web answer")
    mgr = _mock_manager([_record()])
    agent = AgentV2(financial_manager=mgr)
    result = asyncio.run(agent.run(query="What was NCB revenue?", enable_financial_data=False))

    mgr.query_data.assert_not_called()
    # Only one (web) call, no Phase-A decision call
    assert mock_genai_client.models.generate_content.call_count == 1
    assert result["response"] == "web answer"


def test_no_manager_skips_financial(mock_genai_client):
    mock_genai_client.models.generate_content.return_value = _text_response("web answer")
    agent = AgentV2()  # financial_manager=None
    result = asyncio.run(agent.run(query="What was NCB revenue?"))
    assert result["response"] == "web answer"
    assert mock_genai_client.models.generate_content.call_count == 1


def test_financial_phase_a_uses_flash_model(mock_genai_client):
    mock_genai_client.models.generate_content.side_effect = [
        _fc_response(),
        _text_response("answer"),
    ]
    agent = AgentV2(financial_manager=_mock_manager([_record()]))
    asyncio.run(agent.run(query="NCB revenue 2023?"))
    first_call_kwargs = mock_genai_client.models.generate_content.call_args_list[0].kwargs
    assert first_call_kwargs["model"] == "gemini-2.5-flash"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_v2_financial.py -k "financial_path or decline or enable_financial_false or no_manager or phase_a_uses_flash" -v`
Expected: FAIL — `AttributeError: 'AgentV2' object has no attribute '_try_financial'` (or assertion failures: `run()` doesn't accept `enable_financial_data` / doesn't branch yet).

- [ ] **Step 3: Write minimal implementation**

In `fastapi_app/app/agent_v2.py`, add the `_try_financial` method just before `run` (after `_extract_grounding_metadata`):

```python
    async def _try_financial(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> Optional[Dict[str, Any]]:
        """Phase A (flash decide/extract) + Phase B (pro synthesize).

        Returns a full result dict if query_financial_data was invoked, or None
        to signal the caller should fall through to the web/plain path (model
        declined, or the path errored).
        """
        try:
            contents = self._build_contents(conversation_history, query)
            system_prompt = FINANCIAL_DECISION_PROMPT.format(
                metadata_context=self._get_metadata_context()
            )

            # Phase A: decide + extract (cheap/fast, AUTO function calling)
            decision = self.client.models.generate_content(
                model=FINANCIAL_DECISION_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[get_financial_tool()],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                    ),
                    temperature=0.3,
                    max_output_tokens=512,
                ),
            )
            self._track_cost(decision, "financial_extraction", model=FINANCIAL_DECISION_MODEL)

            fc = extract_query_financial_data_call(decision)
            if not fc:
                logger.info("AgentV2 financial: tool not called; falling through to web")
                return None

            args = dict(fc.args) if fc.args else {}
            records, filters, chart, sources = await execute_financial_query(
                self.financial_manager, args
            )

            # Phase B: synthesize from the retrieved data only (no tools)
            context = build_financial_context(records)
            synth_contents = self._build_contents(conversation_history, query)
            synth_contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=(
                                "Financial data retrieved from the JSE database:\n"
                                f"{context}\n\n"
                                "Answer the user's question using ONLY this data. "
                                "Use J$ and include a brief investment disclaimer."
                            )
                        )
                    ],
                )
            )
            synthesis = self.client.models.generate_content(
                model=self.model_name,
                contents=synth_contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT_NO_SEARCH,
                    temperature=0.3,
                    max_output_tokens=8192,
                ),
            )
            self._track_cost(synthesis, "synthesis")

            response_text = synthesis.text if synthesis.text else ""
            if not response_text:
                response_text = (
                    "I found financial data but could not generate a response. "
                    "Please try again."
                )

            updated_history = list(conversation_history) if conversation_history else []
            updated_history.append({"role": "user", "content": query})
            updated_history.append({"role": "assistant", "content": response_text})
            if len(updated_history) > 20:
                updated_history = updated_history[-20:]

            return {
                "response": response_text,
                "data_found": len(records) > 0,
                "record_count": len(records),
                "needs_clarification": False,
                "clarification_question": None,
                "tools_executed": ["query_financial_data"],
                "sources": sources if sources else None,
                "filters_used": filters,
                "data_preview": records[:10] if records else None,
                "chart": chart,
                "web_search_results": None,
                "suggestions": None,
                "conversation_history": updated_history,
                "warnings": None,
                "cost_summary": self._build_cost_summary(),
            }

        except Exception as e:
            logger.error(f"AgentV2 financial path failed: {e}", exc_info=True)
            return None
```

Now update `run`'s signature and insert the financial branch. Change the signature (around line 267) to add the parameter:

```python
    async def run(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        enable_web_search: bool = True,
        enable_financial_data: bool = True,
    ) -> Dict[str, Any]:
```

Then, immediately after the existing `self._reset_cost_tracking()` line inside `run` (and before the `try:`), insert:

```python
        # Financial tool path (Phase A decide -> execute -> Phase B synthesize).
        # Falls through to the web/plain path if the model declines or it errors.
        if enable_financial_data and self.financial_manager:
            financial_result = await self._try_financial(query, conversation_history)
            if financial_result is not None:
                logger.info(
                    f"AgentV2 financial path used. records={financial_result['record_count']}"
                )
                return financial_result
```

Also update the `run` docstring `Args:` block to document `enable_financial_data` (add a line):

```python
            enable_financial_data: When True and a financial_manager is set, first
                attempts the query_financial_data path; falls through to web/plain
                if the model declines.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_v2_financial.py -v`
Expected: PASS (all ~10 tests).

- [ ] **Step 5: Regression guard — existing web path still covered**

The web/plain path has no prior tests, but the new file now exercises it via
`test_decline_falls_through_to_web`, `test_no_manager_skips_financial`, and
`test_enable_financial_false_skips_financial` (all run the existing single-call
`google_search` path). Confirm the whole new file is green:

Run (from `fastapi_app/`): `python -m pytest tests/test_agent_v2_financial.py -v`
Expected: PASS (all ~10).

- [ ] **Step 6: Commit**

```bash
git add fastapi_app/app/agent_v2.py fastapi_app/tests/test_agent_v2_financial.py
git commit -m "feat(agent_v2): sequential query_financial_data path in run()"
```

---

## Task 5: Wire `/chat/stream` endpoint

**Files:**
- Modify: `fastapi_app/app/main.py` (the `chat_stream` handler, ~line 949-991)
- Test: `fastapi_app/tests/test_chat_stream_financial.py`

- [ ] **Step 1: Write the failing test**

Create `fastapi_app/tests/test_chat_stream_financial.py`:

```python
"""Endpoint wiring tests for /chat/stream financial-data integration (ATS-334)."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


def _result():
    return {
        "response": "NCB 2023 revenue was J$123.5M.",
        "data_found": True,
        "record_count": 1,
        "needs_clarification": False,
        "clarification_question": None,
        "tools_executed": ["query_financial_data"],
        "sources": None,
        "filters_used": None,
        "data_preview": None,
        "chart": None,
        "web_search_results": None,
        "suggestions": None,
        "conversation_history": None,
        "warnings": None,
        "cost_summary": None,
    }


def test_chat_stream_forwards_enable_financial_data():
    from app.main import app

    with patch("app.main.AgentV2") as MockAgent:
        instance = MockAgent.return_value
        instance.run = AsyncMock(return_value=_result())
        client = TestClient(app)
        resp = client.post(
            "/chat/stream",
            json={"query": "What was NCB revenue in 2023?", "enable_financial_data": True},
        )

    assert resp.status_code == 200
    assert resp.json()["tools_executed"] == ["query_financial_data"]
    # enable_financial_data forwarded to AgentV2.run
    run_kwargs = instance.run.call_args.kwargs
    assert run_kwargs["enable_financial_data"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_chat_stream_financial.py -v`
Expected: FAIL — `KeyError: 'enable_financial_data'` (the endpoint does not yet forward it to `run`).

- [ ] **Step 3: Write minimal implementation**

In `fastapi_app/app/main.py`, change the `chat_stream` handler signature (~line 951) to inject the manager:

```python
async def chat_stream(
    request: AgentChatRequest,
    financial_manager: Any = Depends(get_financial_manager),
):
```

Then change the agent construction + run call (~line 984-991) from:

```python
        # Create simplified agent (uses Gemini 2.5 Pro with Google Search grounding)
        agent = AgentV2()

        # Run the agent
        result = await agent.run(
            query=request.query,
            conversation_history=request.conversation_history,
            enable_web_search=request.enable_web_search,
        )
```

to:

```python
        # Create agent with the financial manager so query_financial_data is available
        agent = AgentV2(financial_manager=financial_manager)

        # Run the agent
        result = await agent.run(
            query=request.query,
            conversation_history=request.conversation_history,
            enable_web_search=request.enable_web_search,
            enable_financial_data=request.enable_financial_data,
        )
```

(`Depends`, `Any`, and `get_financial_manager` are already imported/defined in `main.py` — no new imports.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_chat_stream_financial.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fastapi_app/app/main.py fastapi_app/tests/test_chat_stream_financial.py
git commit -m "feat(main): /chat/stream injects financial_manager, forwards enable_financial_data"
```

---

## Task 6: Update archive README pointer

**Files:**
- Modify: `fastapi_app/app/_archive/README.md`

- [ ] **Step 1: Edit the README**

In `fastapi_app/app/_archive/README.md`, under the `## Restoring` section, add a note at the top:

```markdown
> **ATS-334 update (2026-05-31):** The financial function-calling primitives
> (`get_financial_data_tool_declaration`, `execute_financial_query`,
> `build_financial_context`) now live in `app/financial_tool.py` and are used by
> `AgentV2` on `/chat/stream` (Option 2). The 3-phase `AgentOrchestrator` below
> was **not** restored (Option 1, PR #32, was closed in favor of the smaller
> AgentV2 path). This file remains archived for reference only.
```

- [ ] **Step 2: Commit**

```bash
git add fastapi_app/app/_archive/README.md
git commit -m "docs(_archive): point to app/financial_tool.py for restored primitives"
```

---

## Task 7: Add financial `chat_stream` eval personas

**Files:**
- Create: `evals/personas/chatstream_ncb_revenue_lookup.yaml`
- Create: `evals/personas/chatstream_compare_gk_ncb_profit.yaml`
- Create: `evals/personas/chatstream_mixed_financial_and_news.yaml`

> These exercise the financial tool on `/chat/stream` (the existing chat_stream
> personas are qualitative and barely touch it). **Ported verbatim from closed PR
> #32** — they match the real `PersonaSpec` schema (`character` / `goal` /
> `expected_facts` / `api_options` / `opening_style`; `name`/`notes` are accepted
> because `PersonaSpec` ignores extra fields) and carry manual-verification notes.
> The eval client reads `api_options.enable_financial_data` (see
> `evals/client/agent_stream.py:48`). Note: the mixed persona's "both tools"
> ideal is **not** met by the sequential design (financial OR web per turn) — its
> `expected_facts` are still listed for aspiration, but it is judged at its
> minimum bar (DB figure present). See spec "Known limitation".
>
> To re-fetch verbatim instead of copying from this plan:
> `git show origin/claude/ecstatic-lovelace-a785e5:evals/personas/chatstream_<id>.yaml`

- [ ] **Step 1: Create `chatstream_ncb_revenue_lookup.yaml`**

```yaml
id: chatstream_ncb_revenue_lookup
name: "Retail investor — NCB revenue lookup (chat_stream)"
category: positive
endpoint: chat_stream
character: |
  You're a self-directed retail investor checking up on a stock you already
  own. You ask direct, specific questions about figures and expect a number
  back, not a pointer to "go read the annual report". You're polite but you
  notice when an answer dodges the question.
goal: |
  Find out NCB Financial Group's total revenue for the 2023 fiscal year, then
  ask how that compares to the year before. You're satisfied when you have a
  concrete revenue figure for 2023 grounded in actual data.
max_turns: 4
expected_facts:
  - "NCB Financial Group is mentioned by name or symbol (NCBFG)"
  - "A concrete total revenue figure for the 2023 fiscal year"
api_options:
  memory_enabled: true
  enable_web_search: true
  enable_financial_data: true
opening_style: direct_question
notes: |
  Regression coverage for ATS-334: a financial-metric query on /chat/stream
  must route to the query_financial_data tool (BigQuery), not fall back to web
  grounding. Verified manually to return ~95 records for NCB revenue 2023.
```

- [ ] **Step 2: Create `chatstream_compare_gk_ncb_profit.yaml`**

```yaml
id: chatstream_compare_gk_ncb_profit
name: "Investor comparing GK vs NCB profit (chat_stream)"
category: positive
endpoint: chat_stream
character: |
  You're weighing two blue-chip JSE stocks against each other and you think in
  side-by-side comparisons. You ask the bot to compare specific metrics for
  specific companies and years, and you follow up if only one side is answered.
  You speak plainly and expect both figures, clearly labelled.
goal: |
  Compare GraceKennedy and NCB Financial Group on net profit for the 2022
  fiscal year, then ask which of the two grew faster. You're satisfied when you
  have a labelled net-profit figure for each company for 2022.
max_turns: 5
expected_facts:
  - "Both GraceKennedy (GK) and NCB Financial Group (NCBFG) are named"
  - "A net profit figure for each company for the 2022 fiscal year"
api_options:
  memory_enabled: true
  enable_web_search: true
  enable_financial_data: true
opening_style: direct_question
notes: |
  Regression coverage for ATS-334: a multi-company comparison on /chat/stream
  must resolve both symbols and call query_financial_data. Verified manually
  that "compare GK and NCB net profit 2022" resolves symbols=['GK','NCBFG'].
```

- [ ] **Step 3: Create `chatstream_mixed_financial_and_news.yaml`**

```yaml
id: chatstream_mixed_financial_and_news
name: "Investor mixing financials and news (chat_stream)"
category: positive
endpoint: chat_stream
character: |
  You're an engaged investor who jumps between hard numbers and current events
  in the same conversation. First you want a specific financial figure, then
  you pivot to "what's the latest news". You expect the bot to handle both
  without losing the thread of which company you're discussing.
goal: |
  Get NCB Financial Group's net profit for 2023, then ask for the latest news
  about the company. You're satisfied when the financial figure is grounded in
  data and the news answer cites current web sources.
max_turns: 5
expected_facts:
  - "A net profit figure for NCB Financial Group for 2023"
  - "A news/current-events answer that references recent sources"
api_options:
  memory_enabled: true
  enable_web_search: true
  enable_financial_data: true
opening_style: direct_question
notes: |
  Regression coverage for ATS-334: exercises both routing branches on
  /chat/stream within one conversation — the metric question must hit
  query_financial_data while the news question must route to google_search.
  NOTE (sequential design): v1 answers financial OR web per turn, so this is
  judged at its minimum bar (the net-profit figure is DB-grounded); the
  "both tools in one turn" ideal is a future enhancement (see spec).
```

- [ ] **Step 4: Schema-validate the personas with the real loader**

Run (from repo root): `python -c "from evals.persona import load_persona; import glob; [load_persona(f) for f in glob.glob('evals/personas/chatstream_*.yaml')]; print('ok')"`
Expected: `ok` (validates against `PersonaSpec`, not just YAML well-formedness — catches a bad `opening_style`/`category`). If `evals` isn't importable from the worktree, run with `PYTHONPATH=.`.

- [ ] **Step 5: Commit**

```bash
git add evals/personas/chatstream_ncb_revenue_lookup.yaml evals/personas/chatstream_compare_gk_ncb_profit.yaml evals/personas/chatstream_mixed_financial_and_news.yaml
git commit -m "test(evals): add financial chat_stream personas for ATS-334"
```

---

## Task 8: Full regression + manual smoke

**Files:** none (verification only)

- [ ] **Step 1: Run the full unit suite (excluding integration/uat, matching repo baseline)**

Run (from `fastapi_app/`): `python -m pytest tests/ --ignore=tests/integration --ignore=tests/uat -q`
Expected: all previously-passing tests still pass, plus the new `test_financial_tool.py`, `test_agent_v2_financial.py`, `test_chat_stream_financial.py`. Pre-existing credential-dependent failures (if any) should match the prior baseline count — do not treat unchanged pre-existing failures as regressions, but DO confirm the count didn't grow.

- [ ] **Step 2: Confirm no import cycle**

Run (from `fastapi_app/`): `python -c "import app.main; import app.agent_v2; import app.financial_tool; print('imports ok')"`
Expected: `imports ok` (financial_tool ← agent_v2 ← main; no cycle since financial_tool imports only charting/models/logging).

- [ ] **Step 3: Commit (if any lint/format fixups were needed; otherwise skip)**

```bash
git add -A
git commit -m "chore(ATS-334): regression fixups"
```

---

## Evaluation (run after implementation; live + billable)

> Prereqs (from memory `reference-eval-suite`): start uvicorn against the **main**
> project dir's `.env` (BigQuery + Gemini creds) — the worktree has no `.env`. Use
> `dotenv_values()` + alias `CHATBOT_API_KEY`→`GOOGLE_API_KEY`; launch uvicorn from a
> small Python script injecting the parsed `.env`. Server at `http://localhost:8000`.

- [ ] **E1: Direct tool-fire verification** (cheapest signal first). With the server up:

```bash
curl -s -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"What was NCB revenue in 2023?","enable_financial_data":true}' | python -m json.tool
```
Expected: `tools_executed` contains `query_financial_data`, `record_count > 0`, response has a specific J$ figure, latency in single-digit seconds (not ~87s).

- [ ] **E2: Negative/declination check** — confirm a news query still uses web:

```bash
curl -s -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"Any recent news about the JSE Main Market?","enable_financial_data":true}' | python -m json.tool
```
Expected: `query_financial_data` NOT in `tools_executed` (web/plain path); a reasonable answer.

- [ ] **E3: Persona eval** — groundedness + latency on chat_stream (incl. the 3 new personas):

```bash
python scripts/run_eval.py --endpoint chat_stream
python scripts/analyze_eval_run.py evals/runs/<timestamp>
```
Expected: the 3 `chatstream_*` financial personas show `query_financial_data` in per-turn `chatbot_metadata.tools_executed`; groundedness on financial personas up vs. a baseline run; chat_stream latency down on DB-answerable turns. **Confirm a baseline run dir actually exists before any before/after comparison — do not assume baseline numbers.**

- [ ] **E4: Record results** in the PR description (tool-fire confirmation, latency delta, persona score delta, and the mixed-persona caveat).

---

## Done criteria

- All new unit tests pass; existing suite shows no new failures.
- `/chat/stream` answers "What was NCB revenue in 2023?" from the DB with
  `query_financial_data` in `tools_executed`, in seconds.
- News/qualitative queries still route to web (no financial tool).
- `enable_financial_data=false` reproduces today's behavior (no extra call).
- `/fast_chat_v2` and `streaming_financial_chat.py` untouched and green.
