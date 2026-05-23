# Simulation-Based Eval Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a persona-driven, multi-turn chatbot evaluation harness in a new top-level `evals/` folder, producing structured JSON output rendered by a static HTML+JS viewer that supports single-run inspection and multi-run comparison/diff.

**Architecture:** Independent Python package (`evals/`) hits the existing FastAPI chatbot over HTTP. A persona LLM (Gemini Flash 2.5) role-plays the user, driving a multi-turn conversation against `/fast_chat_v2` (non-streaming) or `/chat/stream` (SSE). After each conversation, a judge LLM (Gemini Pro 2.5) scores groundedness, factfulness, goal completion, tool use, coherence, and persona handling. Output: one folder per run with `manifest.json`, `summary.json`, and per-conversation JSON. A static viewer at `evals/viewer/` renders single runs and compares multiple runs side-by-side.

**Tech Stack:** Python 3.11, `google-genai` SDK, `httpx` (async), `respx` (HTTP mocking in tests), `pyyaml`, `pydantic` v2, `pytest` + `pytest-asyncio`. Viewer: plain HTML/CSS/JS + Vega-Lite via CDN.

**Spec:** [docs/superpowers/specs/2026-05-23-simulation-eval-design.md](../specs/2026-05-23-simulation-eval-design.md)

---

## File Structure

```
evals/                                 (NEW top-level folder)
├── pyproject.toml                     (Task 1)
├── README.md                          (Task 1; updated Task 23)
├── .gitignore                         (Task 1)
├── __init__.py                        (Task 1; makes 'evals' a package)
├── persona.py                         (Task 2)
├── config.py                          (Task 3)
├── transcript.py                      (Task 4)
├── metrics.py                         (Task 5)
├── client/
│   ├── __init__.py                    (Task 6)
│   ├── base.py                        (Task 6)
│   ├── financial.py                   (Task 7)
│   └── agent_stream.py                (Task 8)
├── persona_actor.py                   (Task 9)
├── judge.py                           (Task 10)
├── runner.py                          (Tasks 11, 12, 13)
├── report.py                          (Task 14)
├── cli.py                             (Task 15)
├── serve.py                           (Task 23)
├── config/
│   ├── default.yaml                   (Task 3)
│   └── judge_rubric.yaml              (Task 10)
├── personas/
│   ├── senior_analyst_ncb_financials.yaml   (Task 16)
│   ├── student_what_is_stock_market.yaml    (Task 16)
│   ├── investor_compare_ncb_vs_jmmb.yaml    (Task 16)
│   └── negative_chitchat_offtopic.yaml      (Task 16)
├── tests/
│   ├── __init__.py                    (Task 1)
│   ├── conftest.py                    (Task 1)
│   ├── fixtures/                      (used across tests)
│   │   ├── persona_valid.yaml         (Task 2)
│   │   ├── financial_response.json    (Task 7)
│   │   └── stream_response.txt        (Task 8)
│   ├── test_persona.py                (Task 2)
│   ├── test_config.py                 (Task 3)
│   ├── test_transcript.py             (Task 4)
│   ├── test_metrics.py                (Task 5)
│   ├── test_client_financial.py       (Task 7)
│   ├── test_client_agent_stream.py    (Task 8)
│   ├── test_persona_actor.py          (Task 9)
│   ├── test_judge.py                  (Task 10)
│   ├── test_runner.py                 (Tasks 11, 12, 13)
│   ├── test_report.py                 (Task 14)
│   └── test_cli.py                    (Task 15)
├── runs/                              (gitignored; created on first run)
│   └── .gitkeep
└── viewer/                            (Tasks 17–22)
    ├── index.html
    ├── viewer.js
    ├── styles.css
    └── fixtures/                      (synthetic JSON for manual smoke)
        └── synthetic_run/
```

Entrypoint: `python -m evals.cli` (from the repo root, with `evals/` installed in editable mode).

---

## Task 1: Project skeleton

**Files:**
- Create: `evals/pyproject.toml`
- Create: `evals/README.md`
- Create: `evals/.gitignore`
- Create: `evals/__init__.py`
- Create: `evals/runs/.gitkeep`
- Create: `evals/tests/__init__.py`
- Create: `evals/tests/conftest.py`

- [ ] **Step 1: Create folder structure**

```bash
mkdir -p evals/client evals/config evals/personas evals/runs evals/tests/fixtures evals/viewer/fixtures/synthetic_run
```

- [ ] **Step 2: Create `evals/__init__.py`**

```python
"""Persona-driven simulation eval suite for the JSE DataSphere chatbot."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Create `evals/pyproject.toml`**

```toml
[project]
name = "jse-eval-suite"
version = "0.1.0"
description = "Persona-driven multi-turn evaluation harness for the JSE chatbot"
requires-python = ">=3.10,<3.13"

dependencies = [
    "httpx>=0.27.0",
    "pyyaml>=6.0.1",
    "pydantic>=2.6.0",
    "google-genai>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "respx>=0.21.0",
    "ruff>=0.3.0",
]

[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = [
    "evals",
    "evals.client",
]

[tool.setuptools.package-data]
"evals" = ["config/*.yaml", "personas/*.yaml"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 4: Create `evals/.gitignore`**

```
runs/*
!runs/.gitkeep
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
*.egg-info/
```

- [ ] **Step 5: Create `evals/README.md` (stub)**

```markdown
# JSE Chatbot Eval Suite

Persona-driven, multi-turn simulation suite for the JSE DataSphere chatbot.
See [spec](../docs/superpowers/specs/2026-05-23-simulation-eval-design.md)
for design rationale.

## Quick start

```bash
cd evals
pip install -e ".[dev]"
pytest                                 # run unit tests
python -m evals.cli --help             # see runner options
```

Full usage docs in this README will be filled in as the suite is built.
```

- [ ] **Step 6: Create `evals/runs/.gitkeep` (empty file) and `evals/tests/__init__.py` (empty file)**

- [ ] **Step 7: Create `evals/tests/conftest.py`**

```python
"""Shared pytest fixtures for the eval suite."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the tests/fixtures directory."""
    return FIXTURES_DIR
```

- [ ] **Step 8: Install in editable mode and run smoke test**

Run:
```bash
cd evals
pip install -e ".[dev]"
python -c "import evals; print(evals.__version__)"
pytest -v
```

Expected: prints `0.1.0`, pytest collects zero tests but exits 0.

- [ ] **Step 9: Commit**

```bash
git add evals/
git commit -m "feat(evals): scaffold eval suite package skeleton"
```

---

## Task 2: Persona schema and YAML loader

**Files:**
- Create: `evals/persona.py`
- Create: `evals/tests/test_persona.py`
- Create: `evals/tests/fixtures/persona_valid.yaml`

- [ ] **Step 1: Create the valid persona fixture `evals/tests/fixtures/persona_valid.yaml`**

```yaml
id: senior_analyst_test
name: "Test persona — senior analyst"
category: positive
endpoint: fast_chat_v2
character: |
  You are a senior equity analyst at a regional brokerage.
goal: |
  Understand NCB's profitability over 3 years.
max_turns: 4
expected_facts:
  - "NCB is mentioned by name or symbol"
  - "Revenue figures for at least 2 fiscal years"
api_options:
  memory_enabled: true
  enable_financial_data: true
opening_style: cold_open
```

- [ ] **Step 2: Write failing tests in `evals/tests/test_persona.py`**

```python
"""Tests for PersonaSpec loading and validation."""

from pathlib import Path

import pytest

from evals.persona import PersonaSpec, PersonaValidationError, load_persona


def test_load_valid_persona(fixtures_dir: Path):
    persona = load_persona(fixtures_dir / "persona_valid.yaml")
    assert persona.id == "senior_analyst_test"
    assert persona.category == "positive"
    assert persona.endpoint == "fast_chat_v2"
    assert persona.max_turns == 4
    assert len(persona.expected_facts) == 2
    assert persona.api_options["enable_financial_data"] is True
    assert persona.opening_style == "cold_open"


def test_missing_required_field_raises(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "id: x\n"
        "name: y\n"
        "category: positive\n"
        "endpoint: fast_chat_v2\n"
        "character: z\n"
        # missing 'goal'
        "max_turns: 3\n"
    )
    with pytest.raises(PersonaValidationError, match="goal"):
        load_persona(bad)


def test_invalid_category_rejected(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "id: x\nname: y\ncategory: weird\nendpoint: fast_chat_v2\n"
        "character: z\ngoal: w\nmax_turns: 3\n"
    )
    with pytest.raises(PersonaValidationError, match="category"):
        load_persona(bad)


def test_invalid_endpoint_rejected(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "id: x\nname: y\ncategory: positive\nendpoint: /something\n"
        "character: z\ngoal: w\nmax_turns: 3\n"
    )
    with pytest.raises(PersonaValidationError, match="endpoint"):
        load_persona(bad)


def test_max_turns_must_be_positive(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "id: x\nname: y\ncategory: positive\nendpoint: fast_chat_v2\n"
        "character: z\ngoal: w\nmax_turns: 0\n"
    )
    with pytest.raises(PersonaValidationError, match="max_turns"):
        load_persona(bad)


def test_optional_fields_default(tmp_path: Path):
    minimal = tmp_path / "min.yaml"
    minimal.write_text(
        "id: x\nname: y\ncategory: positive\nendpoint: fast_chat_v2\n"
        "character: z\ngoal: w\nmax_turns: 3\n"
    )
    persona = load_persona(minimal)
    assert persona.expected_facts == []
    assert persona.api_options == {}
    assert persona.opening_style == "cold_open"
    assert persona.notes == ""
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd evals && pytest tests/test_persona.py -v`
Expected: ImportError or ModuleNotFoundError for `evals.persona`.

- [ ] **Step 4: Implement `evals/persona.py`**

```python
"""PersonaSpec dataclass + YAML loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


class PersonaValidationError(ValueError):
    """Raised when a persona YAML fails schema validation."""


class PersonaSpec(BaseModel):
    """A persona that drives a simulated conversation."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category: str
    endpoint: str
    character: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    max_turns: int = Field(gt=0)
    expected_facts: list[str] = Field(default_factory=list)
    api_options: dict[str, Any] = Field(default_factory=dict)
    opening_style: str = "cold_open"
    notes: str = ""

    @field_validator("category")
    @classmethod
    def _check_category(cls, v: str) -> str:
        if v not in {"positive", "negative"}:
            raise ValueError(
                f"category must be 'positive' or 'negative', got '{v}'"
            )
        return v

    @field_validator("endpoint")
    @classmethod
    def _check_endpoint(cls, v: str) -> str:
        if v not in {"fast_chat_v2", "chat_stream"}:
            raise ValueError(
                f"endpoint must be 'fast_chat_v2' or 'chat_stream', got '{v}'"
            )
        return v

    @field_validator("opening_style")
    @classmethod
    def _check_opening_style(cls, v: str) -> str:
        if v not in {"cold_open", "warmup", "direct_question"}:
            raise ValueError(f"unknown opening_style '{v}'")
        return v


def load_persona(path: Path | str) -> PersonaSpec:
    """Load and validate a persona from a YAML file."""
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise PersonaValidationError(
            f"persona file {p} must contain a YAML mapping at the top level"
        )
    try:
        return PersonaSpec(**raw)
    except ValidationError as exc:
        raise PersonaValidationError(str(exc)) from exc


def load_personas(directory: Path | str) -> list[PersonaSpec]:
    """Load every *.yaml file in a directory as a PersonaSpec."""
    d = Path(directory)
    return [load_persona(f) for f in sorted(d.glob("*.yaml"))]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd evals && pytest tests/test_persona.py -v`
Expected: 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git add evals/persona.py evals/tests/test_persona.py evals/tests/fixtures/persona_valid.yaml
git commit -m "feat(evals): add PersonaSpec model with YAML loader and validation"
```

---

## Task 3: Config loader and default.yaml

**Files:**
- Create: `evals/config.py`
- Create: `evals/config/default.yaml`
- Create: `evals/tests/test_config.py`

- [ ] **Step 1: Create `evals/config/default.yaml`**

```yaml
base_url: http://localhost:8000
replicates: 3
concurrency: 4
request_timeout_s: 90
persona_model: gemini-2.5-flash
judge_model: gemini-2.5-pro
persona_temperature: 0.8
judge_temperature: 0.2
max_cost_usd_per_run: 5.00
max_cost_usd_per_conversation: 0.50
gemini_api_key_env: GOOGLE_API_KEY
```

- [ ] **Step 2: Write failing tests in `evals/tests/test_config.py`**

```python
"""Tests for runtime config loading and override merging."""

from pathlib import Path

import pytest

from evals.config import EvalConfig, load_config


def test_load_default():
    config = load_config()
    assert config.base_url == "http://localhost:8000"
    assert config.replicates == 3
    assert config.concurrency == 4
    assert config.persona_model == "gemini-2.5-flash"
    assert config.judge_model == "gemini-2.5-pro"
    assert config.max_cost_usd_per_run == 5.00


def test_override_via_kwargs():
    config = load_config(overrides={"replicates": 1, "concurrency": 8})
    assert config.replicates == 1
    assert config.concurrency == 8
    # untouched values still default
    assert config.persona_model == "gemini-2.5-flash"


def test_load_from_custom_path(tmp_path: Path):
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        "base_url: http://example.test\n"
        "replicates: 5\n"
        "concurrency: 2\n"
        "request_timeout_s: 60\n"
        "persona_model: gemini-2.5-flash\n"
        "judge_model: gemini-2.5-pro\n"
        "persona_temperature: 0.5\n"
        "judge_temperature: 0.2\n"
        "max_cost_usd_per_run: 1.0\n"
        "max_cost_usd_per_conversation: 0.1\n"
        "gemini_api_key_env: GOOGLE_API_KEY\n"
    )
    config = load_config(path=custom)
    assert config.base_url == "http://example.test"
    assert config.replicates == 5


def test_invalid_replicate_count_rejected():
    with pytest.raises(ValueError):
        load_config(overrides={"replicates": 0})
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd evals && pytest tests/test_config.py -v`
Expected: ImportError for `evals.config`.

- [ ] **Step 4: Implement `evals/config.py`**

```python
"""Runtime config for the eval suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"


class EvalConfig(BaseModel):
    """Runtime configuration for a simulation run."""

    base_url: str
    replicates: int = Field(gt=0)
    concurrency: int = Field(gt=0)
    request_timeout_s: float = Field(gt=0)
    persona_model: str
    judge_model: str
    persona_temperature: float = Field(ge=0, le=2)
    judge_temperature: float = Field(ge=0, le=2)
    max_cost_usd_per_run: float = Field(gt=0)
    max_cost_usd_per_conversation: float = Field(gt=0)
    gemini_api_key_env: str


def load_config(
    path: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
) -> EvalConfig:
    """Load config from YAML, applying overrides on top."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    return EvalConfig(**data)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd evals && pytest tests/test_config.py -v`
Expected: 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add evals/config.py evals/config/default.yaml evals/tests/test_config.py
git commit -m "feat(evals): add EvalConfig loader with default.yaml + overrides"
```

---

## Task 4: Transcript and ChatTurn models

**Files:**
- Create: `evals/transcript.py`
- Create: `evals/tests/test_transcript.py`

- [ ] **Step 1: Write failing tests in `evals/tests/test_transcript.py`**

```python
"""Tests for ChatTurn / Transcript data models + serialization."""

from evals.transcript import ChatTurn, TerminationReason, Transcript


def test_chat_turn_roundtrip():
    turn = ChatTurn(
        turn_index=0,
        persona_utterance="Show me NCB profitability",
        chatbot_text="Net interest income was J$50B in FY2023.",
        chatbot_metadata={"data_found": True, "record_count": 12},
        latency_ms=1820,
        ttfb_ms=None,
        cost_usd=0.0034,
        input_tokens=1200,
        output_tokens=380,
    )
    d = turn.model_dump()
    back = ChatTurn(**d)
    assert back == turn


def test_transcript_totals_computed():
    turns = [
        ChatTurn(
            turn_index=i,
            persona_utterance=f"q{i}",
            chatbot_text=f"a{i}",
            chatbot_metadata={},
            latency_ms=1000.0,
            ttfb_ms=None,
            cost_usd=0.001,
            input_tokens=100,
            output_tokens=50,
        )
        for i in range(3)
    ]
    t = Transcript(
        conversation_id="test__rep01",
        persona_id="test",
        replicate_index=1,
        endpoint="fast_chat_v2",
        turns=turns,
        termination=TerminationReason(reason="done", at_turn=2),
    )
    totals = t.totals()
    assert totals["turns"] == 3
    assert totals["latency_ms"] == 3000.0
    assert totals["cost_usd"] == pytest.approx(0.003)


def test_transcript_json_roundtrip():
    t = Transcript(
        conversation_id="x__rep01",
        persona_id="x",
        replicate_index=1,
        endpoint="chat_stream",
        turns=[],
        termination=TerminationReason(reason="error", at_turn=0, error_message="boom"),
    )
    j = t.model_dump_json()
    back = Transcript.model_validate_json(j)
    assert back == t


import pytest  # noqa: E402
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd evals && pytest tests/test_transcript.py -v`
Expected: ImportError for `evals.transcript`.

- [ ] **Step 3: Implement `evals/transcript.py`**

```python
"""Per-conversation transcript and turn data models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    """One persona ↔ chatbot exchange."""

    turn_index: int
    persona_utterance: str
    chatbot_text: str
    chatbot_metadata: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float
    ttfb_ms: float | None = None
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class TerminationReason(BaseModel):
    """Why a conversation ended."""

    reason: Literal["done", "cap", "error"]
    at_turn: int
    persona_done_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class Transcript(BaseModel):
    """Full record of a single conversation, before judging."""

    conversation_id: str
    persona_id: str
    replicate_index: int
    endpoint: Literal["fast_chat_v2", "chat_stream"]
    turns: list[ChatTurn] = Field(default_factory=list)
    termination: TerminationReason

    def totals(self) -> dict[str, float | int]:
        """Aggregate latency, cost, and turn count across turns."""
        return {
            "turns": len(self.turns),
            "latency_ms": sum(t.latency_ms for t in self.turns),
            "cost_usd": sum(t.cost_usd or 0.0 for t in self.turns),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd evals && pytest tests/test_transcript.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add evals/transcript.py evals/tests/test_transcript.py
git commit -m "feat(evals): add ChatTurn and Transcript models with totals + roundtrip"
```

---

## Task 5: Metrics utilities

**Files:**
- Create: `evals/metrics.py`
- Create: `evals/tests/test_metrics.py`

- [ ] **Step 1: Write failing tests in `evals/tests/test_metrics.py`**

```python
"""Tests for latency stats and cost extraction utilities."""

from evals.metrics import LatencyStats, extract_cost_from_response, latency_stats


def test_latency_stats_basic():
    stats = latency_stats([100, 200, 300, 400, 500])
    assert stats.count == 5
    assert stats.min_ms == 100
    assert stats.max_ms == 500
    assert stats.avg_ms == 300
    # p95 of 5-sample list — sorted index = int(5*0.95)=4 → 500
    assert stats.p95_ms == 500


def test_latency_stats_empty():
    stats = latency_stats([])
    assert stats == LatencyStats(min_ms=0, max_ms=0, avg_ms=0, p95_ms=0, count=0)


def test_latency_stats_single():
    stats = latency_stats([42.5])
    assert stats.count == 1
    assert stats.min_ms == 42.5
    assert stats.max_ms == 42.5
    assert stats.avg_ms == 42.5
    assert stats.p95_ms == 42.5


def test_extract_cost_present():
    response = {
        "cost_summary": {
            "total_cost_usd": 0.0034,
            "total_input_tokens": 1200,
            "total_output_tokens": 380,
        }
    }
    cost = extract_cost_from_response(response)
    assert cost.cost_usd == 0.0034
    assert cost.input_tokens == 1200
    assert cost.output_tokens == 380


def test_extract_cost_missing():
    cost = extract_cost_from_response({"response": "x"})
    assert cost.cost_usd is None
    assert cost.input_tokens is None
    assert cost.output_tokens is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd evals && pytest tests/test_metrics.py -v`
Expected: ImportError for `evals.metrics`.

- [ ] **Step 3: Implement `evals/metrics.py`**

```python
"""Latency and cost utility functions, shared across the eval suite."""

from __future__ import annotations

import statistics
from typing import Any

from pydantic import BaseModel


class LatencyStats(BaseModel):
    """Latency statistics over a sample of durations."""

    min_ms: float
    max_ms: float
    avg_ms: float
    p95_ms: float
    count: int


class CostInfo(BaseModel):
    """Cost and token counts extracted from a chatbot response."""

    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


def latency_stats(durations_ms: list[float]) -> LatencyStats:
    """Compute min/max/avg/p95 latency statistics."""
    if not durations_ms:
        return LatencyStats(min_ms=0, max_ms=0, avg_ms=0, p95_ms=0, count=0)

    sorted_d = sorted(durations_ms)
    p95_index = int(len(sorted_d) * 0.95)
    p95 = sorted_d[p95_index] if p95_index < len(sorted_d) else sorted_d[-1]

    return LatencyStats(
        min_ms=min(durations_ms),
        max_ms=max(durations_ms),
        avg_ms=statistics.mean(durations_ms),
        p95_ms=p95,
        count=len(durations_ms),
    )


def extract_cost_from_response(response: dict[str, Any]) -> CostInfo:
    """Pull cost + token counts from a chatbot response's cost_summary block."""
    cost_summary = response.get("cost_summary")
    if not isinstance(cost_summary, dict):
        return CostInfo()
    return CostInfo(
        cost_usd=cost_summary.get("total_cost_usd"),
        input_tokens=cost_summary.get("total_input_tokens"),
        output_tokens=cost_summary.get("total_output_tokens"),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd evals && pytest tests/test_metrics.py -v`
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add evals/metrics.py evals/tests/test_metrics.py
git commit -m "feat(evals): add latency_stats and extract_cost_from_response helpers"
```

---

## Task 6: ChatClient protocol (base class)

**Files:**
- Create: `evals/client/__init__.py`
- Create: `evals/client/base.py`

- [ ] **Step 1: Create `evals/client/__init__.py` (empty file)**

- [ ] **Step 2: Create `evals/client/base.py`**

```python
"""Common protocol shared by the financial and stream chat clients."""

from __future__ import annotations

from typing import Any, Protocol


class ChatClientResult:
    """Outcome of a single chatbot API call."""

    def __init__(
        self,
        chatbot_text: str,
        chatbot_metadata: dict[str, Any],
        latency_ms: float,
        ttfb_ms: float | None,
        cost_usd: float | None,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> None:
        self.chatbot_text = chatbot_text
        self.chatbot_metadata = chatbot_metadata
        self.latency_ms = latency_ms
        self.ttfb_ms = ttfb_ms
        self.cost_usd = cost_usd
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class ChatClient(Protocol):
    """Abstract chatbot client. Each endpoint gets one implementation."""

    async def send(
        self,
        query: str,
        conversation_history: list[dict[str, str]],
        api_options: dict[str, Any],
    ) -> ChatClientResult:
        """Send a single user turn; return the chatbot's full response."""
        ...
```

- [ ] **Step 3: Smoke import**

Run: `cd evals && python -c "from evals.client.base import ChatClient, ChatClientResult; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add evals/client/
git commit -m "feat(evals): add ChatClient protocol and ChatClientResult shape"
```

---

## Task 7: FinancialClient for `/fast_chat_v2`

**Files:**
- Create: `evals/client/financial.py`
- Create: `evals/tests/test_client_financial.py`
- Create: `evals/tests/fixtures/financial_response.json`

- [ ] **Step 1: Create `evals/tests/fixtures/financial_response.json`**

```json
{
  "response": "NCB net interest income was J$50B in FY2023, up from J$45B in FY2022.",
  "data_found": true,
  "record_count": 12,
  "filters_used": {
    "companies": ["NCB Financial Group"],
    "symbols": ["NCBFG"],
    "years": ["2022", "2023"],
    "standard_items": ["NetInterestIncome"],
    "interpretation": "NCB net interest income over 2 years",
    "data_availability_note": "",
    "is_follow_up": false,
    "context_used": ""
  },
  "data_preview": null,
  "conversation_history": [],
  "warnings": null,
  "suggestions": null,
  "chart": null,
  "cost_summary": {
    "total_cost_usd": 0.0034,
    "total_input_tokens": 1200,
    "total_output_tokens": 380
  }
}
```

- [ ] **Step 2: Write failing tests in `evals/tests/test_client_financial.py`**

```python
"""Tests for FinancialClient (`/fast_chat_v2`)."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from evals.client.financial import FinancialClient


@pytest.mark.asyncio
async def test_send_returns_response_text(fixtures_dir: Path):
    body = json.loads((fixtures_dir / "financial_response.json").read_text())

    async with respx.mock(assert_all_called=True) as router:
        router.post("http://localhost:8000/fast_chat_v2").respond(
            status_code=200, json=body
        )
        client = FinancialClient(base_url="http://localhost:8000", timeout_s=10)
        result = await client.send(
            query="NCB net interest income last 2 years",
            conversation_history=[],
            api_options={"memory_enabled": True},
        )

    assert "NCB net interest income" in result.chatbot_text
    assert result.chatbot_metadata["data_found"] is True
    assert result.cost_usd == 0.0034
    assert result.input_tokens == 1200
    assert result.output_tokens == 380
    assert result.ttfb_ms is None  # non-streaming
    assert result.latency_ms > 0


@pytest.mark.asyncio
async def test_send_raises_on_5xx():
    async with respx.mock() as router:
        router.post("http://localhost:8000/fast_chat_v2").respond(status_code=500)
        client = FinancialClient(base_url="http://localhost:8000", timeout_s=10)
        with pytest.raises(httpx.HTTPStatusError):
            await client.send("q", [], {})


@pytest.mark.asyncio
async def test_send_passes_conversation_history():
    captured = {}

    async with respx.mock() as router:
        def callback(request):
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "response": "ok",
                    "data_found": False,
                    "record_count": 0,
                    "filters_used": {
                        "companies": [], "symbols": [], "years": [],
                        "standard_items": [], "interpretation": "",
                        "data_availability_note": "", "is_follow_up": False,
                        "context_used": "",
                    },
                },
            )

        router.post("http://localhost:8000/fast_chat_v2").mock(side_effect=callback)
        client = FinancialClient(base_url="http://localhost:8000", timeout_s=10)
        await client.send(
            "follow up",
            [{"role": "user", "content": "first"}, {"role": "assistant", "content": "answer"}],
            {"memory_enabled": False},
        )

    assert captured["payload"]["query"] == "follow up"
    assert len(captured["payload"]["conversation_history"]) == 2
    assert captured["payload"]["memory_enabled"] is False
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd evals && pytest tests/test_client_financial.py -v`
Expected: ImportError for `evals.client.financial`.

- [ ] **Step 4: Implement `evals/client/financial.py`**

```python
"""HTTP client for the chatbot's `/fast_chat_v2` (non-streaming) endpoint."""

from __future__ import annotations

import time
from typing import Any

import httpx

from evals.client.base import ChatClientResult
from evals.metrics import extract_cost_from_response


class FinancialClient:
    """Non-streaming client targeting `POST /fast_chat_v2`."""

    def __init__(self, base_url: str, timeout_s: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def send(
        self,
        query: str,
        conversation_history: list[dict[str, str]],
        api_options: dict[str, Any],
    ) -> ChatClientResult:
        payload = {
            "query": query,
            "conversation_history": conversation_history,
            "memory_enabled": api_options.get("memory_enabled", True),
        }
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(f"{self._base_url}/fast_chat_v2", json=payload)
            response.raise_for_status()
            data = response.json()
        elapsed_ms = (time.perf_counter() - start) * 1000

        cost = extract_cost_from_response(data)

        return ChatClientResult(
            chatbot_text=data.get("response", ""),
            chatbot_metadata=data,
            latency_ms=elapsed_ms,
            ttfb_ms=None,
            cost_usd=cost.cost_usd,
            input_tokens=cost.input_tokens,
            output_tokens=cost.output_tokens,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd evals && pytest tests/test_client_financial.py -v`
Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add evals/client/financial.py evals/tests/test_client_financial.py evals/tests/fixtures/financial_response.json
git commit -m "feat(evals): add FinancialClient for /fast_chat_v2 endpoint"
```

---

## Task 8: AgentStreamClient for `/chat/stream`

**Files:**
- Create: `evals/client/agent_stream.py`
- Create: `evals/tests/test_client_agent_stream.py`
- Create: `evals/tests/fixtures/stream_response.txt`

The `/chat/stream` endpoint returns SSE chunks. Each chunk is `data: <JSON>\n\n`. The final chunk contains the full assembled `AgentChatResponse`. This client consumes the stream and returns the final assembled payload.

- [ ] **Step 1: Create `evals/tests/fixtures/stream_response.txt`**

A minimal multi-chunk SSE stream the test will replay. Each event is `data: <json>\n\n`. The last event contains the final response payload under a `final` key, matching what the FastAPI app emits when streaming completes.

```
data: {"type": "progress", "step": "routing"}

data: {"type": "progress", "step": "tool_call", "tool": "financial_data_query"}

data: {"type": "progress", "step": "synthesizing"}

data: {"type": "final", "payload": {"response": "NCB grew net interest income to J$50B in FY2023.", "tools_executed": ["financial_data_query"], "sources": [{"title": "NCB FY2023 Annual Report"}], "needs_clarification": false, "data_found": true, "record_count": 12, "cost_summary": {"total_cost_usd": 0.012, "total_input_tokens": 4500, "total_output_tokens": 800}}}

```

- [ ] **Step 2: Write failing tests in `evals/tests/test_client_agent_stream.py`**

```python
"""Tests for AgentStreamClient (`/chat/stream`)."""

from pathlib import Path

import httpx
import pytest
import respx

from evals.client.agent_stream import AgentStreamClient


@pytest.mark.asyncio
async def test_send_assembles_final_payload(fixtures_dir: Path):
    raw = (fixtures_dir / "stream_response.txt").read_text()

    async with respx.mock(assert_all_called=True) as router:
        router.post("http://localhost:8000/chat/stream").respond(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=raw.encode("utf-8"),
        )
        client = AgentStreamClient(base_url="http://localhost:8000", timeout_s=10)
        result = await client.send(
            query="How did NCB do in FY2023?",
            conversation_history=[],
            api_options={"enable_financial_data": True, "enable_web_search": False},
        )

    assert "J$50B" in result.chatbot_text
    assert result.chatbot_metadata["tools_executed"] == ["financial_data_query"]
    assert result.chatbot_metadata["data_found"] is True
    assert result.cost_usd == 0.012
    assert result.ttfb_ms is not None
    assert result.ttfb_ms <= result.latency_ms


@pytest.mark.asyncio
async def test_send_raises_when_no_final_event():
    no_final = "data: {\"type\": \"progress\", \"step\": \"x\"}\n\n"

    async with respx.mock() as router:
        router.post("http://localhost:8000/chat/stream").respond(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=no_final.encode("utf-8"),
        )
        client = AgentStreamClient(base_url="http://localhost:8000", timeout_s=10)
        with pytest.raises(RuntimeError, match="no final"):
            await client.send("q", [], {})


@pytest.mark.asyncio
async def test_send_passes_options():
    captured = {}

    async with respx.mock() as router:
        def cb(request):
            import json as _json
            captured["payload"] = _json.loads(request.content)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b'data: {"type": "final", "payload": {"response": "ok"}}\n\n',
            )

        router.post("http://localhost:8000/chat/stream").mock(side_effect=cb)
        client = AgentStreamClient(base_url="http://localhost:8000", timeout_s=10)
        await client.send(
            "test",
            [],
            {"enable_web_search": True, "enable_financial_data": False, "memory_enabled": True},
        )

    assert captured["payload"]["enable_web_search"] is True
    assert captured["payload"]["enable_financial_data"] is False
    assert captured["payload"]["memory_enabled"] is True
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd evals && pytest tests/test_client_agent_stream.py -v`
Expected: ImportError for `evals.client.agent_stream`.

- [ ] **Step 4: Implement `evals/client/agent_stream.py`**

```python
"""HTTP client for the chatbot's `/chat/stream` (SSE) endpoint."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from evals.client.base import ChatClientResult
from evals.metrics import extract_cost_from_response


class AgentStreamClient:
    """Streaming client targeting `POST /chat/stream` (SSE).

    Consumes `data: <json>` events; the event with `type == "final"` carries
    the assembled `AgentChatResponse` under `payload`. Records TTFB and
    total elapsed time separately.
    """

    def __init__(self, base_url: str, timeout_s: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def send(
        self,
        query: str,
        conversation_history: list[dict[str, str]],
        api_options: dict[str, Any],
    ) -> ChatClientResult:
        payload = {
            "query": query,
            "conversation_history": conversation_history,
            "memory_enabled": api_options.get("memory_enabled", True),
            "enable_web_search": api_options.get("enable_web_search", True),
            "enable_financial_data": api_options.get("enable_financial_data", True),
        }

        final_payload: dict[str, Any] | None = None
        start = time.perf_counter()
        ttfb_ms: float | None = None

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            async with client.stream(
                "POST", f"{self._base_url}/chat/stream", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if ttfb_ms is None:
                        ttfb_ms = (time.perf_counter() - start) * 1000
                    if not line.startswith("data:"):
                        continue
                    body = line[len("data:") :].strip()
                    if not body:
                        continue
                    try:
                        event = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "final":
                        final_payload = event.get("payload") or {}

        elapsed_ms = (time.perf_counter() - start) * 1000

        if final_payload is None:
            raise RuntimeError(
                "stream ended with no final event; the chatbot may have failed mid-stream"
            )

        cost = extract_cost_from_response(final_payload)
        return ChatClientResult(
            chatbot_text=final_payload.get("response", ""),
            chatbot_metadata=final_payload,
            latency_ms=elapsed_ms,
            ttfb_ms=ttfb_ms,
            cost_usd=cost.cost_usd,
            input_tokens=cost.input_tokens,
            output_tokens=cost.output_tokens,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd evals && pytest tests/test_client_agent_stream.py -v`
Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add evals/client/agent_stream.py evals/tests/test_client_agent_stream.py evals/tests/fixtures/stream_response.txt
git commit -m "feat(evals): add AgentStreamClient for /chat/stream SSE endpoint"
```

> **Note on stream shape:** the test fixture assumes events of shape
> `{"type": "progress", ...}` and a final `{"type": "final", "payload": ...}`.
> If the real `/chat/stream` emits a different envelope, run a smoke test
> against the live endpoint (Task 16, Step 5) and adjust the parser. The
> implementation deliberately tolerates unknown event types — only `final`
> is load-bearing.

---

## Task 9: PersonaActor (Gemini Flash 2.5)

**Files:**
- Create: `evals/persona_actor.py`
- Create: `evals/tests/test_persona_actor.py`

- [ ] **Step 1: Write failing tests in `evals/tests/test_persona_actor.py`**

```python
"""Tests for PersonaActor — the LLM that role-plays the user persona."""

from unittest.mock import MagicMock

import pytest

from evals.persona import PersonaSpec
from evals.persona_actor import PersonaActor, PersonaTurn


def _make_persona() -> PersonaSpec:
    return PersonaSpec(
        id="p1",
        name="P1",
        category="positive",
        endpoint="fast_chat_v2",
        character="A skeptical analyst.",
        goal="Find NCB revenue growth.",
        max_turns=4,
    )


def _mock_genai_response(text_json: str) -> MagicMock:
    response = MagicMock()
    response.text = text_json
    return response


@pytest.mark.asyncio
async def test_act_returns_structured_turn():
    persona = _make_persona()
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock()
    fake_client.aio.models.generate_content.return_value = _async_value(
        _mock_genai_response('{"utterance": "Show me NCB revenue.", "done": false, "done_reason": null}')
    )

    actor = PersonaActor(
        client=fake_client,
        model="gemini-2.5-flash",
        temperature=0.8,
    )
    turn = await actor.act(persona=persona, transcript_history=[], replicate_index=0)

    assert isinstance(turn, PersonaTurn)
    assert turn.utterance == "Show me NCB revenue."
    assert turn.done is False
    assert turn.done_reason is None


@pytest.mark.asyncio
async def test_act_parses_done_signal():
    persona = _make_persona()
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock()
    fake_client.aio.models.generate_content.return_value = _async_value(
        _mock_genai_response(
            '{"utterance": "Thanks, that\'s clear.", "done": true, "done_reason": "got the breakdown"}'
        )
    )

    actor = PersonaActor(client=fake_client, model="gemini-2.5-flash", temperature=0.8)
    turn = await actor.act(persona=persona, transcript_history=[], replicate_index=1)

    assert turn.done is True
    assert turn.done_reason == "got the breakdown"


@pytest.mark.asyncio
async def test_act_retries_once_on_malformed_json():
    persona = _make_persona()
    fake_client = MagicMock()
    call_results = [
        _mock_genai_response("not json at all"),
        _mock_genai_response('{"utterance": "Recovery question.", "done": false, "done_reason": null}'),
    ]
    fake_client.aio.models.generate_content = MagicMock(
        side_effect=[_async_value(r) for r in call_results]
    )

    actor = PersonaActor(client=fake_client, model="gemini-2.5-flash", temperature=0.8)
    turn = await actor.act(persona=persona, transcript_history=[], replicate_index=0)
    assert turn.utterance == "Recovery question."


@pytest.mark.asyncio
async def test_act_raises_after_second_malformed_json():
    persona = _make_persona()
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock(
        side_effect=[
            _async_value(_mock_genai_response("still not json")),
            _async_value(_mock_genai_response("still still not json")),
        ]
    )
    actor = PersonaActor(client=fake_client, model="gemini-2.5-flash", temperature=0.8)
    with pytest.raises(RuntimeError, match="persona_malformed"):
        await actor.act(persona=persona, transcript_history=[], replicate_index=0)


def _async_value(value):
    """Coerce a sync value into an awaitable returning that value."""
    async def _coro():
        return value

    return _coro()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd evals && pytest tests/test_persona_actor.py -v`
Expected: ImportError for `evals.persona_actor`.

- [ ] **Step 3: Implement `evals/persona_actor.py`**

```python
"""LLM persona actor — role-plays the user side of a simulated conversation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel

from evals.persona import PersonaSpec


class PersonaTurn(BaseModel):
    """One persona-side utterance plus the optional done signal."""

    utterance: str
    done: bool
    done_reason: str | None = None


_SYSTEM_TEMPLATE = """You are role-playing a user interacting with a financial chatbot.

# Your character
{character}

# Your goal
{goal}

# Rules
- Stay in character. Use your character's voice, vocabulary, and level of detail.
- Each turn, respond with ONE message you would send to the chatbot.
- Set "done": true ONLY when you believe your goal has been satisfied,
  OR you have decided it cannot be (and explain in done_reason).
- Do not pretend to be the chatbot. Do not narrate. Only produce what you
  would actually type as a user.
- Output ONLY JSON of the schema below.

# Output JSON schema
{{"utterance": "<the message you send>", "done": <true|false>, "done_reason": "<short reason or null>"}}
"""


def _seed_for(persona_id: str, replicate_index: int) -> int:
    digest = hashlib.sha256(persona_id.encode("utf-8")).digest()
    base = int.from_bytes(digest[:4], "big")
    return (base + replicate_index) & 0x7FFFFFFF


def _format_history(turns: list[dict[str, str]]) -> str:
    """Format prior turns the persona can see (text only, no metadata)."""
    if not turns:
        return "(this is your first message — no history yet)"
    lines = []
    for t in turns:
        lines.append(f"You: {t['persona_utterance']}")
        lines.append(f"Bot: {t['chatbot_text']}")
    return "\n".join(lines)


class PersonaActor:
    """Wraps a google-genai client to act as the persona."""

    def __init__(self, client: Any, model: str, temperature: float) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    async def act(
        self,
        persona: PersonaSpec,
        transcript_history: list[dict[str, str]],
        replicate_index: int,
    ) -> PersonaTurn:
        system_text = _SYSTEM_TEMPLATE.format(
            character=persona.character.strip(),
            goal=persona.goal.strip(),
        )
        history_text = _format_history(transcript_history)
        seed = _seed_for(persona.id, replicate_index)

        for attempt in range(2):
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=[
                    {"role": "user", "parts": [{"text": system_text + "\n\n" + history_text}]}
                ],
                config={
                    "temperature": self._temperature if attempt == 0 else 0.0,
                    "response_mime_type": "application/json",
                    "seed": seed,
                },
            )
            try:
                data = json.loads(response.text)
                return PersonaTurn(**data)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        raise RuntimeError(
            f"persona_malformed: persona {persona.id} returned unparseable JSON twice"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd evals && pytest tests/test_persona_actor.py -v`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add evals/persona_actor.py evals/tests/test_persona_actor.py
git commit -m "feat(evals): add PersonaActor with structured JSON output + retry"
```

> **Note on `google-genai` API:** the test mocks `client.aio.models.generate_content`
> which is the documented async entrypoint for the unified SDK. The `config`
> dict passes `temperature`, `response_mime_type`, and `seed`. If your
> installed `google-genai` version exposes these slightly differently (e.g.,
> as a `GenerateContentConfig` object), the implementation file is the only
> place to adapt; the test contract stays the same.

---

## Task 10: Judge (Gemini Pro 2.5)

**Files:**
- Create: `evals/judge.py`
- Create: `evals/config/judge_rubric.yaml`
- Create: `evals/tests/test_judge.py`

- [ ] **Step 1: Create `evals/config/judge_rubric.yaml`**

```yaml
# Rubric injected into the judge prompt. Edit prose freely; keys are load-bearing.

dimensions:
  groundedness:
    description: |
      Are the chatbot's factual claims supported by the sources it returned?
      A score of 5 means every numerical claim and named entity is traceable
      to an item in the response's `sources` (or `data_preview` for
      financial endpoint). A score of 1 means most claims are unsupported
      or contradict the sources.
  factfulness:
    description: |
      Score only if the persona declares `expected_facts`. For each fact,
      decide whether the conversation as a whole satisfies it (binary).
      Score 5 if every expected fact was satisfied; lower proportionally.
      If the persona has no `expected_facts`, leave `score` null.
  goal_completion:
    description: |
      Did the persona achieve the goal stated in the persona YAML? 5 = goal
      clearly met by the final turn; 1 = no progress at all.
  tool_use_appropriateness:
    description: |
      Did the chatbot fire the right tools at the right times, without
      over-calling or skipping necessary calls? 5 = every tool call was
      justified and complete coverage; 1 = clearly wrong tools or missing
      obvious ones.
  coherence:
    description: |
      Multi-turn consistency. 5 = no contradictions, references prior turns
      correctly, maintains topic. 1 = contradicts itself or loses thread.
  persona_handling:
    description: |
      For positive personas: did the bot meet the user at their level (jargon
      for analyst, plain English for novice)? For negative personas: did the
      bot decline or redirect off-topic / adversarial requests?

verdict_weights:
  positive:
    groundedness: 0.30
    goal_completion: 0.30
    factfulness: 0.20
    tool_use_appropriateness: 0.10
    coherence: 0.05
    persona_handling: 0.05
  negative:
    persona_handling: 0.50
    goal_completion_inverted: 0.30
    coherence: 0.10
    groundedness: 0.10
```

- [ ] **Step 2: Write failing tests in `evals/tests/test_judge.py`**

```python
"""Tests for the Judge component."""

import json
from unittest.mock import MagicMock

import pytest

from evals.judge import Judge, JudgeOutput
from evals.persona import PersonaSpec
from evals.transcript import ChatTurn, TerminationReason, Transcript


def _persona() -> PersonaSpec:
    return PersonaSpec(
        id="p1",
        name="P1",
        category="positive",
        endpoint="fast_chat_v2",
        character="Senior analyst.",
        goal="Find NCB revenue growth.",
        max_turns=4,
        expected_facts=["NCB is mentioned", "Revenue figures included"],
    )


def _transcript() -> Transcript:
    return Transcript(
        conversation_id="p1__rep01",
        persona_id="p1",
        replicate_index=0,
        endpoint="fast_chat_v2",
        turns=[
            ChatTurn(
                turn_index=0,
                persona_utterance="Show me NCB revenue last 2 years.",
                chatbot_text="NCB revenue was J$50B in FY2023, J$45B in FY2022.",
                chatbot_metadata={
                    "data_found": True,
                    "record_count": 2,
                    "sources": [{"title": "NCB FY2023 Annual Report"}],
                    "tools_executed": ["financial_data_query"],
                },
                latency_ms=1800,
                cost_usd=0.003,
            ),
        ],
        termination=TerminationReason(reason="done", at_turn=0, persona_done_reason="got the numbers"),
    )


def _judge_response_json() -> str:
    return json.dumps({
        "scores": {
            "groundedness": {"score": 5, "justification": "Both numbers cited."},
            "factfulness": {
                "score": 5,
                "facts_satisfied": [True, True],
                "justification": "Both expected facts present.",
            },
            "goal_completion": {"score": 5, "justification": "Got NCB revenue."},
            "tool_use_appropriateness": {
                "score": 5,
                "observed_tools": ["financial_data_query"],
                "justification": "Correct tool fired.",
            },
            "coherence": {"score": 5, "justification": "One turn, no contradictions."},
            "persona_handling": {"score": 4, "justification": "Crisp, matches analyst style."},
        },
        "verdict": "pass",
        "verdict_reason": "All dimensions strong.",
        "notable_moments": [
            {"turn": 0, "type": "good_citation", "note": "Cites annual report."}
        ],
    })


def _mock_genai_response(text: str) -> MagicMock:
    r = MagicMock()
    r.text = text
    return r


def _async_value(v):
    async def _c():
        return v
    return _c()


@pytest.mark.asyncio
async def test_evaluate_returns_structured_output():
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock(
        return_value=_async_value(_mock_genai_response(_judge_response_json()))
    )
    judge = Judge(
        client=fake_client,
        model="gemini-2.5-pro",
        temperature=0.2,
    )
    output = await judge.evaluate(persona=_persona(), transcript=_transcript())

    assert isinstance(output, JudgeOutput)
    assert output.verdict == "pass"
    assert output.scores.groundedness.score == 5
    assert output.scores.factfulness.score == 5
    assert output.scores.factfulness.facts_satisfied == [True, True]
    assert len(output.notable_moments) == 1


@pytest.mark.asyncio
async def test_factfulness_null_when_no_expected_facts():
    persona = _persona().model_copy(update={"expected_facts": []})
    body = json.loads(_judge_response_json())
    body["scores"]["factfulness"]["score"] = None
    body["scores"]["factfulness"]["facts_satisfied"] = []
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock(
        return_value=_async_value(_mock_genai_response(json.dumps(body)))
    )

    judge = Judge(client=fake_client, model="gemini-2.5-pro", temperature=0.2)
    output = await judge.evaluate(persona=persona, transcript=_transcript())
    assert output.scores.factfulness.score is None
    assert output.scores.factfulness.facts_satisfied == []


@pytest.mark.asyncio
async def test_evaluate_retries_on_invalid_schema():
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock(
        side_effect=[
            _async_value(_mock_genai_response("not parseable")),
            _async_value(_mock_genai_response(_judge_response_json())),
        ]
    )
    judge = Judge(client=fake_client, model="gemini-2.5-pro", temperature=0.2)
    output = await judge.evaluate(persona=_persona(), transcript=_transcript())
    assert output.verdict == "pass"


@pytest.mark.asyncio
async def test_evaluate_raises_after_second_failure():
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = MagicMock(
        side_effect=[
            _async_value(_mock_genai_response("bad")),
            _async_value(_mock_genai_response("still bad")),
        ]
    )
    judge = Judge(client=fake_client, model="gemini-2.5-pro", temperature=0.2)
    with pytest.raises(RuntimeError, match="judge_failed"):
        await judge.evaluate(persona=_persona(), transcript=_transcript())
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd evals && pytest tests/test_judge.py -v`
Expected: ImportError for `evals.judge`.

- [ ] **Step 4: Implement `evals/judge.py`**

```python
"""LLM judge — scores a completed conversation against persona expectations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from evals.persona import PersonaSpec
from evals.transcript import Transcript

DEFAULT_RUBRIC_PATH = Path(__file__).parent / "config" / "judge_rubric.yaml"


class DimensionScore(BaseModel):
    score: int | None
    justification: str


class FactfulnessScore(BaseModel):
    score: int | None
    facts_satisfied: list[bool] = Field(default_factory=list)
    justification: str


class ToolUseScore(BaseModel):
    score: int
    observed_tools: list[str] = Field(default_factory=list)
    justification: str


class JudgeScores(BaseModel):
    groundedness: DimensionScore
    factfulness: FactfulnessScore
    goal_completion: DimensionScore
    tool_use_appropriateness: ToolUseScore
    coherence: DimensionScore
    persona_handling: DimensionScore


class NotableMoment(BaseModel):
    turn: int
    type: str
    note: str


class JudgeOutput(BaseModel):
    scores: JudgeScores
    verdict: Literal["pass", "fail", "partial"]
    verdict_reason: str
    notable_moments: list[NotableMoment] = Field(default_factory=list)


def _load_rubric(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


_PROMPT_TEMPLATE = """You are an expert evaluator of a financial chatbot.

# Persona under test
- id: {persona_id}
- category: {persona_category}
- character: {character}
- goal: {goal}
- expected_facts:
{expected_facts_block}

# Endpoint hit
{endpoint}

# Run metadata
- total_turns: {total_turns}
- termination: {termination}
- total_latency_ms: {total_latency_ms}
- total_cost_usd: {total_cost_usd}

# Full transcript (each turn includes the chatbot's raw API metadata so you can
# verify groundedness against `sources`, `tools_executed`, `filters_used`, etc.)
{transcript_block}

# Scoring rubric
{rubric_block}

# Output instructions
Score each dimension on a 1–5 integer scale. For `factfulness`, set score to
null and facts_satisfied to [] if expected_facts is empty.
Produce strictly valid JSON matching this schema:

{{
  "scores": {{
    "groundedness": {{"score": int, "justification": str}},
    "factfulness": {{"score": int | null, "facts_satisfied": [bool], "justification": str}},
    "goal_completion": {{"score": int, "justification": str}},
    "tool_use_appropriateness": {{"score": int, "observed_tools": [str], "justification": str}},
    "coherence": {{"score": int, "justification": str}},
    "persona_handling": {{"score": int, "justification": str}}
  }},
  "verdict": "pass" | "fail" | "partial",
  "verdict_reason": str,
  "notable_moments": [{{"turn": int, "type": str, "note": str}}]
}}

Output ONLY the JSON, no commentary.
"""


def _format_transcript(transcript: Transcript) -> str:
    lines = []
    for t in transcript.turns:
        lines.append(f"--- Turn {t.turn_index} ---")
        lines.append(f"USER: {t.persona_utterance}")
        lines.append(f"BOT TEXT: {t.chatbot_text}")
        lines.append(
            f"BOT METADATA (sources, tools, filters):\n{json.dumps(t.chatbot_metadata, indent=2)[:4000]}"
        )
    return "\n".join(lines)


def _format_facts(facts: list[str]) -> str:
    if not facts:
        return "  (none — set factfulness.score to null)"
    return "\n".join(f"  - {f}" for f in facts)


def _format_rubric(rubric: dict[str, Any]) -> str:
    dims = rubric.get("dimensions") or {}
    parts: list[str] = []
    for name, body in dims.items():
        parts.append(f"## {name}\n{body.get('description', '').strip()}")
    return "\n\n".join(parts)


class Judge:
    """Wraps the google-genai client to score one conversation."""

    def __init__(
        self,
        client: Any,
        model: str,
        temperature: float,
        rubric_path: Path | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._rubric = _load_rubric(rubric_path or DEFAULT_RUBRIC_PATH)

    async def evaluate(
        self,
        persona: PersonaSpec,
        transcript: Transcript,
    ) -> JudgeOutput:
        totals = transcript.totals()
        prompt = _PROMPT_TEMPLATE.format(
            persona_id=persona.id,
            persona_category=persona.category,
            character=persona.character.strip(),
            goal=persona.goal.strip(),
            expected_facts_block=_format_facts(persona.expected_facts),
            endpoint=persona.endpoint,
            total_turns=totals["turns"],
            termination=transcript.termination.reason,
            total_latency_ms=int(totals["latency_ms"]),
            total_cost_usd=round(totals["cost_usd"], 6),
            transcript_block=_format_transcript(transcript),
            rubric_block=_format_rubric(self._rubric),
        )

        for attempt in range(2):
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config={
                    "temperature": self._temperature if attempt == 0 else 0.0,
                    "response_mime_type": "application/json",
                },
            )
            try:
                data = json.loads(response.text)
                return JudgeOutput(**data)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        raise RuntimeError(
            f"judge_failed: conversation {transcript.conversation_id} judge returned unparseable JSON twice"
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd evals && pytest tests/test_judge.py -v`
Expected: 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add evals/judge.py evals/config/judge_rubric.yaml evals/tests/test_judge.py
git commit -m "feat(evals): add Judge with rubric-driven prompt and structured scoring"
```

---

## Task 11: Single-conversation runner loop

**Files:**
- Create: `evals/runner.py`
- Create: `evals/tests/test_runner.py`

- [ ] **Step 1: Write the first failing test in `evals/tests/test_runner.py`**

```python
"""Tests for the runner — the conversation loop and orchestration."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from evals.client.base import ChatClientResult
from evals.persona import PersonaSpec
from evals.runner import run_conversation


def _persona(max_turns: int = 3) -> PersonaSpec:
    return PersonaSpec(
        id="p1",
        name="P1",
        category="positive",
        endpoint="fast_chat_v2",
        character="A skeptical analyst.",
        goal="Find NCB revenue growth.",
        max_turns=max_turns,
    )


def _client_result(text: str = "OK answer", cost: float = 0.001) -> ChatClientResult:
    return ChatClientResult(
        chatbot_text=text,
        chatbot_metadata={"data_found": True},
        latency_ms=500,
        ttfb_ms=None,
        cost_usd=cost,
        input_tokens=100,
        output_tokens=40,
    )


@pytest.mark.asyncio
async def test_run_conversation_persona_signals_done():
    """A 2-turn conversation where the persona says done after the 2nd reply."""
    persona = _persona(max_turns=5)
    client = MagicMock()
    client.send = AsyncMock(side_effect=[_client_result("a1"), _client_result("a2")])

    actor = MagicMock()
    from evals.persona_actor import PersonaTurn
    actor.act = AsyncMock(
        side_effect=[
            PersonaTurn(utterance="q1", done=False),
            PersonaTurn(utterance="q2", done=True, done_reason="satisfied"),
        ]
    )

    transcript = await run_conversation(
        persona=persona,
        replicate_index=0,
        chat_client=client,
        persona_actor=actor,
        max_cost_usd=1.0,
    )

    assert len(transcript.turns) == 2
    assert transcript.termination.reason == "done"
    assert transcript.termination.persona_done_reason == "satisfied"
    assert transcript.turns[0].persona_utterance == "q1"
    assert transcript.turns[1].chatbot_text == "a2"


@pytest.mark.asyncio
async def test_run_conversation_hits_max_turns():
    persona = _persona(max_turns=2)
    client = MagicMock()
    client.send = AsyncMock(return_value=_client_result())

    from evals.persona_actor import PersonaTurn
    actor = MagicMock()
    actor.act = AsyncMock(
        side_effect=[PersonaTurn(utterance=f"q{i}", done=False) for i in range(5)]
    )

    transcript = await run_conversation(
        persona=persona,
        replicate_index=0,
        chat_client=client,
        persona_actor=actor,
        max_cost_usd=1.0,
    )
    assert len(transcript.turns) == 2
    assert transcript.termination.reason == "cap"
    assert transcript.termination.at_turn == 1


@pytest.mark.asyncio
async def test_run_conversation_aborts_on_api_error():
    persona = _persona()
    client = MagicMock()
    client.send = AsyncMock(side_effect=RuntimeError("HTTP 500"))

    from evals.persona_actor import PersonaTurn
    actor = MagicMock()
    actor.act = AsyncMock(return_value=PersonaTurn(utterance="q", done=False))

    transcript = await run_conversation(
        persona=persona,
        replicate_index=0,
        chat_client=client,
        persona_actor=actor,
        max_cost_usd=1.0,
    )
    assert transcript.termination.reason == "error"
    assert "HTTP 500" in transcript.termination.error_message


@pytest.mark.asyncio
async def test_run_conversation_respects_per_convo_cost_cap():
    persona = _persona(max_turns=10)
    expensive = _client_result(cost=0.6)
    client = MagicMock()
    client.send = AsyncMock(return_value=expensive)

    from evals.persona_actor import PersonaTurn
    actor = MagicMock()
    actor.act = AsyncMock(
        side_effect=[PersonaTurn(utterance=f"q{i}", done=False) for i in range(10)]
    )

    transcript = await run_conversation(
        persona=persona,
        replicate_index=0,
        chat_client=client,
        persona_actor=actor,
        max_cost_usd=0.5,
    )
    # First turn already exceeds the cap; loop exits after capturing it.
    assert transcript.termination.reason == "error"
    assert "cost cap" in transcript.termination.error_message.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd evals && pytest tests/test_runner.py -v`
Expected: ImportError for `evals.runner`.

- [ ] **Step 3: Implement `evals/runner.py`**

```python
"""Conversation runner — orchestrates persona ↔ chatbot turns."""

from __future__ import annotations

from typing import Any

from evals.client.base import ChatClient
from evals.persona import PersonaSpec
from evals.persona_actor import PersonaActor
from evals.transcript import ChatTurn, TerminationReason, Transcript


async def run_conversation(
    persona: PersonaSpec,
    replicate_index: int,
    chat_client: ChatClient,
    persona_actor: PersonaActor,
    max_cost_usd: float,
) -> Transcript:
    """Run one persona ↔ chatbot conversation to completion."""
    conversation_id = f"{persona.id}__rep{replicate_index + 1:02d}"
    turns: list[ChatTurn] = []
    chatbot_history: list[dict[str, str]] = []
    persona_history: list[dict[str, str]] = []
    running_cost = 0.0

    termination: TerminationReason | None = None

    for turn_index in range(persona.max_turns):
        try:
            persona_turn = await persona_actor.act(
                persona=persona,
                transcript_history=persona_history,
                replicate_index=replicate_index,
            )
        except Exception as exc:
            termination = TerminationReason(
                reason="error",
                at_turn=turn_index,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            break

        try:
            result = await chat_client.send(
                query=persona_turn.utterance,
                conversation_history=chatbot_history,
                api_options=persona.api_options,
            )
        except Exception as exc:
            termination = TerminationReason(
                reason="error",
                at_turn=turn_index,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            break

        chat_turn = ChatTurn(
            turn_index=turn_index,
            persona_utterance=persona_turn.utterance,
            chatbot_text=result.chatbot_text,
            chatbot_metadata=result.chatbot_metadata,
            latency_ms=result.latency_ms,
            ttfb_ms=result.ttfb_ms,
            cost_usd=result.cost_usd,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        turns.append(chat_turn)
        chatbot_history.append({"role": "user", "content": persona_turn.utterance})
        chatbot_history.append({"role": "assistant", "content": result.chatbot_text})
        persona_history.append(
            {"persona_utterance": persona_turn.utterance, "chatbot_text": result.chatbot_text}
        )
        running_cost += result.cost_usd or 0.0

        if running_cost > max_cost_usd:
            termination = TerminationReason(
                reason="error",
                at_turn=turn_index,
                error_type="CostCapExceeded",
                error_message=f"per-conversation cost cap ${max_cost_usd:.2f} exceeded at ${running_cost:.4f}",
            )
            break

        if persona_turn.done:
            termination = TerminationReason(
                reason="done",
                at_turn=turn_index,
                persona_done_reason=persona_turn.done_reason,
            )
            break

    if termination is None:
        termination = TerminationReason(reason="cap", at_turn=persona.max_turns - 1)

    return Transcript(
        conversation_id=conversation_id,
        persona_id=persona.id,
        replicate_index=replicate_index,
        endpoint=persona.endpoint,
        turns=turns,
        termination=termination,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd evals && pytest tests/test_runner.py -v`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add evals/runner.py evals/tests/test_runner.py
git commit -m "feat(evals): add run_conversation loop with cap/done/error termination"
```

---

## Task 12: Run-level orchestration with concurrency

**Files:**
- Modify: `evals/runner.py`
- Modify: `evals/tests/test_runner.py`

- [ ] **Step 1: Add new failing tests to `evals/tests/test_runner.py`**

Append to the file:

```python
import asyncio  # noqa: E402

from evals.runner import RunArtifacts, run_simulation  # noqa: E402


@pytest.mark.asyncio
async def test_run_simulation_produces_one_transcript_per_replicate_per_persona():
    persona_a = _persona(max_turns=1).model_copy(update={"id": "a"})
    persona_b = _persona(max_turns=1).model_copy(update={"id": "b"})

    from evals.persona_actor import PersonaTurn
    actor = MagicMock()
    actor.act = AsyncMock(return_value=PersonaTurn(utterance="q", done=True))

    client = MagicMock()
    client.send = AsyncMock(return_value=_client_result())

    fake_judge = MagicMock()
    from evals.judge import (
        DimensionScore, FactfulnessScore, JudgeOutput, JudgeScores, ToolUseScore,
    )
    fake_judge.evaluate = AsyncMock(
        return_value=JudgeOutput(
            scores=JudgeScores(
                groundedness=DimensionScore(score=4, justification="x"),
                factfulness=FactfulnessScore(score=None, facts_satisfied=[], justification="n/a"),
                goal_completion=DimensionScore(score=4, justification="x"),
                tool_use_appropriateness=ToolUseScore(score=4, justification="x"),
                coherence=DimensionScore(score=4, justification="x"),
                persona_handling=DimensionScore(score=4, justification="x"),
            ),
            verdict="pass",
            verdict_reason="ok",
        )
    )

    def client_for(endpoint: str) -> Any:
        return client

    artifacts = await run_simulation(
        personas=[persona_a, persona_b],
        replicates=2,
        concurrency=2,
        max_cost_usd_per_run=10.0,
        max_cost_usd_per_conversation=1.0,
        chat_client_factory=client_for,
        persona_actor=actor,
        judge=fake_judge,
    )

    assert isinstance(artifacts, RunArtifacts)
    assert len(artifacts.conversations) == 4  # 2 personas × 2 replicates
    ids = {c.transcript.conversation_id for c in artifacts.conversations}
    assert ids == {"a__rep01", "a__rep02", "b__rep01", "b__rep02"}
    assert all(c.judge_output is not None for c in artifacts.conversations)
    assert artifacts.cost_capped is False


@pytest.mark.asyncio
async def test_run_simulation_respects_concurrency_limit():
    """No more than `concurrency` conversations should run simultaneously."""
    persona = _persona(max_turns=1)
    in_flight = 0
    max_observed = 0
    lock = asyncio.Lock()

    async def slow_send(*args, **kwargs):
        nonlocal in_flight, max_observed
        async with lock:
            in_flight += 1
            max_observed = max(max_observed, in_flight)
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        return _client_result()

    from evals.persona_actor import PersonaTurn
    actor = MagicMock()
    actor.act = AsyncMock(return_value=PersonaTurn(utterance="q", done=True))

    client = MagicMock()
    client.send = slow_send

    fake_judge = MagicMock()
    from evals.judge import JudgeOutput, JudgeScores, DimensionScore, FactfulnessScore, ToolUseScore
    fake_judge.evaluate = AsyncMock(
        return_value=JudgeOutput(
            scores=JudgeScores(
                groundedness=DimensionScore(score=4, justification="x"),
                factfulness=FactfulnessScore(score=None, facts_satisfied=[], justification="n/a"),
                goal_completion=DimensionScore(score=4, justification="x"),
                tool_use_appropriateness=ToolUseScore(score=4, justification="x"),
                coherence=DimensionScore(score=4, justification="x"),
                persona_handling=DimensionScore(score=4, justification="x"),
            ),
            verdict="pass",
            verdict_reason="ok",
        )
    )

    await run_simulation(
        personas=[persona],
        replicates=8,
        concurrency=3,
        max_cost_usd_per_run=10.0,
        max_cost_usd_per_conversation=1.0,
        chat_client_factory=lambda _: client,
        persona_actor=actor,
        judge=fake_judge,
    )

    assert max_observed <= 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd evals && pytest tests/test_runner.py::test_run_simulation_produces_one_transcript_per_replicate_per_persona -v`
Expected: ImportError for `RunArtifacts` / `run_simulation`.

- [ ] **Step 3: Extend `evals/runner.py`**

Append the following to the existing file:

```python
import asyncio
from typing import Awaitable, Callable

from evals.judge import Judge, JudgeOutput


class ConversationArtifact:
    """One transcript plus its judge output (if successful)."""

    def __init__(
        self,
        transcript: Transcript,
        judge_output: JudgeOutput | None,
        judge_failed: bool,
        judge_error: str | None,
    ) -> None:
        self.transcript = transcript
        self.judge_output = judge_output
        self.judge_failed = judge_failed
        self.judge_error = judge_error


class RunArtifacts:
    """All conversations from a single run, plus a cost-cap flag."""

    def __init__(
        self,
        conversations: list[ConversationArtifact],
        cost_capped: bool,
    ) -> None:
        self.conversations = conversations
        self.cost_capped = cost_capped


async def run_simulation(
    personas: list[PersonaSpec],
    replicates: int,
    concurrency: int,
    max_cost_usd_per_run: float,
    max_cost_usd_per_conversation: float,
    chat_client_factory: Callable[[str], ChatClient],
    persona_actor: PersonaActor,
    judge: Judge,
) -> RunArtifacts:
    """Run all personas × replicates concurrently with a global cost cap."""
    semaphore = asyncio.Semaphore(concurrency)
    judge_semaphore = asyncio.Semaphore(concurrency * 2)

    running_cost = 0.0
    cost_lock = asyncio.Lock()
    cost_capped = False
    cancel_event = asyncio.Event()

    async def one(persona: PersonaSpec, rep: int) -> ConversationArtifact | None:
        nonlocal running_cost, cost_capped
        if cancel_event.is_set():
            return None
        async with semaphore:
            if cancel_event.is_set():
                return None
            chat_client = chat_client_factory(persona.endpoint)
            transcript = await run_conversation(
                persona=persona,
                replicate_index=rep,
                chat_client=chat_client,
                persona_actor=persona_actor,
                max_cost_usd=max_cost_usd_per_conversation,
            )

            convo_cost = float(transcript.totals()["cost_usd"])
            async with cost_lock:
                running_cost += convo_cost
                if running_cost > max_cost_usd_per_run:
                    cost_capped = True
                    cancel_event.set()

        async with judge_semaphore:
            try:
                output = await judge.evaluate(persona=persona, transcript=transcript)
                return ConversationArtifact(transcript, output, False, None)
            except RuntimeError as exc:
                return ConversationArtifact(transcript, None, True, str(exc))

    tasks = [
        asyncio.create_task(one(persona, rep))
        for persona in personas
        for rep in range(replicates)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    conversations = [r for r in results if r is not None]

    return RunArtifacts(conversations=conversations, cost_capped=cost_capped)
```

- [ ] **Step 4: Run all runner tests**

Run: `cd evals && pytest tests/test_runner.py -v`
Expected: 6 tests pass (4 from Task 11 + 2 new).

- [ ] **Step 5: Commit**

```bash
git add evals/runner.py evals/tests/test_runner.py
git commit -m "feat(evals): add run_simulation with concurrency and judge orchestration"
```

---

## Task 13: Run-level cost cap behavior

**Files:**
- Modify: `evals/tests/test_runner.py`

(The cost-cap mechanism was already coded in Task 12; this task adds the
explicit test that exercises and locks down that behavior.)

- [ ] **Step 1: Add failing test to `evals/tests/test_runner.py`**

Append:

```python
@pytest.mark.asyncio
async def test_run_simulation_cost_cap_marks_artifacts_and_skips_remaining():
    persona = _persona(max_turns=1)
    expensive = _client_result(cost=0.6)
    client = MagicMock()
    client.send = AsyncMock(return_value=expensive)

    from evals.persona_actor import PersonaTurn
    actor = MagicMock()
    actor.act = AsyncMock(return_value=PersonaTurn(utterance="q", done=True))

    from evals.judge import JudgeOutput, JudgeScores, DimensionScore, FactfulnessScore, ToolUseScore
    fake_judge = MagicMock()
    fake_judge.evaluate = AsyncMock(
        return_value=JudgeOutput(
            scores=JudgeScores(
                groundedness=DimensionScore(score=4, justification="x"),
                factfulness=FactfulnessScore(score=None, facts_satisfied=[], justification="n/a"),
                goal_completion=DimensionScore(score=4, justification="x"),
                tool_use_appropriateness=ToolUseScore(score=4, justification="x"),
                coherence=DimensionScore(score=4, justification="x"),
                persona_handling=DimensionScore(score=4, justification="x"),
            ),
            verdict="pass",
            verdict_reason="ok",
        )
    )

    artifacts = await run_simulation(
        personas=[persona],
        replicates=10,
        concurrency=1,
        max_cost_usd_per_run=1.0,                 # only ~1-2 convos fit
        max_cost_usd_per_conversation=1.0,
        chat_client_factory=lambda _: client,
        persona_actor=actor,
        judge=fake_judge,
    )

    assert artifacts.cost_capped is True
    assert len(artifacts.conversations) < 10
    assert len(artifacts.conversations) >= 1
```

- [ ] **Step 2: Run test to confirm pass**

Run: `cd evals && pytest tests/test_runner.py::test_run_simulation_cost_cap_marks_artifacts_and_skips_remaining -v`
Expected: PASS (cost-cap logic from Task 12 already satisfies this).

- [ ] **Step 3: Commit**

```bash
git add evals/tests/test_runner.py
git commit -m "test(evals): lock in run-level cost cap behavior"
```

---

## Task 14: Report writer

**Files:**
- Create: `evals/report.py`
- Create: `evals/tests/test_report.py`

- [ ] **Step 1: Write failing tests in `evals/tests/test_report.py`**

```python
"""Tests for the report writer."""

import json
from pathlib import Path

import pytest

from evals.judge import (
    DimensionScore, FactfulnessScore, JudgeOutput, JudgeScores, ToolUseScore,
)
from evals.persona import PersonaSpec
from evals.report import write_run
from evals.runner import ConversationArtifact, RunArtifacts
from evals.transcript import ChatTurn, TerminationReason, Transcript


def _persona(id: str = "p1") -> PersonaSpec:
    return PersonaSpec(
        id=id,
        name=id.upper(),
        category="positive",
        endpoint="fast_chat_v2",
        character="X",
        goal="Y",
        max_turns=3,
        expected_facts=["fact one"],
    )


def _transcript(persona_id: str, rep: int, cost: float = 0.001) -> Transcript:
    return Transcript(
        conversation_id=f"{persona_id}__rep{rep + 1:02d}",
        persona_id=persona_id,
        replicate_index=rep,
        endpoint="fast_chat_v2",
        turns=[
            ChatTurn(
                turn_index=0,
                persona_utterance="q",
                chatbot_text="a",
                chatbot_metadata={"data_found": True},
                latency_ms=500,
                cost_usd=cost,
                input_tokens=100,
                output_tokens=40,
            )
        ],
        termination=TerminationReason(reason="done", at_turn=0),
    )


def _judge(score: int = 4, verdict: str = "pass") -> JudgeOutput:
    return JudgeOutput(
        scores=JudgeScores(
            groundedness=DimensionScore(score=score, justification="g"),
            factfulness=FactfulnessScore(score=score, facts_satisfied=[True], justification="f"),
            goal_completion=DimensionScore(score=score, justification="gc"),
            tool_use_appropriateness=ToolUseScore(score=score, justification="t"),
            coherence=DimensionScore(score=score, justification="c"),
            persona_handling=DimensionScore(score=score, justification="ph"),
        ),
        verdict=verdict,
        verdict_reason="r",
    )


def test_write_run_creates_expected_files(tmp_path: Path):
    persona = _persona()
    artifacts = RunArtifacts(
        conversations=[
            ConversationArtifact(_transcript("p1", 0), _judge(), False, None),
            ConversationArtifact(_transcript("p1", 1), _judge(score=3, verdict="partial"), False, None),
        ],
        cost_capped=False,
    )
    run_dir = write_run(
        artifacts=artifacts,
        personas=[persona],
        config={"base_url": "http://localhost:8000"},
        run_id="r1",
        git_sha="abc123",
        output_root=tmp_path,
    )
    assert run_dir == tmp_path / "r1"
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "summary.json").exists()
    convo_files = list((run_dir / "conversations").glob("*.json"))
    assert len(convo_files) == 2


def test_summary_aggregates_per_persona(tmp_path: Path):
    persona = _persona()
    artifacts = RunArtifacts(
        conversations=[
            ConversationArtifact(_transcript("p1", 0), _judge(score=4), False, None),
            ConversationArtifact(_transcript("p1", 1), _judge(score=5), False, None),
            ConversationArtifact(_transcript("p1", 2), _judge(score=3, verdict="partial"), False, None),
        ],
        cost_capped=False,
    )
    run_dir = write_run(
        artifacts=artifacts,
        personas=[persona],
        config={"base_url": "http://localhost:8000"},
        run_id="r2",
        git_sha=None,
        output_root=tmp_path,
    )
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["conversation_count"] == 3
    p1 = summary["by_persona"]["p1"]
    assert p1["mean_groundedness"] == pytest.approx(4.0)
    assert p1["std_groundedness"] >= 0
    assert summary["overall"]["verdict_counts"]["pass"] == 2
    assert summary["overall"]["verdict_counts"]["partial"] == 1


def test_manifest_captures_cost_cap_flag(tmp_path: Path):
    persona = _persona()
    artifacts = RunArtifacts(
        conversations=[ConversationArtifact(_transcript("p1", 0), _judge(), False, None)],
        cost_capped=True,
    )
    run_dir = write_run(
        artifacts=artifacts,
        personas=[persona],
        config={},
        run_id="r3",
        git_sha=None,
        output_root=tmp_path,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["cost_capped"] is True


def test_conversation_json_inlines_persona_and_judge(tmp_path: Path):
    persona = _persona()
    artifacts = RunArtifacts(
        conversations=[ConversationArtifact(_transcript("p1", 0), _judge(), False, None)],
        cost_capped=False,
    )
    run_dir = write_run(
        artifacts=artifacts,
        personas=[persona],
        config={},
        run_id="r4",
        git_sha=None,
        output_root=tmp_path,
    )
    convo_path = run_dir / "conversations" / "p1__rep01.json"
    data = json.loads(convo_path.read_text())
    assert data["persona"]["id"] == "p1"
    assert data["judge"]["verdict"] == "pass"
    assert data["totals"]["turns"] == 1
    assert data["endpoint"] == "fast_chat_v2"


def test_judge_failed_serialized(tmp_path: Path):
    persona = _persona()
    artifacts = RunArtifacts(
        conversations=[
            ConversationArtifact(
                _transcript("p1", 0),
                judge_output=None,
                judge_failed=True,
                judge_error="parse error",
            )
        ],
        cost_capped=False,
    )
    run_dir = write_run(
        artifacts=artifacts,
        personas=[persona],
        config={},
        run_id="r5",
        git_sha=None,
        output_root=tmp_path,
    )
    convo = json.loads((run_dir / "conversations" / "p1__rep01.json").read_text())
    assert convo["judge"]["judge_failed"] is True
    assert "parse error" in convo["judge"]["error"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd evals && pytest tests/test_report.py -v`
Expected: ImportError for `evals.report`.

- [ ] **Step 3: Implement `evals/report.py`**

```python
"""Writers for manifest.json, summary.json, and per-conversation JSON files."""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.judge import JudgeOutput
from evals.persona import PersonaSpec
from evals.runner import ConversationArtifact, RunArtifacts


def write_run(
    artifacts: RunArtifacts,
    personas: list[PersonaSpec],
    config: dict[str, Any],
    run_id: str,
    git_sha: str | None,
    output_root: Path,
    started_at: str | None = None,
    ended_at: str | None = None,
) -> Path:
    """Write manifest.json, summary.json, and per-conversation JSON files."""
    run_dir = output_root / run_id
    (run_dir / "conversations").mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "run_id": run_id,
        "started_at": started_at or now,
        "ended_at": ended_at or now,
        "git_sha": git_sha,
        "config": config,
        "personas_run": [p.id for p in personas],
        "replicates": _detect_replicates(artifacts),
        "cost_capped": artifacts.cost_capped,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    persona_by_id = {p.id: p for p in personas}
    for c in artifacts.conversations:
        persona = persona_by_id.get(c.transcript.persona_id)
        path = run_dir / "conversations" / f"{c.transcript.conversation_id}.json"
        path.write_text(
            json.dumps(_convo_payload(c, persona), indent=2),
            encoding="utf-8",
        )

    summary = _summarize(artifacts, personas, run_id, manifest)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return run_dir


def _detect_replicates(artifacts: RunArtifacts) -> int:
    if not artifacts.conversations:
        return 0
    by_persona: dict[str, int] = {}
    for c in artifacts.conversations:
        by_persona.setdefault(c.transcript.persona_id, 0)
        by_persona[c.transcript.persona_id] += 1
    return max(by_persona.values())


def _convo_payload(c: ConversationArtifact, persona: PersonaSpec | None) -> dict[str, Any]:
    t = c.transcript
    payload = {
        "conversation_id": t.conversation_id,
        "persona": persona.model_dump() if persona else None,
        "endpoint": t.endpoint,
        "turns": [turn.model_dump() for turn in t.turns],
        "termination": t.termination.model_dump(),
        "totals": t.totals(),
    }
    if c.judge_failed:
        payload["judge"] = {"judge_failed": True, "error": c.judge_error}
    elif c.judge_output is not None:
        payload["judge"] = c.judge_output.model_dump()
    else:
        payload["judge"] = None
    payload["errors"] = []
    return payload


def _summarize(
    artifacts: RunArtifacts,
    personas: list[PersonaSpec],
    run_id: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    convos = artifacts.conversations

    by_persona: dict[str, dict[str, Any]] = {}
    for p in personas:
        ps = [c for c in convos if c.transcript.persona_id == p.id]
        by_persona[p.id] = _persona_stats(ps)

    by_endpoint = {
        endpoint: _persona_stats([c for c in convos if c.transcript.endpoint == endpoint])
        for endpoint in {c.transcript.endpoint for c in convos}
    }

    by_category = {
        cat: _persona_stats(
            [c for c in convos if _category_for(c.transcript.persona_id, personas) == cat]
        )
        for cat in {p.category for p in personas}
    }

    overall = _overall_stats(convos)

    return {
        "run_id": run_id,
        "started_at": manifest["started_at"],
        "ended_at": manifest["ended_at"],
        "git_sha": manifest["git_sha"],
        "config": manifest["config"],
        "conversation_count": len(convos),
        "by_persona": by_persona,
        "by_endpoint": by_endpoint,
        "by_category": by_category,
        "overall": overall,
    }


def _category_for(persona_id: str, personas: list[PersonaSpec]) -> str:
    for p in personas:
        if p.id == persona_id:
            return p.category
    return "unknown"


def _persona_stats(convos: list[ConversationArtifact]) -> dict[str, Any]:
    if not convos:
        return {"count": 0}

    judged = [c for c in convos if c.judge_output is not None]
    if not judged:
        return {"count": len(convos), "judged_count": 0}

    def dim(field: str) -> list[float]:
        out: list[float] = []
        for c in judged:
            s = getattr(c.judge_output.scores, field)
            if s.score is not None:
                out.append(float(s.score))
        return out

    def mean_std(field: str) -> tuple[float | None, float | None]:
        values = dim(field)
        if not values:
            return None, None
        m = statistics.mean(values)
        s = statistics.stdev(values) if len(values) > 1 else 0.0
        return m, s

    out: dict[str, Any] = {"count": len(convos), "judged_count": len(judged)}
    for d in (
        "groundedness",
        "factfulness",
        "goal_completion",
        "tool_use_appropriateness",
        "coherence",
        "persona_handling",
    ):
        m, s = mean_std(d)
        out[f"mean_{d}"] = m
        out[f"std_{d}"] = s

    verdict_counts: dict[str, int] = {"pass": 0, "partial": 0, "fail": 0}
    for c in judged:
        verdict_counts[c.judge_output.verdict] = verdict_counts.get(c.judge_output.verdict, 0) + 1
    out["verdict_counts"] = verdict_counts

    out["mean_turns"] = statistics.mean(len(c.transcript.turns) for c in convos)
    out["mean_latency_ms"] = statistics.mean(c.transcript.totals()["latency_ms"] for c in convos)
    out["total_cost_usd"] = sum(c.transcript.totals()["cost_usd"] for c in convos)
    return out


def _overall_stats(convos: list[ConversationArtifact]) -> dict[str, Any]:
    if not convos:
        return {}

    judged = [c for c in convos if c.judge_output is not None]

    def values(field: str) -> list[float]:
        return [
            float(getattr(c.judge_output.scores, field).score)
            for c in judged
            if getattr(c.judge_output.scores, field).score is not None
        ]

    overall: dict[str, Any] = {}
    for d in (
        "groundedness",
        "factfulness",
        "goal_completion",
        "tool_use_appropriateness",
        "coherence",
        "persona_handling",
    ):
        v = values(d)
        overall[f"mean_{d}"] = statistics.mean(v) if v else None
        overall[f"std_{d}"] = statistics.stdev(v) if len(v) > 1 else (0.0 if v else None)

    verdict_counts = {"pass": 0, "partial": 0, "fail": 0}
    for c in judged:
        verdict_counts[c.judge_output.verdict] += 1
    overall["verdict_counts"] = verdict_counts

    overall["judge_failed_count"] = sum(1 for c in convos if c.judge_failed)
    overall["incomplete_count"] = sum(
        1 for c in convos if c.transcript.termination.reason == "error"
    )
    overall["mean_turns"] = statistics.mean(len(c.transcript.turns) for c in convos)
    overall["mean_latency_ms"] = statistics.mean(
        c.transcript.totals()["latency_ms"] for c in convos
    )
    overall["total_cost_usd"] = sum(c.transcript.totals()["cost_usd"] for c in convos)
    return overall
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd evals && pytest tests/test_report.py -v`
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add evals/report.py evals/tests/test_report.py
git commit -m "feat(evals): add report writer for manifest, summary, and conversations"
```

---

## Task 15: CLI entrypoint

**Files:**
- Create: `evals/cli.py`
- Create: `evals/tests/test_cli.py`

- [ ] **Step 1: Write failing tests in `evals/tests/test_cli.py`**

```python
"""Tests for the eval-suite CLI argparse layer."""

import pytest

from evals.cli import build_arg_parser, parse_args_to_overrides


def test_parser_accepts_all_documented_flags():
    parser = build_arg_parser()
    ns = parser.parse_args(
        [
            "--base-url", "http://x:9000",
            "--persona", "a",
            "--persona", "b",
            "--category", "positive",
            "--endpoint", "fast_chat_v2",
            "--replicates", "2",
            "--concurrency", "8",
            "--max-cost-usd", "3.5",
            "--run-id", "smoke",
            "--request-timeout-s", "45",
        ]
    )
    assert ns.base_url == "http://x:9000"
    assert ns.personas == ["a", "b"]
    assert ns.category == "positive"
    assert ns.endpoint == "fast_chat_v2"
    assert ns.replicates == 2
    assert ns.concurrency == 8
    assert ns.max_cost_usd == 3.5
    assert ns.run_id == "smoke"
    assert ns.request_timeout_s == 45.0


def test_overrides_skip_none_values():
    parser = build_arg_parser()
    ns = parser.parse_args(["--replicates", "1"])
    overrides = parse_args_to_overrides(ns)
    assert overrides["replicates"] == 1
    assert "base_url" not in overrides
    assert "concurrency" not in overrides


def test_invalid_endpoint_rejected():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--endpoint", "bogus"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd evals && pytest tests/test_cli.py -v`
Expected: ImportError for `evals.cli`.

- [ ] **Step 3: Implement `evals/cli.py`**

```python
"""CLI entrypoint for the eval suite (`python -m evals.cli`)."""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai

from evals.client.agent_stream import AgentStreamClient
from evals.client.base import ChatClient
from evals.client.financial import FinancialClient
from evals.config import load_config
from evals.judge import Judge
from evals.persona import PersonaSpec, load_personas
from evals.persona_actor import PersonaActor
from evals.report import write_run
from evals.runner import run_simulation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evals", description="Simulation eval suite")
    parser.add_argument("--base-url")
    parser.add_argument("--persona", action="append", dest="personas", default=[])
    parser.add_argument("--category", choices=["positive", "negative"])
    parser.add_argument("--endpoint", choices=["fast_chat_v2", "chat_stream"])
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--max-cost-usd", type=float, dest="max_cost_usd")
    parser.add_argument("--request-timeout-s", type=float, dest="request_timeout_s")
    parser.add_argument("--config", dest="config_path", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--personas-dir",
        default=str(Path(__file__).parent / "personas"),
        help="Directory of persona YAML files",
    )
    return parser


def parse_args_to_overrides(ns: argparse.Namespace) -> dict[str, Any]:
    """Convert CLI namespace into config override dict, skipping unset values."""
    mapping = {
        "base_url": ns.base_url,
        "replicates": ns.replicates,
        "concurrency": ns.concurrency,
        "max_cost_usd_per_run": ns.max_cost_usd,
        "request_timeout_s": ns.request_timeout_s,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _filter_personas(
    all_personas: list[PersonaSpec],
    ids: list[str],
    category: str | None,
    endpoint: str | None,
) -> list[PersonaSpec]:
    out = all_personas
    if ids:
        out = [p for p in out if p.id in ids]
    if category:
        out = [p for p in out if p.category == category]
    if endpoint:
        out = [p for p in out if p.endpoint == endpoint]
    return out


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


async def _amain(ns: argparse.Namespace) -> int:
    overrides = parse_args_to_overrides(ns)
    config = load_config(path=ns.config_path, overrides=overrides)

    api_key = os.environ.get(config.gemini_api_key_env)
    if not api_key:
        print(f"ERROR: env var {config.gemini_api_key_env} not set")
        return 2

    personas = _filter_personas(
        load_personas(ns.personas_dir),
        ids=ns.personas,
        category=ns.category,
        endpoint=ns.endpoint,
    )
    if not personas:
        print("ERROR: no personas matched the filters")
        return 2

    genai_client = genai.Client(api_key=api_key)
    persona_actor = PersonaActor(
        client=genai_client,
        model=config.persona_model,
        temperature=config.persona_temperature,
    )
    judge = Judge(
        client=genai_client,
        model=config.judge_model,
        temperature=config.judge_temperature,
    )

    def client_factory(endpoint: str) -> ChatClient:
        if endpoint == "fast_chat_v2":
            return FinancialClient(base_url=config.base_url, timeout_s=config.request_timeout_s)
        return AgentStreamClient(base_url=config.base_url, timeout_s=config.request_timeout_s)

    started_at = datetime.now(timezone.utc).isoformat()
    print(
        f"Running {len(personas)} persona(s) × {config.replicates} replicate(s) "
        f"= {len(personas) * config.replicates} conversation(s)..."
    )
    artifacts = await run_simulation(
        personas=personas,
        replicates=config.replicates,
        concurrency=config.concurrency,
        max_cost_usd_per_run=config.max_cost_usd_per_run,
        max_cost_usd_per_conversation=config.max_cost_usd_per_conversation,
        chat_client_factory=client_factory,
        persona_actor=persona_actor,
        judge=judge,
    )
    ended_at = datetime.now(timezone.utc).isoformat()

    run_id = ns.run_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    output_root = Path(ns.output_dir) if ns.output_dir else Path(__file__).parent / "runs"
    run_dir = write_run(
        artifacts=artifacts,
        personas=personas,
        config=config.model_dump(),
        run_id=run_id,
        git_sha=_git_sha(),
        output_root=output_root,
        started_at=started_at,
        ended_at=ended_at,
    )
    print(f"Wrote run to {run_dir}")
    if artifacts.cost_capped:
        print("WARNING: cost cap reached; run is partial")
    return 0


def main() -> int:
    ns = build_arg_parser().parse_args()
    return asyncio.run(_amain(ns))


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd evals && pytest tests/test_cli.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add evals/cli.py evals/tests/test_cli.py
git commit -m "feat(evals): add CLI entrypoint wiring config, personas, and runner"
```

---

## Task 16: Example personas + live smoke test

**Files:**
- Create: `evals/personas/senior_analyst_ncb_financials.yaml`
- Create: `evals/personas/student_what_is_stock_market.yaml`
- Create: `evals/personas/investor_compare_ncb_vs_jmmb.yaml`
- Create: `evals/personas/negative_chitchat_offtopic.yaml`

- [ ] **Step 1: Create `evals/personas/senior_analyst_ncb_financials.yaml`**

```yaml
id: senior_analyst_ncb_financials
name: "Senior Equity Analyst — NCB deep-dive"
category: positive
endpoint: fast_chat_v2
character: |
  You are a senior equity analyst at a regional brokerage. You speak crisply,
  use financial jargon naturally (EPS, ROE, NIM, YoY), and push back when an
  answer feels surface-level. You don't apologize, you don't pad.
goal: |
  Understand NCB Financial Group's profitability trend over the last 3 fiscal
  years and decide whether revenue growth is being driven by net interest
  income or non-interest income. You're satisfied when you have a clear,
  numbers-backed answer to that specific question.
max_turns: 6
expected_facts:
  - "NCB Financial Group is mentioned by name or symbol"
  - "Revenue or net interest income figures for at least 2 fiscal years"
  - "Some breakdown between interest and non-interest income"
api_options:
  memory_enabled: true
  enable_financial_data: true
opening_style: cold_open
```

- [ ] **Step 2: Create `evals/personas/student_what_is_stock_market.yaml`**

```yaml
id: student_what_is_stock_market
name: "University student — first time on JSE"
category: positive
endpoint: chat_stream
character: |
  You're a first-year university student who has never invested before. You
  ask basic questions, you don't know acronyms, and you appreciate plain
  English. You're not in a hurry.
goal: |
  Understand what the Jamaica Stock Exchange is, why someone would buy a
  stock, and what makes one company a better investment than another. You're
  satisfied when you feel you could explain it to a classmate.
max_turns: 5
expected_facts:
  - "Some explanation of what a stock represents (ownership / equity)"
  - "Some mention of the Jamaica Stock Exchange specifically"
api_options:
  memory_enabled: true
  enable_web_search: true
  enable_financial_data: true
opening_style: cold_open
```

- [ ] **Step 3: Create `evals/personas/investor_compare_ncb_vs_jmmb.yaml`**

```yaml
id: investor_compare_ncb_vs_jmmb
name: "Retail investor — NCB vs JMMB head-to-head"
category: positive
endpoint: fast_chat_v2
character: |
  You're a retail investor in Kingston with about J$2M to allocate. You
  read business news but you're not a professional analyst. You're trying
  to decide between two financial-sector stocks.
goal: |
  Compare NCB Financial Group (NCBFG) and JMMB Group across revenue, net
  income, and EPS for the last 2 fiscal years. You want a concrete
  recommendation or at least a clear summary of which one has been
  performing better.
max_turns: 6
expected_facts:
  - "Both NCB and JMMB (or NCBFG and JMMBGL) are mentioned"
  - "Side-by-side figures for at least one shared metric across both"
  - "Both fiscal years are referenced"
api_options:
  memory_enabled: true
  enable_financial_data: true
opening_style: direct_question
```

- [ ] **Step 4: Create `evals/personas/negative_chitchat_offtopic.yaml`**

```yaml
id: negative_chitchat_offtopic
name: "Off-topic chit-chat probe"
category: negative
endpoint: chat_stream
character: |
  You're a curious user trying to use this chatbot like ChatGPT — asking it
  to write a poem, debug Python code, or chat about your weekend.
goal: |
  Get the bot to engage in off-topic conversation. You "succeed" if it does.
max_turns: 4
expected_facts:
  - "Response declines off-topic requests or redirects to JSE/financial topics"
api_options:
  memory_enabled: true
  enable_web_search: true
  enable_financial_data: true
opening_style: cold_open
```

- [ ] **Step 5: Live smoke test against running API (manual)**

> Requires `GOOGLE_API_KEY` set and the FastAPI app running locally on
> port 8000. Skip this step in CI; it's the only test that hits the real
> Gemini API and the real chatbot.

```bash
# In one terminal: start the chatbot
cd fastapi_app && uvicorn main:app --reload --port 8000

# In another terminal: run a 1-replicate, 1-persona smoke
cd evals
python -m evals.cli \
  --persona senior_analyst_ncb_financials \
  --replicates 1 \
  --run-id smoke_$(date +%s)
```

Expected: prints "Wrote run to evals/runs/smoke_…/". That folder contains
`manifest.json`, `summary.json`, and `conversations/senior_analyst_ncb_financials__rep01.json`.
Open the conversation JSON and verify:
  - `turns[]` has 1+ entries with non-empty `chatbot_text`
  - `judge.verdict` is one of `pass` / `partial` / `fail`
  - `judge.scores.groundedness.score` is a number 1–5

If the stream endpoint persona (e.g., `student_what_is_stock_market`)
fails with "stream ended with no final event", inspect the raw response
shape and adjust `AgentStreamClient` in `evals/client/agent_stream.py`
to match the real envelope (see Task 8 note).

- [ ] **Step 6: Commit**

```bash
git add evals/personas/
git commit -m "feat(evals): add four example personas (analyst, student, investor, negative)"
```

---

## Task 17: Viewer scaffold

**Files:**
- Create: `evals/viewer/index.html`
- Create: `evals/viewer/styles.css`
- Create: `evals/viewer/viewer.js`
- Create: `evals/viewer/fixtures/synthetic_run/manifest.json`
- Create: `evals/viewer/fixtures/synthetic_run/summary.json`
- Create: `evals/viewer/fixtures/synthetic_run/conversations/synthetic__rep01.json`

The viewer is plain HTML/CSS/JS. We build it up across tasks 17–22; this
task gets the shell working with drag-and-drop and a tiny synthetic
fixture so iteration is fast.

- [ ] **Step 1: Create `evals/viewer/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>JSE Eval Viewer</title>
<link rel="stylesheet" href="styles.css" />
<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
</head>
<body>
<header>
  <h1>JSE Eval Viewer</h1>
  <nav id="tabs"></nav>
</header>

<main id="main">
  <section id="dropzone" class="dropzone">
    <p>Drop a <code>runs/&lt;run_id&gt;/</code> folder here, or drop multiple folders for comparison.</p>
    <input type="file" id="file-input" webkitdirectory multiple />
  </section>
  <section id="status"></section>
  <section id="view-root"></section>
</main>

<script src="viewer.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `evals/viewer/styles.css`**

```css
:root {
  --bg: #fafaf9;
  --fg: #1f2937;
  --muted: #6b7280;
  --accent: #2563eb;
  --pass: #16a34a;
  --partial: #ca8a04;
  --fail: #dc2626;
  --neutral: #9ca3af;
  --border: #e5e7eb;
}

* { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, -apple-system, sans-serif; color: var(--fg); background: var(--bg); }
header { padding: 1rem 2rem; border-bottom: 1px solid var(--border); display: flex; align-items: baseline; gap: 2rem; }
h1 { margin: 0; font-size: 1.25rem; }
nav#tabs button { background: none; border: none; padding: 0.5rem 1rem; cursor: pointer; color: var(--muted); }
nav#tabs button.active { color: var(--accent); border-bottom: 2px solid var(--accent); }
main { padding: 1.5rem 2rem; }
.dropzone { border: 2px dashed var(--border); border-radius: 0.5rem; padding: 2rem; text-align: center; }
.dropzone.drag { border-color: var(--accent); background: rgba(37, 99, 235, 0.05); }

table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
th, td { padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid var(--border); font-size: 0.875rem; }
th { background: #f3f4f6; font-weight: 600; cursor: pointer; }
.verdict-pass { color: var(--pass); }
.verdict-partial { color: var(--partial); }
.verdict-fail { color: var(--fail); }
.verdict-judgefailed { color: var(--neutral); }

.bubble { padding: 0.5rem 0.75rem; border-radius: 0.5rem; margin: 0.25rem 0; max-width: 80%; }
.bubble.user { background: #e0e7ff; align-self: flex-end; }
.bubble.bot  { background: #f3f4f6; align-self: flex-start; }
.turn-row { display: flex; flex-direction: column; margin: 0.5rem 0; }
.turn-meta { color: var(--muted); font-size: 0.75rem; }

.delta-up { color: var(--pass); font-weight: 600; }
.delta-down { color: var(--fail); font-weight: 600; }
.delta-noise { color: var(--neutral); }

.turn-drawer { margin: 0.25rem 0 1rem; font-size: 0.8rem; }
.turn-drawer summary { cursor: pointer; color: var(--muted); }
.turn-drawer pre { background: #f3f4f6; padding: 0.5rem; border-radius: 0.25rem; overflow-x: auto; max-height: 20rem; }
.turn-drawer h5 { margin: 0.5rem 0 0.25rem; }
.turn-drawer ul { margin: 0; padding-left: 1.25rem; }
```

- [ ] **Step 3: Create `evals/viewer/viewer.js` (scaffold + drag-drop)**

```javascript
"use strict";

const state = {
  runs: [],            // [{ runId, manifest, summary, conversations: [...] }]
  activeView: null,    // 'overview' | 'list' | 'detail' | 'compare' | 'diff'
  activeConvoId: null, // for detail view
  diff: { a: null, b: null },
};

// ---------- File loading ----------

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const statusEl = document.getElementById("status");
const viewRoot = document.getElementById("view-root");

["dragenter", "dragover"].forEach(evt =>
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.add("drag");
  })
);
["dragleave", "drop"].forEach(evt =>
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  })
);
dropzone.addEventListener("drop", async e => {
  const items = Array.from(e.dataTransfer.items || []);
  const files = [];
  for (const item of items) {
    const entry = item.webkitGetAsEntry?.();
    if (entry) await collectEntryFiles(entry, files);
  }
  if (files.length) await ingestFiles(files);
});
fileInput.addEventListener("change", async () => {
  const files = Array.from(fileInput.files);
  if (files.length) await ingestFiles(files);
});

async function collectEntryFiles(entry, out, basePath = "") {
  if (entry.isFile) {
    const file = await new Promise(res => entry.file(res));
    file._relPath = basePath + entry.name;
    out.push(file);
  } else if (entry.isDirectory) {
    const reader = entry.createReader();
    const entries = await new Promise(res => reader.readEntries(res));
    for (const sub of entries) {
      await collectEntryFiles(sub, out, basePath + entry.name + "/");
    }
  }
}

async function ingestFiles(files) {
  const runs = bucketByRun(files);
  for (const [runId, fileMap] of Object.entries(runs)) {
    const run = await parseRun(runId, fileMap);
    if (run) state.runs.push(run);
  }
  renderApp();
}

function bucketByRun(files) {
  // Detect run roots by finding manifest.json files
  const runs = {};
  for (const f of files) {
    const rel = f._relPath || f.webkitRelativePath || f.name;
    const parts = rel.split("/");
    const idx = parts.findIndex(p => /^[\w.\-:T]+$/.test(p) && parts[parts.indexOf(p) + 1]);
    // Simpler heuristic: the first path segment under the dropped root is the run id
    const runId = parts[0];
    runs[runId] ||= {};
    runs[runId][rel] = f;
  }
  return runs;
}

async function parseRun(runId, fileMap) {
  const manifestKey = Object.keys(fileMap).find(k => k.endsWith("manifest.json"));
  const summaryKey = Object.keys(fileMap).find(k => k.endsWith("summary.json"));
  if (!manifestKey || !summaryKey) {
    setStatus(`Skipped ${runId}: no manifest.json/summary.json`);
    return null;
  }
  const manifest = await readJson(fileMap[manifestKey]);
  const summary = await readJson(fileMap[summaryKey]);
  const convoFiles = Object.entries(fileMap).filter(([k]) => k.includes("/conversations/"));
  const conversations = await Promise.all(convoFiles.map(([, f]) => readJson(f)));
  return { runId: manifest.run_id || runId, manifest, summary, conversations };
}

function readJson(file) {
  return file.text().then(t => JSON.parse(t));
}

function setStatus(msg) {
  statusEl.textContent = msg;
}

// ---------- Rendering ----------

function renderApp() {
  renderTabs();
  if (!state.activeView) {
    state.activeView = state.runs.length > 1 ? "compare" : "overview";
  }
  renderActiveView();
}

function renderTabs() {
  const tabs = document.getElementById("tabs");
  tabs.innerHTML = "";
  const opts = state.runs.length > 1
    ? ["compare", "diff", "list", "detail"]
    : ["overview", "list", "detail"];
  for (const opt of opts) {
    const btn = document.createElement("button");
    btn.textContent = opt;
    btn.className = state.activeView === opt ? "active" : "";
    btn.onclick = () => {
      state.activeView = opt;
      renderApp();
    };
    tabs.appendChild(btn);
  }
}

function renderActiveView() {
  if (!state.runs.length) {
    viewRoot.innerHTML = "<p>No runs loaded yet.</p>";
    return;
  }
  if (state.activeView === "overview") renderOverview();
  else viewRoot.innerHTML = `<p>(${state.activeView} view not implemented yet)</p>`;
}

function renderOverview() {
  const run = state.runs[0];
  const s = run.summary;
  viewRoot.innerHTML = `
    <h2>${run.runId}</h2>
    <p>git: <code>${run.manifest.git_sha || "(unknown)"}</code> · conversations: ${s.conversation_count} · total cost: $${(s.overall.total_cost_usd || 0).toFixed(4)}</p>
    <p>(Overview table coming in next task)</p>
  `;
}
```

- [ ] **Step 4: Create synthetic run fixture for testing the viewer offline**

`evals/viewer/fixtures/synthetic_run/manifest.json`:

```json
{
  "run_id": "synthetic_run",
  "started_at": "2026-05-23T14:00:00Z",
  "ended_at": "2026-05-23T14:02:00Z",
  "git_sha": "abc1234",
  "config": {"base_url": "http://localhost:8000", "replicates": 1},
  "personas_run": ["synthetic"],
  "replicates": 1,
  "cost_capped": false
}
```

`evals/viewer/fixtures/synthetic_run/summary.json`:

```json
{
  "run_id": "synthetic_run",
  "started_at": "2026-05-23T14:00:00Z",
  "ended_at": "2026-05-23T14:02:00Z",
  "git_sha": "abc1234",
  "config": {"base_url": "http://localhost:8000"},
  "conversation_count": 1,
  "by_persona": {
    "synthetic": {
      "count": 1, "judged_count": 1,
      "mean_groundedness": 4.0, "std_groundedness": 0.0,
      "mean_factfulness": 5.0, "std_factfulness": 0.0,
      "mean_goal_completion": 4.0, "std_goal_completion": 0.0,
      "mean_tool_use_appropriateness": 4.0, "std_tool_use_appropriateness": 0.0,
      "mean_coherence": 5.0, "std_coherence": 0.0,
      "mean_persona_handling": 4.0, "std_persona_handling": 0.0,
      "verdict_counts": {"pass": 1, "partial": 0, "fail": 0},
      "mean_turns": 1, "mean_latency_ms": 1500, "total_cost_usd": 0.003
    }
  },
  "by_endpoint": {},
  "by_category": {},
  "overall": {
    "mean_groundedness": 4.0, "std_groundedness": 0.0,
    "mean_factfulness": 5.0, "std_factfulness": 0.0,
    "mean_goal_completion": 4.0, "std_goal_completion": 0.0,
    "mean_tool_use_appropriateness": 4.0, "std_tool_use_appropriateness": 0.0,
    "mean_coherence": 5.0, "std_coherence": 0.0,
    "mean_persona_handling": 4.0, "std_persona_handling": 0.0,
    "verdict_counts": {"pass": 1, "partial": 0, "fail": 0},
    "judge_failed_count": 0, "incomplete_count": 0,
    "mean_turns": 1, "mean_latency_ms": 1500, "total_cost_usd": 0.003
  }
}
```

`evals/viewer/fixtures/synthetic_run/conversations/synthetic__rep01.json`:

```json
{
  "conversation_id": "synthetic__rep01",
  "persona": {
    "id": "synthetic",
    "name": "Synthetic test persona",
    "category": "positive",
    "endpoint": "fast_chat_v2",
    "character": "test character",
    "goal": "test goal",
    "max_turns": 3,
    "expected_facts": ["fact one"]
  },
  "endpoint": "fast_chat_v2",
  "turns": [{
    "turn_index": 0,
    "persona_utterance": "What's NCB revenue?",
    "chatbot_text": "NCB revenue was J$50B in FY2023.",
    "chatbot_metadata": {"data_found": true, "record_count": 1, "tools_executed": ["financial_data_query"]},
    "latency_ms": 1500,
    "ttfb_ms": null,
    "cost_usd": 0.003,
    "input_tokens": 1000,
    "output_tokens": 300
  }],
  "termination": {"reason": "done", "at_turn": 0, "persona_done_reason": "got it"},
  "totals": {"turns": 1, "latency_ms": 1500, "cost_usd": 0.003},
  "judge": {
    "scores": {
      "groundedness": {"score": 4, "justification": "Mostly grounded."},
      "factfulness": {"score": 5, "facts_satisfied": [true], "justification": "Fact met."},
      "goal_completion": {"score": 4, "justification": "Goal mostly met."},
      "tool_use_appropriateness": {"score": 4, "observed_tools": ["financial_data_query"], "justification": "Correct tool."},
      "coherence": {"score": 5, "justification": "Coherent."},
      "persona_handling": {"score": 4, "justification": "Good match."}
    },
    "verdict": "pass",
    "verdict_reason": "All dimensions strong.",
    "notable_moments": []
  },
  "errors": []
}
```

- [ ] **Step 5: Manual smoke test — open viewer with the fixture**

Open `evals/viewer/index.html` directly in a browser (Chrome/Edge work
best for `webkitdirectory`). Drop the `evals/viewer/fixtures/synthetic_run/`
folder onto the dropzone. Expected:
- "synthetic_run" appears in the header
- Overview shows "git: abc1234 · conversations: 1 · total cost: $0.0030"

- [ ] **Step 6: Commit**

```bash
git add evals/viewer/
git commit -m "feat(evals/viewer): scaffold viewer with drag-drop + synthetic fixture"
```

---

## Task 18: Single-run overview, conversation list, conversation detail

**Files:**
- Modify: `evals/viewer/viewer.js`

- [ ] **Step 1: Replace the placeholder `renderActiveView` and `renderOverview` in `evals/viewer/viewer.js`**

Replace the existing `renderActiveView` and `renderOverview` with the full single-run views, and add `renderList` and `renderDetail`:

```javascript
function renderActiveView() {
  if (!state.runs.length) {
    viewRoot.innerHTML = "<p>No runs loaded yet.</p>";
    return;
  }
  if (state.activeView === "overview") renderOverview();
  else if (state.activeView === "list") renderList();
  else if (state.activeView === "detail") renderDetail();
  else viewRoot.innerHTML = `<p>(${state.activeView} view not implemented yet)</p>`;
}

const DIMENSIONS = [
  "groundedness",
  "factfulness",
  "goal_completion",
  "tool_use_appropriateness",
  "coherence",
  "persona_handling",
];

function renderOverview() {
  const run = state.runs[0];
  const s = run.summary;
  const ov = s.overall;
  viewRoot.innerHTML = `
    <h2>${run.runId}</h2>
    <p><strong>git:</strong> <code>${run.manifest.git_sha || "(unknown)"}</code>
       · <strong>started:</strong> ${run.manifest.started_at}
       · <strong>conversations:</strong> ${s.conversation_count}
       · <strong>total cost:</strong> $${(ov.total_cost_usd || 0).toFixed(4)}
       · <strong>mean latency:</strong> ${Math.round(ov.mean_latency_ms || 0)} ms</p>

    <h3>Verdict mix</h3>
    <p>
      <span class="verdict-pass">pass: ${ov.verdict_counts?.pass || 0}</span> ·
      <span class="verdict-partial">partial: ${ov.verdict_counts?.partial || 0}</span> ·
      <span class="verdict-fail">fail: ${ov.verdict_counts?.fail || 0}</span>
      ${ov.judge_failed_count ? `· <span class="verdict-judgefailed">judge_failed: ${ov.judge_failed_count}</span>` : ""}
    </p>

    <h3>Per-persona</h3>
    ${renderPersonaTable(s.by_persona)}
  `;
}

function renderPersonaTable(byPersona) {
  const personas = Object.keys(byPersona);
  if (!personas.length) return "<p>(no personas)</p>";
  const header = ["persona", "count", ...DIMENSIONS.map(d => d.replace(/_/g, " ")), "verdicts"];
  const rows = personas.map(pid => {
    const p = byPersona[pid];
    return `<tr>
      <td>${pid}</td>
      <td>${p.count}</td>
      ${DIMENSIONS.map(d => `<td>${fmtMeanStd(p[`mean_${d}`], p[`std_${d}`])}</td>`).join("")}
      <td>${fmtVerdicts(p.verdict_counts)}</td>
    </tr>`;
  });
  return `<table>
    <thead><tr>${header.map(h => `<th>${h}</th>`).join("")}</tr></thead>
    <tbody>${rows.join("")}</tbody>
  </table>`;
}

function fmtMeanStd(mean, std) {
  if (mean == null) return "—";
  if (std == null) return mean.toFixed(2);
  return `${mean.toFixed(2)} ± ${std.toFixed(2)}`;
}

function fmtVerdicts(counts) {
  if (!counts) return "—";
  return `<span class="verdict-pass">${counts.pass || 0}</span>/` +
         `<span class="verdict-partial">${counts.partial || 0}</span>/` +
         `<span class="verdict-fail">${counts.fail || 0}</span>`;
}

function renderList() {
  const run = state.runs[0];
  const rows = run.conversations.map(c => {
    const verdict = c.judge?.verdict || (c.judge?.judge_failed ? "judge_failed" : "unknown");
    return `<tr class="row-clickable" data-id="${c.conversation_id}">
      <td>${c.conversation_id}</td>
      <td>${c.persona?.id || ""}</td>
      <td>${c.endpoint}</td>
      <td class="verdict-${verdict.replace("_", "")}">${verdict}</td>
      <td>${c.totals?.turns ?? "—"}</td>
      <td>${Math.round(c.totals?.latency_ms || 0)} ms</td>
      <td>$${(c.totals?.cost_usd || 0).toFixed(4)}</td>
    </tr>`;
  });
  viewRoot.innerHTML = `<table>
    <thead><tr><th>conversation</th><th>persona</th><th>endpoint</th>
      <th>verdict</th><th>turns</th><th>latency</th><th>cost</th></tr></thead>
    <tbody>${rows.join("")}</tbody>
  </table>`;
  viewRoot.querySelectorAll(".row-clickable").forEach(tr => {
    tr.addEventListener("click", () => {
      state.activeConvoId = tr.dataset.id;
      state.activeView = "detail";
      renderApp();
    });
  });
}

function renderDetail() {
  const run = state.runs[0];
  const c = state.activeConvoId
    ? run.conversations.find(x => x.conversation_id === state.activeConvoId)
    : run.conversations[0];
  if (!c) { viewRoot.innerHTML = "<p>(no conversation selected)</p>"; return; }
  const transcript = c.turns.map(renderTurn).join("");
  const judge = c.judge?.judge_failed
    ? `<p class="verdict-judgefailed">Judge failed: ${c.judge.error}</p>`
    : renderJudge(c.judge);
  viewRoot.innerHTML = `
    <h2>${c.conversation_id}</h2>
    <p>endpoint: ${c.endpoint} · termination: ${c.termination.reason} (turn ${c.termination.at_turn})</p>
    <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 1rem;">
      <div>${transcript}</div>
      <div>${judge}</div>
    </div>
  `;
  c.turns.forEach((t, i) => {
    const chartSpec = t.chatbot_metadata?.chart?.vega_lite;
    if (chartSpec) {
      vegaEmbed(`#chart-turn-${i}`, chartSpec, { actions: false });
    }
  });
}

function renderTurn(t, i) {
  const chartSlot = t.chatbot_metadata?.chart?.vega_lite
    ? `<div id="chart-turn-${i}" style="margin-top: 0.5rem;"></div>` : "";
  const tools = (t.chatbot_metadata?.tools_executed || []).join(", ");
  const sources = t.chatbot_metadata?.sources || [];
  const filters = t.chatbot_metadata?.filters_used;
  const drawer = `
    <details class="turn-drawer">
      <summary>raw metadata</summary>
      ${sources.length ? `<h5>sources</h5><ul>${sources.map(s =>
        `<li>${escapeHtml(s.title || JSON.stringify(s))}</li>`
      ).join("")}</ul>` : ""}
      ${filters ? `<h5>filters_used</h5><pre>${escapeHtml(JSON.stringify(filters, null, 2))}</pre>` : ""}
      <h5>full response</h5>
      <pre>${escapeHtml(JSON.stringify(t.chatbot_metadata, null, 2))}</pre>
    </details>
  `;
  return `
    <div class="turn-row">
      <div class="bubble user">${escapeHtml(t.persona_utterance)}</div>
      <div class="bubble bot">${escapeHtml(t.chatbot_text)}${chartSlot}</div>
      <div class="turn-meta">turn ${t.turn_index} · ${Math.round(t.latency_ms)} ms${t.cost_usd ? ` · $${t.cost_usd.toFixed(4)}` : ""}${tools ? ` · tools: ${tools}` : ""}</div>
      ${drawer}
    </div>
  `;
}

function renderJudge(j) {
  if (!j) return "<p>(not judged)</p>";
  const scoreRows = DIMENSIONS.map(d => {
    const s = j.scores?.[d];
    if (!s) return "";
    const score = s.score == null ? "—" : s.score;
    return `<tr><td>${d.replace(/_/g, " ")}</td><td>${score}</td><td>${escapeHtml(s.justification || "")}</td></tr>`;
  }).join("");
  const notable = (j.notable_moments || []).map(m =>
    `<li>turn ${m.turn} (${m.type}): ${escapeHtml(m.note)}</li>`
  ).join("");
  return `
    <h3 class="verdict-${j.verdict}">verdict: ${j.verdict}</h3>
    <p>${escapeHtml(j.verdict_reason || "")}</p>
    <table>
      <thead><tr><th>dimension</th><th>score</th><th>justification</th></tr></thead>
      <tbody>${scoreRows}</tbody>
    </table>
    ${notable ? `<h4>Notable moments</h4><ul>${notable}</ul>` : ""}
  `;
}

function escapeHtml(s) {
  return String(s || "").replace(/[&<>"']/g, c => (
    {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]
  ));
}
```

- [ ] **Step 2: Manual smoke test**

Reopen `evals/viewer/index.html`. Drop the synthetic fixture again.
Expected:
- Overview tab: a per-persona table with `synthetic` row showing 4.00 / 5.00 / 4.00 / 4.00 / 5.00 / 4.00 across the dimensions and 1/0/0 verdict counts.
- List tab: one row `synthetic__rep01` with verdict `pass`. Clicking it opens the Detail tab with the user/bot transcript and the judge table.

- [ ] **Step 3: Commit**

```bash
git add evals/viewer/viewer.js
git commit -m "feat(evals/viewer): render overview, conversation list, and detail views"
```

---

## Task 19: Multi-run loading and run roster

**Files:**
- Modify: `evals/viewer/viewer.js`
- Create: `evals/viewer/fixtures/synthetic_run_b/` (a second synthetic run with slightly different scores)

- [ ] **Step 1: Create a second synthetic run for comparison**

Copy the three files from `evals/viewer/fixtures/synthetic_run/` to
`evals/viewer/fixtures/synthetic_run_b/`, then edit:

`evals/viewer/fixtures/synthetic_run_b/manifest.json`:

```json
{
  "run_id": "synthetic_run_b",
  "started_at": "2026-05-24T14:00:00Z",
  "ended_at": "2026-05-24T14:02:00Z",
  "git_sha": "def5678",
  "config": {"base_url": "http://localhost:8000", "replicates": 1},
  "personas_run": ["synthetic"],
  "replicates": 1,
  "cost_capped": false
}
```

`evals/viewer/fixtures/synthetic_run_b/summary.json` — change ALL `_run` to `_run_b`, set `git_sha` to `def5678`, and change scores so the second run scores lower on groundedness (use 3.0 instead of 4.0 and bump some std to 0.5):

```json
{
  "run_id": "synthetic_run_b",
  "started_at": "2026-05-24T14:00:00Z",
  "ended_at": "2026-05-24T14:02:00Z",
  "git_sha": "def5678",
  "config": {"base_url": "http://localhost:8000"},
  "conversation_count": 1,
  "by_persona": {
    "synthetic": {
      "count": 1, "judged_count": 1,
      "mean_groundedness": 3.0, "std_groundedness": 0.0,
      "mean_factfulness": 4.0, "std_factfulness": 0.0,
      "mean_goal_completion": 4.0, "std_goal_completion": 0.0,
      "mean_tool_use_appropriateness": 4.0, "std_tool_use_appropriateness": 0.0,
      "mean_coherence": 5.0, "std_coherence": 0.0,
      "mean_persona_handling": 4.0, "std_persona_handling": 0.0,
      "verdict_counts": {"pass": 0, "partial": 1, "fail": 0},
      "mean_turns": 1, "mean_latency_ms": 1700, "total_cost_usd": 0.004
    }
  },
  "by_endpoint": {},
  "by_category": {},
  "overall": {
    "mean_groundedness": 3.0, "std_groundedness": 0.0,
    "mean_factfulness": 4.0, "std_factfulness": 0.0,
    "mean_goal_completion": 4.0, "std_goal_completion": 0.0,
    "mean_tool_use_appropriateness": 4.0, "std_tool_use_appropriateness": 0.0,
    "mean_coherence": 5.0, "std_coherence": 0.0,
    "mean_persona_handling": 4.0, "std_persona_handling": 0.0,
    "verdict_counts": {"pass": 0, "partial": 1, "fail": 0},
    "judge_failed_count": 0, "incomplete_count": 0,
    "mean_turns": 1, "mean_latency_ms": 1700, "total_cost_usd": 0.004
  }
}
```

Copy `evals/viewer/fixtures/synthetic_run/conversations/synthetic__rep01.json` to `evals/viewer/fixtures/synthetic_run_b/conversations/synthetic__rep01.json` and change `groundedness.score` from `4` to `3` and `verdict` from `"pass"` to `"partial"`.

- [ ] **Step 2: Add the run roster + multi-run mode to `evals/viewer/viewer.js`**

Add this near the top, after `state`:

```javascript
// ---------- Run roster rendering (multi-run mode) ----------

function renderRoster() {
  if (state.runs.length < 2) return "";
  const baselineId = state.diff.a || state.runs[0].runId;
  const candidateId = state.diff.b || (state.runs[1] && state.runs[1].runId);
  const rows = state.runs.map(r => {
    const role = r.runId === baselineId ? "baseline" :
                 r.runId === candidateId ? "candidate" : "loaded";
    return `<tr>
      <td>${r.runId}</td>
      <td><code>${r.manifest.git_sha || "—"}</code></td>
      <td>${r.manifest.started_at}</td>
      <td>${r.summary.conversation_count}</td>
      <td>$${(r.summary.overall.total_cost_usd || 0).toFixed(4)}</td>
      <td>${role}</td>
    </tr>`;
  });
  return `<h3>Loaded runs (${state.runs.length})</h3>
    <table>
      <thead><tr><th>run id</th><th>git</th><th>started</th><th>convos</th><th>cost</th><th>role</th></tr></thead>
      <tbody>${rows.join("")}</tbody>
    </table>`;
}
```

Then modify `renderActiveView` to prepend the roster in multi-run mode:

```javascript
function renderActiveView() {
  if (!state.runs.length) {
    viewRoot.innerHTML = "<p>No runs loaded yet.</p>";
    return;
  }
  const roster = state.runs.length > 1 ? renderRoster() : "";
  if (state.activeView === "overview") { renderOverview(); }
  else if (state.activeView === "list") { renderList(); }
  else if (state.activeView === "detail") { renderDetail(); }
  else if (state.activeView === "compare") { viewRoot.innerHTML = roster + "<p>(compare view coming in next task)</p>"; }
  else if (state.activeView === "diff") { viewRoot.innerHTML = roster + "<p>(diff view coming in next task)</p>"; }
  else { viewRoot.innerHTML = `<p>(${state.activeView} view not implemented yet)</p>`; }
  if (roster && (state.activeView === "compare" || state.activeView === "diff")) return;
  if (roster) viewRoot.insertAdjacentHTML("afterbegin", roster);
}
```

- [ ] **Step 3: Manual smoke test**

Reopen `evals/viewer/index.html`. Drop BOTH
`evals/viewer/fixtures/synthetic_run/` and
`evals/viewer/fixtures/synthetic_run_b/` together (Cmd-click /
Ctrl-click both folders or drop sequentially).
Expected:
- Tabs become: compare · diff · list · detail
- Compare and Diff tabs both show the "Loaded runs (2)" roster table with one row marked `baseline` and one marked `candidate`.

- [ ] **Step 4: Commit**

```bash
git add evals/viewer/viewer.js evals/viewer/fixtures/synthetic_run_b/
git commit -m "feat(evals/viewer): support loading multiple runs and roster rendering"
```

---

## Task 20: Comparison overview view

**Files:**
- Modify: `evals/viewer/viewer.js`

- [ ] **Step 1: Add `renderCompare` to `evals/viewer/viewer.js`**

Replace the placeholder `compare` branch in `renderActiveView`:

```javascript
function renderActiveView() {
  if (!state.runs.length) {
    viewRoot.innerHTML = "<p>No runs loaded yet.</p>";
    return;
  }
  const roster = state.runs.length > 1 ? renderRoster() : "";
  if (state.activeView === "overview") { renderOverview(); }
  else if (state.activeView === "list") { renderList(); }
  else if (state.activeView === "detail") { renderDetail(); }
  else if (state.activeView === "compare") { renderCompare(roster); }
  else if (state.activeView === "diff") { viewRoot.innerHTML = roster + "<p>(diff view coming next task)</p>"; }
  if (roster && state.activeView !== "compare" && state.activeView !== "diff") {
    viewRoot.insertAdjacentHTML("afterbegin", roster);
  }
}

function renderCompare(roster) {
  // Union of personas across all runs
  const allPersonas = new Set();
  state.runs.forEach(r => Object.keys(r.summary.by_persona || {}).forEach(p => allPersonas.add(p)));

  const baseline = state.runs[0];
  const scorecards = state.runs.map(r => {
    const ov = r.summary.overall;
    return `<div style="border: 1px solid var(--border); padding: 0.75rem; border-radius: 0.5rem;">
      <h4>${r.runId} ${r.runId === baseline.runId ? "<small>(baseline)</small>" : ""}</h4>
      <p>convos: ${r.summary.conversation_count} · cost: $${(ov.total_cost_usd || 0).toFixed(4)} · turns: ${(ov.mean_turns || 0).toFixed(1)}</p>
      <p>
        <span class="verdict-pass">${ov.verdict_counts?.pass || 0}</span>/
        <span class="verdict-partial">${ov.verdict_counts?.partial || 0}</span>/
        <span class="verdict-fail">${ov.verdict_counts?.fail || 0}</span>
      </p>
    </div>`;
  });

  // Per-persona table: rows = personas; columns grouped by run
  const headerCells = ["persona"];
  state.runs.forEach(r => DIMENSIONS.forEach(d => headerCells.push(`${r.runId.slice(0, 8)}<br><small>${d}</small>`)));
  const rows = [...allPersonas].map(pid => {
    const cells = [pid];
    state.runs.forEach(r => {
      const p = r.summary.by_persona?.[pid];
      DIMENSIONS.forEach(d => {
        const mean = p?.[`mean_${d}`];
        const std = p?.[`std_${d}`];
        const cell = mean == null ? "—" : fmtMeanStd(mean, std);
        // delta vs baseline
        if (r.runId !== baseline.runId) {
          const bMean = baseline.summary.by_persona?.[pid]?.[`mean_${d}`];
          const bStd = baseline.summary.by_persona?.[pid]?.[`std_${d}`];
          if (mean != null && bMean != null) {
            const delta = mean - bMean;
            const noise = Math.max(0.5, (std || 0) + (bStd || 0));
            const cls = Math.abs(delta) < noise ? "delta-noise"
                       : (delta > 0 ? "delta-up" : "delta-down");
            cells.push(`${cell}<br><span class="${cls}">Δ ${delta >= 0 ? "+" : ""}${delta.toFixed(2)}</span>`);
            return;
          }
        }
        cells.push(cell);
      });
    });
    return `<tr>${cells.map(c => `<td>${c}</td>`).join("")}</tr>`;
  });

  viewRoot.innerHTML = `
    ${roster}
    <h3>Side-by-side scorecards</h3>
    <div style="display: grid; grid-template-columns: repeat(${state.runs.length}, 1fr); gap: 1rem;">
      ${scorecards.join("")}
    </div>
    <h3>Per-persona dimensions vs baseline (${baseline.runId})</h3>
    <table>
      <thead><tr>${headerCells.map(h => `<th>${h}</th>`).join("")}</tr></thead>
      <tbody>${rows.join("")}</tbody>
    </table>
  `;
}
```

- [ ] **Step 2: Manual smoke test**

Reload the viewer with both fixtures. Switch to the `compare` tab.
Expected:
- Two scorecards side-by-side.
- A per-persona table with the `synthetic` row showing `4.00 ± 0.00` under run A and `3.00 ± 0.00` with `Δ -1.00` (in red because `|Δ|=1.0 > σ=0.0+0.0=0.0; noise floor=0.5; 1.0 > 0.5`) under run B for `groundedness`.

- [ ] **Step 3: Commit**

```bash
git add evals/viewer/viewer.js
git commit -m "feat(evals/viewer): add comparison overview with baseline delta highlighting"
```

---

## Task 21: Diff view with biggest movers and conversation pair

**Files:**
- Modify: `evals/viewer/viewer.js`

- [ ] **Step 1: Add `renderDiff` to `evals/viewer/viewer.js`**

Replace the placeholder `diff` branch in `renderActiveView`:

```javascript
function renderActiveView() {
  if (!state.runs.length) {
    viewRoot.innerHTML = "<p>No runs loaded yet.</p>";
    return;
  }
  const roster = state.runs.length > 1 ? renderRoster() : "";
  if (state.activeView === "overview") { renderOverview(); }
  else if (state.activeView === "list") { renderList(); }
  else if (state.activeView === "detail") { renderDetail(); }
  else if (state.activeView === "compare") { renderCompare(roster); }
  else if (state.activeView === "diff") { renderDiff(roster); }
  if (roster && state.activeView !== "compare" && state.activeView !== "diff") {
    viewRoot.insertAdjacentHTML("afterbegin", roster);
  }
}

function renderDiff(roster) {
  const a = state.diff.a ? state.runs.find(r => r.runId === state.diff.a) : state.runs[0];
  const b = state.diff.b ? state.runs.find(r => r.runId === state.diff.b) : state.runs[1];
  if (!a || !b) { viewRoot.innerHTML = roster + "<p>Need two runs loaded.</p>"; return; }

  const pickers = `
    <p>
      A (baseline): <select id="diff-a">${state.runs.map(r => `<option value="${r.runId}"${r.runId === a.runId ? " selected" : ""}>${r.runId}</option>`).join("")}</select>
      &nbsp;B (candidate): <select id="diff-b">${state.runs.map(r => `<option value="${r.runId}"${r.runId === b.runId ? " selected" : ""}>${r.runId}</option>`).join("")}</select>
    </p>
  `;

  const personas = new Set([
    ...Object.keys(a.summary.by_persona || {}),
    ...Object.keys(b.summary.by_persona || {}),
  ]);

  // Per-persona delta cells
  const rows = [...personas].map(pid => {
    const ap = a.summary.by_persona[pid];
    const bp = b.summary.by_persona[pid];
    const cells = [pid];
    DIMENSIONS.forEach(d => {
      const aMean = ap?.[`mean_${d}`];
      const bMean = bp?.[`mean_${d}`];
      const aStd = ap?.[`std_${d}`] || 0;
      const bStd = bp?.[`std_${d}`] || 0;
      if (aMean == null || bMean == null) { cells.push("—"); return; }
      const delta = bMean - aMean;
      const noise = Math.max(0.5, aStd + bStd);
      const cls = Math.abs(delta) < noise ? "delta-noise" : (delta > 0 ? "delta-up" : "delta-down");
      cells.push(`<span class="${cls}">${delta >= 0 ? "+" : ""}${delta.toFixed(2)}</span>`);
    });
    return `<tr><td>${pid}</td>${cells.slice(1).map(c => `<td>${c}</td>`).join("")}</tr>`;
  });

  // Top movers across all (persona × dimension) cells
  const movers = [];
  for (const pid of personas) {
    for (const d of DIMENSIONS) {
      const aMean = a.summary.by_persona[pid]?.[`mean_${d}`];
      const bMean = b.summary.by_persona[pid]?.[`mean_${d}`];
      if (aMean == null || bMean == null) continue;
      movers.push({ pid, dim: d, delta: bMean - aMean });
    }
  }
  movers.sort((x, y) => y.delta - x.delta);
  const top = movers.slice(0, 5);
  const bottom = movers.slice(-5).reverse();

  const pairOptions = [...personas].map(p => `<option value="${p}">${p}</option>`).join("");

  viewRoot.innerHTML = `
    ${roster}
    ${pickers}
    <h3>Per-persona dimension deltas (B − A)</h3>
    <table>
      <thead><tr><th>persona</th>${DIMENSIONS.map(d => `<th>${d.replace(/_/g, " ")}</th>`).join("")}</tr></thead>
      <tbody>${rows.join("")}</tbody>
    </table>

    <h3>Biggest improvements</h3>
    <ol>${top.map(m => `<li><strong>${m.pid}</strong> · ${m.dim}: <span class="delta-up">+${m.delta.toFixed(2)}</span></li>`).join("")}</ol>

    <h3>Biggest regressions</h3>
    <ol>${bottom.map(m => `<li><strong>${m.pid}</strong> · ${m.dim}: <span class="delta-down">${m.delta.toFixed(2)}</span></li>`).join("")}</ol>

    <h3>Conversation pair</h3>
    <p>Persona: <select id="pair-persona">${pairOptions}</select>
       Replicate: <input id="pair-rep" type="number" min="1" max="20" value="1" /></p>
    <div id="pair-display"></div>
  `;

  document.getElementById("diff-a").onchange = e => { state.diff.a = e.target.value; renderApp(); };
  document.getElementById("diff-b").onchange = e => { state.diff.b = e.target.value; renderApp(); };

  function renderPair() {
    const pid = document.getElementById("pair-persona").value;
    const rep = parseInt(document.getElementById("pair-rep").value, 10) || 1;
    const cid = `${pid}__rep${String(rep).padStart(2, "0")}`;
    const aC = a.conversations.find(c => c.conversation_id === cid);
    const bC = b.conversations.find(c => c.conversation_id === cid);
    const pairEl = document.getElementById("pair-display");
    pairEl.innerHTML = `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
        <div><h4>A: ${a.runId}</h4>${aC ? aC.turns.map(renderTurn).join("") : "<p>(not found)</p>"}
          ${aC?.judge ? renderJudge(aC.judge) : ""}</div>
        <div><h4>B: ${b.runId}</h4>${bC ? bC.turns.map(renderTurn).join("") : "<p>(not found)</p>"}
          ${bC?.judge ? renderJudge(bC.judge) : ""}</div>
      </div>
    `;
  }
  document.getElementById("pair-persona").onchange = renderPair;
  document.getElementById("pair-rep").onchange = renderPair;
  renderPair();
}
```

- [ ] **Step 2: Manual smoke test**

Reload viewer with both fixtures loaded. Switch to `diff` tab.
Expected:
- Two `<select>` dropdowns for A and B with both runs listed.
- Per-persona delta table; `synthetic` row shows `groundedness: -1.00` in red.
- Biggest improvements list shows highest positive movers (or empty / `+0.00` ties).
- Biggest regressions list lists `synthetic · groundedness: -1.00` and `synthetic · factfulness: -1.00`.
- Conversation pair section auto-shows `synthetic__rep01` from both runs side-by-side with both judge tables.

- [ ] **Step 3: Commit**

```bash
git add evals/viewer/viewer.js
git commit -m "feat(evals/viewer): add diff view with deltas, top movers, conversation pairs"
```

---

## Task 22: Serve helper and README finalization

**Files:**
- Create: `evals/serve.py`
- Modify: `evals/README.md`

- [ ] **Step 1: Implement `evals/serve.py`**

```python
"""Tiny dev server for the eval viewer + runs."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class CorsHandler(SimpleHTTPRequestHandler):
    """Adds permissive CORS headers so viewer can fetch from the same port."""

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the eval viewer locally")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", default=str(Path(__file__).parent))
    ns = parser.parse_args()

    import os

    os.chdir(ns.root)
    server = ThreadingHTTPServer(("127.0.0.1", ns.port), CorsHandler)
    print(f"Serving {ns.root} at http://127.0.0.1:{ns.port}/viewer/")
    print("Use http://127.0.0.1:{port}/viewer/?run=<run_id> to auto-load a run.".format(port=ns.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Add `?run=` and `?runs=` auto-load support to `evals/viewer/viewer.js`**

Append to the end of the file (after the existing code):

```javascript
// ---------- Auto-load from query string ----------

async function autoLoadFromQuery() {
  const params = new URLSearchParams(window.location.search);
  const single = params.get("run");
  const many = params.get("runs");
  const ids = single ? [single] : (many ? many.split(",") : []);
  for (const id of ids) {
    try {
      const manifest = await (await fetch(`../runs/${id}/manifest.json`)).json();
      const summary = await (await fetch(`../runs/${id}/summary.json`)).json();
      // Fetch conversation index by listing — best effort using manifest.personas_run × replicates
      const conversations = [];
      for (const p of manifest.personas_run || []) {
        for (let r = 1; r <= (manifest.replicates || 1); r++) {
          const cid = `${p}__rep${String(r).padStart(2, "0")}`;
          try {
            const c = await (await fetch(`../runs/${id}/conversations/${cid}.json`)).json();
            conversations.push(c);
          } catch (_e) { /* missing — skip */ }
        }
      }
      state.runs.push({ runId: manifest.run_id || id, manifest, summary, conversations });
    } catch (e) {
      setStatus(`Failed to load ${id}: ${e.message}`);
    }
  }
  if (state.runs.length) renderApp();
}

autoLoadFromQuery();
```

- [ ] **Step 3: Replace `evals/README.md` with full usage docs**

```markdown
# JSE Chatbot Eval Suite

Persona-driven, multi-turn simulation suite for the JSE DataSphere chatbot.
See the [design spec](../docs/superpowers/specs/2026-05-23-simulation-eval-design.md)
for rationale.

## Install

```bash
cd evals
pip install -e ".[dev]"
```

Set `GOOGLE_API_KEY` (or whatever `gemini_api_key_env` is set to in
`config/default.yaml`) to a Gemini API key.

## Run a simulation

The FastAPI chatbot must be reachable at `--base-url`. Example:

```bash
# 1. Start the chatbot in another terminal
cd ../fastapi_app && uvicorn main:app --port 8000

# 2. Run the suite
cd evals
python -m evals.cli --replicates 3
# → writes evals/runs/<timestamp>/
```

Common variations:

```bash
# Run a single persona to iterate quickly
python -m evals.cli --persona senior_analyst_ncb_financials --replicates 1

# Restrict to negative personas
python -m evals.cli --category negative

# Restrict to one endpoint
python -m evals.cli --endpoint fast_chat_v2

# Override defaults
python -m evals.cli --concurrency 6 --max-cost-usd 2.0 --run-id smoke

# Use a custom config file
python -m evals.cli --config path/to/my-config.yaml
```

## View results

Two ways to open the viewer:

**(a) Drag-and-drop** — open `evals/viewer/index.html` in your browser
(Chrome / Edge work best). Drag the `evals/runs/<run_id>/` folder onto the
dropzone. To compare runs, drop two or more folders.

**(b) Local server** — run a tiny static server:

```bash
python -m evals.serve --port 8765
```

Then open:
- `http://localhost:8765/viewer/?run=<run_id>` to auto-load a single run
- `http://localhost:8765/viewer/?runs=<id1>,<id2>` to load multiple runs

## Authoring personas

Each persona is a YAML file in `evals/personas/`. See
[the design spec](../docs/superpowers/specs/2026-05-23-simulation-eval-design.md#4-persona-schema)
for the full schema. The repo ships four examples to start from:

- `senior_analyst_ncb_financials.yaml` — analyst, `/fast_chat_v2`
- `student_what_is_stock_market.yaml` — novice, `/chat/stream`
- `investor_compare_ncb_vs_jmmb.yaml` — retail investor, head-to-head
- `negative_chitchat_offtopic.yaml` — negative; bot should decline

## Configuration

`evals/config/default.yaml` controls models, concurrency, and cost caps.
`evals/config/judge_rubric.yaml` controls the prompt the judge sees;
edit dimension descriptions to tune scoring without touching code.

## Tests

```bash
pytest                                  # all unit tests (no network)
```

The CLI's `--persona` mode is the de-facto live integration test — it
hits the real Gemini API and the real chatbot. Keep `--replicates 1`
for quick iteration.

## Layout

```
evals/
├── persona.py / persona_actor.py / judge.py / runner.py / report.py / cli.py
├── client/{base,financial,agent_stream}.py
├── config/{default,judge_rubric}.yaml
├── personas/*.yaml
├── tests/
├── viewer/{index.html, viewer.js, styles.css}
└── runs/<run_id>/{manifest,summary}.json + conversations/*.json   # gitignored
```
```

- [ ] **Step 4: Smoke test the serve helper**

```bash
cd evals
python -m evals.serve --port 8765
```

Open `http://localhost:8765/viewer/?run=synthetic_run` (after first
copying `evals/viewer/fixtures/synthetic_run/` into `evals/runs/synthetic_run/`
for this test). Expected: viewer auto-loads the run without dragging.

- [ ] **Step 5: Commit**

```bash
git add evals/serve.py evals/viewer/viewer.js evals/README.md
git commit -m "feat(evals): add serve helper, query-string auto-load, and README"
```

---

## Task 23: Final pass — run full test suite and live A/A check

- [ ] **Step 1: Run full test suite**

```bash
cd evals && pytest -v
```

Expected: all tests pass (covering persona, config, transcript, metrics,
both clients, persona_actor, judge, runner, report, cli).

- [ ] **Step 2: Optional — characterize judge variance with an A/A run**

> Only do this if you want a noise baseline. Costs ~$0.30-$0.50.

```bash
# Run #1 (HEAD)
python -m evals.cli --replicates 3 --run-id aa_baseline

# Run #2 (same HEAD, second run)
python -m evals.cli --replicates 3 --run-id aa_second
```

Open the viewer and load both into diff mode. The deltas observed here
are pure noise — anything within those bounds in a real comparison is
likely not signal.

- [ ] **Step 3: Final commit** (only if Step 1 surfaced any test adjustments)

```bash
git add -A
git status   # verify nothing unexpected staged
git commit -m "test(evals): final pass — full suite green"
```

---

## Self-review checklist (done during plan authoring)

- **Spec §3 architecture:** covered by Tasks 1–15 (Python modules) and 17–22 (viewer).
- **Spec §4 persona schema:** Task 2 implements model; Task 16 ships examples.
- **Spec §5 conversation loop:** Task 11 implements; Task 12 orchestrates.
- **Spec §6 judge + rubric:** Task 10 implements; rubric YAML committed alongside.
- **Spec §7 runner CLI + concurrency + cost cap:** Tasks 12, 13, 15.
- **Spec §8 output format:** Task 14 writes manifest/summary/conversations matching the documented shape.
- **Spec §10 viewer single-run:** Tasks 17–18.
- **Spec §10 viewer multi-run + diff:** Tasks 19–21.
- **Spec §11 operational details:** `.gitignore` in Task 1, git SHA capture in Task 15 (`_git_sha`), env var in Task 15, serve helper in Task 22.
- **Spec §12 cross-run noise risk:** addressed by the `|Δ| > σ_A + σ_B` heuristic in Tasks 20–21.
- **Placeholder scan:** every step contains code or a concrete command; no TBDs / vague "implement appropriate".
- **Type consistency:** `ConversationArtifact`, `RunArtifacts`, `Transcript`, `ChatTurn`, `JudgeOutput`, `PersonaTurn`, `PersonaSpec`, `EvalConfig`, `ChatClient`, `ChatClientResult` names match across tasks.
