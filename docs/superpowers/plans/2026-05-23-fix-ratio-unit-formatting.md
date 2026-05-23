# Fix Dividend / Ratio Unit Formatting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the blanket `item_type == "ratio"` → `%` rule with name-based classification so `dividend_per_share` renders as `J$1.90`, fractional ratios like `roe`/`roa`/`dividend_payout_ratio` render as correct percentages, and pure ratios like `current_ratio` render as plain decimals.

**Architecture:** Add three `frozenset` constants near the top of `financial_utils.py` that classify `standard_item` names into currency / fractional-percent / pure-ratio buckets. Replace lines 864–865 (the single-line ratio override) with a name-based dispatch. Use a value threshold (≤1.5) to distinguish fraction-stored vs already-percent items within the fractional bucket, handling the GraceKennedy (0.343) vs BPOW (21.41) `dividend_payout_ratio` split.

**Tech Stack:** Python 3.11, pytest, google-cloud-bigquery mock via monkeypatch.

---

### Task 1: Add failing tests for all three bug cases

**Files:**
- Modify: `fastapi_app/tests/test_financial_utils.py`

These tests exercise the formatting logic end-to-end through `query_data()` using inline mock rows (no CSV). They must fail on the current code before we touch `financial_utils.py`.

- [ ] **Step 1: Append this test function to `fastapi_app/tests/test_financial_utils.py`**

```python
# ── Ratio / dividend formatting tests (ATS-325) ──────────────────────────────


class _Row:
    """Minimal BQ row stub for formatting tests."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _manager_with_rows(monkeypatch, rows):
    """Return a FinancialDataManager whose BQ client yields the given rows."""

    class _Result:
        def result(self):
            return iter(rows)

    class _Client:
        def query(self, *a, **kw):
            return _Result()

    monkeypatch.setattr(
        "fastapi_app.app.financial_utils.bigquery.Client", lambda *a, **kw: _Client()
    )
    from fastapi_app.app.financial_utils import FinancialDataManager

    return FinancialDataManager()


def test_dividend_per_share_formats_as_currency(monkeypatch):
    """Caribbean Cement: dividend_per_share 1.8976 → J$1.90, not 1.90%."""
    row = _Row(
        Company="Caribbean Cement Company Limited",
        Symbol="CCC",
        Year=2023,
        standard_item="dividend_per_share",
        item=1.8976,
        unit_multiplier=1.0,
        item_type="ratio",
        item_name="Dividend Per Share",
    )
    manager = _manager_with_rows(monkeypatch, [row])
    from fastapi_app.app.models import FinancialDataFilters

    results = manager.query_data(FinancialDataFilters())
    assert len(results) == 1
    fv = results[0].formatted_value
    assert fv == "J$1.90", f"Expected 'J$1.90', got '{fv}'"


def test_dividend_payout_ratio_fraction_multiplied(monkeypatch):
    """GraceKennedy: dividend_payout_ratio 0.343 (fraction) → 34.30%, not 0.34%."""
    row = _Row(
        Company="GraceKennedy Limited",
        Symbol="GK",
        Year=2023,
        standard_item="dividend_payout_ratio",
        item=0.343,
        unit_multiplier=1.0,
        item_type="ratio",
        item_name="Dividend Payout Ratio",
    )
    manager = _manager_with_rows(monkeypatch, [row])
    from fastapi_app.app.models import FinancialDataFilters

    results = manager.query_data(FinancialDataFilters())
    assert len(results) == 1
    fv = results[0].formatted_value
    assert fv == "34.30%", f"Expected '34.30%', got '{fv}'"


def test_dividend_payout_ratio_already_percent(monkeypatch):
    """BPOW: dividend_payout_ratio 21.41 (already %) → 21.41%, not 2141%."""
    row = _Row(
        Company="Barita Power Limited",
        Symbol="BPOW",
        Year=2023,
        standard_item="dividend_payout_ratio",
        item=21.41,
        unit_multiplier=1.0,
        item_type="ratio",
        item_name="Dividend Payout Ratio",
    )
    manager = _manager_with_rows(monkeypatch, [row])
    from fastapi_app.app.models import FinancialDataFilters

    results = manager.query_data(FinancialDataFilters())
    assert len(results) == 1
    fv = results[0].formatted_value
    assert fv == "21.41%", f"Expected '21.41%', got '{fv}'"


def test_roe_fraction_multiplied(monkeypatch):
    """ROE 0.53 (fraction) → 53.00%."""
    row = _Row(
        Company="Dolla Financial Services Limited",
        Symbol="DOLLA",
        Year=2023,
        standard_item="roe",
        item=0.53,
        unit_multiplier=1.0,
        item_type="ratio",
        item_name="Return on Equity",
    )
    manager = _manager_with_rows(monkeypatch, [row])
    from fastapi_app.app.models import FinancialDataFilters

    results = manager.query_data(FinancialDataFilters())
    assert len(results) == 1
    fv = results[0].formatted_value
    assert fv == "53.00%", f"Expected '53.00%', got '{fv}'"


def test_current_ratio_no_percent_suffix(monkeypatch):
    """current_ratio 1.11 → '1.11' (plain decimal, no %)."""
    row = _Row(
        Company="Some Company",
        Symbol="SCO",
        Year=2023,
        standard_item="current_ratio",
        item=1.11,
        unit_multiplier=1.0,
        item_type="ratio",
        item_name="Current Ratio",
    )
    manager = _manager_with_rows(monkeypatch, [row])
    from fastapi_app.app.models import FinancialDataFilters

    results = manager.query_data(FinancialDataFilters())
    assert len(results) == 1
    fv = results[0].formatted_value
    assert fv == "1.11", f"Expected '1.11', got '{fv}'"
```

- [ ] **Step 2: Run the tests to confirm they fail**

```
cd fastapi_app
pytest tests/test_financial_utils.py::test_dividend_per_share_formats_as_currency \
       tests/test_financial_utils.py::test_dividend_payout_ratio_fraction_multiplied \
       tests/test_financial_utils.py::test_dividend_payout_ratio_already_percent \
       tests/test_financial_utils.py::test_roe_fraction_multiplied \
       tests/test_financial_utils.py::test_current_ratio_no_percent_suffix \
       -v
```

Expected: all 5 FAIL (wrong formatted_value assertions).

---

### Task 2: Add item-classification constants to `financial_utils.py`

**Files:**
- Modify: `fastapi_app/app/financial_utils.py` — after line 24 (`USE_DSPY = ...`)

- [ ] **Step 1: Insert constants immediately after the `USE_DSPY` line**

After:
```python
USE_DSPY = os.getenv("USE_DSPY", "false").lower() in ("true", "1", "yes")
```

Insert:
```python

# Items stored as a monetary value per share (J$/share) — format with J$ prefix
_CURRENCY_RATIO_ITEMS = frozenset(
    {
        "dividend_per_share",
        "eps",
    }
)

# Items stored as a 0–1 fraction that should be displayed as a percentage.
# If abs(value) <= 1.5 the value is a fraction and is multiplied by 100.
# If abs(value) > 1.5 the value is already in percent form and is used as-is.
_FRACTIONAL_PERCENT_ITEMS = frozenset(
    {
        "roe",
        "roa",
        "gross_profit_margin",
        "net_profit_margin",
        "operating_profit_margin",
        "efficiency_ratio",
        "dividend_payout_ratio",
    }
)

# Items that are dimensionless ratios — display as plain decimal, no % suffix.
_PURE_RATIO_ITEMS = frozenset(
    {
        "current_ratio",
        "debt_to_equity_ratio",
    }
)
```

---

### Task 3: Replace the single-line ratio override with name-based formatting

**Files:**
- Modify: `fastapi_app/app/financial_utils.py:859–865`

The existing block is:
```python
                item_type = (
                    get_row_attr(row, "item_type")
                    if hasattr(row, "item_type") or "item_type" in dir(row)
                    else ""
                )
                if item_type == "ratio" and unit_multiplier == 1.0 and item is not None:
                    formatted_value = f"{item:.2f}%"
```

- [ ] **Step 1: Replace those 7 lines with the name-based dispatch**

```python
                standard_item_key = str(
                    get_row_attr(row, "standard_item") if item is not None else ""
                ).lower()
                item_type = (
                    get_row_attr(row, "item_type")
                    if hasattr(row, "item_type") or "item_type" in dir(row)
                    else ""
                )
                if item_type == "ratio" and unit_multiplier == 1.0 and item is not None:
                    if standard_item_key in _CURRENCY_RATIO_ITEMS:
                        formatted_value = f"J${item:,.2f}"
                    elif standard_item_key in _FRACTIONAL_PERCENT_ITEMS:
                        pct = item * 100 if abs(item) <= 1.5 else item
                        formatted_value = f"{pct:.2f}%"
                    elif standard_item_key in _PURE_RATIO_ITEMS:
                        formatted_value = f"{item:.2f}"
                    else:
                        formatted_value = f"{item:.2f}%"
```

Note: `standard_item_key` is extracted here rather than reusing the `standard_item` from the `FinancialDataRecord` constructor below because the constructor hasn't run yet at this point in the loop.

---

### Task 4: Run all tests and confirm green

- [ ] **Step 1: Run only the new formatting tests**

```
cd fastapi_app
pytest tests/test_financial_utils.py::test_dividend_per_share_formats_as_currency \
       tests/test_financial_utils.py::test_dividend_payout_ratio_fraction_multiplied \
       tests/test_financial_utils.py::test_dividend_payout_ratio_already_percent \
       tests/test_financial_utils.py::test_roe_fraction_multiplied \
       tests/test_financial_utils.py::test_current_ratio_no_percent_suffix \
       -v
```

Expected: all 5 PASS.

- [ ] **Step 2: Run the full test suite to check for regressions**

```
cd fastapi_app
pytest tests/ -v --tb=short
```

Expected: all previously passing tests still PASS.

---

### Task 5: Commit

- [ ] **Step 1: Stage and commit**

```bash
git add fastapi_app/app/financial_utils.py fastapi_app/tests/test_financial_utils.py
git commit -m "fix(financial_utils): format ratio items by standard_item name not item_type

- dividend_per_share renders as J\$X.XX (currency) not X.XX%
- roe/roa/margins multiply fraction by 100 before adding %
- dividend_payout_ratio uses value threshold (<=1.5) to handle
  both fraction-stored (GraceKennedy 0.343) and percent-stored
  (BPOW 21.41) variants
- current_ratio/debt_to_equity_ratio render as plain decimals

Closes ATS-325"
```
