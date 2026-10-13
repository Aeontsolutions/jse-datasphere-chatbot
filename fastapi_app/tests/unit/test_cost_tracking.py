"""Unit tests for cost tracking utilities."""

from unittest.mock import MagicMock, Mock

import pytest

from app.utils.cost_tracking import (
    DEFAULT_FLASH_INPUT_PRICE,
    DEFAULT_FLASH_OUTPUT_PRICE,
    DEFAULT_PRO_INPUT_PRICE,
    DEFAULT_PRO_OUTPUT_PRICE,
    CostResult,
    TokenUsage,
    calculate_cost,
    calculate_cost_from_response,
    get_cost_config,
    get_pricing_for_model,
)


@pytest.mark.unit
class TestTokenUsage:
    """Test cases for TokenUsage dataclass."""

    def test_default_values(self):
        """Test TokenUsage has correct default values."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cached_tokens == 0
        assert usage.total_tokens == 0

    def test_from_response_with_full_metadata(self):
        """Test extracting tokens from response with complete usage metadata."""
        mock_response = Mock()
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.cached_content_token_count = 20
        mock_response.usage_metadata.total_token_count = 150

        usage = TokenUsage.from_response(mock_response)

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cached_tokens == 20
        assert usage.total_tokens == 150

    def test_from_response_without_metadata(self):
        """Test extracting tokens from response without usage metadata."""
        mock_response = Mock()
        mock_response.usage_metadata = None

        usage = TokenUsage.from_response(mock_response)

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cached_tokens == 0
        assert usage.total_tokens == 0

    def test_from_response_with_partial_metadata(self):
        """Test extracting tokens from response with partial metadata."""
        mock_response = Mock()
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        # No cached_content_token_count attribute
        del mock_response.usage_metadata.cached_content_token_count
        mock_response.usage_metadata.total_token_count = None

        usage = TokenUsage.from_response(mock_response)

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cached_tokens == 0
        # total_tokens should be calculated from input + output when total is None
        assert usage.total_tokens == 150

    def test_from_response_no_usage_metadata_attribute(self):
        """Test extracting tokens from response without usage_metadata attribute."""
        mock_response = Mock(spec=[])  # No attributes

        usage = TokenUsage.from_response(mock_response)

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0


@pytest.mark.unit
class TestGetPricingForModel:
    """Test cases for pricing lookup function."""

    def test_flash_model_pricing(self):
        """Test pricing lookup for Flash model."""
        pricing = get_pricing_for_model("gemini-2.5-flash")

        assert pricing["input_per_million"] == DEFAULT_FLASH_INPUT_PRICE
        assert pricing["output_per_million"] == DEFAULT_FLASH_OUTPUT_PRICE

    def test_pro_model_pricing(self):
        """Test pricing lookup for Pro model."""
        pricing = get_pricing_for_model("gemini-3-pro-preview")

        assert pricing["input_per_million"] == DEFAULT_PRO_INPUT_PRICE
        assert pricing["output_per_million"] == DEFAULT_PRO_OUTPUT_PRICE

    def test_flash_model_case_insensitive(self):
        """Test Flash model detection is case insensitive."""
        pricing = get_pricing_for_model("GEMINI-2.5-FLASH")
        assert pricing["input_per_million"] == DEFAULT_FLASH_INPUT_PRICE

    def test_pro_model_case_insensitive(self):
        """Test Pro model detection is case insensitive."""
        pricing = get_pricing_for_model("Gemini-3-Pro-Preview")
        assert pricing["input_per_million"] == DEFAULT_PRO_INPUT_PRICE

    def test_unknown_model_uses_pro_pricing(self):
        """Test that unknown models use conservative pro pricing as fallback."""
        pricing = get_pricing_for_model("gemini-unknown-model")

        assert pricing["input_per_million"] == DEFAULT_PRO_INPUT_PRICE
        assert pricing["output_per_million"] == DEFAULT_PRO_OUTPUT_PRICE


@pytest.mark.unit
class TestCalculateCost:
    """Test cases for cost calculation functions."""

    def test_calculate_cost_flash_model(self):
        """Test cost calculation for Flash model."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        result = calculate_cost("gemini-2.5-flash", usage, phase="test")

        # Flash pricing: $0.15/1M input, $0.60/1M output
        expected_input_cost = (1000 / 1_000_000) * DEFAULT_FLASH_INPUT_PRICE
        expected_output_cost = (500 / 1_000_000) * DEFAULT_FLASH_OUTPUT_PRICE
        expected_total = expected_input_cost + expected_output_cost

        assert result.input_cost == pytest.approx(expected_input_cost, rel=1e-6)
        assert result.output_cost == pytest.approx(expected_output_cost, rel=1e-6)
        assert result.total_cost == pytest.approx(expected_total, rel=1e-6)
        assert result.phase == "test"
        assert result.model == "gemini-2.5-flash"

    def test_calculate_cost_pro_model(self):
        """Test cost calculation for Pro model."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        result = calculate_cost("gemini-3-pro-preview", usage, phase="synthesis")

        # Pro pricing: $1.25/1M input, $5.00/1M output
        expected_input_cost = (1000 / 1_000_000) * DEFAULT_PRO_INPUT_PRICE
        expected_output_cost = (500 / 1_000_000) * DEFAULT_PRO_OUTPUT_PRICE
        expected_total = expected_input_cost + expected_output_cost

        assert result.input_cost == pytest.approx(expected_input_cost, rel=1e-6)
        assert result.output_cost == pytest.approx(expected_output_cost, rel=1e-6)
        assert result.total_cost == pytest.approx(expected_total, rel=1e-6)
        assert result.phase == "synthesis"

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        usage = TokenUsage(input_tokens=0, output_tokens=0)
        result = calculate_cost("gemini-2.5-flash", usage, phase="empty")

        assert result.input_cost == 0.0
        assert result.output_cost == 0.0
        assert result.total_cost == 0.0

    def test_calculate_cost_large_token_count(self):
        """Test cost calculation with large token counts."""
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=500_000)
        result = calculate_cost("gemini-2.5-flash", usage, phase="large")

        # 1M input tokens at $0.15/1M = $0.15
        # 500K output tokens at $0.60/1M = $0.30
        expected_input = DEFAULT_FLASH_INPUT_PRICE
        expected_output = 0.5 * DEFAULT_FLASH_OUTPUT_PRICE

        assert result.input_cost == pytest.approx(expected_input, rel=1e-6)
        assert result.output_cost == pytest.approx(expected_output, rel=1e-6)

    def test_calculate_cost_includes_token_usage(self):
        """Test that CostResult includes the original TokenUsage."""
        usage = TokenUsage(input_tokens=100, output_tokens=50, cached_tokens=10)
        result = calculate_cost("gemini-2.5-flash", usage, phase="test")

        assert result.token_usage.input_tokens == 100
        assert result.token_usage.output_tokens == 50
        assert result.token_usage.cached_tokens == 10


@pytest.mark.unit
class TestCalculateCostFromResponse:
    """Test cases for calculate_cost_from_response function."""

    def test_calculate_cost_from_response_with_metadata(self):
        """Test calculating cost directly from a Gemini response."""
        mock_response = Mock()
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.cached_content_token_count = 0
        mock_response.usage_metadata.total_token_count = 150

        result = calculate_cost_from_response(
            model="gemini-2.5-flash", response=mock_response, phase="classification"
        )

        assert result.token_usage.input_tokens == 100
        assert result.token_usage.output_tokens == 50
        assert result.phase == "classification"
        assert result.model == "gemini-2.5-flash"
        assert result.total_cost > 0

    def test_calculate_cost_from_response_without_metadata(self):
        """Test calculating cost from response without usage metadata."""
        mock_response = Mock()
        mock_response.usage_metadata = None

        result = calculate_cost_from_response(
            model="gemini-3-pro-preview", response=mock_response, phase="synthesis"
        )

        assert result.token_usage.input_tokens == 0
        assert result.token_usage.output_tokens == 0
        assert result.total_cost == 0.0


@pytest.mark.unit
class TestCostResult:
    """Test cases for CostResult dataclass."""

    def test_default_values(self):
        """Test CostResult has correct default values."""
        result = CostResult()

        assert result.input_cost == 0.0
        assert result.output_cost == 0.0
        assert result.total_cost == 0.0
        assert result.model == ""
        assert result.phase == ""
        assert isinstance(result.token_usage, TokenUsage)

    def test_custom_values(self):
        """Test CostResult with custom values."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        result = CostResult(
            input_cost=0.01,
            output_cost=0.02,
            total_cost=0.03,
            model="gemini-2.5-flash",
            phase="test",
            token_usage=usage,
        )

        assert result.input_cost == 0.01
        assert result.output_cost == 0.02
        assert result.total_cost == 0.03
        assert result.model == "gemini-2.5-flash"
        assert result.phase == "test"
        assert result.token_usage.input_tokens == 100


@pytest.mark.unit
class TestCostTrackingConfig:
    """Test cases for CostTrackingConfig."""

    def test_default_config_values(self):
        """Test that config loads with expected defaults."""
        config = get_cost_config()

        assert config.gemini_flash_input_price == DEFAULT_FLASH_INPUT_PRICE
        assert config.gemini_flash_output_price == DEFAULT_FLASH_OUTPUT_PRICE
        assert config.gemini_pro_input_price == DEFAULT_PRO_INPUT_PRICE
        assert config.gemini_pro_output_price == DEFAULT_PRO_OUTPUT_PRICE


@pytest.mark.unit
class TestThinkingTokens:
    """Thinking tokens are billed as output and count against max_output_tokens,
    but nothing read them — see issue #72."""

    def test_from_response_captures_thinking_tokens(self):
        resp = Mock()
        resp.usage_metadata = Mock()
        resp.usage_metadata.prompt_token_count = 392
        resp.usage_metadata.candidates_token_count = 9
        resp.usage_metadata.cached_content_token_count = 0
        resp.usage_metadata.total_token_count = 644
        resp.usage_metadata.thoughts_token_count = 243

        usage = TokenUsage.from_response(resp)
        assert usage.thinking_tokens == 243
        assert usage.total_tokens == 644

    def test_thinking_tokens_default_zero_when_absent(self):
        assert TokenUsage().thinking_tokens == 0

    def test_total_fallback_includes_thinking(self):
        """With no API total, input + output + thinking is what was billed."""
        resp = Mock()
        resp.usage_metadata = Mock()
        resp.usage_metadata.prompt_token_count = 100
        resp.usage_metadata.candidates_token_count = 50
        resp.usage_metadata.cached_content_token_count = 0
        resp.usage_metadata.total_token_count = None
        resp.usage_metadata.thoughts_token_count = 30

        assert TokenUsage.from_response(resp).total_tokens == 180

    def test_non_int_counters_are_treated_as_unreported(self):
        """Auto-created Mock attributes must not leak into token arithmetic."""
        resp = Mock()
        resp.usage_metadata = Mock()  # every counter is a Mock, not an int
        usage = TokenUsage.from_response(resp)
        assert (usage.input_tokens, usage.output_tokens, usage.thinking_tokens) == (0, 0, 0)


@pytest.mark.unit
class TestThinkingTokensAreBilled:
    """Issue #84: thinking is billed as output; cost_usd used to drop it."""

    def test_thinking_billed_at_output_rate(self):
        usage = TokenUsage(input_tokens=1000, output_tokens=100, thinking_tokens=400)
        result = calculate_cost("gemini-2.5-flash", usage, phase="generation")

        expected_output = (500 / 1_000_000) * DEFAULT_FLASH_OUTPUT_PRICE
        assert result.output_cost == pytest.approx(expected_output)
        assert result.total_cost == pytest.approx(
            (1000 / 1_000_000) * DEFAULT_FLASH_INPUT_PRICE + expected_output
        )

    def test_cost_from_response_includes_thoughts_token_count(self):
        resp = Mock()
        resp.usage_metadata = Mock()
        resp.usage_metadata.prompt_token_count = 0
        resp.usage_metadata.candidates_token_count = 9
        resp.usage_metadata.cached_content_token_count = 0
        resp.usage_metadata.total_token_count = 252
        resp.usage_metadata.thoughts_token_count = 243

        result = calculate_cost_from_response("gemini-2.5-flash", resp, phase="refusal")
        assert result.output_cost == pytest.approx((252 / 1_000_000) * DEFAULT_FLASH_OUTPUT_PRICE)


@pytest.fixture
def clean_pricing_state(monkeypatch):
    """Isolate the price-config singleton and the warn-once memory per test."""
    from app.utils import cost_tracking

    for var in ("GEMINI_MODEL_PRICES_JSON", "GEMINI_GROUNDING_PRICE_PER_THOUSAND_QUERIES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(cost_tracking, "_cost_config", None)
    monkeypatch.setattr(cost_tracking, "_warned_unpriced_models", set())
    yield cost_tracking
    cost_tracking._cost_config = None


@pytest.mark.unit
class TestPerModelPricing:
    """A new model generation must not silently inherit another's price."""

    def test_override_prices_exact_model_id(self, clean_pricing_state, monkeypatch):
        monkeypatch.setenv(
            "GEMINI_MODEL_PRICES_JSON", '{"Gemini-3.7-Flash": {"input": 0.75, "output": 3.75}}'
        )
        pricing = get_pricing_for_model("gemini-3.7-flash")
        assert pricing == {"input_per_million": 0.75, "output_per_million": 3.75}

    def test_override_wins_over_builtin_entry(self, clean_pricing_state, monkeypatch):
        monkeypatch.setenv(
            "GEMINI_MODEL_PRICES_JSON", '{"gemini-2.5-flash": {"input": 1, "output": 2}}'
        )
        assert get_pricing_for_model("gemini-2.5-flash")["output_per_million"] == 2.0

    def test_builtin_entries_do_not_warn(self, clean_pricing_state, monkeypatch):
        spy = Mock()
        monkeypatch.setattr(clean_pricing_state, "logger", spy)
        get_pricing_for_model("gemini-2.5-flash")
        get_pricing_for_model("gemini-2.5-pro")
        spy.warning.assert_not_called()

    def test_unpriced_model_warns_once_per_process(self, clean_pricing_state, monkeypatch):
        # The app logger is structlog and bypasses caplog, so spy on it directly.
        spy = Mock()
        monkeypatch.setattr(clean_pricing_state, "logger", spy)
        for _ in range(3):
            get_pricing_for_model("gemini-3.7-flash")
        get_pricing_for_model("gemini-3.6-flash")

        assert spy.warning.call_count == 2  # one per distinct model, not one per call
        assert "gemini-3.7-flash" in spy.warning.call_args_list[0].args[0]
        assert "GEMINI_MODEL_PRICES_JSON" in spy.warning.call_args_list[0].args[0]

    def test_unpriced_flash_model_uses_flash_family_rate(self, clean_pricing_state):
        pricing = get_pricing_for_model("gemini-3.7-flash")
        assert pricing["output_per_million"] == DEFAULT_FLASH_OUTPUT_PRICE

    @pytest.mark.parametrize(
        "raw",
        ["not json", "[1, 2]", '{"gemini-x": {"input": 1}}', '{"gemini-x": "free"}'],
    )
    def test_bad_override_is_ignored_not_raised(self, clean_pricing_state, monkeypatch, raw):
        monkeypatch.setenv("GEMINI_MODEL_PRICES_JSON", raw)
        # Falls through to the family rate instead of raising on the request path.
        assert get_pricing_for_model("gemini-x-flash")["input_per_million"] == (
            DEFAULT_FLASH_INPUT_PRICE
        )

    def test_good_entries_survive_a_bad_neighbour(self, clean_pricing_state, monkeypatch):
        monkeypatch.setenv(
            "GEMINI_MODEL_PRICES_JSON",
            '{"bad": {"input": 1}, "gemini-3.7-flash": {"input": 2, "output": 4}}',
        )
        assert get_pricing_for_model("gemini-3.7-flash")["output_per_million"] == 4.0


@pytest.mark.unit
class TestGroundingQueries:
    """Search grounding is billed per executed query, not per token."""

    @staticmethod
    def _grounded_response(queries):
        resp = Mock()
        resp.usage_metadata = None
        candidate = Mock()
        candidate.grounding_metadata = Mock()
        candidate.grounding_metadata.web_search_queries = queries
        resp.candidates = [candidate]
        return resp

    def test_counts_queries_without_charging_by_default(self, clean_pricing_state):
        result = calculate_cost_from_response(
            "gemini-2.5-pro", self._grounded_response(["q1", "q2", "q3"]), phase="generation"
        )
        assert result.grounding_queries == 3
        assert result.grounding_cost == 0.0
        assert result.total_cost == 0.0

    def test_charges_when_a_price_is_configured(self, clean_pricing_state, monkeypatch):
        monkeypatch.setenv("GEMINI_GROUNDING_PRICE_PER_THOUSAND_QUERIES", "14")
        result = calculate_cost_from_response(
            "gemini-2.5-pro", self._grounded_response(["q1", "q2"]), phase="generation"
        )
        assert result.grounding_cost == pytest.approx(0.028)
        assert result.total_cost == pytest.approx(0.028)

    @pytest.mark.parametrize("response", [Mock(), Mock(spec=[]), Mock(candidates=[])])
    def test_unexpected_shapes_count_as_zero(self, clean_pricing_state, response):
        assert calculate_cost_from_response("gemini-2.5-pro", response).grounding_queries == 0


@pytest.mark.unit
class TestLogRecord:
    def test_to_log_record_carries_the_per_phase_breakdown(self):
        result = calculate_cost(
            "gemini-2.5-flash",
            TokenUsage(input_tokens=100, output_tokens=20, thinking_tokens=30, cached_tokens=5),
            phase="routing",
            grounding_queries=2,
        )
        record = result.to_log_record()

        assert record["phase"] == "routing"
        assert record["model"] == "gemini-2.5-flash"
        assert (record["input_tokens"], record["output_tokens"]) == (100, 20)
        assert (record["thinking_tokens"], record["cached_tokens"]) == (30, 5)
        assert record["grounding_queries"] == 2
        assert record["total_cost_usd"] == pytest.approx(result.total_cost)


@pytest.mark.unit
def test_evals_price_table_matches_the_app_defaults():
    """evals/metrics.py keeps its own copy (it cannot import `app`); fail on drift."""
    import importlib.util
    from pathlib import Path

    metrics_path = Path(__file__).resolve().parents[3] / "evals" / "metrics.py"
    if not metrics_path.exists():
        pytest.skip("evals/ not present in this checkout")
    spec = importlib.util.spec_from_file_location("_evals_metrics", metrics_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.GEMINI_PRICING_PER_MILLION == {
        "flash": {"input": DEFAULT_FLASH_INPUT_PRICE, "output": DEFAULT_FLASH_OUTPUT_PRICE},
        "pro": {"input": DEFAULT_PRO_INPUT_PRICE, "output": DEFAULT_PRO_OUTPUT_PRICE},
    }
