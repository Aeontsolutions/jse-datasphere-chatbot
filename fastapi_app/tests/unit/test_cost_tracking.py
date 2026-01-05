"""Unit tests for cost tracking utilities."""

import pytest
from unittest.mock import Mock, MagicMock

from app.utils.cost_tracking import (
    TokenUsage,
    CostResult,
    calculate_cost,
    calculate_cost_from_response,
    get_pricing_for_model,
    get_cost_config,
    DEFAULT_FLASH_INPUT_PRICE,
    DEFAULT_FLASH_OUTPUT_PRICE,
    DEFAULT_PRO_INPUT_PRICE,
    DEFAULT_PRO_OUTPUT_PRICE,
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
