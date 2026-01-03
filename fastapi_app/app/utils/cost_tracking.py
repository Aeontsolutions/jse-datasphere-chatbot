"""
Cost tracking utilities for Gemini API calls.

Internal-only module for ops/developer monitoring.
Tracks token usage and calculates costs per API call and phase.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# COST TRACKING CONFIGURATION
# =============================================================================

# Default pricing per 1M tokens (USD) - as of late 2024
DEFAULT_FLASH_INPUT_PRICE = 0.15
DEFAULT_FLASH_OUTPUT_PRICE = 0.60
DEFAULT_PRO_INPUT_PRICE = 1.25
DEFAULT_PRO_OUTPUT_PRICE = 5.00


class CostTrackingConfig(BaseSettings):
    """
    Optional environment variable overrides for Gemini pricing.

    Set these env vars to override default pricing:
    - GEMINI_FLASH_INPUT_PRICE_PER_MILLION
    - GEMINI_FLASH_OUTPUT_PRICE_PER_MILLION
    - GEMINI_PRO_INPUT_PRICE_PER_MILLION
    - GEMINI_PRO_OUTPUT_PRICE_PER_MILLION
    """

    gemini_flash_input_price: float = Field(
        default=DEFAULT_FLASH_INPUT_PRICE,
        validation_alias="GEMINI_FLASH_INPUT_PRICE_PER_MILLION",
    )
    gemini_flash_output_price: float = Field(
        default=DEFAULT_FLASH_OUTPUT_PRICE,
        validation_alias="GEMINI_FLASH_OUTPUT_PRICE_PER_MILLION",
    )
    gemini_pro_input_price: float = Field(
        default=DEFAULT_PRO_INPUT_PRICE,
        validation_alias="GEMINI_PRO_INPUT_PRICE_PER_MILLION",
    )
    gemini_pro_output_price: float = Field(
        default=DEFAULT_PRO_OUTPUT_PRICE,
        validation_alias="GEMINI_PRO_OUTPUT_PRICE_PER_MILLION",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Singleton for config
_cost_config: Optional[CostTrackingConfig] = None


def get_cost_config() -> CostTrackingConfig:
    """Get cost tracking configuration singleton."""
    global _cost_config
    if _cost_config is None:
        _cost_config = CostTrackingConfig()
    return _cost_config


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class TokenUsage:
    """Token usage for a single API call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_response(cls, response: Any) -> "TokenUsage":
        """
        Extract token usage from Gemini response.

        Args:
            response: Gemini GenerateContentResponse object

        Returns:
            TokenUsage with extracted token counts
        """
        if not hasattr(response, "usage_metadata") or not response.usage_metadata:
            return cls()

        metadata = response.usage_metadata
        input_tokens = getattr(metadata, "prompt_token_count", 0) or 0
        output_tokens = getattr(metadata, "candidates_token_count", 0) or 0
        cached_tokens = getattr(metadata, "cached_content_token_count", 0) or 0
        total_tokens = getattr(metadata, "total_token_count", 0) or 0

        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens or (input_tokens + output_tokens),
        )


@dataclass
class CostResult:
    """Cost calculation result for a single API call."""

    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    model: str = ""
    phase: str = ""
    token_usage: TokenUsage = field(default_factory=TokenUsage)


# =============================================================================
# COST CALCULATION
# =============================================================================


def get_pricing_for_model(model: str) -> Dict[str, float]:
    """
    Get pricing for a model based on model name.

    Args:
        model: Model name (e.g., "gemini-2.5-flash", "gemini-3-pro-preview")

    Returns:
        Dict with "input_per_million" and "output_per_million" keys
    """
    config = get_cost_config()

    model_lower = model.lower()

    if "flash" in model_lower:
        return {
            "input_per_million": config.gemini_flash_input_price,
            "output_per_million": config.gemini_flash_output_price,
        }
    elif "pro" in model_lower:
        return {
            "input_per_million": config.gemini_pro_input_price,
            "output_per_million": config.gemini_pro_output_price,
        }
    else:
        # Unknown model - use conservative (higher) pricing
        logger.warning(f"Unknown model for pricing: {model}, using pro pricing as fallback")
        return {
            "input_per_million": config.gemini_pro_input_price,
            "output_per_million": config.gemini_pro_output_price,
        }


def calculate_cost(
    model: str,
    token_usage: TokenUsage,
    phase: str = "unknown",
) -> CostResult:
    """
    Calculate cost for an API call.

    Args:
        model: Model name (e.g., "gemini-2.5-flash")
        token_usage: Token usage from the response
        phase: Agent phase name for tracking

    Returns:
        CostResult with calculated costs
    """
    pricing = get_pricing_for_model(model)

    input_cost = (token_usage.input_tokens / 1_000_000) * pricing["input_per_million"]
    output_cost = (token_usage.output_tokens / 1_000_000) * pricing["output_per_million"]
    total_cost = input_cost + output_cost

    return CostResult(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total_cost,
        model=model,
        phase=phase,
        token_usage=token_usage,
    )


def calculate_cost_from_response(
    model: str,
    response: Any,
    phase: str = "unknown",
) -> CostResult:
    """
    Calculate cost directly from a Gemini response.

    Args:
        model: Model name
        response: Gemini response object
        phase: Agent phase name

    Returns:
        CostResult with calculated costs
    """
    token_usage = TokenUsage.from_response(response)
    return calculate_cost(model, token_usage, phase)
