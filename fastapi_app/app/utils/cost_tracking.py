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


def _int_attr(obj: Any, name: str) -> int:
    """Read an integer counter off an SDK object, defaulting to 0.

    Deliberately strict about the type: usage-metadata fields come back as
    `None` on some responses and are absent entirely on others, and test
    doubles auto-create truthy attributes. Anything that isn't an int is
    treated as "not reported" so token arithmetic can never raise.
    """
    value = getattr(obj, name, 0)
    return value if isinstance(value, int) else 0


@dataclass
class TokenUsage:
    """Token usage for a single API call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    # Gemini 2.5 reports reasoning tokens separately from `candidates_token_count`.
    # They are billed as output and, critically, they count against
    # `max_output_tokens` — so a call can hit its ceiling while `output_tokens`
    # still looks tiny. Tracking this is what makes that failure visible.
    thinking_tokens: int = 0

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
        input_tokens = _int_attr(metadata, "prompt_token_count")
        output_tokens = _int_attr(metadata, "candidates_token_count")
        cached_tokens = _int_attr(metadata, "cached_content_token_count")
        total_tokens = _int_attr(metadata, "total_token_count")
        thinking_tokens = _int_attr(metadata, "thoughts_token_count")

        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens or (input_tokens + output_tokens + thinking_tokens),
            thinking_tokens=thinking_tokens,
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
    finish_reason: Optional[str] = None
    truncated: bool = False


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
    # Thinking tokens bill at the output rate. Omitting them under-reported the
    # cost of every reasoning-heavy call — see ADR 0002, where a refusal spent
    # 241 thinking tokens against 13 visible ones.
    billable_output = token_usage.output_tokens + token_usage.thinking_tokens
    output_cost = (billable_output / 1_000_000) * pricing["output_per_million"]
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
    result = calculate_cost(model, token_usage, phase)
    result.finish_reason = get_finish_reason(response)
    result.truncated = result.finish_reason == TRUNCATED_FINISH_REASON
    return result


# =============================================================================
# TRUNCATION DETECTION
# =============================================================================

# The finish reason Gemini reports when generation stopped because it hit
# `max_output_tokens` rather than finishing its answer.
TRUNCATED_FINISH_REASON = "MAX_TOKENS"


def get_finish_reason(response: Any) -> Optional[str]:
    """Return the first candidate's finish reason as a plain string, if any.

    The SDK returns an enum whose `str()` is `FinishReason.MAX_TOKENS`; callers
    (and BigQuery) want the bare name.
    """
    candidates = getattr(response, "candidates", None)
    # Be strict about the shape: this runs on the response path, and a missing
    # or oddly-typed `candidates` must degrade to "unknown", never raise.
    if not isinstance(candidates, (list, tuple)) or not candidates:
        return None

    reason = getattr(candidates[0], "finish_reason", None)
    if reason is None:
        return None

    # Enum members expose `.name`; raw strings pass through unchanged.
    name = getattr(reason, "name", None)
    if isinstance(name, str):
        return name
    return str(reason).rsplit(".", 1)[-1] if not isinstance(reason, str) else reason


def was_truncated(response: Any) -> bool:
    """True when generation was cut short by the output-token ceiling.

    This is the signal that was missing when interaction
    7dba1a26bc014fdebfdb2b9567012c52 was logged as `success: true` despite
    ending mid-sentence.
    """
    return get_finish_reason(response) == TRUNCATED_FINISH_REASON
