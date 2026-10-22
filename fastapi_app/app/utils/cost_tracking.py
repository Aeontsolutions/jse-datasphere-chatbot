"""
Cost tracking utilities for Gemini API calls.

Internal-only module for ops/developer monitoring.
Tracks token usage and calculates costs per API call and phase.
"""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Optional, Set

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

    The flash/pro rows are only a *family fallback*. Price a specific model
    exactly with GEMINI_MODEL_PRICES_JSON, e.g.
    '{"gemini-3.7-flash": {"input": <usd>, "output": <usd>}}' (USD per 1M
    tokens; "output" is applied to visible output and thinking tokens alike).

    Google Search grounding is billed per executed search query, not per token.
    Set GEMINI_GROUNDING_PRICE_PER_THOUSAND_QUERIES to include it in cost_usd;
    at the default of 0 the query count is still recorded but costs nothing.
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
    gemini_model_prices_json: Optional[str] = Field(
        default=None, validation_alias="GEMINI_MODEL_PRICES_JSON"
    )
    gemini_grounding_price_per_thousand_queries: float = Field(
        default=0.0, validation_alias="GEMINI_GROUNDING_PRICE_PER_THOUSAND_QUERIES"
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

    Usage-metadata fields come back as `None` on some responses and are absent
    on others, and loosely-mocked test doubles auto-create truthy non-int
    attributes. Anything that is not an int counts as "not reported", so token
    arithmetic can never raise.
    """
    value = getattr(obj, name, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


@dataclass
class TokenUsage:
    """Token usage for a single API call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    # Billed as output and counted against max_output_tokens, but reported
    # separately from candidates_token_count — so a call can exhaust its budget
    # on thinking and return almost no visible text (issue #72).
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
            # The API total includes thinking; the fallback must too.
            total_tokens=total_tokens or (input_tokens + output_tokens + thinking_tokens),
            thinking_tokens=thinking_tokens,
        )


@dataclass
class CostResult:
    """Cost calculation result for a single API call."""

    input_cost: float = 0.0
    # Covers visible output AND thinking tokens (both bill at the output rate).
    output_cost: float = 0.0
    total_cost: float = 0.0
    model: str = ""
    phase: str = ""
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    # Search queries the model executed for grounding. Billed per query, not per
    # token; folded into total_cost only when a grounding price is configured.
    grounding_queries: int = 0
    grounding_cost: float = 0.0

    def to_log_record(self) -> Dict[str, Any]:
        """Flat per-phase record for the interactions table's phase_costs_json.

        Field names mirror `models.PhaseCost` so both endpoints log one shape.
        """
        return {
            "phase": self.phase,
            "model": self.model,
            "input_tokens": self.token_usage.input_tokens,
            "output_tokens": self.token_usage.output_tokens,
            "thinking_tokens": self.token_usage.thinking_tokens,
            "cached_tokens": self.token_usage.cached_tokens,
            "grounding_queries": self.grounding_queries,
            "input_cost_usd": self.input_cost,
            "output_cost_usd": self.output_cost,
            "grounding_cost_usd": self.grounding_cost,
            "total_cost_usd": self.total_cost,
        }


# =============================================================================
# COST CALCULATION
# =============================================================================


@lru_cache(maxsize=8)
def _parse_model_price_overrides(raw: Optional[str]) -> Dict[str, Dict[str, float]]:
    """Parse GEMINI_MODEL_PRICES_JSON into {lowercase model id: {input, output}}.

    Bad JSON or a bad entry is logged and skipped, never raised: a pricing typo
    must not take down the request path.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        logger.error(f"GEMINI_MODEL_PRICES_JSON is not valid JSON, ignoring it: {exc}")
        return {}
    if not isinstance(parsed, dict):
        logger.error("GEMINI_MODEL_PRICES_JSON must be a JSON object, ignoring it")
        return {}

    prices: Dict[str, Dict[str, float]] = {}
    for model_id, entry in parsed.items():
        try:
            prices[str(model_id).lower()] = {
                "input": float(entry["input"]),
                "output": float(entry["output"]),
            }
        except (KeyError, TypeError, ValueError):
            logger.error(
                f"GEMINI_MODEL_PRICES_JSON entry for {model_id!r} needs numeric "
                '"input" and "output"; skipping it'
            )
    return prices


# Models already warned about, so a model with no price entry logs once per
# process rather than once per request.
_warned_unpriced_models: Set[str] = set()


def get_pricing_for_model(model: str) -> Dict[str, float]:
    """
    Get pricing for a model.

    Resolution order:
      1. an exact model id in GEMINI_MODEL_PRICES_JSON
      2. an exact model id with a built-in entry (gemini-2.5-flash / -pro)
      3. the flash/pro family rate, with a warning the first time that model is
         seen. Any model that lands here is priced with a stale guess: a new
         generation (gemini-3.x) must not silently inherit an old generation's
         rates, so set GEMINI_MODEL_PRICES_JSON for it.

    Args:
        model: Model name (e.g., "gemini-2.5-flash", "gemini-3.7-flash")

    Returns:
        Dict with "input_per_million" and "output_per_million" keys
    """
    config = get_cost_config()
    model_lower = model.lower()

    flash = {
        "input_per_million": config.gemini_flash_input_price,
        "output_per_million": config.gemini_flash_output_price,
    }
    pro = {
        "input_per_million": config.gemini_pro_input_price,
        "output_per_million": config.gemini_pro_output_price,
    }

    override = _parse_model_price_overrides(config.gemini_model_prices_json).get(model_lower)
    if override is not None:
        return {
            "input_per_million": override["input"],
            "output_per_million": override["output"],
        }

    # Carried over unchanged from the previous family defaults; not re-verified
    # against Google's current price list.
    builtin = {"gemini-2.5-flash": flash, "gemini-2.5-pro": pro}
    if model_lower in builtin:
        return builtin[model_lower]

    if "flash" in model_lower:
        family, fallback = "flash", flash
    else:
        # "pro", or a name matching neither: use the higher rate so an unknown
        # model overstates cost rather than hiding it.
        family, fallback = "pro", pro

    if model_lower not in _warned_unpriced_models:
        _warned_unpriced_models.add(model_lower)
        logger.warning(
            f"No price entry for model {model!r}; using the {family} family rate. "
            "Costs for this model are estimates until it is priced in GEMINI_MODEL_PRICES_JSON."
        )
    return fallback


def calculate_cost(
    model: str,
    token_usage: TokenUsage,
    phase: str = "unknown",
    grounding_queries: int = 0,
) -> CostResult:
    """
    Calculate cost for an API call.

    Thinking tokens are billed by Google at the output rate but reported apart
    from `candidates_token_count`, so they are added to the output side here.

    Args:
        model: Model name (e.g., "gemini-2.5-flash")
        token_usage: Token usage from the response
        phase: Agent phase name for tracking
        grounding_queries: Google Search queries the model executed

    Returns:
        CostResult with calculated costs
    """
    pricing = get_pricing_for_model(model)

    billable_output = token_usage.output_tokens + token_usage.thinking_tokens
    input_cost = (token_usage.input_tokens / 1_000_000) * pricing["input_per_million"]
    output_cost = (billable_output / 1_000_000) * pricing["output_per_million"]
    grounding_cost = (
        grounding_queries / 1000
    ) * get_cost_config().gemini_grounding_price_per_thousand_queries
    total_cost = input_cost + output_cost + grounding_cost

    return CostResult(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total_cost,
        model=model,
        phase=phase,
        token_usage=token_usage,
        grounding_queries=grounding_queries,
        grounding_cost=grounding_cost,
    )


def extract_model_version(response: Any) -> Optional[str]:
    """Name of the model that actually served `response`, or None.

    Google can serve a different model than the one requested (measured
    2026-10-22: gemini-3.7-flash is served by gemini-3.8-flash), and says so
    only in the response's `model_version`. Anything that is not a non-empty
    string (absent, None, a test double's auto-created attribute) is "unknown".
    """
    value = getattr(response, "model_version", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def count_grounding_queries(response: Any) -> int:
    """Count the search queries a grounded response executed.

    Strict about shape (this runs on the response path): anything that is not a
    list of queries counts as zero rather than raising.
    """
    candidates = getattr(response, "candidates", None)
    if not isinstance(candidates, (list, tuple)) or not candidates:
        return 0
    metadata = getattr(candidates[0], "grounding_metadata", None)
    queries = getattr(metadata, "web_search_queries", None)
    return len(queries) if isinstance(queries, (list, tuple)) else 0


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
    return calculate_cost(
        model, token_usage, phase, grounding_queries=count_grounding_queries(response)
    )
