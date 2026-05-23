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
