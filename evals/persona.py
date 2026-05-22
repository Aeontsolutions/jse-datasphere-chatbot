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
