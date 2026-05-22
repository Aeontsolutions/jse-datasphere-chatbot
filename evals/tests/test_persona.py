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
