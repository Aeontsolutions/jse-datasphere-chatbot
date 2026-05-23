"""Tests for runtime config loading and override merging."""

from pathlib import Path

import pytest

from evals.config import EvalConfig, load_config


def test_load_default():
    config = load_config()
    assert config.base_url == "http://localhost:8000"
    assert config.replicates == 3
    assert config.concurrency == 2
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
