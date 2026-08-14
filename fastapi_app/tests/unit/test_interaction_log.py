"""Unit tests for InteractionLogger."""

from unittest.mock import MagicMock, patch

import pytest

from app.interaction_log import InteractionLogger, build_interaction_logger


def test_disabled_when_no_client():
    logger_ = InteractionLogger(bq_client=None, dataset="ds", table="interactions", enabled=True)
    assert logger_.enabled is False


def test_disabled_when_flag_off():
    client = MagicMock()
    logger_ = InteractionLogger(bq_client=client, dataset="ds", table="interactions", enabled=False)
    assert logger_.enabled is False


def test_disabled_when_dataset_missing():
    client = MagicMock()
    logger_ = InteractionLogger(bq_client=client, dataset=None, table="interactions", enabled=True)
    assert logger_.enabled is False


def test_build_row_shape():
    row = InteractionLogger.build_row(
        endpoint="fast_chat_v2",
        query="revenue for nyc holdings",
        response="Here is the revenue.",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        cache_hit=False,
        success=True,
        timestamp="2026-08-14T00:00:00Z",
    )
    assert row["endpoint"] == "fast_chat_v2"
    assert row["total_tokens"] == 150
    assert row["cache_hit"] is False
    assert row["success"] is True
    assert "interaction_id" in row and row["interaction_id"]


@pytest.mark.asyncio
async def test_log_noop_when_disabled():
    logger_ = InteractionLogger(bq_client=None, dataset="ds", table="interactions", enabled=True)
    await logger_.log({"query": "x"})  # must not raise, must not touch a client


@pytest.mark.asyncio
async def test_log_calls_insert_rows_json():
    client = MagicMock()
    client.insert_rows_json.return_value = []
    logger_ = InteractionLogger(bq_client=client, dataset="ds", table="interactions", enabled=True)
    await logger_.log({"query": "x"})
    client.insert_rows_json.assert_called_once()
    args, _ = client.insert_rows_json.call_args
    assert args[0] == "ds.interactions"
    assert args[1] == [{"query": "x", "environment": None}]


@pytest.mark.asyncio
async def test_log_stamps_configured_environment_onto_every_row():
    client = MagicMock()
    client.insert_rows_json.return_value = []
    logger_ = InteractionLogger(
        bq_client=client, dataset="ds", table="interactions", enabled=True, environment="dev"
    )
    await logger_.log({"query": "x"})
    args, _ = client.insert_rows_json.call_args
    assert args[1] == [{"query": "x", "environment": "dev"}]


@pytest.mark.asyncio
async def test_log_swallows_insert_errors():
    client = MagicMock()
    client.insert_rows_json.side_effect = Exception("insert failed")
    logger_ = InteractionLogger(bq_client=client, dataset="ds", table="interactions", enabled=True)
    await logger_.log({"query": "x"})  # must not raise


@pytest.mark.asyncio
async def test_log_reports_row_level_errors_without_raising():
    client = MagicMock()
    client.insert_rows_json.return_value = [{"index": 0, "errors": ["bad row"]}]
    logger_ = InteractionLogger(bq_client=client, dataset="ds", table="interactions", enabled=True)
    await logger_.log({"query": "x"})  # must not raise


def test_build_interaction_logger_disabled_when_logging_disabled():
    mock_config = MagicMock()
    mock_config.bigquery.logging_enabled = False
    mock_config.bigquery.resolved_interactions_dataset = "ds"
    mock_config.bigquery.interactions_table = "interactions"
    mock_config.gcp.project_id = "proj"
    mock_config.environment = None
    with patch("app.config.get_config", return_value=mock_config):
        logger_ = build_interaction_logger()
    assert logger_.enabled is False


def test_build_interaction_logger_passes_environment_through():
    mock_config = MagicMock()
    mock_config.bigquery.logging_enabled = False
    mock_config.bigquery.resolved_interactions_dataset = "ds"
    mock_config.bigquery.interactions_table = "interactions"
    mock_config.gcp.project_id = "proj"
    mock_config.environment = "staging"
    with patch("app.config.get_config", return_value=mock_config):
        logger_ = build_interaction_logger()
    assert logger_._environment == "staging"
