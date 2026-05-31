"""Tests for the AgentV2 financial-data path (ATS-334)."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.agent_v2 import FINANCIAL_DECISION_MODEL, AgentV2


@pytest.fixture
def mock_genai_client():
    with patch("app.agent_v2.get_genai_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        yield mock_client


def test_constructor_stores_financial_manager(mock_genai_client):
    mgr = MagicMock()
    agent = AgentV2(financial_manager=mgr)
    assert agent.financial_manager is mgr


def test_constructor_defaults_financial_manager_none(mock_genai_client):
    agent = AgentV2()
    assert agent.financial_manager is None


def test_metadata_context_includes_symbols_and_years(mock_genai_client):
    mgr = MagicMock()
    mgr.metadata = {"symbols": ["NCB", "GK"], "years": ["2022", "2023"]}
    agent = AgentV2(financial_manager=mgr)
    ctx = agent._get_metadata_context()
    assert "NCB" in ctx and "2023" in ctx


def test_metadata_context_empty_when_no_manager(mock_genai_client):
    agent = AgentV2()
    assert agent._get_metadata_context() == ""


def test_decision_model_is_flash():
    assert FINANCIAL_DECISION_MODEL == "gemini-2.5-flash"