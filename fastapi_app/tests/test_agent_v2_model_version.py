"""The model that SERVED a call is logged next to the model we REQUESTED.

Google routes some requested names to a different model: measured 2026-10-22,
`gemini-3.7-flash` is served by `gemini-3.8-flash` and `gemini-3.5-flash` by
`gemini-3.6-flash`. The response says so in `model_version`. Until now the
interactions table only held the name we asked for, so a silent re-route (and
any latency or verbosity change it brings) was invisible in our own data.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent_v2 import AgentV2
from app.models import PhaseCost
from app.utils.cost_tracking import extract_model_version


@pytest.fixture
def mock_genai_client():
    with patch("app.agent_v2.get_genai_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock()
        mock_get_client.return_value = mock_client
        yield mock_client


def _response(text, model_version="gemini-3.8-flash"):
    resp = MagicMock()
    resp.text = text
    resp.candidates = []
    resp.usage_metadata = None
    resp.model_version = model_version
    return resp


def test_extract_model_version_reads_the_serving_model():
    assert extract_model_version(_response("x", "gemini-3.8-flash")) == "gemini-3.8-flash"


@pytest.mark.parametrize("value", [None, "", "   ", 42])
def test_extract_model_version_is_none_when_absent_or_not_a_name(value):
    assert extract_model_version(_response("x", value)) is None


def test_extract_model_version_ignores_responses_without_the_field():
    class Bare:
        pass

    assert extract_model_version(Bare()) is None


def test_extract_model_version_never_raises_on_a_mock_attribute():
    """An auto-created MagicMock attribute is not a model name."""
    assert extract_model_version(MagicMock()) is None


def test_phase_cost_model_version_defaults_to_none_and_is_dumped():
    assert PhaseCost(phase="routing", model="m").model_version is None
    dumped = PhaseCost(phase="routing", model="m", model_version="v").model_dump()
    assert dumped["model_version"] == "v"


@pytest.mark.asyncio
async def test_each_phase_records_the_model_that_served_it(mock_genai_client):
    mock_genai_client.aio.models.generate_content.side_effect = [
        _response("ALLOW", "gemini-3.8-flash"),
        _response("Answer.", "gemini-3.6-flash"),
    ]

    result = await AgentV2().run("What is the JSE?", financial_manager=None)

    phases = {p.phase: p for p in result["cost_summary"].phases}
    # Requested name is untouched; the served name rides alongside it.
    assert phases["routing"].model_version == "gemini-3.8-flash"
    assert phases["generation"].model_version == "gemini-3.6-flash"
    assert phases["routing"].model == phases["generation"].model


def test_chat_stream_logs_model_version_in_phase_costs_json(mock_genai_client):
    from fastapi.testclient import TestClient

    from app.main import app, get_interaction_logger_dep

    mock_genai_client.aio.models.generate_content.side_effect = [
        _response("ALLOW", "gemini-3.8-flash"),
        _response("Answer.", "gemini-3.8-flash"),
    ]
    fake_logger = MagicMock()
    fake_logger.log = AsyncMock()
    app.dependency_overrides[get_interaction_logger_dep] = lambda: fake_logger
    try:
        resp = TestClient(app).post("/chat/stream", json={"query": "What is the JSE?"})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    row = fake_logger.log.call_args.args[0]
    phases = {p["phase"]: p for p in json.loads(row["phase_costs_json"])}
    assert phases["routing"]["model_version"] == "gemini-3.8-flash"
    assert phases["generation"]["model_version"] == "gemini-3.8-flash"
