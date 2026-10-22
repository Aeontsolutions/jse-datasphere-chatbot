"""Per-phase wall-clock timing on AgentV2's cost summary.

The interactions table logs one `latency_ms` per request, so a 58s request and
a 9s request look identical beyond the number: nothing says whether the router,
the ticker-extraction call or the grounded generation ate the time. Each
`PhaseCost` now carries the wall-clock `latency_ms` of its phase, and
`phase_costs_json` (which dumps PhaseCost) records it with no schema change.

These tests drive `run()` with calls that take a known, real amount of time
and assert the timing lands on the right phase -- not that some number exists.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent_v2 import AgentV2
from app.models import PhaseCost

ROUTER_DELAY_S = 0.05
GENERATION_DELAY_S = 0.15
EXTRACTION_DELAY_S = 0.10


@pytest.fixture
def mock_genai_client():
    with patch("app.agent_v2.get_genai_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock()
        mock_get_client.return_value = mock_client
        yield mock_client


def _response(text):
    resp = MagicMock()
    resp.text = text
    resp.candidates = []
    resp.usage_metadata = None
    return resp


def _slow(delay_s, result):
    async def call(*args, **kwargs):
        await asyncio.sleep(delay_s)
        return result

    return call


def _script(client, *steps):
    """Make successive generate_content calls take (delay_s, text) each."""
    calls = iter([_slow(delay, _response(text)) for delay, text in steps])

    async def next_call(*args, **kwargs):
        return await next(calls)(*args, **kwargs)

    client.aio.models.generate_content.side_effect = next_call


def _financial_manager():
    return MagicMock(
        metadata={
            "companies": ["Medical Disposables & Supplies Limited"],
            "symbols": ["MDS"],
            "associations": {
                "symbol_to_company": {"MDS": ["Medical Disposables & Supplies Limited"]}
            },
        }
    )


def _phases(result):
    return {p.phase: p for p in result["cost_summary"].phases}


def test_phase_cost_latency_defaults_to_zero():
    """A phase recorded without a timing (older callers) must not crash or lie."""
    assert PhaseCost(phase="routing", model="m").latency_ms == 0.0


def test_phase_cost_dump_includes_latency_for_the_interactions_log():
    """/chat/stream logs `[p.model_dump() for p in phases]`; the field must ride along."""
    assert PhaseCost(phase="routing", model="m", latency_ms=12.5).model_dump()["latency_ms"] == 12.5


@pytest.mark.asyncio
async def test_routing_and_generation_are_timed_separately(mock_genai_client):
    _script(mock_genai_client, (ROUTER_DELAY_S, "ALLOW"), (GENERATION_DELAY_S, "Answer."))

    result = await AgentV2().run("What is the JSE?", financial_manager=None)

    phases = _phases(result)
    routing_ms = phases["routing"].latency_ms
    generation_ms = phases["generation"].latency_ms
    assert routing_ms >= ROUTER_DELAY_S * 1000 * 0.9
    assert generation_ms >= GENERATION_DELAY_S * 1000 * 0.9
    # Not one shared number: the slower call must read as slower.
    assert generation_ms > routing_ms + 50


@pytest.mark.asyncio
async def test_refusal_path_times_routing_and_refusal(mock_genai_client):
    _script(
        mock_genai_client,
        (ROUTER_DELAY_S, "REFUSE"),
        (GENERATION_DELAY_S, "I can only help with the JSE."),
    )

    result = await AgentV2().run("Best stocks on the NYSE?", financial_manager=None)

    phases = _phases(result)
    assert phases["routing"].latency_ms >= ROUTER_DELAY_S * 1000 * 0.9
    assert phases["refusal"].latency_ms >= GENERATION_DELAY_S * 1000 * 0.9


@pytest.mark.asyncio
async def test_symbol_extraction_is_recorded_as_its_own_timed_phase(mock_genai_client, monkeypatch):
    monkeypatch.delenv("DISABLE_REGISTRY_GROUNDING", raising=False)
    _script(mock_genai_client, (ROUTER_DELAY_S, "ALLOW"), (GENERATION_DELAY_S, "Answer."))

    with patch(
        "app.agent_v2.extract_companies_from_query_async",
        side_effect=_slow(EXTRACTION_DELAY_S, {"symbols": ["MDS"]}),
    ):
        result = await AgentV2().run("MDS revenue?", financial_manager=_financial_manager())

    phases = _phases(result)
    assert "symbol_extraction" in phases
    # Its own duration -- not the router's, and not the whole request's.
    assert phases["symbol_extraction"].latency_ms >= EXTRACTION_DELAY_S * 1000 * 0.9
    assert phases["symbol_extraction"].latency_ms < (EXTRACTION_DELAY_S + 0.5) * 1000
    # Zero cost: this call's usage is not tracked today; the record is for timing only.
    assert phases["symbol_extraction"].total_cost_usd == 0.0


@pytest.mark.asyncio
async def test_no_symbol_extraction_phase_when_registry_grounding_disabled(
    mock_genai_client, monkeypatch
):
    monkeypatch.setenv("DISABLE_REGISTRY_GROUNDING", "true")
    mock_genai_client.aio.models.generate_content.side_effect = [
        _response("ALLOW"),
        _response("Answer."),
    ]

    result = await AgentV2().run("MDS revenue?", financial_manager=_financial_manager())

    assert "symbol_extraction" not in _phases(result)


@pytest.mark.asyncio
async def test_failed_generation_does_not_leave_a_fabricated_timing(mock_genai_client):
    """A call that raised never produced a phase record; don't invent one."""
    mock_genai_client.aio.models.generate_content.side_effect = [
        _response("ALLOW"),
        RuntimeError("upstream 503"),
    ]

    result = await AgentV2().run("What is the JSE?", financial_manager=None)

    phases = _phases(result)
    assert "routing" in phases
    assert "generation" not in phases


def test_chat_stream_logs_per_phase_latency_in_phase_costs_json(mock_genai_client):
    """End to end: the real AgentV2 -> /chat/stream -> the row written to BigQuery."""
    import json

    from fastapi.testclient import TestClient

    from app.main import app, get_interaction_logger_dep

    _script(mock_genai_client, (ROUTER_DELAY_S, "ALLOW"), (GENERATION_DELAY_S, "Answer."))
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
    assert phases["routing"]["latency_ms"] >= ROUTER_DELAY_S * 1000 * 0.9
    assert phases["generation"]["latency_ms"] >= GENERATION_DELAY_S * 1000 * 0.9
    # The per-phase timings are a breakdown of the request, so they cannot exceed it.
    assert sum(p["latency_ms"] for p in phases.values()) <= row["latency_ms"] + 5
