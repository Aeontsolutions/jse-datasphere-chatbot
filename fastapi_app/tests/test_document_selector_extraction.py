"""Tests for company/symbol extraction's thinking-budget and finish_reason handling.

Background: prod's error-rate alarm fired twice on 2026-08-28 from
`company_extraction_failed` JSONDecodeErrors in extract_companies_from_query[_async]
(document_selector.py) — the same failure signature as issue #72 (see
test_agent_v2_truncation.py): a gemini-2.5-flash call with a tight output cap and
thinking left at the model default, so thinking tokens ate the budget and left
truncated/malformed JSON. Nothing read finish_reason, so the cause was invisible
beyond "JSONDecodeError".

These tests pin the same two fixes at this second call site: thinking disabled
(the call is one-shot entity extraction, no deliberation needed) and finish_reason
captured on parse failure so a recurrence is diagnosable from the log line alone.
"""

from unittest.mock import MagicMock, patch

import pytest
from structlog.testing import capture_logs

from app.document_selector import (
    _build_extraction_request,
    extract_companies_from_query,
    extract_companies_from_query_async,
)


def _response(text, finish_reason=None):
    resp = MagicMock()
    resp.text = text
    if finish_reason is None:
        resp.candidates = []
    else:
        candidate = MagicMock()
        candidate.finish_reason = finish_reason
        resp.candidates = [candidate]
    return resp


@pytest.fixture
def mock_genai_client():
    with patch("app.document_selector.get_genai_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        yield mock_client


# ---------------------------------------------------------------------------
# Thinking budget
# ---------------------------------------------------------------------------


def test_extraction_request_disables_thinking():
    """A one-shot company/symbol extraction needs no deliberation; thinking must
    not compete with the 512-token output cap for budget."""
    _model_name, _contents, config = _build_extraction_request(
        "GraceKennedy revenue", ["GraceKennedy"]
    )
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_budget == 0


# ---------------------------------------------------------------------------
# finish_reason surfaced on parse failure
# ---------------------------------------------------------------------------


def test_sync_parse_failure_logs_finish_reason(mock_genai_client):
    mock_genai_client.models.generate_content.return_value = _response(
        '{"companies": ["GraceKenn', finish_reason="MAX_TOKENS"
    )
    with capture_logs() as logs:
        result = extract_companies_from_query("GraceKennedy revenue 2024", ["GraceKennedy"])

    assert result == {"companies": [], "symbols": []}
    entry = next(e for e in logs if e["event"] == "company_extraction_failed")
    assert entry["extra"]["finish_reason"] == "MAX_TOKENS"


async def test_async_parse_failure_logs_finish_reason(mock_genai_client):
    async def _fake_generate(*args, **kwargs):
        return _response('{"companies": ["GraceKenn', finish_reason="MAX_TOKENS")

    mock_genai_client.aio.models.generate_content = _fake_generate

    with capture_logs() as logs:
        result = await extract_companies_from_query_async(
            "GraceKennedy revenue 2024", ["GraceKennedy"]
        )

    assert result == {"companies": [], "symbols": []}
    entry = next(e for e in logs if e["event"] == "company_extraction_failed")
    assert entry["extra"]["finish_reason"] == "MAX_TOKENS"


def test_parse_failure_with_no_candidates_logs_none_finish_reason(mock_genai_client):
    """A response with no candidates at all must not crash the failure-logging path."""
    mock_genai_client.models.generate_content.return_value = _response("not json")
    with capture_logs() as logs:
        extract_companies_from_query("some query", ["GraceKennedy"])

    entry = next(e for e in logs if e["event"] == "company_extraction_failed")
    assert entry["extra"]["finish_reason"] is None


def test_api_call_failure_still_returns_empty_result(mock_genai_client):
    """When the API call itself throws (no response object), extraction must still
    degrade gracefully rather than propagate — there's no finish_reason to read here."""
    mock_genai_client.models.generate_content.side_effect = RuntimeError("network boom")
    with capture_logs() as logs:
        result = extract_companies_from_query("some query", ["GraceKennedy"])

    assert result == {"companies": [], "symbols": []}
    entry = next(e for e in logs if e["event"] == "company_extraction_failed")
    assert entry["extra"]["finish_reason"] is None
