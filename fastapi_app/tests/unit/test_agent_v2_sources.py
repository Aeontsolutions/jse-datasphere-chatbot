"""Unit tests for web source extraction from Gemini grounding metadata.

Gemini returns one grounding chunk per grounding support, so the same URL
appears several times in a single response — the extracted source list must
be deduped. Grounding URIs are vertexaisearch redirects that expire after
about 30 days, so `domain` is captured separately: when the link dies, the
citation degrades to readable text instead of to nothing.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.agent_v2 import AgentV2
from app.models import SourceType


def _web_chunk(uri, title, domain=None):
    """Build a grounding chunk.

    `spec` is required: a bare MagicMock auto-creates any attribute, so
    `getattr(web, "domain", None)` would return a Mock instead of None and the
    fallback path would never be exercised.
    """
    spec = ["uri", "title"] + (["domain"] if domain is not None else [])
    web = MagicMock(spec=spec)
    web.uri = uri
    web.title = title
    if domain is not None:
        web.domain = domain
    chunk = MagicMock(spec=["web"])
    chunk.web = web
    return chunk


def _grounded_response(chunks):
    grounding = MagicMock(spec=["grounding_chunks", "web_search_queries", "search_entry_point"])
    grounding.grounding_chunks = chunks
    grounding.web_search_queries = ["ncb net profit 2023"]
    entry_point = MagicMock(spec=["rendered_content"])
    entry_point.rendered_content = "<div></div>"
    grounding.search_entry_point = entry_point
    candidate = MagicMock(spec=["grounding_metadata"])
    candidate.grounding_metadata = grounding
    response = MagicMock(spec=["candidates"])
    response.candidates = [candidate]
    return response


@pytest.fixture
def agent():
    with patch("app.agent_v2.get_genai_client"):
        yield AgentV2()


@pytest.mark.unit
class TestWebSourceExtraction:
    def test_duplicate_urls_are_deduped(self, agent):
        response = _grounded_response(
            [
                _web_chunk("https://jse.com/a", "jamstockex.com"),
                _web_chunk("https://jse.com/a", "jamstockex.com"),
                _web_chunk("https://jse.com/b", "jamstockex.com"),
            ]
        )
        sources = agent._extract_grounding_metadata(response)["sources"]
        assert len(sources) == 2
        assert [s.url for s in sources] == ["https://jse.com/a", "https://jse.com/b"]

    def test_sources_are_typed_web(self, agent):
        response = _grounded_response([_web_chunk("https://jse.com/a", "jamstockex.com")])
        source = agent._extract_grounding_metadata(response)["sources"][0]
        assert source.type == SourceType.WEB
        assert source.title == "jamstockex.com"

    def test_domain_field_is_preferred_when_present(self, agent):
        response = _grounded_response(
            [_web_chunk("https://x/a", "JSE Annual Review", domain="jamstockex.com")]
        )
        source = agent._extract_grounding_metadata(response)["sources"][0]
        assert source.domain == "jamstockex.com"
        assert source.title == "JSE Annual Review"

    def test_domain_falls_back_to_title(self, agent):
        response = _grounded_response([_web_chunk("https://x/a", "jamstockex.com")])
        source = agent._extract_grounding_metadata(response)["sources"][0]
        assert source.domain == "jamstockex.com"

    def test_retrieved_at_is_populated(self, agent):
        response = _grounded_response([_web_chunk("https://x/a", "jamstockex.com")])
        source = agent._extract_grounding_metadata(response)["sources"][0]
        assert source.retrieved_at is not None
        assert "T" in source.retrieved_at  # ISO 8601

    def test_chunks_without_uri_are_skipped(self, agent):
        response = _grounded_response([_web_chunk("", "jamstockex.com")])
        assert agent._extract_grounding_metadata(response)["sources"] == []

    def test_no_candidates_yields_no_sources(self, agent):
        response = MagicMock(spec=["candidates"])
        response.candidates = []
        assert agent._extract_grounding_metadata(response)["sources"] == []

    def test_search_results_still_populated(self, agent):
        """web_search_results is a separate field and must not change shape."""
        response = _grounded_response([_web_chunk("https://x/a", "jamstockex.com")])
        results = agent._extract_grounding_metadata(response)["search_results"]
        assert results["queries"] == ["ncb net profit 2023"]
        assert results["grounding_chunks"] == [{"title": "jamstockex.com", "uri": "https://x/a"}]
