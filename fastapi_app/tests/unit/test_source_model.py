"""Unit tests for the Source provenance model.

Every chat endpoint returns a `sources` array so users can fact-check an
answer. Only `type` and `title` are required, so a client that understands
nothing else can still render a meaningful citation line.
"""

import pytest
from pydantic import ValidationError

from app.models import (
    AgentChatResponse,
    ChatResponse,
    FinancialDataFilters,
    FinancialDataResponse,
    Source,
    SourceType,
)


@pytest.mark.unit
class TestSourceModel:
    def test_web_source_keeps_legacy_shape(self):
        """Existing clients read type/title/url. Those must serialize verbatim."""
        source = Source(
            type=SourceType.WEB,
            title="Jamaica Stock Exchange",
            url="https://vertexaisearch.example/redirect/abc",
        )
        dumped = source.model_dump()
        assert dumped["type"] == "web"
        assert dumped["title"] == "Jamaica Stock Exchange"
        assert dumped["url"] == "https://vertexaisearch.example/redirect/abc"

    def test_type_and_title_are_sufficient(self):
        source = Source(type=SourceType.DATASET, title="JSE financial statements")
        assert source.url is None
        assert source.document_id is None
        assert source.record_count is None

    def test_title_is_required(self):
        with pytest.raises(ValidationError):
            Source(type=SourceType.WEB)

    def test_agent_response_coerces_source_dicts(self):
        """Redis returns plain dicts; the response model must coerce them."""
        built = AgentChatResponse(
            response="r",
            data_found=True,
            record_count=0,
            sources=[{"type": "web", "title": "T", "url": "https://x"}],
        )
        assert built.sources[0].type == SourceType.WEB
        assert built.sources[0].title == "T"

    def test_financial_response_accepts_sources(self):
        built = FinancialDataResponse(
            response="r",
            data_found=True,
            record_count=2,
            filters_used=FinancialDataFilters(),
            sources=[{"type": "dataset", "title": "JSE financials", "record_count": 2}],
        )
        assert built.sources[0].type == SourceType.DATASET
        assert built.sources[0].record_count == 2

    def test_chat_response_accepts_sources_alongside_documents_loaded(self):
        """documents_loaded is retained for backward compatibility."""
        built = ChatResponse(
            response="r",
            documents_loaded=["ncb_annual_2023.pdf"],
            sources=[{"type": "document", "title": "ncb_annual_2023.pdf", "document_id": "abc123"}],
        )
        assert built.documents_loaded == ["ncb_annual_2023.pdf"]
        assert built.sources[0].document_id == "abc123"
