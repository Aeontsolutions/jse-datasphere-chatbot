"""Unit tests for document provenance returned by the /chat loader.

The loader has always had document_link and the selection reason in scope
and thrown them away, returning bare filenames. Users fact-checking an
answer need to reach the actual PDF, so the loader now returns Source
objects carrying an opaque document_id the resolver endpoint can redeem.
"""

from unittest.mock import Mock, patch

import pytest

from app.document_registry import make_document_id
from app.document_selector import DocumentLoadResult, auto_load_relevant_documents
from app.models import SourceType

S3_PATH = "s3://jse-renamed-docs-copy-2/organized/ncb/annual_2023.pdf"

RECOMMENDATION = {
    "companies_mentioned": ["NCB Financial Group"],
    "documents_to_load": [
        {
            "company": "NCB Financial Group",
            "document_link": S3_PATH,
            "filename": "ncb_annual_report_2023.pdf",
            "reason": "Contains FY2023 net profit",
        }
    ],
}


@pytest.fixture
def loaded():
    with (
        patch("app.document_selector.semantic_document_selection") as mock_select,
        patch("app.document_selector.download_and_extract_from_s3") as mock_download,
    ):
        mock_select.return_value = RECOMMENDATION
        mock_download.return_value = "extracted pdf text"
        yield auto_load_relevant_documents(
            s3_client=Mock(),
            query="NCB net profit 2023",
            metadata={"NCB Financial Group": []},
            current_document_texts={},
        )


@pytest.mark.unit
class TestDocumentSources:
    def test_returns_a_document_load_result(self, loaded):
        assert isinstance(loaded, DocumentLoadResult)

    def test_loaded_docs_still_lists_filenames(self, loaded):
        """Backward compatibility: ChatResponse.documents_loaded is unchanged."""
        assert loaded.loaded_docs == ["ncb_annual_report_2023.pdf"]

    def test_texts_are_keyed_by_filename(self, loaded):
        assert loaded.texts["ncb_annual_report_2023.pdf"] == "extracted pdf text"

    def test_source_is_typed_document(self, loaded):
        assert loaded.sources[0].type == SourceType.DOCUMENT

    def test_source_carries_resolvable_document_id(self, loaded):
        assert loaded.sources[0].document_id == make_document_id(S3_PATH)

    def test_source_never_exposes_the_s3_path(self, loaded):
        """The bucket layout must not reach the browser."""
        dumped = loaded.sources[0].model_dump()
        assert S3_PATH not in str(dumped)
        assert "jse-renamed-docs-copy-2" not in str(dumped)

    def test_source_carries_selection_reason_and_company(self, loaded):
        source = loaded.sources[0]
        assert source.title == "ncb_annual_report_2023.pdf"
        assert source.detail == "Contains FY2023 net profit"
        assert source.company == "NCB Financial Group"
        assert source.retrieved_at is not None

    def test_failed_download_produces_no_source(self):
        """A document we could not read must not be cited as evidence."""
        with (
            patch("app.document_selector.semantic_document_selection") as mock_select,
            patch("app.document_selector.download_and_extract_from_s3") as mock_download,
        ):
            mock_select.return_value = RECOMMENDATION
            mock_download.return_value = None
            result = auto_load_relevant_documents(
                s3_client=Mock(),
                query="NCB net profit 2023",
                metadata={"NCB Financial Group": []},
                current_document_texts={},
            )
        assert result.loaded_docs == []
        assert result.sources == []
