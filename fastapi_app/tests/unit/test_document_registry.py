"""Unit tests for the document registry.

`document_id` is deliberately a one-way hash rather than an encoded S3 path.
If it were reversible (e.g. base64), a client could edit the id and read
arbitrary objects out of the bucket through our own resolver endpoint.
Because resolution is a dict lookup built from metadata.json, no crafted
input can reach an object we did not index.
"""

import pytest

from app.document_registry import (
    build_document_index,
    make_document_id,
    parse_s3_path,
)

METADATA = {
    "NCB Financial Group": [
        {
            "filename": "ncb_annual_report_2023.pdf",
            "document_link": "s3://jse-renamed-docs-copy-2/organized/ncb/annual_2023.pdf",
            "year": "2023",
        },
        {
            "filename": "ncb_q4_2023.pdf",
            "document_link": "s3://jse-renamed-docs-copy-2/organized/ncb/q4_2023.pdf",
            "period": "2023",
        },
    ],
    "GraceKennedy": [
        {
            "filename": "gk_annual_report_2024.pdf",
            "document_link": "s3://jse-renamed-docs-copy-2/organized/gk/annual_2024.pdf",
            "year": "2024",
        }
    ],
}


@pytest.mark.unit
class TestMakeDocumentId:
    def test_is_stable(self):
        path = "s3://bucket/key.pdf"
        assert make_document_id(path) == make_document_id(path)

    def test_is_sixteen_hex_chars(self):
        doc_id = make_document_id("s3://bucket/key.pdf")
        assert len(doc_id) == 16
        assert all(c in "0123456789abcdef" for c in doc_id)

    def test_differs_per_path(self):
        assert make_document_id("s3://bucket/a.pdf") != make_document_id("s3://bucket/b.pdf")

    def test_leaks_no_path_material(self):
        """The id must not reveal bucket or key — it is a lookup key, not a codec."""
        doc_id = make_document_id("s3://secret-bucket/organized/private.pdf")
        assert "secret-bucket" not in doc_id
        assert "private" not in doc_id
        assert "organized" not in doc_id


@pytest.mark.unit
class TestBuildDocumentIndex:
    def test_indexes_every_document(self):
        index = build_document_index(METADATA)
        assert len(index) == 3

    def test_entry_carries_resolution_fields(self):
        index = build_document_index(METADATA)
        doc_id = make_document_id("s3://jse-renamed-docs-copy-2/organized/gk/annual_2024.pdf")
        entry = index[doc_id]
        assert entry["s3_path"] == "s3://jse-renamed-docs-copy-2/organized/gk/annual_2024.pdf"
        assert entry["filename"] == "gk_annual_report_2024.pdf"
        assert entry["company"] == "GraceKennedy"
        assert entry["year"] == "2024"

    def test_period_is_used_when_year_is_absent(self):
        index = build_document_index(METADATA)
        doc_id = make_document_id("s3://jse-renamed-docs-copy-2/organized/ncb/q4_2023.pdf")
        assert index[doc_id]["year"] == "2023"

    def test_documents_without_a_link_are_skipped(self):
        index = build_document_index({"ACME": [{"filename": "no_link.pdf"}]})
        assert index == {}

    def test_none_metadata_yields_empty_index(self):
        assert build_document_index(None) == {}

    def test_malformed_metadata_is_tolerated(self):
        """Metadata comes from S3 — a bad shape must not crash startup."""
        assert build_document_index({"ACME": "not-a-list"}) == {}
        assert build_document_index({"ACME": ["not-a-dict"]}) == {}


@pytest.mark.unit
class TestParseS3Path:
    def test_splits_bucket_and_key(self):
        assert parse_s3_path("s3://my-bucket/a/b/c.pdf") == ("my-bucket", "a/b/c.pdf")

    def test_rejects_non_s3_scheme(self):
        with pytest.raises(ValueError):
            parse_s3_path("https://my-bucket/a.pdf")

    def test_rejects_bucket_without_key(self):
        with pytest.raises(ValueError):
            parse_s3_path("s3://my-bucket")
