"""Tests for GET /documents/{document_id}.

The resolver is the only path from a citation to the underlying PDF. It must
resolve exactly the documents present in metadata.json and nothing else — an
unknown or tampered id gets a 404, never a presigned URL.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.document_registry import build_document_index, make_document_id
from app.main import app

S3_PATH = "s3://jse-renamed-docs-copy-2/organized/ncb/annual_2023.pdf"
METADATA = {
    "NCB Financial Group": [
        {
            "filename": "ncb_annual_report_2023.pdf",
            "document_link": S3_PATH,
            "year": "2023",
        }
    ]
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app.state, "metadata", METADATA, raising=False)
    monkeypatch.setattr(app.state, "document_index", build_document_index(METADATA), raising=False)
    mock_s3 = MagicMock()
    mock_s3.generate_presigned_url.return_value = "https://s3.example/presigned?sig=abc"
    monkeypatch.setattr(app.state, "s3_client", mock_s3, raising=False)
    yield TestClient(app)


@pytest.mark.integration
class TestDocumentResolver:
    def test_known_id_redirects_to_presigned_url(self, client):
        doc_id = make_document_id(S3_PATH)
        response = client.get(f"/documents/{doc_id}", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "https://s3.example/presigned?sig=abc"

    def test_presigned_url_is_short_lived(self, client):
        doc_id = make_document_id(S3_PATH)
        client.get(f"/documents/{doc_id}", follow_redirects=False)
        _, kwargs = app.state.s3_client.generate_presigned_url.call_args
        assert kwargs["ExpiresIn"] == 300
        assert kwargs["Params"]["Bucket"] == "jse-renamed-docs-copy-2"
        assert kwargs["Params"]["Key"] == "organized/ncb/annual_2023.pdf"

    def test_unknown_id_returns_404(self, client):
        response = client.get("/documents/0000000000000000", follow_redirects=False)
        assert response.status_code == 404
        app.state.s3_client.generate_presigned_url.assert_not_called()

    def test_tampered_id_returns_404(self, client):
        """Flipping one character must not resolve to anything."""
        doc_id = make_document_id(S3_PATH)
        tampered = ("0" if doc_id[0] != "0" else "1") + doc_id[1:]
        response = client.get(f"/documents/{tampered}", follow_redirects=False)
        assert response.status_code == 404
        app.state.s3_client.generate_presigned_url.assert_not_called()

    def test_path_traversal_attempt_does_not_resolve(self, client):
        response = client.get("/documents/..%2F..%2Fetc%2Fpasswd", follow_redirects=False)
        assert response.status_code == 404
        app.state.s3_client.generate_presigned_url.assert_not_called()
