"""
Integration tests for the /chroma/query endpoint to verify it handles
single-clause filters correctly and returns 200 status codes.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import os
import sys

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from app.main import app


class TestChromaQueryEndpoint:
    """Integration tests for the /chroma/query endpoint."""

    def setup_method(self):
        """Set up test client and mocks."""
        self.client = TestClient(app)

        # Mock the ChromaDB collection
        self.mock_collection = Mock()
        self.mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Document 1 content", "Document 2 content"]],
            "metadatas": [
                [
                    {
                        "id": "doc1",
                        "year": "2023",
                        "company_name": "Test Co",
                        "file_type": "financial",
                    },
                    {
                        "id": "doc2",
                        "year": "2022",
                        "company_name": "Test Co",
                        "file_type": "financial",
                    },
                ]
            ],
        }

    @patch("app.main.get_chroma_collection")
    @patch("app.chroma_utils.get_companies_from_query")
    @patch("app.chroma_utils.get_doctype_from_query")
    def test_single_company_filter_returns_200(
        self, mock_get_doctype, mock_get_companies, mock_get_collection
    ):
        """Test that single company filter (previously failing) returns 200."""
        # Mock dependencies
        mock_get_collection.return_value = self.mock_collection
        mock_get_companies.return_value = ["Test Company"]
        mock_get_doctype.return_value = ["unknown"]  # This will be ignored

        # Make request
        response = self.client.post(
            "/chroma/query", json={"query": "What is Test Company's revenue?", "n_results": 5}
        )

        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert "documents" in response_data
        assert "ids" in response_data
        assert "metadatas" in response_data
        assert len(response_data["documents"]) == 2

        # Verify the collection was called with single filter (no $and)
        self.mock_collection.query.assert_called()
        call_args = self.mock_collection.query.call_args
        expected_filter = {"company_name": {"$in": ["Test Company"]}}
        assert call_args[1]["where"] == expected_filter

    @patch("app.main.get_chroma_collection")
    @patch("app.chroma_utils.get_companies_from_query")
    @patch("app.chroma_utils.get_doctype_from_query")
    def test_single_doctype_filter_returns_200(
        self, mock_get_doctype, mock_get_companies, mock_get_collection
    ):
        """Test that single doctype filter (previously failing) returns 200."""
        # Mock dependencies
        mock_get_collection.return_value = self.mock_collection
        mock_get_companies.return_value = []  # No company matches
        mock_get_doctype.return_value = ["financial"]

        # Make request
        response = self.client.post(
            "/chroma/query", json={"query": "Show me financial documents", "n_results": 5}
        )

        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert "documents" in response_data
        assert len(response_data["documents"]) == 2

        # Verify the collection was called with single filter (no $and)
        self.mock_collection.query.assert_called()
        call_args = self.mock_collection.query.call_args
        expected_filter = {"file_type": {"$in": ["financial"]}}
        assert call_args[1]["where"] == expected_filter

    @patch("app.main.get_chroma_collection")
    @patch("app.chroma_utils.get_companies_from_query")
    @patch("app.chroma_utils.get_doctype_from_query")
    def test_multiple_filters_use_and_returns_200(
        self, mock_get_doctype, mock_get_companies, mock_get_collection
    ):
        """Test that multiple filters correctly use $and and return 200."""
        # Mock dependencies
        mock_get_collection.return_value = self.mock_collection
        mock_get_companies.return_value = ["Test Company"]
        mock_get_doctype.return_value = ["financial"]

        # Make request
        response = self.client.post(
            "/chroma/query",
            json={"query": "What is Test Company's financial information?", "n_results": 5},
        )

        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert "documents" in response_data
        assert len(response_data["documents"]) == 2

        # Verify the collection was called with $and filter
        self.mock_collection.query.assert_called()
        call_args = self.mock_collection.query.call_args
        expected_filter = {
            "$and": [
                {"company_name": {"$in": ["Test Company"]}},
                {"file_type": {"$in": ["financial"]}},
            ]
        }
        assert call_args[1]["where"] == expected_filter

    @patch("app.main.get_chroma_collection")
    @patch("app.chroma_utils.get_companies_from_query")
    @patch("app.chroma_utils.get_doctype_from_query")
    def test_no_filters_returns_200(
        self, mock_get_doctype, mock_get_companies, mock_get_collection
    ):
        """Test that queries with no filters return 200."""
        # Mock dependencies
        mock_get_collection.return_value = self.mock_collection
        mock_get_companies.return_value = []
        mock_get_doctype.return_value = ["unknown"]

        # Make request
        response = self.client.post(
            "/chroma/query", json={"query": "General information query", "n_results": 5}
        )

        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert "documents" in response_data
        assert len(response_data["documents"]) == 2

        # Verify the collection was called with no filter
        self.mock_collection.query.assert_called()
        call_args = self.mock_collection.query.call_args
        assert call_args[1]["where"] is None

    def test_missing_query_returns_422(self):
        """Test that requests missing required query field return 422."""
        response = self.client.post(
            "/chroma/query",
            json={
                "n_results": 5
                # Missing "query" field
            },
        )

        assert response.status_code == 422

    def test_invalid_json_returns_422(self):
        """Test that invalid JSON returns 422."""
        response = self.client.post(
            "/chroma/query", data="invalid json", headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422


class TestChromaMetaEndpoints:
    """Integration tests for the new metadata collection endpoints."""

    def setup_method(self):
        """Set up test client and mocks."""
        self.client = TestClient(app)

        # Mock the metadata collection
        self.mock_meta_collection = Mock()
        self.mock_meta_collection.query.return_value = {
            "ids": [["doc1.txt", "doc2.txt"]],
            "documents": [["Company A - financial - 2023", "Company B - non-financial - 2022"]],
            "metadatas": [
                [
                    {
                        "filename": "doc1.txt",
                        "company": "Company A",
                        "period": "2023",
                        "type": "financial",
                    },
                    {
                        "filename": "doc2.txt",
                        "company": "Company B",
                        "period": "2022",
                        "type": "non-financial",
                    },
                ]
            ],
        }

    @patch("app.main.get_meta_collection")
    @patch("app.chroma_utils.add_documents")
    def test_meta_update_returns_200(self, mock_add_documents, mock_get_meta_collection):
        """Test that meta update endpoint returns 200."""
        # Mock dependencies
        mock_get_meta_collection.return_value = self.mock_meta_collection
        mock_add_documents.return_value = ["doc1.txt", "doc2.txt"]

        # Test data
        request_data = {
            "documents": [
                {
                    "filename": "doc1.txt",
                    "company": "Company A",
                    "period": "2023",
                    "type": "financial",
                    "description": "Company A - financial - 2023",
                },
                {
                    "filename": "doc2.txt",
                    "company": "Company B",
                    "period": "2022",
                    "type": "non-financial",
                    "description": "Company B - non-financial - 2022",
                },
            ]
        }

        # Make request
        response = self.client.post("/chroma/meta/update", json=request_data)

        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        assert len(response_data["ids"]) == 2

        # Verify add_documents was called correctly
        mock_add_documents.assert_called_once()
        call_args = mock_add_documents.call_args
        assert len(call_args[0][1]) == 2  # documents list
        assert len(call_args[0][2]) == 2  # metadatas list
        assert len(call_args[0][3]) == 2  # ids list

    @patch("app.main.get_meta_collection")
    @patch("app.chroma_utils.query_collection")
    def test_meta_query_returns_200(self, mock_query_collection, mock_get_meta_collection):
        """Test that meta query endpoint returns 200."""
        # Mock dependencies
        mock_get_meta_collection.return_value = self.mock_meta_collection
        mock_query_collection.return_value = (
            [
                ({"filename": "doc1.txt", "company": "Company A"}, "Company A - financial - 2023"),
                (
                    {"filename": "doc2.txt", "company": "Company B"},
                    "Company B - non-financial - 2022",
                ),
            ],
            "context",
        )

        # Test data
        request_data = {"query": "financial reports for Company A", "n_results": 5}

        # Make request
        response = self.client.post("/chroma/meta/query", json=request_data)

        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert "ids" in response_data
        assert "documents" in response_data
        assert "metadatas" in response_data
        assert len(response_data["documents"]) == 2

    def test_meta_update_missing_required_fields_returns_422(self):
        """Test that meta update with missing fields returns 422."""
        request_data = {
            "documents": [
                {
                    "filename": "doc1.txt",
                    # Missing required fields
                }
            ]
        }

        response = self.client.post("/chroma/meta/update", json=request_data)
        assert response.status_code == 422

    def test_meta_query_missing_query_returns_422(self):
        """Test that meta query without query field returns 422."""
        request_data = {
            "n_results": 5
            # Missing "query" field
        }

        response = self.client.post("/chroma/meta/query", json=request_data)
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
