"""
Unit tests for chroma_utils.py focusing on the single-clause $and filter fix.

These tests verify that the query_collection function correctly handles:
1. Zero filter clauses (no metadata filters)
2. One filter clause (single metadata filter - should not use $and)
3. Two+ filter clauses (multiple metadata filters - should use $and)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Add the app directory to Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.chroma_utils import query_collection


class TestQueryCollectionFilterLogic:
    """Test the core filter logic for ChromaDB queries."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock collection object
        self.mock_collection = Mock()
        self.mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Document 1 content", "Document 2 content"]],
            "metadatas": [[{"year": "2023", "company_name": "Test Co"}, {"year": "2022", "company_name": "Test Co"}]]
        }
        
    @patch('app.chroma_utils.get_companies_from_query')
    @patch('app.chroma_utils.get_doctype_from_query')
    def test_zero_clause_filter(self, mock_get_doctype, mock_get_companies):
        """Test that zero filter clauses result in where_filter=None."""
        # Setup: No company matches, no document type
        mock_get_companies.return_value = []
        mock_get_doctype.return_value = ["unknown"]
        
        # Execute
        query_collection(self.mock_collection, "test query", n_results=5)
        
        # Verify: Should call collection.query with where=None
        self.mock_collection.query.assert_called()
        call_args = self.mock_collection.query.call_args
        assert call_args[1]['where'] is None
        
    @patch('app.chroma_utils.get_companies_from_query')
    @patch('app.chroma_utils.get_doctype_from_query')
    def test_one_clause_filter_company_only(self, mock_get_doctype, mock_get_companies):
        """Test that one filter clause (company only) is passed directly without $and."""
        # Setup: One company match, no document type
        mock_get_companies.return_value = ["Test Company"]
        mock_get_doctype.return_value = ["unknown"]
        
        # Execute
        query_collection(self.mock_collection, "test query", n_results=5)
        
        # Verify: Should call collection.query with single filter clause (no $and)
        self.mock_collection.query.assert_called()
        call_args = self.mock_collection.query.call_args
        expected_filter = {"company_name": {"$in": ["Test Company"]}}
        assert call_args[1]['where'] == expected_filter
        assert '$and' not in call_args[1]['where']
        
    @patch('app.chroma_utils.get_companies_from_query')
    @patch('app.chroma_utils.get_doctype_from_query')
    def test_one_clause_filter_doctype_only(self, mock_get_doctype, mock_get_companies):
        """Test that one filter clause (doctype only) is passed directly without $and."""
        # Setup: No company matches, one document type
        mock_get_companies.return_value = []
        mock_get_doctype.return_value = ["financial"]
        
        # Execute
        query_collection(self.mock_collection, "test query", n_results=5)
        
        # Verify: Should call collection.query with single filter clause (no $and)
        self.mock_collection.query.assert_called()
        call_args = self.mock_collection.query.call_args
        expected_filter = {"file_type": {"$in": ["financial"]}}
        assert call_args[1]['where'] == expected_filter
        assert '$and' not in call_args[1]['where']
        
    @patch('app.chroma_utils.get_companies_from_query')
    @patch('app.chroma_utils.get_doctype_from_query')
    def test_two_clause_filter_with_and(self, mock_get_doctype, mock_get_companies):
        """Test that two filter clauses are properly wrapped in $and."""
        # Setup: One company match and one document type
        mock_get_companies.return_value = ["Test Company"]
        mock_get_doctype.return_value = ["financial"]
        
        # Execute
        query_collection(self.mock_collection, "test query", n_results=5)
        
        # Verify: Should call collection.query with $and wrapping both clauses
        self.mock_collection.query.assert_called()
        call_args = self.mock_collection.query.call_args
        expected_filter = {
            "$and": [
                {"company_name": {"$in": ["Test Company"]}},
                {"file_type": {"$in": ["financial"]}}
            ]
        }
        assert call_args[1]['where'] == expected_filter
        assert '$and' in call_args[1]['where']
        assert len(call_args[1]['where']['$and']) == 2
        
    def test_explicit_where_filter_bypasses_logic(self):
        """Test that providing explicit 'where' parameter bypasses filter building logic."""
        explicit_filter = {"custom_field": {"$eq": "custom_value"}}
        
        # Execute
        query_collection(self.mock_collection, "test query", n_results=5, where=explicit_filter)
        
        # Verify: Should use the explicit filter as-is
        self.mock_collection.query.assert_called()
        call_args = self.mock_collection.query.call_args
        assert call_args[1]['where'] == explicit_filter
        
    @patch('app.chroma_utils.get_companies_from_query')
    @patch('app.chroma_utils.get_doctype_from_query')  
    def test_fallback_behavior_no_results(self, mock_get_doctype, mock_get_companies):
        """Test fallback behavior when no results are found."""
        # Setup: Both company and doctype filters
        mock_get_companies.return_value = ["Test Company"]
        mock_get_doctype.return_value = ["financial"]
        
        # Mock collection to return no results initially, then results on fallback
        self.mock_collection.query.side_effect = [
            {"ids": [[]], "documents": [[]], "metadatas": [[]]},  # No results with full filter
            {"ids": [["doc1"]], "documents": [["Content"]], "metadatas": [[{"year": "2023"}]]}  # Results with company-only
        ]
        
        # Execute
        query_collection(self.mock_collection, "test query", n_results=5)
        
        # Verify: Should make two calls - first with $and, then with company-only fallback
        assert self.mock_collection.query.call_count == 2
        
        # First call should have $and filter
        first_call = self.mock_collection.query.call_args_list[0]
        assert '$and' in first_call[1]['where']
        
        # Second call should have company-only filter
        second_call = self.mock_collection.query.call_args_list[1]
        assert second_call[1]['where'] == {"company_name": {"$in": ["Test Company"]}}


class TestQueryCollectionEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_collection = Mock()
        self.mock_collection.query.return_value = {
            "ids": [["doc1"]],
            "documents": [["Document content"]],
            "metadatas": [[{"year": "2023", "company_name": "Test Co"}]]
        }
        
    @patch('app.chroma_utils.get_companies_from_query')
    @patch('app.chroma_utils.get_doctype_from_query')
    def test_empty_company_list_handling(self, mock_get_doctype, mock_get_companies):
        """Test handling of empty company list."""
        mock_get_companies.return_value = []
        mock_get_doctype.return_value = ["financial"]
        
        query_collection(self.mock_collection, "test query", n_results=5)
        
        # Should only have doctype filter
        call_args = self.mock_collection.query.call_args
        expected_filter = {"file_type": {"$in": ["financial"]}}
        assert call_args[1]['where'] == expected_filter
        
    @patch('app.chroma_utils.get_companies_from_query')
    @patch('app.chroma_utils.get_doctype_from_query')
    def test_unknown_doctype_handling(self, mock_get_doctype, mock_get_companies):
        """Test handling of unknown document type."""
        mock_get_companies.return_value = ["Test Company"]
        mock_get_doctype.return_value = ["unknown"]
        
        query_collection(self.mock_collection, "test query", n_results=5)
        
        # Should only have company filter (unknown doctype ignored)
        call_args = self.mock_collection.query.call_args
        expected_filter = {"company_name": {"$in": ["Test Company"]}}
        assert call_args[1]['where'] == expected_filter


if __name__ == "__main__":
    pytest.main([__file__, "-v"])