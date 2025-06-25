"""
Unit tests for chroma_utils.py focusing on the refactored filename-only filtering.

These tests verify that the query_collection function correctly handles:
1. No metadata filters (where=None)
2. Explicit filename filters (where parameter)
3. Fallback behavior when filtered queries return no results
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Add the fastapi_app directory to Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.chroma_utils import query_collection


class TestQueryCollectionFilenameFiltering:
    """Test the refactored filename-only filtering logic for ChromaDB queries."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock collection object
        self.mock_collection = Mock()
        self.mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Document 1 content", "Document 2 content"]],
            "metadatas": [[{"year": "2023", "filename": "test1.pdf"}, {"year": "2022", "filename": "test2.pdf"}]]
        }
        
    def test_no_where_filter(self):
        """Test that no where filter results in where_filter=None."""
        # Execute
        query_collection(self.mock_collection, "test query", n_results=5)
        
        # Verify: Should call collection.query with where=None
        self.mock_collection.query.assert_called()
        call_args = self.mock_collection.query.call_args
        assert call_args[1]['where'] is None
        
    def test_explicit_filename_filter(self):
        """Test that explicit filename filter is passed through correctly."""
        filename_filter = {"filename": {"$in": ["test.pdf", "report.pdf"]}}
        
        # Execute
        query_collection(self.mock_collection, "test query", n_results=5, where=filename_filter)
        
        # Verify: Should call collection.query with the exact filter provided
        self.mock_collection.query.assert_called()
        call_args = self.mock_collection.query.call_args
        assert call_args[1]['where'] == filename_filter
        
    def test_explicit_custom_filter(self):
        """Test that any custom filter is passed through correctly."""
        custom_filter = {"custom_field": {"$eq": "custom_value"}}
        
        # Execute
        query_collection(self.mock_collection, "test query", n_results=5, where=custom_filter)
        
        # Verify: Should use the explicit filter as-is
        self.mock_collection.query.assert_called()
        call_args = self.mock_collection.query.call_args
        assert call_args[1]['where'] == custom_filter
        
    def test_fallback_behavior_with_filter(self):
        """Test fallback behavior when filtered query returns no results."""
        filename_filter = {"filename": {"$in": ["nonexistent.pdf"]}}
        
        # Mock collection to return no results initially, then results on fallback
        self.mock_collection.query.side_effect = [
            {"ids": [[]], "documents": [[]], "metadatas": [[]]},  # No results with filter
            {"ids": [["doc1"]], "documents": [["Content"]], "metadatas": [[{"year": "2023", "filename": "test.pdf"}]]}  # Results without filter
        ]
        
        # Execute
        query_collection(self.mock_collection, "test query", n_results=5, where=filename_filter)
        
        # Verify: Should make two calls - first with filter, then without
        assert self.mock_collection.query.call_count == 2
        
        # First call should have the filter
        first_call = self.mock_collection.query.call_args_list[0]
        assert first_call[1]['where'] == filename_filter
        
        # Second call should have no filter (fallback)
        second_call = self.mock_collection.query.call_args_list[1]
        # The fallback call omits the where parameter entirely
        assert 'where' not in second_call[1] or second_call[1]['where'] is None
        
    def test_no_fallback_when_no_filter_provided(self):
        """Test that no fallback occurs when no filter is provided and no results found."""
        # Mock collection to return no results
        self.mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]]
        }
        
        # Execute
        query_collection(self.mock_collection, "test query", n_results=5)
        
        # Verify: Should make only one call since no filter was provided
        assert self.mock_collection.query.call_count == 1
        call_args = self.mock_collection.query.call_args
        assert call_args[1]['where'] is None


class TestQueryCollectionSorting:
    """Test the document sorting functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_collection = Mock()
        
    def test_document_sorting_by_year(self):
        """Test that documents are sorted by year in descending order."""
        # Setup: Return documents with different years
        self.mock_collection.query.return_value = {
            "ids": [["doc1", "doc2", "doc3"]],
            "documents": [["2021 content", "2023 content", "2022 content"]],
            "metadatas": [[
                {"year": "2021", "filename": "old.pdf"},
                {"year": "2023", "filename": "new.pdf"},
                {"year": "2022", "filename": "mid.pdf"}
            ]]
        }
        
        # Execute
        sorted_results, context = query_collection(self.mock_collection, "test query", n_results=5)
        
        # Verify: Results should be sorted by year (2023, 2022, 2021)
        assert len(sorted_results) == 3
        assert sorted_results[0][0]["year"] == "2023"  # Most recent first
        assert sorted_results[1][0]["year"] == "2022"
        assert sorted_results[2][0]["year"] == "2021"  # Oldest last
        
        # Verify context string is properly formed
        assert "2023 content" in context
        assert "2022 content" in context
        assert "2021 content" in context


class TestQueryCollectionRobustness:
    """Test robustness for multiple sequential queries."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_collection = Mock()
        
    def test_multiple_sequential_queries(self):
        """Test that multiple queries work correctly without side effects."""
        # Setup: Different responses for different queries
        self.mock_collection.query.side_effect = [
            {
                "ids": [["doc1"]],
                "documents": [["First query content"]],
                "metadatas": [[{"year": "2023", "filename": "first.pdf"}]]
            },
            {
                "ids": [["doc2"]],
                "documents": [["Second query content"]],
                "metadatas": [[{"year": "2022", "filename": "second.pdf"}]]
            },
            {
                "ids": [["doc3"]],
                "documents": [["Third query content"]],
                "metadatas": [[{"year": "2021", "filename": "third.pdf"}]]
            }
        ]
        
        # Execute: Multiple queries in sequence
        result1, context1 = query_collection(self.mock_collection, "first query", n_results=5)
        result2, context2 = query_collection(self.mock_collection, "second query", n_results=5)
        result3, context3 = query_collection(self.mock_collection, "third query", n_results=5)
        
        # Verify: Each query should work independently
        assert len(result1) == 1
        assert "First query content" in context1
        assert result1[0][0]["filename"] == "first.pdf"
        
        assert len(result2) == 1
        assert "Second query content" in context2
        assert result2[0][0]["filename"] == "second.pdf"
        
        assert len(result3) == 1
        assert "Third query content" in context3
        assert result3[0][0]["filename"] == "third.pdf"
        
        # Verify all three calls were made
        assert self.mock_collection.query.call_count == 3
        
    def test_query_with_changing_filters(self):
        """Test queries with different filename filters work correctly."""
        # Setup: Return different results based on the filter
        def mock_query_side_effect(*args, **kwargs):
            where_filter = kwargs.get('where')
            if where_filter and 'filename' in where_filter:
                filenames = where_filter['filename']['$in']
                if 'test1.pdf' in filenames:
                    return {
                        "ids": [["doc1"]],
                        "documents": [["Test1 content"]],
                        "metadatas": [[{"year": "2023", "filename": "test1.pdf"}]]
                    }
                elif 'test2.pdf' in filenames:
                    return {
                        "ids": [["doc2"]],
                        "documents": [["Test2 content"]],
                        "metadatas": [[{"year": "2022", "filename": "test2.pdf"}]]
                    }
            return {"ids": [[]], "documents": [[]], "metadatas": [[]]}
        
        self.mock_collection.query.side_effect = mock_query_side_effect
        
        # Execute: Different filename filters
        filter1 = {"filename": {"$in": ["test1.pdf"]}}
        filter2 = {"filename": {"$in": ["test2.pdf"]}}
        
        result1, context1 = query_collection(self.mock_collection, "query", n_results=5, where=filter1)
        result2, context2 = query_collection(self.mock_collection, "query", n_results=5, where=filter2)
        
        # Verify: Each filter works correctly
        assert "Test1 content" in context1
        assert result1[0][0]["filename"] == "test1.pdf"
        
        assert "Test2 content" in context2
        assert result2[0][0]["filename"] == "test2.pdf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])