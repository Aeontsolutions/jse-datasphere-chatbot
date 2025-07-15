"""
Unit tests for the metadata collection functionality.

These tests verify:
1. query_meta_collection function 
2. New API endpoints for metadata collection
3. Embedding-based semantic document selection
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Add the fastapi_app directory to Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.chroma_utils import query_meta_collection
from app.utils import semantic_document_selection


class TestQueryMetaCollection:
    """Test the query_meta_collection function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_collection = Mock()
        
        # Mock successful query response
        self.mock_collection.query.return_value = {
            "ids": [["doc1.txt", "doc2.txt"]],
            "documents": [["Company A - financial - 2023", "Company B - non-financial - 2022"]],
            "metadatas": [[
                {
                    "filename": "doc1.txt",
                    "company": "Company A", 
                    "period": "2023",
                    "type": "financial"
                },
                {
                    "filename": "doc2.txt",
                    "company": "Company B",
                    "period": "2022", 
                    "type": "non-financial"
                }
            ]]
        }
    
    def test_query_meta_collection_success(self):
        """Test successful query to metadata collection."""
        result = query_meta_collection(
            meta_collection=self.mock_collection,
            query="financial reports for Company A",
            n_results=5
        )
        
        # Verify the collection was queried
        self.mock_collection.query.assert_called_once()
        call_args = self.mock_collection.query.call_args
        assert call_args[1]['query_texts'] == ["financial reports for Company A"]
        assert call_args[1]['n_results'] == 5
        
        # Verify the result format
        assert result is not None
        assert "companies_mentioned" in result
        assert "documents_to_load" in result
        
        # Check companies
        assert "Company A" in result["companies_mentioned"]
        assert "Company B" in result["companies_mentioned"]
        
        # Check documents
        assert len(result["documents_to_load"]) == 2
        doc1 = result["documents_to_load"][0]
        assert doc1["filename"] == "doc1.txt"
        assert doc1["company"] == "Company A"
    
    def test_query_meta_collection_with_conversation_history(self):
        """Test query with conversation history enhancement."""
        conversation_history = [
            {"role": "user", "content": "Tell me about Company A"},
            {"role": "assistant", "content": "Here's info about Company A"},
            {"role": "user", "content": "What about their financials?"}
        ]
        
        query_meta_collection(
            meta_collection=self.mock_collection,
            query="latest report",
            conversation_history=conversation_history
        )
        
        # Verify query was enhanced with conversation context
        call_args = self.mock_collection.query.call_args
        query_text = call_args[1]['query_texts'][0]
        assert "Tell me about Company A" in query_text
        assert "What about their financials?" in query_text
        assert "latest report" in query_text
    
    def test_query_meta_collection_no_results(self):
        """Test behavior when no results are returned."""
        self.mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]]
        }
        
        result = query_meta_collection(
            meta_collection=self.mock_collection,
            query="nonexistent company"
        )
        
        assert result is None
    
    def test_query_meta_collection_with_where_filter(self):
        """Test query with metadata filter."""
        where_filter = {"company": "Company A"}
        
        query_meta_collection(
            meta_collection=self.mock_collection,
            query="financial report",
            where=where_filter
        )
        
        # Verify filter was passed through
        call_args = self.mock_collection.query.call_args
        assert call_args[1]['where'] == where_filter


class TestSemanticDocumentSelectionEmbedding:
    """Test the refactored semantic_document_selection function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_meta_collection = Mock()
        self.sample_metadata = [
            {
                "company_name": "Company A",
                "documents": [
                    {
                        "filename": "company_a_2023.pdf",
                        "type": "financial",
                        "period": "2023"
                    }
                ]
            }
        ]
    
    def test_semantic_selection_fallback_no_meta_collection(self):
        """Test fallback to LLM when no meta_collection is provided."""
        with patch('app.utils.semantic_document_selection_llm_fallback') as mock_llm_fallback:
            mock_llm_fallback.return_value = {
                "companies_mentioned": ["Company A"],
                "documents_to_load": [{"filename": "doc.pdf"}]
            }
            
            result = semantic_document_selection(
                query="Show me Company A financials",
                metadata=self.sample_metadata,
                meta_collection=None  # No meta collection
            )
            
            # Verify only LLM approach was used
            mock_llm_fallback.assert_called_once()
            
            assert result == mock_llm_fallback.return_value
    
    def test_semantic_selection_with_meta_collection_integration(self):
        """Integration test with mock meta collection."""
        # Mock successful query response in the collection
        self.mock_meta_collection.query.return_value = {
            "ids": [["doc1.txt"]],
            "documents": [["Company A - financial - 2023"]],
            "metadatas": [[
                {
                    "filename": "doc1.txt",
                    "company": "Company A", 
                    "period": "2023",
                    "type": "financial"
                }
            ]]
        }
        
        # Mock get_companies_from_query to return empty list (no companies detected)
        with patch('app.utils.get_companies_from_query') as mock_get_companies:
            mock_get_companies.return_value = []  # No companies detected, should use broader search
            
            # Mock query_meta_collection to return the expected result
            with patch('app.utils.query_meta_collection') as mock_query_meta:
                def mock_query_meta_side_effect(meta_collection, query, n_results=5, where=None, conversation_history=None):
                    # Should be called with n_results=15 for broader search when no companies detected
                    results = self.mock_meta_collection.query.return_value
                    metadatas = results["metadatas"][0] 
                    companies_mentioned = list(set(meta.get("company", "") for meta in metadatas if meta.get("company")))
                    documents_to_load = []
                    
                    for meta in metadatas:
                        if meta.get("filename") and meta.get("company"):
                            documents_to_load.append({
                                "company": meta["company"],
                                "document_link": f"s3://document-path/{meta['filename']}",
                                "filename": meta["filename"],
                                "reason": f"Relevant {meta.get('type', 'document')} for {meta.get('period', 'period')}"
                            })
                    
                    return {
                        "companies_mentioned": companies_mentioned,
                        "documents_to_load": documents_to_load
                    }
                
                mock_query_meta.side_effect = mock_query_meta_side_effect
                
                result = semantic_document_selection(
                    query="Show me Company A financials",
                    metadata=self.sample_metadata,
                    meta_collection=self.mock_meta_collection
                )
                
                # Verify result format
                assert result is not None
                assert "companies_mentioned" in result
                assert "documents_to_load" in result
                assert len(result["documents_to_load"]) == 1
                assert result["companies_mentioned"] == ["Company A"]
    
    def test_semantic_selection_multi_company_query(self):
        """Test that multi-company queries return documents from all mentioned companies."""
        # Mock the get_companies_from_query function
        with patch('app.utils.get_companies_from_query') as mock_get_companies:
            mock_get_companies.return_value = ["Access Bank", "NCB"]
            
            # Mock the meta collection to return different results for each company
            def mock_query_side_effect(*args, **kwargs):
                where_clause = kwargs.get('where', {})
                if where_clause.get('company', {}).get('$eq') == "Access Bank":
                    return {
                        "ids": [["access_doc.txt"]],
                        "documents": [["Access Bank - financial - 2023"]],
                        "metadatas": [[{
                            "filename": "access_doc.txt",
                            "company": "Access Bank",
                            "period": "2023",
                            "type": "financial"
                        }]]
                    }
                elif where_clause.get('company', {}).get('$eq') == "NCB":
                    return {
                        "ids": [["ncb_doc.txt"]],
                        "documents": [["NCB - financial - 2023"]],
                        "metadatas": [[{
                            "filename": "ncb_doc.txt", 
                            "company": "NCB",
                            "period": "2023",
                            "type": "financial"
                        }]]
                    }
                else:
                    return {"ids": [[]], "documents": [[]], "metadatas": [[]]}
            
            self.mock_meta_collection.query.side_effect = mock_query_side_effect
            
            # Mock query_meta_collection to call the actual function logic but with mocked collection
            with patch('app.utils.query_meta_collection') as mock_query_meta:
                def mock_query_meta_side_effect(meta_collection, query, n_results=5, where=None, conversation_history=None):
                    # Simulate what query_meta_collection would return
                    results = mock_query_side_effect(query_texts=[query], n_results=n_results, where=where)
                    if not results or not results.get("metadatas") or not results["metadatas"][0]:
                        return None
                    
                    metadatas = results["metadatas"][0] 
                    companies_mentioned = list(set(meta.get("company", "") for meta in metadatas if meta.get("company")))
                    documents_to_load = []
                    
                    for meta in metadatas:
                        if meta.get("filename") and meta.get("company"):
                            documents_to_load.append({
                                "company": meta["company"],
                                "document_link": f"s3://document-path/{meta['filename']}",
                                "filename": meta["filename"],
                                "reason": f"Relevant {meta.get('type', 'document')} for {meta.get('period', 'period')}"
                            })
                    
                    return {
                        "companies_mentioned": companies_mentioned,
                        "documents_to_load": documents_to_load
                    }
                
                mock_query_meta.side_effect = mock_query_meta_side_effect
                
                result = semantic_document_selection(
                    query="How did Access Bank compare to NCB in 2023?",
                    metadata=self.sample_metadata,
                    meta_collection=self.mock_meta_collection
                )
                
                # Verify result contains documents from both companies
                assert result is not None
                assert "companies_mentioned" in result
                assert "documents_to_load" in result
                
                # Should have documents from both companies
                assert len(result["documents_to_load"]) == 2
                companies_in_results = [doc["company"] for doc in result["documents_to_load"]]
                assert "Access Bank" in companies_in_results
                assert "NCB" in companies_in_results
                
                # Verify companies mentioned includes both
                assert set(result["companies_mentioned"]) >= {"Access Bank", "NCB"}
    
    def test_semantic_selection_deduplication(self):
        """Test that duplicate documents are properly deduplicated."""
        with patch('app.utils.get_companies_from_query') as mock_get_companies:
            mock_get_companies.return_value = ["Company A", "Company B"]
            
            # Mock to return the same document for both companies (simulating duplication)
            def mock_query_side_effect(*args, **kwargs):
                return {
                    "ids": [["shared_doc.txt"]],
                    "documents": [["Shared document"]],
                    "metadatas": [[{
                        "filename": "shared_doc.txt",
                        "company": kwargs.get('where', {}).get('company', {}).get('$eq', 'Unknown'),
                        "period": "2023",
                        "type": "financial"
                    }]]
                }
            
            self.mock_meta_collection.query.side_effect = mock_query_side_effect
            
            with patch('app.utils.query_meta_collection') as mock_query_meta:
                def mock_query_meta_side_effect(meta_collection, query, n_results=5, where=None, conversation_history=None):
                    results = mock_query_side_effect(query_texts=[query], n_results=n_results, where=where)
                    if not results or not results.get("metadatas") or not results["metadatas"][0]:
                        return None
                    
                    metadatas = results["metadatas"][0]
                    companies_mentioned = list(set(meta.get("company", "") for meta in metadatas if meta.get("company")))
                    documents_to_load = []
                    
                    for meta in metadatas:
                        if meta.get("filename") and meta.get("company"):
                            documents_to_load.append({
                                "company": meta["company"],
                                "document_link": f"s3://document-path/{meta['filename']}",
                                "filename": meta["filename"],
                                "reason": f"Relevant {meta.get('type', 'document')} for {meta.get('period', 'period')}"
                            })
                    
                    return {
                        "companies_mentioned": companies_mentioned,
                        "documents_to_load": documents_to_load
                    }
                
                mock_query_meta.side_effect = mock_query_meta_side_effect
                
                result = semantic_document_selection(
                    query="Compare Company A and Company B",
                    metadata=self.sample_metadata,
                    meta_collection=self.mock_meta_collection
                )
                
                # Verify deduplication occurred - should only have one document despite two companies
                assert result is not None
                assert len(result["documents_to_load"]) == 1
                assert result["documents_to_load"][0]["filename"] == "shared_doc.txt"
    
    def test_semantic_selection_broader_fallback(self):
        """Test that broader search is used when no companies are detected."""
        with patch('app.utils.get_companies_from_query') as mock_get_companies:
            mock_get_companies.return_value = []  # No companies detected
            
            # Mock successful broad search
            self.mock_meta_collection.query.return_value = {
                "ids": [["doc1.txt", "doc2.txt"]],
                "documents": [["Generic doc 1", "Generic doc 2"]],
                "metadatas": [[
                    {"filename": "doc1.txt", "company": "Company X", "period": "2023"},
                    {"filename": "doc2.txt", "company": "Company Y", "period": "2022"}
                ]]
            }
            
            with patch('app.utils.query_meta_collection') as mock_query_meta:
                def mock_query_meta_side_effect(meta_collection, query, n_results=5, where=None, conversation_history=None):
                    # Should be called with n_results=15 for broader search
                    assert n_results == 15
                    assert where is None  # No company filter for broad search
                    
                    results = self.mock_meta_collection.query.return_value
                    metadatas = results["metadatas"][0]
                    companies_mentioned = list(set(meta.get("company", "") for meta in metadatas if meta.get("company")))
                    documents_to_load = []
                    
                    for meta in metadatas:
                        if meta.get("filename") and meta.get("company"):
                            documents_to_load.append({
                                "company": meta["company"],
                                "document_link": f"s3://document-path/{meta['filename']}",
                                "filename": meta["filename"],
                                "reason": f"Relevant document for {meta.get('period', 'period')}"
                            })
                    
                    return {
                        "companies_mentioned": companies_mentioned,
                        "documents_to_load": documents_to_load
                    }
                
                mock_query_meta.side_effect = mock_query_meta_side_effect
                
                result = semantic_document_selection(
                    query="Show me some financial reports",
                    metadata=self.sample_metadata,
                    meta_collection=self.mock_meta_collection
                )
                
                # Verify broader search was used
                assert result is not None
                assert len(result["documents_to_load"]) == 2