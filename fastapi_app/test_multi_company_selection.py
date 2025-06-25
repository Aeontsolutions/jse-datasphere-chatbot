"""
Unit tests for multi-company semantic document selection.

These tests verify that the refactored semantic_document_selection function correctly handles:
1. Multi-company queries (e.g., "Access vs NCB")
2. Single-company queries (backward compatibility)
3. Fallback behavior when no companies are detected
4. Deduplication of results
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import sys
import json

# Add the fastapi_app directory to Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.utils import semantic_document_selection


class TestMultiCompanySemanticSelection:
    """Test the multi-company document selection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Sample metadata for testing
        self.sample_metadata = [
            {
                "company": "Access Bank Plc",
                "filename": "access_annual_2022.txt",
                "document_link": "s3://bucket/access_annual_2022.txt",
                "year": "2022",
                "document_type": "Annual Report"
            },
            {
                "company": "Access Bank Plc",
                "filename": "access_quarterly_q2_2022.txt",
                "document_link": "s3://bucket/access_quarterly_q2_2022.txt",
                "year": "2022",
                "document_type": "Quarterly Report"
            },
            {
                "company": "NCB Financial Group Limited",
                "filename": "ncb_annual_2022.txt",
                "document_link": "s3://bucket/ncb_annual_2022.txt",
                "year": "2022",
                "document_type": "Annual Report"
            },
            {
                "company": "NCB Financial Group Limited",
                "filename": "ncb_quarterly_q2_2022.txt",
                "document_link": "s3://bucket/ncb_quarterly_q2_2022.txt",
                "year": "2022",
                "document_type": "Quarterly Report"
            },
            {
                "company": "Barita Investments Limited",
                "filename": "barita_annual_2022.txt",
                "document_link": "s3://bucket/barita_annual_2022.txt",
                "year": "2022",
                "document_type": "Annual Report"
            }
        ]
    
    @patch('app.utils.get_companies_from_query')
    @patch('app.utils.GenerativeModel')
    def test_multi_company_query_access_vs_ncb(self, mock_model_class, mock_get_companies):
        """Test that a multi-company query returns documents for both companies."""
        # Setup: Mock company detection to return both Access and NCB
        mock_get_companies.return_value = ["Access Bank Plc", "NCB Financial Group Limited"]
        
        # Mock LLM responses for each company
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        # Mock responses for Access Bank documents
        access_response = MagicMock()
        access_response.text = json.dumps([
            {
                "company": "Access Bank Plc",
                "document_link": "s3://bucket/access_annual_2022.txt",
                "filename": "access_annual_2022.txt",
                "reason": "Annual report contains comprehensive financial data for 2022"
            }
        ])
        
        # Mock responses for NCB documents
        ncb_response = MagicMock()
        ncb_response.text = json.dumps([
            {
                "company": "NCB Financial Group Limited",
                "document_link": "s3://bucket/ncb_annual_2022.txt",
                "filename": "ncb_annual_2022.txt",
                "reason": "Annual report for comparison with Access Bank"
            }
        ])
        
        # Set up mock to return different responses for different calls
        mock_model.generate_content.side_effect = [access_response, ncb_response]
        
        # Execute
        result = semantic_document_selection(
            "How did Access Bank compare to NCB in 2022?",
            self.sample_metadata
        )
        
        # Verify: Should detect both companies
        mock_get_companies.assert_called_once_with("How did Access Bank compare to NCB in 2022?")
        
        # Verify: Should return documents for both companies
        assert result is not None
        assert "companies_mentioned" in result
        assert "documents_to_load" in result
        
        assert "Access Bank Plc" in result["companies_mentioned"]
        assert "NCB Financial Group Limited" in result["companies_mentioned"]
        
        # Check that documents from both companies are included
        filenames = [doc["filename"] for doc in result["documents_to_load"]]
        assert "access_annual_2022.txt" in filenames
        assert "ncb_annual_2022.txt" in filenames
        
        # Verify both companies had their documents processed
        assert mock_model.generate_content.call_count == 2
    
    @patch('app.utils.get_companies_from_query')
    @patch('app.utils.GenerativeModel')
    def test_single_company_query_backward_compatibility(self, mock_model_class, mock_get_companies):
        """Test that single-company queries still work (backward compatibility)."""
        # Setup: Mock company detection to return only Access
        mock_get_companies.return_value = ["Access Bank Plc"]
        
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        # Mock response for Access Bank documents
        access_response = MagicMock()
        access_response.text = json.dumps([
            {
                "company": "Access Bank Plc",
                "document_link": "s3://bucket/access_annual_2022.txt",
                "filename": "access_annual_2022.txt",
                "reason": "Annual report contains earnings information"
            }
        ])
        
        mock_model.generate_content.return_value = access_response
        
        # Execute
        result = semantic_document_selection(
            "How much did Access earn in 2022?",
            self.sample_metadata
        )
        
        # Verify: Should work as before for single company
        assert result is not None
        assert result["companies_mentioned"] == ["Access Bank Plc"]
        assert len(result["documents_to_load"]) == 1
        assert result["documents_to_load"][0]["filename"] == "access_annual_2022.txt"
    
    @patch('app.utils.get_companies_from_query')
    @patch('app.utils.GenerativeModel')
    def test_no_companies_detected_fallback(self, mock_model_class, mock_get_companies):
        """Test fallback behavior when no companies are detected."""
        # Setup: Mock company detection to return empty list
        mock_get_companies.return_value = []
        
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        # Mock fallback response
        fallback_response = MagicMock()
        fallback_response.text = json.dumps({
            "companies_mentioned": ["Access Bank Plc", "NCB Financial Group Limited"],
            "documents_to_load": [
                {
                    "company": "Access Bank Plc",
                    "document_link": "s3://bucket/access_annual_2022.txt",
                    "filename": "access_annual_2022.txt",
                    "reason": "Relevant financial document"
                },
                {
                    "company": "NCB Financial Group Limited",
                    "document_link": "s3://bucket/ncb_annual_2022.txt",
                    "filename": "ncb_annual_2022.txt",
                    "reason": "Comparative financial document"
                }
            ]
        })
        
        mock_model.generate_content.return_value = fallback_response
        
        # Execute
        result = semantic_document_selection(
            "What are the financial trends in banking?",
            self.sample_metadata
        )
        
        # Verify: Should use fallback approach
        assert result is not None
        assert "companies_mentioned" in result
        assert "documents_to_load" in result
        
        # Should use broader search (called once for fallback, not per company)
        assert mock_model.generate_content.call_count == 1
    
    @patch('app.utils.get_companies_from_query')
    @patch('app.utils.GenerativeModel')
    def test_deduplication_of_results(self, mock_model_class, mock_get_companies):
        """Test that duplicate documents are removed from results."""
        # Setup: Mock company detection
        mock_get_companies.return_value = ["Access Bank Plc"]
        
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        # Mock response with duplicate filenames
        duplicate_response = MagicMock()
        duplicate_response.text = json.dumps([
            {
                "company": "Access Bank Plc",
                "document_link": "s3://bucket/access_annual_2022.txt",
                "filename": "access_annual_2022.txt",
                "reason": "First occurrence"
            },
            {
                "company": "Access Bank Plc",
                "document_link": "s3://bucket/access_annual_2022.txt",
                "filename": "access_annual_2022.txt",
                "reason": "Duplicate occurrence"
            }
        ])
        
        mock_model.generate_content.return_value = duplicate_response
        
        # Execute
        result = semantic_document_selection(
            "Access Bank information",
            self.sample_metadata
        )
        
        # Verify: Should deduplicate based on filename
        assert result is not None
        assert len(result["documents_to_load"]) == 1
        assert result["documents_to_load"][0]["filename"] == "access_annual_2022.txt"
    
    @patch('app.utils.get_companies_from_query')
    @patch('app.utils.GenerativeModel')
    def test_company_not_in_metadata(self, mock_model_class, mock_get_companies):
        """Test behavior when detected company has no documents in metadata."""
        # Setup: Mock company detection to return a company not in metadata
        mock_get_companies.return_value = ["Unknown Company"]
        
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        # Mock fallback response
        fallback_response = MagicMock()
        fallback_response.text = json.dumps({
            "companies_mentioned": [],
            "documents_to_load": []
        })
        
        mock_model.generate_content.return_value = fallback_response
        
        # Execute
        result = semantic_document_selection(
            "Tell me about Unknown Company",
            self.sample_metadata
        )
        
        # Verify: Should fall back to broader search
        assert result is not None
        # Should not process the unknown company, should go to fallback
        assert mock_model.generate_content.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])