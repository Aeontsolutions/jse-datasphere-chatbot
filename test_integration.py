"""
Integration test to verify multi-company functionality works end-to-end.
This test creates a mock scenario to validate the complete flow.
"""

import json
from unittest.mock import patch, MagicMock

# Simple integration test to validate our changes work
def test_multi_company_integration():
    """Test that multi-company queries work end-to-end"""
    
    # Import our function
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'fastapi_app'))
    
    from fastapi_app.app.utils import semantic_document_selection
    
    # Sample metadata
    metadata = [
        {
            "company": "Access Bank Plc",
            "filename": "access_2022.txt",
            "document_link": "s3://bucket/access_2022.txt",
            "year": "2022"
        },
        {
            "company": "NCB Financial Group Limited", 
            "filename": "ncb_2022.txt",
            "document_link": "s3://bucket/ncb_2022.txt",
            "year": "2022"
        }
    ]
    
    # Mock the dependencies
    with patch('fastapi_app.app.utils.get_companies_from_query') as mock_get_companies, \
         patch('fastapi_app.app.utils.GenerativeModel') as mock_model_class:
        
        # Setup mocks
        mock_get_companies.return_value = ["Access Bank Plc", "NCB Financial Group Limited"]
        
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        # Mock LLM responses for each company
        responses = [
            # Access Bank response
            MagicMock(text=json.dumps([{
                "company": "Access Bank Plc",
                "document_link": "s3://bucket/access_2022.txt", 
                "filename": "access_2022.txt",
                "reason": "Access Bank annual report for 2022"
            }])),
            # NCB response
            MagicMock(text=json.dumps([{
                "company": "NCB Financial Group Limited",
                "document_link": "s3://bucket/ncb_2022.txt",
                "filename": "ncb_2022.txt", 
                "reason": "NCB annual report for comparison"
            }]))
        ]
        
        mock_model.generate_content.side_effect = responses
        
        # Test the multi-company query
        result = semantic_document_selection(
            "How did Access Bank compare to NCB in 2022?",
            metadata
        )
        
        # Verify results
        assert result is not None
        assert "companies_mentioned" in result
        assert "documents_to_load" in result
        
        # Check both companies were detected
        companies = result["companies_mentioned"]
        assert "Access Bank Plc" in companies
        assert "NCB Financial Group Limited" in companies
        
        # Check documents for both companies were returned
        docs = result["documents_to_load"]
        filenames = [doc["filename"] for doc in docs]
        assert "access_2022.txt" in filenames
        assert "ncb_2022.txt" in filenames
        
        print("✅ Multi-company integration test passed!")
        print(f"Companies found: {companies}")
        print(f"Documents returned: {filenames}")
        
        return True

if __name__ == "__main__":
    test_multi_company_integration()